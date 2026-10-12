#include "common/hostException.h"

#include "common/logging/log.h"
#include <atomic>
#include <cstdio>
#include <cstring>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#else
#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <initializer_list>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/ucontext.h>
#else
#include <ucontext.h> // IWYU pragma: keep
#endif
#endif

// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <excpt.h>
// IWYU pragma: no_include <minwinbase.h>
// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <wtypes.h>

namespace Common::HostException {

#if !defined(__APPLE__)

static std::atomic<Handler> g_handler {nullptr};
static std::atomic_uint32_t g_install_state {0};

static_assert(decltype(g_handler)::is_always_lock_free);
static_assert(decltype(g_install_state)::is_always_lock_free);
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX

// macOS uses the same POSIX platform setting and needs a signal stack too.
class ThreadSignalStack {
public:
        ThreadSignalStack() {
                const auto page_size = static_cast<size_t>(::getpagesize());
                const auto stack_size =
                    (std::max<size_t>(64 * 1024, MINSIGSTKSZ) + page_size - 1) & ~(page_size - 1);
                if (::posix_memalign(&m_memory, page_size, stack_size) != 0) {
                        return;
                }

                stack_t stack {};
                stack.ss_sp   = m_memory;
                stack.ss_size = stack_size;
                if (::sigaltstack(&stack, &m_previous) != 0) {
                        std::free(m_memory);
                        m_memory = nullptr;
                }
        }

        ~ThreadSignalStack() {
                if (m_memory != nullptr && ::sigaltstack(&m_previous, nullptr) == 0) {
                        std::free(m_memory);
                }
        }

        [[nodiscard]] bool IsInitialized() const { return m_memory != nullptr; }

        KYTY_CLASS_NO_COPY(ThreadSignalStack)

private:
        void*   m_memory = nullptr;
        stack_t m_previous {};
};

bool InitializeThreadSignalStack() {
        // Keep fault handling off guest stacks, which GPU tracking can make read-only.
        thread_local ThreadSignalStack signal_stack;
        return signal_stack.IsInitialized();
}

#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

// Static 64 KB zero page used to redirect null-page faulting instructions.
// BSS-allocated, no VirtualAlloc dependency. Reads from this page return 0.
static uint8_t g_zero_page[65536] = {};

// Decode the faulting x86-64 instruction at RIP and identify the base
// register used for the memory operand. Returns the register index 0-15
// or -1 if the instruction can't be safely patched (e.g. RIP-relative,
// SIB without a base register, or register-direct operand).
static int FindFaultingBaseRegister(const uint8_t* code) {
        size_t i = 0;
        // Skip legacy prefixes.
        while (i < 15) {
                uint8_t b = code[i];
                if (b == 0x26 || b == 0x2e || b == 0x36 || b == 0x3e ||
                    b == 0x64 || b == 0x65 || b == 0x66 || b == 0x67 ||
                    b == 0xf0 || b == 0xf2 || b == 0xf3) {
                        i++;
                } else {
                        break;
                }
        }
        // Parse REX prefix.
        uint8_t rex = 0;
        if (i < 15 && (code[i] & 0xF0) == 0x40) {
                rex = code[i++];
        }
        const bool rex_b = (rex & 0x01) != 0;  // extends rm to r8-r15
        // Skip opcode byte (1 byte; this is a simplification that ignores
        // multi-byte 0F-prefixed opcodes, but for null-page faults the
        // important part is the ModR/M byte which follows the opcode).
        if (i >= 15) return -1;
        uint8_t opcode = code[i++];
        (void)opcode;
        // Handle 0F-prefixed two-byte opcodes.
        if (opcode == 0x0f) {
                if (i >= 15) return -1;
                opcode = code[i++];
                // Some 0F opcodes have an extra prefix byte (0x38, 0x3a) for
                // three-byte forms. Skip if present.
                if ((opcode == 0x38 || opcode == 0x3a) && i < 15) {
                        opcode = code[i++];
                }
        }
        // Read ModR/M byte.
        if (i >= 15) return -1;
        uint8_t modrm = code[i++];
        uint8_t mod = (modrm >> 6) & 0x03;
        uint8_t rm = modrm & 0x07;
        // mod == 3 means register-direct, no memory operand.
        if (mod == 0x03) return -1;
        // mod == 0 + rm == 5 means RIP-relative, no base register to patch.
        if (mod == 0x00 && rm == 0x05) return -1;
        // rm == 4 means SIB byte follows.
        if (rm == 0x04) {
                if (i >= 15) return -1;
                uint8_t sib = code[i++];
                uint8_t base = sib & 0x07;
                // SIB base == 5 with mod == 0 means disp32 only, no base register.
                if (mod == 0x00 && base == 0x05) return -1;
                // SIB base == 4 is RSP, which is the stack pointer. Don't patch
                // RSP — corrupting the stack would be much worse than the crash.
                if (base == 0x04) return -1;
                return base | (rex_b ? 0x08 : 0x00);
        }
        // rm is the base register (extended by REX.B if present).
        // Don't patch RSP (index 4) — same reason as SIB base.
        if (rm == 0x04) return -1;
        return rm | (rex_b ? 0x08 : 0x00);
}

static LONG WINAPI ExceptionFilter(PEXCEPTION_POINTERS exception) noexcept {
        auto* exception_record = exception->ExceptionRecord;

        if (exception_record->ExceptionCode == DBG_PRINTEXCEPTION_C ||
            exception_record->ExceptionCode == DBG_PRINTEXCEPTION_WIDE_C) {
                return EXCEPTION_CONTINUE_SEARCH;
        }

        if (exception_record->ExceptionCode == 0x406D1388) {
                // Set a thread name.
                return EXCEPTION_CONTINUE_EXECUTION;
        }

        ExceptionInfo info {};
        info.exception_address = reinterpret_cast<uint64_t>(exception_record->ExceptionAddress);
        info.native_code       = exception_record->ExceptionCode;
        info.native_context    = exception->ContextRecord;

        if (exception_record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
                info.type = ExceptionType::AccessViolation;
                switch (exception_record->ExceptionInformation[0]) {
                        case 0: info.access_violation_type = AccessViolationType::Read; break;
                        case 1: info.access_violation_type = AccessViolationType::Write; break;
                        case 8: info.access_violation_type = AccessViolationType::Execute; break;
                        default: info.access_violation_type = AccessViolationType::Unknown; break;
                }
                info.access_violation_vaddr = exception_record->ExceptionInformation[1];
        } else if (exception_record->ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION) {
                info.type = ExceptionType::IllegalInstruction;
        } else {
                return EXCEPTION_CONTINUE_SEARCH;
        }

        // Null-page recovery: when the guest shader walker or a C runtime helper
        // (vcruntime's std::sort, std::partition, etc.) dereferences a null+
        // small-offset pointer, instead of crashing the emulator via EXIT(),
        // redirect the faulting base register to the static g_zero_page (64 KB of
        // zeros in BSS) and resume execution. The faulting instruction then reads
        // 0 instead of crashing, the caller proceeds with degraded data, and the
        // game continues running. No bucket counter: an infinite loop here would
        // indicate a deeper bug and would just hang the game (the user can kill
        // the process). RSP is never patched to avoid corrupting the stack.
        if (info.type == ExceptionType::AccessViolation &&
            info.access_violation_vaddr < 0x10000ull &&
            info.access_violation_vaddr != 0) {
                const uint8_t* code = reinterpret_cast<const uint8_t*>(info.exception_address);
                int base_reg = FindFaultingBaseRegister(code);
                if (base_reg >= 0 && base_reg < 16 && base_reg != 4) {  // skip RSP
                        auto* ctx = exception->ContextRecord;
                        uint64_t* regs[] = {
                                &ctx->Rax, &ctx->Rcx, &ctx->Rdx, &ctx->Rbx,
                                &ctx->Rsp, &ctx->Rbp, &ctx->Rsi, &ctx->Rdi,
                                &ctx->R8,  &ctx->R9,  &ctx->R10, &ctx->R11,
                                &ctx->R12, &ctx->R13, &ctx->R14, &ctx->R15,
                        };
                        *regs[base_reg] = reinterpret_cast<uint64_t>(g_zero_page);
                        static std::atomic<uint32_t> recovery_log_count{0};
                        if (recovery_log_count.fetch_add(1, std::memory_order_relaxed) < 4) {
                                LOGF("NullPageRecovery: recovered access violation at pc=0x%016llx "
                                     "addr=0x%llx by redirecting reg%d to zero_page\n",
                                     (unsigned long long)info.exception_address,
                                     (unsigned long long)info.access_violation_vaddr, base_reg);
                        }
                        return EXCEPTION_CONTINUE_EXECUTION;
                }
        }

        info.rax = exception->ContextRecord->Rax;
        info.rbx = exception->ContextRecord->Rbx;
        info.rcx = exception->ContextRecord->Rcx;
        info.rdx = exception->ContextRecord->Rdx;
        info.rsi = exception->ContextRecord->Rsi;
        info.rdi = exception->ContextRecord->Rdi;
        info.rbp = exception->ContextRecord->Rbp;
        info.rsp = exception->ContextRecord->Rsp;
        info.r8  = exception->ContextRecord->R8;
        info.r9  = exception->ContextRecord->R9;
        info.r10 = exception->ContextRecord->R10;
        info.r11 = exception->ContextRecord->R11;
        info.r12 = exception->ContextRecord->R12;
        info.r13 = exception->ContextRecord->R13;
        info.r14 = exception->ContextRecord->R14;
        info.r15 = exception->ContextRecord->R15;

        const auto handler = g_handler.load(std::memory_order_acquire);
        if (handler != nullptr && handler(info)) {
                return EXCEPTION_CONTINUE_EXECUTION;
        }
        return EXCEPTION_CONTINUE_SEARCH;
}

#elif defined(__APPLE__)

static std::atomic<Handler> g_handler {nullptr};
static std::atomic_uint32_t g_install_state {0};

static_assert(decltype(g_handler)::is_always_lock_free);
static_assert(decltype(g_install_state)::is_always_lock_free);

// Translate the x86-64 page-fault error code (mcontext __es.__err) into an access type.
// bit 1 (0x2) = write, bit 4 (0x10) = instruction fetch, otherwise a read.
static AccessViolationType DecodeAccess(uint64_t err) {
        if ((err & 0x10u) != 0) {
                return AccessViolationType::Execute;
        }
        if ((err & 0x2u) != 0) {
                return AccessViolationType::Write;
        }
        return AccessViolationType::Read;
}

// POSIX signal handler that mirrors the Windows vectored handler: build an ExceptionInfo
// from the mcontext and dispatch. A resolved fault (handler returns true) simply returns,
// re-executing the faulting instruction against the now-fixed protection. An unresolved
// fault restores the default disposition so the retry terminates the process.
static void SignalHandler(int sig, siginfo_t* si, void* uctx) {
        auto*       uc = static_cast<ucontext_t*>(uctx);
        const auto* mc = uc->uc_mcontext;
        const auto& ss = mc->__ss;

        ExceptionInfo info {};
        info.exception_address = ss.__rip;
        info.native_code       = static_cast<uint32_t>(si->si_code);
        info.native_context    = uctx;

        if (sig == SIGILL) {
                info.type = ExceptionType::IllegalInstruction;
        } else {
                info.type                   = ExceptionType::AccessViolation;
                info.access_violation_type  = DecodeAccess(mc->__es.__err);
                info.access_violation_vaddr = reinterpret_cast<uint64_t>(si->si_addr);
        }

        info.rax = ss.__rax;
        info.rbx = ss.__rbx;
        info.rcx = ss.__rcx;
        info.rdx = ss.__rdx;
        info.rsi = ss.__rsi;
        info.rdi = ss.__rdi;
        info.rbp = ss.__rbp;
        info.rsp = ss.__rsp;
        info.r8  = ss.__r8;
        info.r9  = ss.__r9;
        info.r10 = ss.__r10;
        info.r11 = ss.__r11;
        info.r12 = ss.__r12;
        info.r13 = ss.__r13;
        info.r14 = ss.__r14;
        info.r15 = ss.__r15;

        const auto handler = g_handler.load(std::memory_order_acquire);
        if (handler != nullptr && handler(info)) {
                return; // retry the faulting instruction against the fixed mapping
        }

        // Unresolved: restore the default action so the re-executed instruction terminates.
        struct sigaction dfl {};
        dfl.sa_handler = SIG_DFL;
        sigemptyset(&dfl.sa_mask);
        sigaction(sig, &dfl, nullptr);
}

#else

// x86-64 page-fault error bits.
constexpr uint64_t PAGE_FAULT_ERROR_WRITE       = 0x02;
constexpr uint64_t PAGE_FAULT_ERROR_INSTRUCTION = 0x10;

// Let the kernel handle an unresolved fault on retry.
static void ChainToDefault(int signal_number) noexcept {
        struct sigaction restore {};
        restore.sa_handler = SIG_DFL;
        sigemptyset(&restore.sa_mask);
        restore.sa_flags = 0;
        ::sigaction(signal_number, &restore, nullptr);
}

static void SignalHandler(int signal_number, siginfo_t* signal_info, void* native_context) {
        auto* context = static_cast<ucontext_t*>(native_context);
        auto* gregs   = context->uc_mcontext.gregs;

        ExceptionInfo info {};
        info.exception_address = static_cast<uint64_t>(gregs[REG_RIP]);
        info.native_code       = static_cast<uint32_t>(signal_number);
        info.native_context    = context;

        if (signal_number == SIGSEGV || signal_number == SIGBUS) {
                info.type             = ExceptionType::AccessViolation;
                const auto error_code = static_cast<uint64_t>(gregs[REG_ERR]);
                if ((error_code & PAGE_FAULT_ERROR_INSTRUCTION) != 0) {
                        info.access_violation_type = AccessViolationType::Execute;
                } else if ((error_code & PAGE_FAULT_ERROR_WRITE) != 0) {
                        info.access_violation_type = AccessViolationType::Write;
                } else {
                        info.access_violation_type = AccessViolationType::Read;
                }
                info.access_violation_vaddr = reinterpret_cast<uint64_t>(signal_info->si_addr);
        } else if (signal_number == SIGILL) {
                info.type = ExceptionType::IllegalInstruction;
        } else {
                ChainToDefault(signal_number);
                return;
        }

        info.rax = static_cast<uint64_t>(gregs[REG_RAX]);
        info.rbx = static_cast<uint64_t>(gregs[REG_RBX]);
        info.rcx = static_cast<uint64_t>(gregs[REG_RCX]);
        info.rdx = static_cast<uint64_t>(gregs[REG_RDX]);
        info.rsi = static_cast<uint64_t>(gregs[REG_RSI]);
        info.rdi = static_cast<uint64_t>(gregs[REG_RDI]);
        info.rbp = static_cast<uint64_t>(gregs[REG_RBP]);
        info.rsp = static_cast<uint64_t>(gregs[REG_RSP]);
        info.r8  = static_cast<uint64_t>(gregs[REG_R8]);
        info.r9  = static_cast<uint64_t>(gregs[REG_R9]);
        info.r10 = static_cast<uint64_t>(gregs[REG_R10]);
        info.r11 = static_cast<uint64_t>(gregs[REG_R11]);
        info.r12 = static_cast<uint64_t>(gregs[REG_R12]);
        info.r13 = static_cast<uint64_t>(gregs[REG_R13]);
        info.r14 = static_cast<uint64_t>(gregs[REG_R14]);
        info.r15 = static_cast<uint64_t>(gregs[REG_R15]);

        const auto handler = g_handler.load(std::memory_order_acquire);
        if (handler != nullptr && handler(info)) {
                return;
        }

        ChainToDefault(signal_number);
}

#endif

bool InstallHandler(Handler handler) {
        if (handler == nullptr) {
                return false;
        }

        uint32_t expected_state = 0;
        if (!g_install_state.compare_exchange_strong(expected_state, 1, std::memory_order_acq_rel)) {
                return expected_state == 2 && g_handler.load(std::memory_order_acquire) == handler;
        }

        g_handler.store(handler, std::memory_order_release);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
        if (AddVectoredExceptionHandler(1, ExceptionFilter) == nullptr) {
                g_handler.store(nullptr, std::memory_order_release);
                g_install_state.store(0, std::memory_order_release);
                printf("AddVectoredExceptionHandler() failed\n");
                return false;
        }
#elif defined(__APPLE__)
        struct sigaction sa {};
        sa.sa_sigaction = SignalHandler;
        sa.sa_flags     = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&sa.sa_mask);
        // The guest signal-dispatch path (KernelRaiseException) interrupts threads with
        // SIGUSR1; block it while a fault is being resolved so a stop-the-world request
        // cannot preempt the handler between the protection fix and the retry.
        sigaddset(&sa.sa_mask, SIGUSR1);

        // macOS raises SIGBUS for protection faults on some paths and SIGSEGV on others;
        // SIGILL covers instructions the host cannot execute (routed to the x64 emulator).
        bool ok = sigaction(SIGSEGV, &sa, nullptr) == 0 && sigaction(SIGBUS, &sa, nullptr) == 0 &&
                  sigaction(SIGILL, &sa, nullptr) == 0;
        if (!ok) {
                g_handler.store(nullptr, std::memory_order_release);
                g_install_state.store(0, std::memory_order_release);
                printf("sigaction() failed to install the host fault handler\n");
                return false;
        }
#else
        struct sigaction action {};
        action.sa_sigaction = SignalHandler;
        sigemptyset(&action.sa_mask);
        action.sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK;

        for (const int signal_number: {SIGSEGV, SIGBUS, SIGILL}) {
                if (::sigaction(signal_number, &action, nullptr) != 0) {
                        g_handler.store(nullptr, std::memory_order_release);
                        g_install_state.store(0, std::memory_order_release);
                        printf("sigaction(%d) failed\n", signal_number);
                        return false;
                }
        }
#endif

        g_install_state.store(2, std::memory_order_release);
        return true;
}

} // namespace Common::HostException
