//! 32 位 Darwin ABI / 启动栈支持
//!
//! 这个模块负责在 guest 内存里构造一个更接近真实 Darwin 进程的初始栈。
//!
//! 当前构造出的栈布局如下（注意最前面多了一个 dummy 返回地址）：
//!
//!   fake_return_address
//!   argc
//!   argv[0]
//!   argv[1]
//!   ...
//!   NULL
//!   envp[0]
//!   envp[1]
//!   ...
//!   NULL
//!   apple[0]
//!   apple[1]
//!   ...
//!   NULL
//!
//! 其中：
//! - argv[0] 会被设置为可执行文件路径
//! - envp 使用外部传入的环境变量
//! - apple 目前只放最小需要的 `executable_path=...`
//! - fake_return_address 是为了兼容老式启动代码对 `[ebp+8]` 的访问习惯
//!
//! 注意：
//! - 这仍然不是“完整复刻真实 Darwin 内核创建进程时的全部数据”
//! - 但已经比最初只有 argc/argv 的 demo 栈更像样，足以支撑更多真实程序启动

use guestmem::{Error as GuestError, GuestMemory, Prot};

/// guest 栈的最高地址（不包含）
///
/// 合法栈地址范围大致为：
/// [STACK_TOP - STACK_SIZE, STACK_TOP)
pub const STACK_TOP: u32 = 0x7fff_0000;

/// 栈总大小
///
/// 这里给 1 MiB，调试真实程序时会比很小的栈稳定很多。
pub const STACK_SIZE: u32 = 0x0010_0000; // 1 MiB

/// 初始 ESP 不直接顶在 STACK_TOP，给上方留一点余量，
/// 避免某些启动代码一上来就读到未映射上边界。
const STACK_INITIAL_SLACK: u32 = 0x0001_0000; // 64 KiB

/// 写入原始字节到栈中，并向下移动 SP
fn push_bytes(mem: &mut GuestMemory, sp: &mut u32, bytes: &[u8]) -> Result<u32, GuestError> {
    *sp = sp.wrapping_sub(bytes.len() as u32);
    mem.write(*sp, bytes)?;
    Ok(*sp)
}

/// 写入一个以 `\0` 结尾的 C 字符串到栈中，并返回它在 guest 中的地址
fn push_c_string(mem: &mut GuestMemory, sp: &mut u32, s: &str) -> Result<u32, GuestError> {
    let mut buf = Vec::with_capacity(s.len() + 1);
    buf.extend_from_slice(s.as_bytes());
    buf.push(0);
    push_bytes(mem, sp, &buf)
}

/// 栈向下按指定对齐值对齐
fn align_down(value: u32, align: u32) -> u32 {
    debug_assert!(align.is_power_of_two());
    value & !(align - 1)
}

/// 往栈里压一个 32 位值
fn push_u32(mem: &mut GuestMemory, sp: &mut u32, value: u32) -> Result<(), GuestError> {
    *sp = sp.wrapping_sub(4);
    mem.write(*sp, &value.to_le_bytes())?;
    Ok(())
}

/// 构造 32 位 Darwin 风格的初始启动栈
///
/// 参数说明：
/// - `mem`：guest 内存
/// - `exec_path`：可执行文件路径，会作为 argv[0] 和 apple 向量的一部分
/// - `args`：用户参数，不包含 argv[0]
/// - `env`：环境变量数组，每一项形如 `"KEY=VALUE"`
///
/// 返回值：
/// - 构造完成后的初始 ESP
pub fn prepare_stack(
    mem: &mut GuestMemory,
    exec_path: &str,
    args: &[String],
    env: &[String],
    trace: bool,
) -> Result<u32, GuestError> {
    if trace {
        eprintln!(
            "[ABI] STACK_TOP={:#010x} STACK_SIZE={:#010x} STACK_INITIAL_SLACK={:#010x}",
            STACK_TOP, STACK_SIZE, STACK_INITIAL_SLACK
        );
    }
    // ------------------------------------------------------------
    // 1. 映射整块栈区域
    // ------------------------------------------------------------
    mem.map(STACK_TOP - STACK_SIZE, STACK_SIZE, Prot::READ | Prot::WRITE)?;

    // 初始栈顶不直接贴着 STACK_TOP，给顶部留一段缓冲
    let mut sp = STACK_TOP - STACK_INITIAL_SLACK;

    // ------------------------------------------------------------
    // 2. 准备 argv / env / apple 三组字符串
    // ------------------------------------------------------------

    // argv[0] 必须是程序路径，后面再拼接用户传入参数
    let mut argv_strings = Vec::with_capacity(args.len() + 1);
    argv_strings.push(exec_path.to_string());
    argv_strings.extend(args.iter().cloned());

    // apple 向量先保留最小必要项
    let apple_strings = vec![format!("executable_path={}", exec_path)];

    // ------------------------------------------------------------
    // 3. 先把所有字符串写到高地址区域
    // ------------------------------------------------------------
    //
    // 采用“字符串区在上，指针区在下”的布局，
    // 更接近真实进程启动时的内存分布，也更方便调试。
    let mut argv_ptrs = Vec::with_capacity(argv_strings.len());
    for s in argv_strings.iter().rev() {
        let addr = push_c_string(mem, &mut sp, s)?;
        argv_ptrs.push(addr);
    }
    argv_ptrs.reverse();

    let mut env_ptrs = Vec::with_capacity(env.len());
    for s in env.iter().rev() {
        let addr = push_c_string(mem, &mut sp, s)?;
        env_ptrs.push(addr);
    }
    env_ptrs.reverse();

    let mut apple_ptrs = Vec::with_capacity(apple_strings.len());
    for s in apple_strings.iter().rev() {
        let addr = push_c_string(mem, &mut sp, s)?;
        apple_ptrs.push(addr);
    }
    apple_ptrs.reverse();

    // 指针区按 16 字节对齐，通常更稳一点
    sp = align_down(sp, 16);

    // ------------------------------------------------------------
    // 4. 压入 apple[] 和结尾 NULL
    // ------------------------------------------------------------
    push_u32(mem, &mut sp, 0)?;
    for &ptr in apple_ptrs.iter().rev() {
        push_u32(mem, &mut sp, ptr)?;
    }

    // ------------------------------------------------------------
    // 5. 压入 envp[] 和结尾 NULL
    // ------------------------------------------------------------
    push_u32(mem, &mut sp, 0)?;
    for &ptr in env_ptrs.iter().rev() {
        push_u32(mem, &mut sp, ptr)?;
    }

    // ------------------------------------------------------------
    // 6. 压入 argv[] 和结尾 NULL
    // ------------------------------------------------------------
    push_u32(mem, &mut sp, 0)?;
    for &ptr in argv_ptrs.iter().rev() {
        push_u32(mem, &mut sp, ptr)?;
    }

    // ------------------------------------------------------------
    // 7. 压入 argc
    // ------------------------------------------------------------
    let argc = argv_ptrs.len() as u32;
    push_u32(mem, &mut sp, argc)?;

    // ------------------------------------------------------------
    // 8. 压入 fake return address
    // ------------------------------------------------------------
    //
    // 这是这次修复的关键：
    //
    // 这样入口如果执行：
    //   push ebp
    //   mov  ebp, esp
    //
    // 那么：
    //   [ebp + 8]  -> argc
    //   [ebp + 12] -> argv[0]
    //
    // 否则老式启动代码会把参数区整体错位 4 字节。
    push_u32(mem, &mut sp, 0)?;
    if trace {
        eprintln!("[ABI] final initial ESP = {:#010x}", sp);
    }
    Ok(sp)
}
