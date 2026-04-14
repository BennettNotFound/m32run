//! 32 位 Mach-O 程序加载器
//!
//! 这个模块负责把以下几部分串起来：
//! - Mach-O 解析器
//! - guest 内存管理
//! - ABI 启动栈构造
//! - CPU 初始状态
//!
//! 当前加载流程：
//! 1. 解析 Mach-O
//! 2. 按段信息把程序映射进 guest 内存
//! 3. 把文件中的段内容拷贝到 guest 内存
//! 4. 读取宿主环境变量，转换成 guest 的 envp
//! 5. 构造初始栈
//! 6. 初始化 CPU 的 EIP / ESP
//!
//! 当前仍然不处理：
//! - dyld
//! - 重定位
//! - 动态符号绑定
//! - 共享库真正装载
//!
//! 所以它能支撑“主程序加载并开始执行”，
//! 但还不能独立跑复杂动态链接 GUI 程序的全流程。

use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use abi::prepare_stack;
use guestmem::{GuestMemory, Prot};
use macho32::parse;
use thiserror::Error;
use x86core::Cpu;

/// 加载过程中可能出现的错误
#[derive(Error, Debug)]
pub enum LoadError {
    /// 文件读取、seek、Mach-O 解析等 I/O 错误
    #[error(transparent)]
    Io(#[from] io::Error),

    /// guest 内存映射或读写错误
    #[error(transparent)]
    Memory(#[from] guestmem::Error),
}

/// 加载完成后的程序状态
#[derive(Debug)]
pub struct LoadedProgram {
    pub mem: GuestMemory,
    pub cpu: Cpu,
}

/// 把宿主环境变量转换成 guest 栈里使用的 `"KEY=VALUE"` 形式
///
/// 这里使用 `vars_os()` 而不是 `vars()`，这样即使环境变量中含有
/// 非 UTF-8 内容，也不会直接丢失；最后统一用 lossy 方式转成字符串。
fn collect_host_env() -> Vec<String> {
    let mut result = Vec::new();

    for (key, value) in env::vars_os() {
        let key = os_to_lossy_string(key);
        let value = os_to_lossy_string(value);
        result.push(format!("{}={}", key, value));
    }

    // 排序一下，方便调试时保持稳定输出
    result.sort();

    result
}

/// 把 OsString 转成 lossy UTF-8 String
fn os_to_lossy_string(s: OsString) -> String {
    s.to_string_lossy().into_owned()
}

/// 根据 Mach-O 的 initprot 生成“运行期权限”
fn runtime_prot_from_initprot(initprot: u32) -> Prot {
    // Mach VM 权限位：
    // 0x01 = READ
    // 0x02 = WRITE
    // 0x04 = EXEC
    //
    // 当前 GuestMemory/解释器基本都假定段至少可读，
    // 所以这里默认从 READ 开始。
    let mut prot = Prot::READ;

    if (initprot & 0x02) != 0 {
        prot = prot | Prot::WRITE;
    }

    if (initprot & 0x04) != 0 {
        prot = prot | Prot::EXEC;
    }

    prot
}

/// 加载一个 32 位 Mach-O 程序
///
/// 参数：
/// - `path`：可执行文件路径
/// - `args`：用户参数（不包含 argv[0]）
/// - `trace`：是否输出 loader / ABI 调试信息
///
/// 返回：
/// - 成功：LoadedProgram
/// - 失败：LoadError
pub fn load<P: AsRef<Path>>(
    path: P,
    args: &[String],
    trace: bool,
) -> Result<LoadedProgram, LoadError> {
    let path_ref = path.as_ref();

    // ------------------------------------------------------------
    // 1. 解析 Mach-O
    // ------------------------------------------------------------
    let macho = parse(path_ref)?;
    let mut file = File::open(path_ref)?;

    // 新建 guest 内存空间
    let mut mem = GuestMemory::new();

    // ------------------------------------------------------------
    // 2. 把 Mach-O 各段映射到 guest 内存
    // ------------------------------------------------------------
    for seg in &macho.segments {
        // 先算出“运行期想要的权限”
        let runtime_prot = runtime_prot_from_initprot(seg.initprot);

        // 关键修复：
        // 加载阶段必须把文件内容写进段内存，所以临时强制加 WRITE。
        //
        // 否则像 __TEXT 这种 initprot = RX 的段，在 mem.write() 时会直接报：
        // write attempted on non-writable region
        //
        // 以后如果 GuestMemory 支持 reprotect/mprotect，
        // 可以在写完文件内容后再把权限降回 runtime_prot。
        let load_prot = runtime_prot | Prot::WRITE;

        if trace {
            eprintln!(
                "[LOADER] map seg='{}' vmaddr={:#010x} vmsize={:#010x} fileoff={:#010x} filesize={:#010x} initprot={:#x}",
                seg.segname,
                seg.vmaddr,
                seg.vmsize,
                seg.fileoff,
                seg.filesize,
                seg.initprot,
            );
        }

        // 先映射整段虚拟内存
        mem.map(seg.vmaddr, seg.vmsize, load_prot)?;

        // 再把文件中的实际数据读出来覆盖进去
        if seg.filesize > 0 {
            file.seek(SeekFrom::Start(seg.fileoff as u64))?;

            let mut buf = vec![0u8; seg.filesize as usize];
            file.read_exact(&mut buf)?;

            mem.write(seg.vmaddr, &buf)?;
        }

        // 如果 vmsize > filesize，后面的 zero-fill 区域（例如 bss）
        // 会自然保持为 0，因为 map() 默认分配的是零初始化缓冲区。
    }

    // ------------------------------------------------------------
    // 3. 读取宿主真实环境变量
    // ------------------------------------------------------------
    let exec_path = path_ref.to_string_lossy().to_string();
    let env = collect_host_env();

    if trace {
        eprintln!("[LOADER] exec_path = {}", exec_path);
        eprintln!("[LOADER] env count = {}", env.len());
        eprintln!("[LOADER] entry eip = {:#010x}", macho.entry_point);
    }

    // ------------------------------------------------------------
    // 4. 构造 Darwin 风格初始栈
    // ------------------------------------------------------------
    let sp = prepare_stack(&mut mem, &exec_path, args, &env, trace)?;

    if trace {
        eprintln!("[LOADER] initial esp = {:#010x}", sp);
    }

    // ------------------------------------------------------------
    // 5. 初始化 CPU
    // ------------------------------------------------------------
    let mut cpu = Cpu::new();
    cpu.eip = macho.entry_point;
    cpu.esp = sp;

    Ok(LoadedProgram { mem, cpu })
}