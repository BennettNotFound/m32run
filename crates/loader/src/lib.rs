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

use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use abi::prepare_stack;
use guestmem::{GuestMemory, Prot};
use macho32::{parse, ImportKind32, MachO32};
use thiserror::Error;
use x86core::Cpu;

const ERRNO_ENOENT: u32 = 2;
const ERRNO_EIO: u32 = 5;
const ERRNO_EBADF: u32 = 9;
const ERRNO_ENOMEM: u32 = 12;
const ERRNO_EACCES: u32 = 13;
const ERRNO_EFAULT: u32 = 14;
const ERRNO_EEXIST: u32 = 17;
const ERRNO_EINVAL: u32 = 22;
const ERRNO_ERANGE: u32 = 34;

const SECTION_TYPE_MASK: u32 = 0x000000ff;
const S_MOD_INIT_FUNC_POINTERS: u32 = 0x9;

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
    pub dyld: DyldState,
}

#[derive(Debug, Clone)]
struct ImportBinding {
    kind: ImportKind32,
    addr: u32,
    size: u32,
    indirect_symbol_index: u32,
    symbol_name: Option<String>,
    dylib: Option<String>,
    section: String,
}

#[derive(Debug, Clone)]
struct LoadedImage {
    handle: u32,
    path: PathBuf,
    slide: u32,
    exports: HashMap<String, u32>,
    init_funcs: Vec<u32>,
    initializers_ran: bool,
}

impl ImportBinding {
    fn display_name(&self) -> String {
        self.symbol_name
            .clone()
            .unwrap_or_else(|| format!("<indirect:{}>", self.indirect_symbol_index))
    }
}

#[derive(Error, Debug)]
pub enum DyldError {
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error(transparent)]
    Memory(#[from] guestmem::Error),

    #[error("import binding not found for eip={eip:#010x}, indirect_symbol_index={indirect_symbol_index}")]
    BindingNotFound {
        eip: u32,
        indirect_symbol_index: u32,
    },

    #[error("guest stack underflow while returning from import call")]
    InvalidGuestStack,

    #[error("guest requested exit with status {0}")]
    GuestExit(u32),
}

#[derive(Debug)]
pub struct DyldState {
    trace: bool,
    main_exec_path: Option<PathBuf>,
    main_exec_dir: Option<PathBuf>,
    main_exports: HashMap<String, u32>,
    imports_by_addr: HashMap<u32, ImportBinding>,
    imports_by_indirect: HashMap<u32, ImportBinding>,
    scratch_base: u32,
    scratch_size: u32,
    scratch_cursor: u32,
    trampoline_base: u32,
    trampoline_size: u32,
    trampoline_cursor: u32,
    heap_base: u32,
    heap_size: u32,
    heap_cursor: u32,
    heap_allocations: HashMap<u32, u32>,
    last_dlerror: Option<String>,
    last_dlerror_ptr: u32,
    fake_dlopen_next: u32,
    loaded_images: HashMap<u32, LoadedImage>,
    handle_by_path: HashMap<String, u32>,
    next_dylib_base: u32,
    next_dynamic_indirect: u32,
    host_next_fd: u32,
    host_files: HashMap<u32, File>,
    host_streams: HashMap<u32, u32>,
    host_next_mmap: u32,
    errno_slot: u32,
    pthread_self_ptr: u32,
    localtime_tm_ptr: u32,
    stdin_stream_ptr: u32,
    stdout_stream_ptr: u32,
    stderr_stream_ptr: u32,
    stdin_var_ptr: u32,
    stdout_var_ptr: u32,
    stderr_var_ptr: u32,
    stack_chk_guard_var_ptr: u32,
    pthread_next_key: u32,
    pthread_key_values: HashMap<u32, u32>,
    warned_unimplemented: HashSet<String>,
}

impl DyldState {
    fn from_macho(macho: &MachO32, trace: bool) -> Self {
        let mut imports_by_addr = HashMap::new();
        let mut imports_by_indirect = HashMap::new();
        let mut main_exports = HashMap::new();

        for import in &macho.imports {
            let binding = ImportBinding {
                kind: import.kind,
                addr: import.addr,
                size: import.size,
                indirect_symbol_index: import.indirect_symbol_index,
                symbol_name: import.symbol_name.clone(),
                dylib: import.dylib.clone(),
                section: import.section.clone(),
            };
            imports_by_addr.insert(import.addr, binding.clone());
            imports_by_indirect.insert(import.indirect_symbol_index, binding);
        }
        for export in &macho.exports {
            main_exports.insert(export.name.clone(), export.value);
        }

        Self {
            trace,
            main_exec_path: None,
            main_exec_dir: None,
            main_exports,
            imports_by_addr,
            imports_by_indirect,
            scratch_base: 0,
            scratch_size: 0,
            scratch_cursor: 0,
            trampoline_base: 0,
            trampoline_size: 0,
            trampoline_cursor: 0,
            heap_base: 0,
            heap_size: 0,
            heap_cursor: 0,
            heap_allocations: HashMap::new(),
            last_dlerror: None,
            last_dlerror_ptr: 0,
            fake_dlopen_next: 1,
            loaded_images: HashMap::new(),
            handle_by_path: HashMap::new(),
            next_dylib_base: 0x5000_0000,
            next_dynamic_indirect: 0xF000_0000,
            host_next_fd: 3,
            host_files: HashMap::new(),
            host_streams: HashMap::new(),
            host_next_mmap: 0x2400_0000,
            errno_slot: 0,
            pthread_self_ptr: 0,
            localtime_tm_ptr: 0,
            stdin_stream_ptr: 0,
            stdout_stream_ptr: 0,
            stderr_stream_ptr: 0,
            stdin_var_ptr: 0,
            stdout_var_ptr: 0,
            stderr_var_ptr: 0,
            stack_chk_guard_var_ptr: 0,
            pthread_next_key: 1,
            pthread_key_values: HashMap::new(),
            warned_unimplemented: HashSet::new(),
        }
    }

    fn set_main_executable_path(&mut self, path: &Path) {
        self.main_exec_path = Some(path.to_path_buf());
        self.main_exec_dir = path.parent().map(|p| p.to_path_buf());
    }

    fn align_up(value: u32, align: u32) -> u32 {
        debug_assert!(align.is_power_of_two());
        (value.wrapping_add(align - 1)) & !(align - 1)
    }

    fn symbol_candidates(symbol: &str) -> Vec<String> {
        let mut out = Vec::new();
        if symbol.is_empty() {
            return out;
        }
        out.push(symbol.to_string());

        if let Some(stripped) = symbol.strip_prefix('_') {
            if !stripped.is_empty() {
                out.push(stripped.to_string());
            }
        } else {
            out.push(format!("_{}", symbol));
        }
        out
    }

    fn lookup_export(exports: &HashMap<String, u32>, symbol: &str) -> Option<u32> {
        for candidate in Self::symbol_candidates(symbol) {
            if let Some(&addr) = exports.get(&candidate) {
                return Some(addr);
            }
        }
        None
    }

    fn resolve_dlopen_path(&self, raw: &str) -> Option<PathBuf> {
        let mut candidates = Vec::new();
        let raw_path = Path::new(raw);

        if raw_path.is_absolute() {
            candidates.push(raw_path.to_path_buf());
        }

        if let Some(exec_dir) = &self.main_exec_dir {
            if let Some(rest) = raw.strip_prefix("@loader_path/") {
                candidates.push(exec_dir.join(rest));
            }
            if let Some(rest) = raw.strip_prefix("@executable_path/") {
                candidates.push(exec_dir.join(rest));
            }
            if let Some(rest) = raw.strip_prefix("@rpath/") {
                candidates.push(exec_dir.join(rest));
                if let Some(parent) = exec_dir.parent() {
                    candidates.push(parent.join(rest));
                }
            }

            candidates.push(exec_dir.join(raw));

            // 常见 Steam/macOS 布局：真正 32 位库在 bin/osx32 下。
            if let Some(rest) = raw.strip_prefix("bin/") {
                if !raw.contains("/osx32/") {
                    candidates.push(exec_dir.join("bin/osx32").join(rest));
                }
            }

            if let Some(filename) = raw_path.file_name() {
                candidates.push(exec_dir.join("bin/osx32").join(filename));
            }
        }

        candidates.push(raw_path.to_path_buf());

        let mut visited = HashSet::new();
        for cand in candidates {
            if !visited.insert(cand.clone()) {
                continue;
            }
            if cand.exists() {
                if let Ok(canon) = fs::canonicalize(&cand) {
                    return Some(canon);
                }
                return Some(cand);
            }
        }
        None
    }

    fn map_runtime_region(
        &self,
        mem: &mut GuestMemory,
        name: &str,
        size: u32,
        prot: Prot,
        candidates: &[u32],
    ) -> Result<u32, guestmem::Error> {
        let mut last_err = None;
        for &base in candidates {
            match mem.map(base, size, prot) {
                Ok(()) => {
                    if self.trace {
                        eprintln!(
                            "[DYLD] {} memory mapped at {:#010x} (size={:#x})",
                            name, base, size
                        );
                    }
                    return Ok(base);
                }
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or(guestmem::Error::AddressNotMapped(0)))
    }

    fn init_runtime_memory(&mut self, mem: &mut GuestMemory) -> Result<(), guestmem::Error> {
        const SCRATCH_SIZE: u32 = 0x0001_0000;
        const TRAMPOLINE_SIZE: u32 = 0x0002_0000;
        const HEAP_SIZE: u32 = 0x0200_0000;

        const SCRATCH_BASES: [u32; 6] = [
            0x6ff0_0000,
            0x6fe0_0000,
            0x6fd0_0000,
            0x6fc0_0000,
            0x6fb0_0000,
            0x6fa0_0000,
        ];
        const TRAMPOLINE_BASES: [u32; 6] = [
            0x6f80_0000,
            0x6f70_0000,
            0x6f60_0000,
            0x6f50_0000,
            0x6f40_0000,
            0x6f30_0000,
        ];
        const HEAP_BASES: [u32; 5] = [
            0x6d00_0000,
            0x6b00_0000,
            0x6900_0000,
            0x6700_0000,
            0x6500_0000,
        ];

        self.scratch_base = self.map_runtime_region(
            mem,
            "scratch",
            SCRATCH_SIZE,
            Prot::READ | Prot::WRITE,
            &SCRATCH_BASES,
        )?;
        self.scratch_size = SCRATCH_SIZE;
        self.scratch_cursor = self.scratch_base;

        self.trampoline_base = self.map_runtime_region(
            mem,
            "trampoline",
            TRAMPOLINE_SIZE,
            Prot::READ | Prot::WRITE | Prot::EXEC,
            &TRAMPOLINE_BASES,
        )?;
        self.trampoline_size = TRAMPOLINE_SIZE;
        self.trampoline_cursor = self.trampoline_base;

        self.heap_base =
            self.map_runtime_region(mem, "heap", HEAP_SIZE, Prot::READ | Prot::WRITE, &HEAP_BASES)?;
        self.heap_size = HEAP_SIZE;
        self.heap_cursor = self.heap_base;

        Ok(())
    }

    fn alloc_trampoline_stub(&mut self, mem: &mut GuestMemory) -> Result<u32, guestmem::Error> {
        if self.trampoline_base == 0 || self.trampoline_size == 0 {
            return Err(guestmem::Error::AddressNotMapped(0));
        }
        let end = self.trampoline_base + self.trampoline_size;
        // 每个桩占 4 字节，后续如果要扩成小段代码也不需要改布局。
        if self.trampoline_cursor + 4 > end {
            return Err(guestmem::Error::AddressNotMapped(self.trampoline_base));
        }
        let addr = self.trampoline_cursor;
        mem.write(addr, &[0xF4, 0xF4, 0xF4, 0xF4])?;
        self.trampoline_cursor = self.trampoline_cursor.wrapping_add(4);
        Ok(addr)
    }

    fn patch_import_stubs(
        &mut self,
        mem: &mut GuestMemory,
        cpu: &mut Cpu,
    ) -> Result<(), guestmem::Error> {
        let bindings: Vec<ImportBinding> = self.imports_by_addr.values().cloned().collect();
        for binding in bindings {
            match binding.kind {
                ImportKind32::SymbolStub => {
                    let patch_len = binding.size.max(1) as usize;
                    let patch = vec![0xF4u8; patch_len];
                    mem.write(binding.addr, &patch)?;
                    cpu.register_import_stub(binding.addr, binding.indirect_symbol_index);
                    if self.trace {
                        eprintln!(
                            "[DYLD] patched import stub at {:#010x} size={} symbol={} section={}",
                            binding.addr,
                            binding.size,
                            binding.display_name(),
                            binding.section
                        );
                    }
                }
                ImportKind32::LazyPointer | ImportKind32::NonLazyPointer => {
                    if binding.kind == ImportKind32::NonLazyPointer {
                        let symbol = binding.display_name();
                        let symbol_norm = symbol.strip_prefix('_').unwrap_or(symbol.as_str());
                        let special_ptr = match symbol_norm {
                            "__stderrp" | "_stderrp" => Some(self.ensure_stderr_var_ptr(mem)?),
                            "__stdoutp" | "_stdoutp" => Some(self.ensure_stdout_var_ptr(mem)?),
                            "__stdinp" | "_stdinp" => Some(self.ensure_stdin_var_ptr(mem)?),
                            "__stack_chk_guard" | "_stack_chk_guard" => {
                                Some(self.ensure_stack_chk_guard_var_ptr(mem)?)
                            }
                            _ => None,
                        };
                        if let Some(ptr) = special_ptr {
                            mem.write(binding.addr, &ptr.to_le_bytes())?;
                            if self.trace {
                                eprintln!(
                                    "[DYLD] patched import non-lazy pointer at {:#010x} -> {:#010x} symbol={} section={}",
                                    binding.addr,
                                    ptr,
                                    binding.display_name(),
                                    binding.section
                                );
                            }
                            continue;
                        }
                    }

                    let trap = self.alloc_trampoline_stub(mem)?;
                    mem.write(binding.addr, &trap.to_le_bytes())?;
                    cpu.register_import_stub(trap, binding.indirect_symbol_index);
                    if self.trace {
                        eprintln!(
                            "[DYLD] patched import pointer at {:#010x} -> trap {:#010x} symbol={} section={}",
                            binding.addr,
                            trap,
                            binding.display_name(),
                            binding.section
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn patch_single_import_binding(
        &mut self,
        mem: &mut GuestMemory,
        cpu: &mut Cpu,
        binding: &ImportBinding,
    ) -> Result<(), DyldError> {
        match binding.kind {
            ImportKind32::SymbolStub => {
                let patch_len = binding.size.max(1) as usize;
                let patch = vec![0xF4u8; patch_len];
                mem.write(binding.addr, &patch)?;
                cpu.register_import_stub(binding.addr, binding.indirect_symbol_index);
                if self.trace {
                    eprintln!(
                        "[DYLD] patched dynamic import stub at {:#010x} size={} symbol={} section={}",
                        binding.addr,
                        binding.size,
                        binding.display_name(),
                        binding.section
                    );
                }
            }
            ImportKind32::NonLazyPointer => {
                if let Some((target, _handle)) = self.try_resolve_import_symbol(binding, mem, cpu)? {
                    mem.write(binding.addr, &target.to_le_bytes())?;
                    if self.trace {
                        eprintln!(
                            "[DYLD] eagerly bound non-lazy pointer at {:#010x} -> {:#010x} symbol={} section={}",
                            binding.addr,
                            target,
                            binding.display_name(),
                            binding.section
                        );
                    }
                    return Ok(());
                }

                let symbol = binding.display_name();
                let symbol_norm = symbol.strip_prefix('_').unwrap_or(symbol.as_str());
                let var_ptr = match symbol_norm {
                    "__stderrp" | "_stderrp" => self.ensure_stderr_var_ptr(mem)?,
                    "__stdoutp" | "_stdoutp" => self.ensure_stdout_var_ptr(mem)?,
                    "__stdinp" | "_stdinp" => self.ensure_stdin_var_ptr(mem)?,
                    "__stack_chk_guard" | "_stack_chk_guard" => self.ensure_stack_chk_guard_var_ptr(mem)?,
                    _ => {
                        // 保底：给未知非惰性数据符号构造 “变量 -> 对象” 链，避免读到 0xF4F4F4F4。
                        let obj_ptr = self.alloc_heap(mem, 16, true)?;
                        if obj_ptr == 0 {
                            return Err(guestmem::Error::AddressNotMapped(0).into());
                        }
                        mem.write(obj_ptr, &obj_ptr.to_le_bytes())?;
                        let var_ptr = self.alloc_heap(mem, 4, true)?;
                        if var_ptr == 0 {
                            return Err(guestmem::Error::AddressNotMapped(0).into());
                        }
                        mem.write(var_ptr, &obj_ptr.to_le_bytes())?;
                        var_ptr
                    }
                };
                mem.write(binding.addr, &var_ptr.to_le_bytes())?;
                if self.trace {
                    eprintln!(
                        "[DYLD] synthesized non-lazy pointer at {:#010x} -> {:#010x} symbol={} section={}",
                        binding.addr,
                        var_ptr,
                        binding.display_name(),
                        binding.section
                    );
                }
            }
            ImportKind32::LazyPointer => {
                let trap = self.alloc_trampoline_stub(mem)?;
                mem.write(binding.addr, &trap.to_le_bytes())?;
                cpu.register_import_stub(trap, binding.indirect_symbol_index);
                if self.trace {
                    eprintln!(
                        "[DYLD] patched dynamic import pointer at {:#010x} -> trap {:#010x} symbol={} section={}",
                        binding.addr,
                        trap,
                        binding.display_name(),
                        binding.section
                    );
                }
            }
        }
        Ok(())
    }

    fn register_macho_imports_with_slide(
        &mut self,
        macho: &MachO32,
        slide: u32,
        mem: &mut GuestMemory,
        cpu: &mut Cpu,
    ) -> Result<(), DyldError> {
        for import in &macho.imports {
            // 为每个 dylib 导入分配唯一间接索引，避免不同镜像冲突。
            let indirect_symbol_index = self.next_dynamic_indirect;
            self.next_dynamic_indirect = self.next_dynamic_indirect.wrapping_add(1);

            let binding = ImportBinding {
                kind: import.kind,
                addr: import.addr.wrapping_add(slide),
                size: import.size,
                indirect_symbol_index,
                symbol_name: import.symbol_name.clone(),
                dylib: import.dylib.clone(),
                section: import.section.clone(),
            };
            self.patch_single_import_binding(mem, cpu, &binding)?;
            self.imports_by_addr.insert(binding.addr, binding.clone());
            self.imports_by_indirect
                .insert(binding.indirect_symbol_index, binding);
        }
        Ok(())
    }

    fn rebase_dylib_pointers_heuristic(
        &self,
        macho: &MachO32,
        slide: u32,
        mem: &mut GuestMemory,
    ) -> Result<usize, guestmem::Error> {
        let mut patched = 0usize;
        let min_vmaddr = macho.segments.iter().map(|s| s.vmaddr).min().unwrap_or(0);
        let max_vmend = macho
            .segments
            .iter()
            .map(|s| s.vmaddr.wrapping_add(s.vmsize))
            .max()
            .unwrap_or(0);
        let rebase_lo = min_vmaddr.max(0x1000);
        // 某些字符串指针可能不是 4 字节对齐（例如指向 __cstring 的中间位置）。
        // 这里额外放宽这些 section 的候选条件，避免漏掉真实指针。
        let unaligned_pointer_targets: Vec<(u32, u32)> = macho
            .sections
            .iter()
            .filter(|sec| {
                sec.segname == "__TEXT"
                    && matches!(
                        sec.sectname.as_str(),
                        "__cstring"
                            | "__objc_methname"
                            | "__objc_classname"
                            | "__objc_methtype"
                    )
            })
            .map(|sec| (sec.addr, sec.addr.wrapping_add(sec.size)))
            .collect();

        for seg in &macho.segments {
            if seg.segname != "__DATA" && seg.segname != "__DATA_CONST" && seg.segname != "__OBJC" {
                continue;
            }
            if seg.vmsize < 4 {
                continue;
            }
            let base = seg.vmaddr.wrapping_add(slide);
            let words = seg.vmsize / 4;
            for i in 0..words {
                let ptr_addr = base.wrapping_add(i.wrapping_mul(4));
                let raw = Self::read_u32(mem, ptr_addr)?;
                let allow_unaligned = unaligned_pointer_targets
                    .iter()
                    .any(|(lo, hi)| raw >= *lo && raw < *hi);
                if raw >= rebase_lo && raw < max_vmend && ((raw & 0x3) == 0 || allow_unaligned) {
                    let rebased = raw.wrapping_add(slide);
                    mem.write(ptr_addr, &rebased.to_le_bytes())?;
                    patched += 1;
                }
            }
        }
        Ok(patched)
    }

    fn map_dylib_image(
        &mut self,
        path: &Path,
        mem: &mut GuestMemory,
        cpu: &mut Cpu,
    ) -> Result<LoadedImage, DyldError> {
        let macho = parse(path)?;
        let mut file = File::open(path)?;

        let min_vmaddr = macho.segments.iter().map(|s| s.vmaddr).min().unwrap_or(0);
        let max_vmend = macho
            .segments
            .iter()
            .map(|s| s.vmaddr.wrapping_add(s.vmsize))
            .max()
            .unwrap_or(0);
        let image_size = max_vmend.saturating_sub(min_vmaddr).max(0x1000);

        let base = Self::align_up(self.next_dylib_base, 0x1000);
        let slide = base.wrapping_sub(min_vmaddr);
        self.next_dylib_base =
            Self::align_up(base.wrapping_add(image_size).wrapping_add(0x10000), 0x1000);

        for seg in &macho.segments {
            if seg.segname == "__PAGEZERO" {
                continue;
            }
            let load_addr = seg.vmaddr.wrapping_add(slide);
            let runtime_prot = runtime_prot_from_initprot(seg.initprot);
            let load_prot = runtime_prot | Prot::WRITE;
            mem.map(load_addr, seg.vmsize, load_prot)?;

            if seg.filesize > 0 {
                let fileoff_abs = macho.file_base_offset + seg.fileoff as u64;
                file.seek(SeekFrom::Start(fileoff_abs))?;
                let mut buf = vec![0u8; seg.filesize as usize];
                file.read_exact(&mut buf)?;
                mem.write(load_addr, &buf)?;
            }
        }

        let rebased = self.rebase_dylib_pointers_heuristic(&macho, slide, mem)?;
        if self.trace && rebased > 0 {
            eprintln!(
                "[DYLD] heuristic rebased {} pointer(s) in '{}'",
                rebased,
                path.display()
            );
        }

        self.register_macho_imports_with_slide(&macho, slide, mem, cpu)?;

        let mut exports = HashMap::new();
        for export in &macho.exports {
            exports.insert(export.name.clone(), export.value.wrapping_add(slide));
        }

        let mut init_funcs = Vec::new();
        for section in &macho.sections {
            if (section.flags & SECTION_TYPE_MASK) != S_MOD_INIT_FUNC_POINTERS {
                continue;
            }
            let count = (section.size / 4) as usize;
            for i in 0..count {
                let slot_addr = section
                    .addr
                    .wrapping_add(slide)
                    .wrapping_add((i as u32).wrapping_mul(4));
                let Ok(raw_fn) = Self::read_u32(mem, slot_addr) else {
                    continue;
                };
                if raw_fn == 0 {
                    continue;
                }
                let fn_addr = if raw_fn >= min_vmaddr && raw_fn < max_vmend {
                    raw_fn.wrapping_add(slide)
                } else {
                    raw_fn
                };
                let low = min_vmaddr.wrapping_add(slide);
                let high = max_vmend.wrapping_add(slide);
                if fn_addr >= low && fn_addr < high {
                    init_funcs.push(fn_addr);
                }
            }
        }

        let handle = self.fake_dlopen_next;
        self.fake_dlopen_next = self.fake_dlopen_next.wrapping_add(1).max(1);
        Ok(LoadedImage {
            handle,
            path: path.to_path_buf(),
            slide,
            exports,
            init_funcs,
            initializers_ran: false,
        })
    }

    fn ensure_dylib_loaded(
        &mut self,
        dylib_name: &str,
        mem: &mut GuestMemory,
        cpu: &mut Cpu,
    ) -> Result<Option<u32>, DyldError> {
        let Some(resolved_path) = self.resolve_dlopen_path(dylib_name) else {
            return Ok(None);
        };
        let key = resolved_path.to_string_lossy().to_string();
        if let Some(&handle) = self.handle_by_path.get(&key) {
            return Ok(Some(handle));
        }

        match self.map_dylib_image(&resolved_path, mem, cpu) {
            Ok(image) => {
                let handle = image.handle;
                self.handle_by_path.insert(key, handle);
                self.loaded_images.insert(handle, image.clone());
                if self.trace {
                    eprintln!(
                        "[DYLD] auto-loaded dependency '{}' -> handle {} (exports={}, init_funcs={})",
                        image.path.display(),
                        handle,
                        image.exports.len(),
                        image.init_funcs.len()
                    );
                }
                Ok(Some(handle))
            }
            Err(err) => {
                if self.trace {
                    eprintln!(
                        "[DYLD] failed to auto-load dependency '{}' ({})",
                        dylib_name, err
                    );
                }
                Ok(None)
            }
        }
    }

    fn try_resolve_import_symbol(
        &mut self,
        binding: &ImportBinding,
        mem: &mut GuestMemory,
        cpu: &mut Cpu,
    ) -> Result<Option<(u32, u32)>, DyldError> {
        let symbol = match &binding.symbol_name {
            Some(s) => s.as_str(),
            None => return Ok(None),
        };
        let dylib = match &binding.dylib {
            Some(s) => s.as_str(),
            None => return Ok(None),
        };

        let Some(handle) = self.ensure_dylib_loaded(dylib, mem, cpu)? else {
            return Ok(None);
        };
        let Some(image) = self.loaded_images.get(&handle) else {
            return Ok(None);
        };
        Ok(Self::lookup_export(&image.exports, symbol).map(|target| (target, handle)))
    }

    fn take_pending_initializers(&mut self, handle: u32) -> Vec<u32> {
        let Some(image) = self.loaded_images.get_mut(&handle) else {
            return Vec::new();
        };
        if image.initializers_ran || image.init_funcs.is_empty() {
            return Vec::new();
        }
        image.initializers_ran = true;
        image.init_funcs.clone()
    }

    fn schedule_calls_before_target(
        &self,
        cpu: &mut Cpu,
        mem: &mut GuestMemory,
        calls: &[u32],
        final_target: u32,
    ) -> Result<(), guestmem::Error> {
        fn push_u32(cpu: &mut Cpu, mem: &mut GuestMemory, value: u32) -> Result<(), guestmem::Error> {
            cpu.esp = cpu.esp.wrapping_sub(4);
            mem.write(cpu.esp, &value.to_le_bytes())
        }

        if calls.is_empty() {
            cpu.eip = final_target;
            return Ok(());
        }
        push_u32(cpu, mem, final_target)?;
        for &addr in calls.iter().skip(1).rev() {
            push_u32(cpu, mem, addr)?;
        }
        cpu.eip = calls[0];
        Ok(())
    }

    fn read_u32(mem: &GuestMemory, addr: u32) -> Result<u32, guestmem::Error> {
        let mut buf = [0u8; 4];
        mem.read(addr, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(mem: &GuestMemory, addr: u32) -> Result<u64, guestmem::Error> {
        let mut buf = [0u8; 8];
        mem.read(addr, &mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_c_string(
        mem: &GuestMemory,
        addr: u32,
        max_len: usize,
    ) -> Result<String, guestmem::Error> {
        let mut out = Vec::new();
        for i in 0..max_len {
            let mut b = [0u8; 1];
            mem.read(addr.wrapping_add(i as u32), &mut b)?;
            if b[0] == 0 {
                break;
            }
            out.push(b[0]);
        }
        Ok(String::from_utf8_lossy(&out).to_string())
    }

    fn read_arg_u32(mem: &GuestMemory, cpu: &Cpu, arg_index: u32) -> Result<u32, guestmem::Error> {
        let addr = cpu
            .esp
            .wrapping_add(4)
            .wrapping_add(arg_index.wrapping_mul(4));
        Self::read_u32(mem, addr)
    }

    fn read_vararg_u32(mem: &GuestMemory, cursor: &mut u32) -> Result<u32, guestmem::Error> {
        let value = Self::read_u32(mem, *cursor)?;
        *cursor = cursor.wrapping_add(4);
        Ok(value)
    }

    fn read_vararg_u64(mem: &GuestMemory, cursor: &mut u32) -> Result<u64, guestmem::Error> {
        let lo = Self::read_vararg_u32(mem, cursor)? as u64;
        let hi = Self::read_vararg_u32(mem, cursor)? as u64;
        Ok((hi << 32) | lo)
    }

    fn apply_printf_width(mut text: String, width: Option<i32>, left_align: bool, pad_zero: bool) -> String {
        let Some(width) = width else {
            return text;
        };
        let width = width.max(0) as usize;
        if text.len() >= width {
            return text;
        }
        let pad_len = width - text.len();
        let pad_char = if pad_zero { '0' } else { ' ' };
        let padding = std::iter::repeat_n(pad_char, pad_len).collect::<String>();
        if left_align {
            text.push_str(&padding);
            text
        } else if pad_zero && (text.starts_with('-') || text.starts_with('+')) {
            // 让符号位保持在最前，零填充跟在符号位后。
            let sign = text.remove(0);
            let mut out = String::with_capacity(width);
            out.push(sign);
            out.push_str(&padding);
            out.push_str(&text);
            out
        } else {
            let mut out = String::with_capacity(width);
            out.push_str(&padding);
            out.push_str(&text);
            out
        }
    }

    fn format_unsigned(mut value: u64, base: u32, upper: bool) -> String {
        if value == 0 {
            return "0".to_string();
        }
        let digits = if upper {
            b"0123456789ABCDEF"
        } else {
            b"0123456789abcdef"
        };
        let mut buf = [0u8; 64];
        let mut i = buf.len();
        while value != 0 {
            let d = (value % (base as u64)) as usize;
            i -= 1;
            buf[i] = digits[d];
            value /= base as u64;
        }
        String::from_utf8_lossy(&buf[i..]).to_string()
    }

    fn format_integer_with_precision(mut digits: String, precision: Option<usize>) -> String {
        if let Some(precision) = precision {
            if precision == 0 && digits == "0" {
                return String::new();
            }
            if digits.len() < precision {
                let zeros = "0".repeat(precision - digits.len());
                digits = format!("{}{}", zeros, digits);
            }
        }
        digits
    }

    fn format_variadic_printf(
        mem: &GuestMemory,
        fmt: &str,
        vararg_cursor: &mut u32,
    ) -> Result<String, guestmem::Error> {
        let bytes = fmt.as_bytes();
        let mut out = String::new();
        let mut i = 0usize;

        while i < bytes.len() {
            if bytes[i] != b'%' {
                out.push(bytes[i] as char);
                i += 1;
                continue;
            }
            i += 1;
            if i >= bytes.len() {
                out.push('%');
                break;
            }
            if bytes[i] == b'%' {
                out.push('%');
                i += 1;
                continue;
            }

            let mut left_align = false;
            let mut plus_sign = false;
            let mut space_sign = false;
            let mut alt_form = false;
            let mut zero_pad = false;
            while i < bytes.len() {
                match bytes[i] as char {
                    '-' => left_align = true,
                    '+' => plus_sign = true,
                    ' ' => space_sign = true,
                    '#' => alt_form = true,
                    '0' => zero_pad = true,
                    _ => break,
                }
                i += 1;
            }

            let mut width: Option<i32> = None;
            if i < bytes.len() && bytes[i] == b'*' {
                width = Some(Self::read_vararg_u32(mem, vararg_cursor)? as i32);
                i += 1;
            } else {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i > start {
                    width = Some(String::from_utf8_lossy(&bytes[start..i]).parse().unwrap_or(0));
                }
            }
            if let Some(w) = width {
                if w < 0 {
                    width = Some(-w);
                    left_align = true;
                }
            }

            let mut precision: Option<usize> = None;
            if i < bytes.len() && bytes[i] == b'.' {
                i += 1;
                if i < bytes.len() && bytes[i] == b'*' {
                    let p = Self::read_vararg_u32(mem, vararg_cursor)? as i32;
                    if p >= 0 {
                        precision = Some(p as usize);
                    }
                    i += 1;
                } else {
                    let start = i;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    let p = if i > start {
                        String::from_utf8_lossy(&bytes[start..i]).parse().unwrap_or(0)
                    } else {
                        0
                    };
                    precision = Some(p);
                }
            }

            let mut len_h = false;
            let mut len_l = false;
            let mut len_ll = false;
            if i < bytes.len() {
                match bytes[i] as char {
                    'h' => {
                        len_h = true;
                        i += 1;
                        if i < bytes.len() && bytes[i] == b'h' {
                            i += 1;
                        }
                    }
                    'l' => {
                        len_l = true;
                        i += 1;
                        if i < bytes.len() && bytes[i] == b'l' {
                            len_ll = true;
                            i += 1;
                        }
                    }
                    'z' | 't' | 'j' | 'L' => {
                        // i386 下按 32 位对待，L/ll 以外无需特别区分。
                        i += 1;
                    }
                    _ => {}
                }
            }

            if i >= bytes.len() {
                out.push('%');
                break;
            }
            let spec = bytes[i] as char;
            i += 1;

            let mut piece = String::new();
            let mut numeric = false;

            match spec {
                's' => {
                    let ptr = Self::read_vararg_u32(mem, vararg_cursor)?;
                    let max_len = precision.unwrap_or(8192).min(1 << 20);
                    let mut s = if ptr == 0 {
                        "(null)".to_string()
                    } else {
                        Self::read_c_string(mem, ptr, max_len).unwrap_or_else(|_| "<bad-ptr>".to_string())
                    };
                    if let Some(p) = precision {
                        s = s.chars().take(p).collect();
                    }
                    piece = s;
                }
                'c' => {
                    let c = Self::read_vararg_u32(mem, vararg_cursor)? as u8 as char;
                    piece.push(c);
                }
                'd' | 'i' => {
                    numeric = true;
                    let value = if len_ll {
                        Self::read_vararg_u64(mem, vararg_cursor)? as i64
                    } else {
                        let mut v = Self::read_vararg_u32(mem, vararg_cursor)? as i32 as i64;
                        if len_h {
                            v = (v as i16) as i64;
                        }
                        v
                    };
                    let neg = value < 0;
                    let abs = value.unsigned_abs();
                    let mut digits = Self::format_unsigned(abs, 10, false);
                    digits = Self::format_integer_with_precision(digits, precision);
                    if neg {
                        piece.push('-');
                    } else if plus_sign {
                        piece.push('+');
                    } else if space_sign {
                        piece.push(' ');
                    }
                    piece.push_str(&digits);
                }
                'u' => {
                    numeric = true;
                    let mut value = if len_ll {
                        Self::read_vararg_u64(mem, vararg_cursor)?
                    } else {
                        Self::read_vararg_u32(mem, vararg_cursor)? as u64
                    };
                    if len_h {
                        value = (value as u16) as u64;
                    }
                    let mut digits = Self::format_unsigned(value, 10, false);
                    digits = Self::format_integer_with_precision(digits, precision);
                    piece.push_str(&digits);
                }
                'o' => {
                    numeric = true;
                    let value = if len_ll {
                        Self::read_vararg_u64(mem, vararg_cursor)?
                    } else {
                        Self::read_vararg_u32(mem, vararg_cursor)? as u64
                    };
                    let mut digits = Self::format_unsigned(value, 8, false);
                    digits = Self::format_integer_with_precision(digits, precision);
                    if alt_form && !digits.starts_with('0') {
                        piece.push('0');
                    }
                    piece.push_str(&digits);
                }
                'x' | 'X' => {
                    numeric = true;
                    let upper = spec == 'X';
                    let value = if len_ll {
                        Self::read_vararg_u64(mem, vararg_cursor)?
                    } else {
                        Self::read_vararg_u32(mem, vararg_cursor)? as u64
                    };
                    let mut digits = Self::format_unsigned(value, 16, upper);
                    digits = Self::format_integer_with_precision(digits, precision);
                    if alt_form && value != 0 {
                        if upper {
                            piece.push_str("0X");
                        } else {
                            piece.push_str("0x");
                        }
                    }
                    piece.push_str(&digits);
                }
                'p' => {
                    numeric = true;
                    let value = Self::read_vararg_u32(mem, vararg_cursor)?;
                    let _ = write!(piece, "0x{:08x}", value);
                }
                'f' | 'F' | 'e' | 'E' | 'g' | 'G' => {
                    numeric = true;
                    let raw = Self::read_vararg_u64(mem, vararg_cursor)?;
                    let value = f64::from_bits(raw);
                    let prec = precision.unwrap_or(6);
                    let formatted = match spec {
                        'e' => format!("{:.*e}", prec, value),
                        'E' => format!("{:.*E}", prec, value),
                        'g' => format!("{:.*}", prec, value),
                        'G' => format!("{:.*}", prec, value).to_uppercase(),
                        _ => format!("{:.*}", prec, value),
                    };
                    piece.push_str(&formatted);
                }
                _ => {
                    // 遇到未知格式符时保底原样输出，防止日志/路径彻底损坏。
                    piece.push('%');
                    if len_ll {
                        piece.push_str("ll");
                    } else if len_l {
                        piece.push('l');
                    } else if len_h {
                        piece.push('h');
                    }
                    piece.push(spec);
                }
            }

            let pad_zero = zero_pad && !left_align && precision.is_none() && numeric;
            let piece = Self::apply_printf_width(piece, width, left_align, pad_zero);
            out.push_str(&piece);
        }

        Ok(out)
    }

    fn write_c_buffer(
        mem: &mut GuestMemory,
        dst: u32,
        size: u32,
        text: &str,
    ) -> Result<(), guestmem::Error> {
        if size == 0 {
            return Ok(());
        }
        let bytes = text.as_bytes();
        let max_copy = (size - 1) as usize;
        let copy_len = bytes.len().min(max_copy);
        if copy_len != 0 {
            mem.write(dst, &bytes[..copy_len])?;
        }
        mem.write(dst.wrapping_add(copy_len as u32), &[0])?;
        Ok(())
    }

    fn is_ascii_space(b: u8) -> bool {
        matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
    }

    fn skip_ascii_space(bytes: &[u8], pos: &mut usize) {
        while *pos < bytes.len() && Self::is_ascii_space(bytes[*pos]) {
            *pos += 1;
        }
    }

    fn digit_value(b: u8) -> Option<u32> {
        match b {
            b'0'..=b'9' => Some((b - b'0') as u32),
            b'a'..=b'f' => Some((b - b'a') as u32 + 10),
            b'A'..=b'F' => Some((b - b'A') as u32 + 10),
            _ => None,
        }
    }

    fn scan_int_prefix(
        input: &[u8],
        width: Option<usize>,
        spec: char,
    ) -> Option<(i64, usize)> {
        let max = width.unwrap_or(input.len()).min(input.len());
        if max == 0 {
            return None;
        }

        let mut i = 0usize;
        let mut neg = false;
        if i < max {
            match input[i] {
                b'+' => i += 1,
                b'-' => {
                    neg = true;
                    i += 1;
                }
                _ => {}
            }
        }
        if i >= max {
            return None;
        }

        let mut base = match spec {
            'x' | 'X' | 'p' => 16,
            'o' => 8,
            _ => 10,
        };
        let auto_base = spec == 'i';
        if auto_base {
            if i + 1 < max
                && input[i] == b'0'
                && (input[i + 1] == b'x' || input[i + 1] == b'X')
            {
                base = 16;
                i += 2;
            } else if input[i] == b'0' {
                base = 8;
            } else {
                base = 10;
            }
        } else if base == 16
            && i + 1 < max
            && input[i] == b'0'
            && (input[i + 1] == b'x' || input[i + 1] == b'X')
        {
            i += 2;
        }

        let digits_start = i;
        let mut acc: u128 = 0;
        while i < max {
            let Some(d) = Self::digit_value(input[i]) else {
                break;
            };
            if d >= base {
                break;
            }
            acc = acc.saturating_mul(base as u128).saturating_add(d as u128);
            i += 1;
        }
        if i == digits_start {
            return None;
        }

        let mut out = acc.min(i64::MAX as u128) as i64;
        if neg {
            out = out.saturating_neg();
        }
        Some((out, i))
    }

    fn scan_float_prefix(input: &[u8], width: Option<usize>) -> Option<(f64, usize)> {
        let max = width.unwrap_or(input.len()).min(input.len());
        if max == 0 {
            return None;
        }

        let mut i = 0usize;
        if i < max && (input[i] == b'+' || input[i] == b'-') {
            i += 1;
        }

        // 兼容 inf/nan 文本。
        if i + 2 < max {
            let tail = &input[i..max];
            if tail.len() >= 3 && tail[..3].eq_ignore_ascii_case(b"inf") {
                let used = i + 3;
                let text = String::from_utf8_lossy(&input[..used]).to_string();
                if let Ok(v) = text.parse::<f64>() {
                    return Some((v, used));
                }
            }
            if tail.len() >= 3 && tail[..3].eq_ignore_ascii_case(b"nan") {
                let used = i + 3;
                let text = String::from_utf8_lossy(&input[..used]).to_string();
                if let Ok(v) = text.parse::<f64>() {
                    return Some((v, used));
                }
            }
        }

        let int_start = i;
        while i < max && input[i].is_ascii_digit() {
            i += 1;
        }
        let mut has_digit = i > int_start;
        if i < max && input[i] == b'.' {
            i += 1;
            let frac_start = i;
            while i < max && input[i].is_ascii_digit() {
                i += 1;
            }
            has_digit |= i > frac_start;
        }
        if !has_digit {
            return None;
        }

        let exp_pos = i;
        if i < max && (input[i] == b'e' || input[i] == b'E') {
            let mut j = i + 1;
            if j < max && (input[j] == b'+' || input[j] == b'-') {
                j += 1;
            }
            let exp_start = j;
            while j < max && input[j].is_ascii_digit() {
                j += 1;
            }
            if j > exp_start {
                i = j;
            } else {
                // 没有指数数字，回退到 e/E 之前。
                i = exp_pos;
            }
        }

        let text = String::from_utf8_lossy(&input[..i]).to_string();
        text.parse::<f64>().ok().map(|v| (v, i))
    }

    fn unix_time_parts() -> (u32, u32) {
        let dur = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        let secs = dur.as_secs().min(u32::MAX as u64) as u32;
        let usec = dur.subsec_micros();
        (secs, usec)
    }

    fn write_i32(mem: &mut GuestMemory, addr: u32, value: i32) -> Result<(), guestmem::Error> {
        mem.write(addr, &value.to_le_bytes())
    }

    fn write_timeval32(
        mem: &mut GuestMemory,
        addr: u32,
        sec: u32,
        usec: u32,
    ) -> Result<(), guestmem::Error> {
        mem.write(addr, &(sec as i32).to_le_bytes())?;
        mem.write(addr.wrapping_add(4), &(usec as i32).to_le_bytes())
    }

    fn ensure_localtime_tm_ptr(&mut self, mem: &mut GuestMemory) -> Result<u32, guestmem::Error> {
        if self.localtime_tm_ptr != 0 {
            return Ok(self.localtime_tm_ptr);
        }
        // struct tm (32-bit Darwin) 预留 64 字节足够容纳常见字段布局。
        let ptr = self.alloc_heap(mem, 64, true)?;
        if ptr == 0 {
            return Err(guestmem::Error::AddressNotMapped(0));
        }
        self.localtime_tm_ptr = ptr;
        Ok(ptr)
    }

    fn write_tm32(
        mem: &mut GuestMemory,
        tm_ptr: u32,
        secs_since_epoch: u32,
    ) -> Result<(), guestmem::Error> {
        let sec = (secs_since_epoch % 60) as i32;
        let min = ((secs_since_epoch / 60) % 60) as i32;
        let hour = ((secs_since_epoch / 3600) % 24) as i32;
        let days = secs_since_epoch / 86_400;
        let wday = ((days + 4) % 7) as i32; // 1970-01-01 是周四
        let yday = (days % 365) as i32;
        let year = 1970 + (days / 365) as i32;
        let mon = ((yday / 30) % 12) as i32;
        let mday = ((yday % 30) + 1) as i32;

        Self::write_i32(mem, tm_ptr, sec)?;
        Self::write_i32(mem, tm_ptr.wrapping_add(4), min)?;
        Self::write_i32(mem, tm_ptr.wrapping_add(8), hour)?;
        Self::write_i32(mem, tm_ptr.wrapping_add(12), mday)?;
        Self::write_i32(mem, tm_ptr.wrapping_add(16), mon)?;
        Self::write_i32(mem, tm_ptr.wrapping_add(20), year - 1900)?;
        Self::write_i32(mem, tm_ptr.wrapping_add(24), wday)?;
        Self::write_i32(mem, tm_ptr.wrapping_add(28), yday)?;
        Self::write_i32(mem, tm_ptr.wrapping_add(32), 0)?; // tm_isdst
        Self::write_i32(mem, tm_ptr.wrapping_add(36), 0)?; // tm_gmtoff
        Self::write_i32(mem, tm_ptr.wrapping_add(40), 0)?; // tm_zone ptr (NULL)
        Ok(())
    }

    fn c_strlen(mem: &GuestMemory, addr: u32, max_len: usize) -> Result<usize, guestmem::Error> {
        for i in 0..max_len {
            let mut b = [0u8; 1];
            mem.read(addr.wrapping_add(i as u32), &mut b)?;
            if b[0] == 0 {
                return Ok(i);
            }
        }
        Ok(max_len)
    }

    fn open_options_from_flags(flags: u32) -> OpenOptions {
        const O_ACCMODE: u32 = 0x0003;
        const O_WRONLY: u32 = 0x0001;
        const O_RDWR: u32 = 0x0002;
        const O_APPEND: u32 = 0x0008;
        const O_CREAT: u32 = 0x0200;
        const O_TRUNC: u32 = 0x0400;
        const O_EXCL: u32 = 0x0800;

        let mut opts = OpenOptions::new();
        match flags & O_ACCMODE {
            O_WRONLY => {
                opts.write(true);
            }
            O_RDWR => {
                opts.read(true).write(true);
            }
            _ => {
                opts.read(true);
            }
        }
        if (flags & O_CREAT) != 0 {
            opts.create(true);
        }
        if (flags & O_TRUNC) != 0 {
            opts.truncate(true);
        }
        if (flags & O_EXCL) != 0 {
            opts.create_new(true);
        }
        if (flags & O_APPEND) != 0 {
            opts.append(true);
        }
        opts
    }

    fn open_options_from_mode(mode: &str) -> OpenOptions {
        let mut opts = OpenOptions::new();
        let mut chars = mode.chars();
        let first = chars.next().unwrap_or('r');
        let plus = mode.contains('+');

        match first {
            'r' => {
                opts.read(true);
                if plus {
                    opts.write(true);
                }
            }
            'w' => {
                opts.write(true).create(true).truncate(true);
                if plus {
                    opts.read(true);
                }
            }
            'a' => {
                opts.write(true).append(true).create(true);
                if plus {
                    opts.read(true);
                }
            }
            _ => {
                opts.read(true);
            }
        }

        opts
    }

    fn resolve_host_path_candidates(&self, path: &str) -> Vec<PathBuf> {
        let mut candidates = vec![PathBuf::from(path)];
        let raw = Path::new(path);
        if !raw.is_absolute() && !path.is_empty() {
            if let Some(exec_dir) = &self.main_exec_dir {
                candidates.push(exec_dir.join(raw));
            }
        }
        candidates
    }

    fn errno_from_io(err: &io::Error) -> u32 {
        match err.kind() {
            io::ErrorKind::NotFound => ERRNO_ENOENT,
            io::ErrorKind::PermissionDenied => ERRNO_EACCES,
            io::ErrorKind::AlreadyExists => ERRNO_EEXIST,
            io::ErrorKind::InvalidInput => ERRNO_EINVAL,
            _ => ERRNO_EIO,
        }
    }

    fn ensure_errno_slot(&mut self, mem: &mut GuestMemory) -> Result<u32, guestmem::Error> {
        if self.errno_slot != 0 {
            return Ok(self.errno_slot);
        }
        let slot = self.alloc_heap(mem, 4, true)?;
        if slot == 0 {
            return Err(guestmem::Error::AddressNotMapped(0));
        }
        self.errno_slot = slot;
        Ok(slot)
    }

    fn set_errno(&mut self, mem: &mut GuestMemory, errno: u32) -> Result<(), guestmem::Error> {
        let slot = self.ensure_errno_slot(mem)?;
        mem.write(slot, &errno.to_le_bytes())
    }

    fn clear_errno(&mut self, mem: &mut GuestMemory) -> Result<(), guestmem::Error> {
        self.set_errno(mem, 0)
    }

    fn ensure_pthread_self_ptr(&mut self, mem: &mut GuestMemory) -> Result<u32, guestmem::Error> {
        if self.pthread_self_ptr != 0 {
            return Ok(self.pthread_self_ptr);
        }
        let ptr = self.alloc_heap(mem, 0x100, true)?;
        if ptr == 0 {
            return Err(guestmem::Error::AddressNotMapped(0));
        }
        // 简化版 pthread 结构：首字存自身地址，便于调试时快速识别。
        mem.write(ptr, &ptr.to_le_bytes())?;
        self.pthread_self_ptr = ptr;
        Ok(ptr)
    }

    fn alloc_stream_struct(
        &mut self,
        mem: &mut GuestMemory,
        fd: u32,
    ) -> Result<u32, guestmem::Error> {
        let ptr = self.alloc_heap(mem, 8, true)?;
        if ptr == 0 {
            return Err(guestmem::Error::AddressNotMapped(0));
        }
        mem.write(ptr, &fd.to_le_bytes())?;
        self.host_streams.insert(ptr, fd);
        Ok(ptr)
    }

    fn stream_fd(&self, mem: &GuestMemory, stream_ptr: u32) -> Option<u32> {
        if stream_ptr == 0 {
            return None;
        }
        if let Some(&fd) = self.host_streams.get(&stream_ptr) {
            return Some(fd);
        }
        Self::read_u32(mem, stream_ptr).ok()
    }

    fn ensure_stdin_var_ptr(&mut self, mem: &mut GuestMemory) -> Result<u32, guestmem::Error> {
        if self.stdin_var_ptr != 0 {
            return Ok(self.stdin_var_ptr);
        }
        if self.stdin_stream_ptr == 0 {
            self.stdin_stream_ptr = self.alloc_stream_struct(mem, 0)?;
        }
        let var_ptr = self.alloc_heap(mem, 4, true)?;
        if var_ptr == 0 {
            return Err(guestmem::Error::AddressNotMapped(0));
        }
        mem.write(var_ptr, &self.stdin_stream_ptr.to_le_bytes())?;
        self.stdin_var_ptr = var_ptr;
        Ok(var_ptr)
    }

    fn ensure_stdout_var_ptr(&mut self, mem: &mut GuestMemory) -> Result<u32, guestmem::Error> {
        if self.stdout_var_ptr != 0 {
            return Ok(self.stdout_var_ptr);
        }
        if self.stdout_stream_ptr == 0 {
            self.stdout_stream_ptr = self.alloc_stream_struct(mem, 1)?;
        }
        let var_ptr = self.alloc_heap(mem, 4, true)?;
        if var_ptr == 0 {
            return Err(guestmem::Error::AddressNotMapped(0));
        }
        mem.write(var_ptr, &self.stdout_stream_ptr.to_le_bytes())?;
        self.stdout_var_ptr = var_ptr;
        Ok(var_ptr)
    }

    fn ensure_stderr_var_ptr(&mut self, mem: &mut GuestMemory) -> Result<u32, guestmem::Error> {
        if self.stderr_var_ptr != 0 {
            return Ok(self.stderr_var_ptr);
        }
        if self.stderr_stream_ptr == 0 {
            self.stderr_stream_ptr = self.alloc_stream_struct(mem, 2)?;
        }
        let var_ptr = self.alloc_heap(mem, 4, true)?;
        if var_ptr == 0 {
            return Err(guestmem::Error::AddressNotMapped(0));
        }
        mem.write(var_ptr, &self.stderr_stream_ptr.to_le_bytes())?;
        self.stderr_var_ptr = var_ptr;
        Ok(var_ptr)
    }

    fn ensure_stack_chk_guard_var_ptr(
        &mut self,
        mem: &mut GuestMemory,
    ) -> Result<u32, guestmem::Error> {
        if self.stack_chk_guard_var_ptr != 0 {
            return Ok(self.stack_chk_guard_var_ptr);
        }
        let var_ptr = self.alloc_heap(mem, 4, true)?;
        if var_ptr == 0 {
            return Err(guestmem::Error::AddressNotMapped(0));
        }
        let (secs, usec) = Self::unix_time_parts();
        let guard = secs.rotate_left(7) ^ usec.rotate_left(19) ^ 0xa5a5_5a5a;
        mem.write(var_ptr, &guard.to_le_bytes())?;
        self.stack_chk_guard_var_ptr = var_ptr;
        Ok(var_ptr)
    }

    fn next_file_fd(&mut self) -> u32 {
        let fd = self.host_next_fd;
        self.host_next_fd = self.host_next_fd.wrapping_add(1).max(3);
        fd
    }

    fn alloc_heap(
        &mut self,
        mem: &mut GuestMemory,
        size: u32,
        zero: bool,
    ) -> Result<u32, guestmem::Error> {
        if self.heap_base == 0 || self.heap_size == 0 {
            return Ok(0);
        }
        let size = size.max(1);
        let size_aligned = (size.wrapping_add(15)) & !15;
        let end = self.heap_base + self.heap_size;
        if self.heap_cursor.saturating_add(size_aligned) > end {
            return Ok(0);
        }
        let addr = self.heap_cursor;
        self.heap_cursor = self.heap_cursor.wrapping_add(size_aligned);
        self.heap_allocations.insert(addr, size_aligned);
        if zero {
            let zeros = vec![0u8; size_aligned as usize];
            mem.write(addr, &zeros)?;
        }
        Ok(addr)
    }

    fn free_heap(&mut self, ptr: u32) {
        self.heap_allocations.remove(&ptr);
    }

    fn realloc_heap(
        &mut self,
        mem: &mut GuestMemory,
        ptr: u32,
        new_size: u32,
    ) -> Result<u32, guestmem::Error> {
        if ptr == 0 {
            return self.alloc_heap(mem, new_size, false);
        }
        if new_size == 0 {
            self.free_heap(ptr);
            return Ok(0);
        }

        let old_size = *self.heap_allocations.get(&ptr).unwrap_or(&0);
        let new_ptr = self.alloc_heap(mem, new_size, false)?;
        if new_ptr == 0 {
            return Ok(0);
        }
        if old_size != 0 {
            let copy_len = old_size.min(new_size) as usize;
            let mut buf = vec![0u8; copy_len];
            mem.read(ptr, &mut buf)?;
            mem.write(new_ptr, &buf)?;
        }
        self.free_heap(ptr);
        Ok(new_ptr)
    }

    fn alloc_scratch_string(
        &mut self,
        mem: &mut GuestMemory,
        text: &str,
    ) -> Result<u32, guestmem::Error> {
        if self.scratch_base == 0 || self.scratch_size == 0 {
            return Err(guestmem::Error::AddressNotMapped(0));
        }

        let bytes = text.as_bytes();
        let need = (bytes.len() + 1) as u32;
        if need >= self.scratch_size {
            return Err(guestmem::Error::AddressNotMapped(self.scratch_base));
        }

        let end = self.scratch_base + self.scratch_size;
        if self.scratch_cursor + need > end {
            self.scratch_cursor = self.scratch_base;
        }

        let addr = self.scratch_cursor;
        mem.write(addr, bytes)?;
        mem.write(addr + bytes.len() as u32, &[0])?;
        self.scratch_cursor = (self.scratch_cursor + need + 3) & !3;
        Ok(addr)
    }

    fn set_dlerror(
        &mut self,
        mem: &mut GuestMemory,
        msg: Option<String>,
    ) -> Result<(), guestmem::Error> {
        self.last_dlerror = msg;
        if let Some(text) = self.last_dlerror.clone() {
            self.last_dlerror_ptr = self.alloc_scratch_string(mem, &text)?;
        } else {
            self.last_dlerror_ptr = 0;
        }
        Ok(())
    }

    fn import_return(cpu: &mut Cpu, mem: &GuestMemory) -> Result<(), DyldError> {
        let ret_addr = Self::read_u32(mem, cpu.esp)?;
        cpu.esp = cpu.esp.wrapping_add(4);
        cpu.eip = ret_addr;
        Ok(())
    }

    fn noop_import(
        &mut self,
        cpu: &mut Cpu,
        mem: &mut GuestMemory,
        binding: &ImportBinding,
    ) -> Result<(), DyldError> {
        let symbol = binding.display_name();
        let dylib = binding
            .dylib
            .clone()
            .unwrap_or_else(|| "<unknown dylib>".to_string());
        let key = format!("{}@{}", symbol, dylib);

        let mut fallback_eax = 0u32;
        if symbol.starts_with("__ZN")
            && (symbol.contains("C1E")
                || symbol.contains("C2E")
                || symbol.contains("D1E")
                || symbol.contains("D2E"))
        {
            if let Ok(this_ptr) = Self::read_arg_u32(mem, cpu, 0) {
                fallback_eax = this_ptr;
            }
        }

        if self.trace && self.warned_unimplemented.insert(key.clone()) {
            eprintln!(
                "[DYLD] unimplemented import '{}' from '{}', fallback return={:#010x}",
                symbol, dylib, fallback_eax
            );
        }

        cpu.eax = fallback_eax;
        Self::import_return(cpu, mem)?;
        Ok(())
    }

    pub fn describe_import(&self, eip: u32, indirect_symbol_index: u32) -> Option<String> {
        let binding = self
            .imports_by_addr
            .get(&eip)
            .or_else(|| self.imports_by_indirect.get(&indirect_symbol_index))?;
        let symbol = binding.display_name();
        let dylib = binding
            .dylib
            .clone()
            .unwrap_or_else(|| "<unknown dylib>".to_string());
        Some(format!(
            "{} (dylib={}, section={}, indirect={})",
            symbol, dylib, binding.section, binding.indirect_symbol_index
        ))
    }

    pub fn handle_unresolved_import(
        &mut self,
        cpu: &mut Cpu,
        mem: &mut GuestMemory,
        eip: u32,
        indirect_symbol_index: u32,
    ) -> Result<(), DyldError> {
        let binding = self
            .imports_by_addr
            .get(&eip)
            .cloned()
            .or_else(|| {
                self.imports_by_indirect
                    .get(&indirect_symbol_index)
                    .cloned()
            })
            .ok_or(DyldError::BindingNotFound {
                eip,
                indirect_symbol_index,
            })?;

        let symbol = binding.display_name();
        let symbol_norm = symbol.strip_prefix('_').unwrap_or(symbol.as_str());
        let symbol_core = symbol_norm.strip_prefix('_').unwrap_or(symbol_norm);

        // C++ operator new/new[] 的常见 Itanium ABI 名称。
        if matches!(
            symbol_core,
            "Znwj" | "Znwm" | "Znaj" | "Znam" | "ZnwjRKSt9nothrow_t" | "ZnwmRKSt9nothrow_t"
        ) {
            let size = Self::read_arg_u32(mem, cpu, 0)?;
            cpu.eax = self.alloc_heap(mem, size, false)?;
            Self::import_return(cpu, mem)?;
            return Ok(());
        }

        // C++ operator delete/delete[] 的常见名称。
        if symbol_core.starts_with("ZdlPv") || symbol_core.starts_with("ZdaPv") {
            let ptr = Self::read_arg_u32(mem, cpu, 0)?;
            self.free_heap(ptr);
            cpu.eax = 0;
            Self::import_return(cpu, mem)?;
            return Ok(());
        }

        if let Some((target, handle)) = self.try_resolve_import_symbol(&binding, mem, cpu)? {
            let init_funcs = self.take_pending_initializers(handle);
            if self.trace {
                eprintln!(
                    "[DYLD] resolved import '{}' from '{}' -> {:#010x}",
                    binding.display_name(),
                    binding
                        .dylib
                        .clone()
                        .unwrap_or_else(|| "<unknown dylib>".to_string()),
                    target
                );
                if !init_funcs.is_empty() {
                    eprintln!(
                        "[DYLD] running {} initializer(s) before first call into handle {}",
                        init_funcs.len(),
                        handle
                    );
                }
            }
            if matches!(
                binding.kind,
                ImportKind32::LazyPointer | ImportKind32::NonLazyPointer
            ) {
                mem.write(binding.addr, &target.to_le_bytes())?;
            }
            self.schedule_calls_before_target(cpu, mem, &init_funcs, target)?;
            return Ok(());
        }

        match symbol_norm {
            "dlopen" => {
                let path_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let _mode = Self::read_arg_u32(mem, cpu, 1)?;
                let requested = Self::read_c_string(mem, path_ptr, 4096).unwrap_or_default();
                let resolved = self.resolve_dlopen_path(&requested);
                if let Some(resolved_path) = resolved {
                    let key = resolved_path.to_string_lossy().to_string();
                    if let Some(&handle) = self.handle_by_path.get(&key) {
                        self.set_dlerror(mem, None)?;
                        cpu.eax = handle;
                        if self.trace {
                            eprintln!(
                                "[DYLD] dlopen('{}') -> existing handle {}",
                                resolved_path.display(),
                                handle
                            );
                        }
                    } else {
                        match self.map_dylib_image(&resolved_path, mem, cpu) {
                            Ok(image) => {
                                let handle = image.handle;
                                self.handle_by_path.insert(key, handle);
                                self.loaded_images.insert(handle, image.clone());
                                self.set_dlerror(mem, None)?;
                                cpu.eax = handle;
                                if self.trace {
                                    eprintln!(
                                        "[DYLD] dlopen('{}') -> handle {} slide={:#010x} exports={} init_funcs={}",
                                        image.path.display(),
                                        handle,
                                        image.slide,
                                        image.exports.len(),
                                        image.init_funcs.len()
                                    );
                                }
                            }
                            Err(err) => {
                                self.set_dlerror(
                                    mem,
                                    Some(format!(
                                        "dlopen failed for '{}': {}",
                                        resolved_path.display(),
                                        err
                                    )),
                                )?;
                                cpu.eax = 0;
                            }
                        }
                    }
                } else {
                    self.set_dlerror(
                        mem,
                        Some(format!("dlopen failed: cannot resolve '{}'", requested)),
                    )?;
                    cpu.eax = 0;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "dlclose" => {
                let handle = Self::read_arg_u32(mem, cpu, 0)?;
                if self.loaded_images.contains_key(&handle) {
                    // 当前不做真正 unload（无引用计数/无 unmap），保持已映射镜像可继续调用。
                    self.set_dlerror(mem, None)?;
                    cpu.eax = 0;
                } else {
                    self.set_dlerror(
                        mem,
                        Some(format!("dlclose failed: unknown handle {}", handle)),
                    )?;
                    cpu.eax = 1;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "dlsym" => {
                let handle = Self::read_arg_u32(mem, cpu, 0)?;
                let sym_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                let sym = Self::read_c_string(mem, sym_ptr, 1024).unwrap_or_default();
                let mut resolved = None;
                if handle == 0 || handle == 0xffff_fffe {
                    resolved = Self::lookup_export(&self.main_exports, &sym);
                    if resolved.is_none() {
                        for image in self.loaded_images.values() {
                            resolved = Self::lookup_export(&image.exports, &sym);
                            if resolved.is_some() {
                                break;
                            }
                        }
                    }
                } else if let Some(image) = self.loaded_images.get(&handle) {
                    resolved = Self::lookup_export(&image.exports, &sym);
                }

                if let Some(addr) = resolved {
                    self.set_dlerror(mem, None)?;
                    cpu.eax = addr;
                    if self.trace {
                        eprintln!(
                            "[DYLD] dlsym(handle={}, symbol='{}') -> {:#010x}",
                            handle, sym, addr
                        );
                    }
                } else {
                    self.set_dlerror(
                        mem,
                        Some(format!(
                            "dlsym unresolved: handle={} symbol='{}'",
                            handle, sym
                        )),
                    )?;
                    cpu.eax = 0;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "dlerror" => {
                cpu.eax = self.last_dlerror_ptr;
                if self.trace {
                    eprintln!("[DYLD] dlerror() -> {:#010x}", self.last_dlerror_ptr);
                }
                self.last_dlerror = None;
                self.last_dlerror_ptr = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "NSClassFromString" => {
                let _name_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let cls = self.alloc_heap(mem, 16, true)?;
                cpu.eax = cls;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "objc_msgSend" => {
                let receiver = Self::read_arg_u32(mem, cpu, 0)?;
                cpu.eax = if receiver != 0 {
                    receiver
                } else {
                    self.alloc_heap(mem, 16, true)?
                };
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "objc_msgSendSuper" => {
                let super_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let receiver = if super_ptr != 0 {
                    Self::read_u32(mem, super_ptr).unwrap_or(0)
                } else {
                    0
                };
                cpu.eax = if receiver != 0 {
                    receiver
                } else {
                    self.alloc_heap(mem, 16, true)?
                };
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "objc_msgSend_stret" => {
                let ret_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                cpu.eax = ret_ptr;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "objc_msgSend_fpret" => {
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "objc_enumerationMutation" => {
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "__error" | "_error" => {
                cpu.eax = self.ensure_errno_slot(mem)?;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "__cxa_guard_acquire" | "_cxa_guard_acquire" => {
                let guard_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let mut guard = [0u8; 8];
                if mem.read(guard_ptr, &mut guard).is_err() {
                    let _ = self.set_errno(mem, ERRNO_EFAULT);
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                if guard[0] != 0 {
                    cpu.eax = 0;
                } else {
                    // 单线程简化语义：未初始化时抢占并返回 1，后续 release 写入已初始化标记。
                    guard[1] = 1;
                    mem.write(guard_ptr, &guard)?;
                    cpu.eax = 1;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "__cxa_guard_release" | "_cxa_guard_release" => {
                let guard_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let mut guard = [0u8; 8];
                if mem.read(guard_ptr, &mut guard).is_ok() {
                    guard[0] = 1;
                    guard[1] = 0;
                    let _ = mem.write(guard_ptr, &guard);
                }
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "__cxa_guard_abort" | "_cxa_guard_abort" => {
                let guard_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let mut guard = [0u8; 8];
                if mem.read(guard_ptr, &mut guard).is_ok() {
                    guard[1] = 0;
                    let _ = mem.write(guard_ptr, &guard);
                }
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "__udivdi3" | "_udivdi3" => {
                let a_lo = Self::read_arg_u32(mem, cpu, 0)? as u64;
                let a_hi = Self::read_arg_u32(mem, cpu, 1)? as u64;
                let b_lo = Self::read_arg_u32(mem, cpu, 2)? as u64;
                let b_hi = Self::read_arg_u32(mem, cpu, 3)? as u64;
                let a = (a_hi << 32) | a_lo;
                let b = (b_hi << 32) | b_lo;
                let q = if b == 0 { 0 } else { a / b };
                cpu.eax = q as u32;
                cpu.edx = (q >> 32) as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "__umoddi3" | "_umoddi3" => {
                let a_lo = Self::read_arg_u32(mem, cpu, 0)? as u64;
                let a_hi = Self::read_arg_u32(mem, cpu, 1)? as u64;
                let b_lo = Self::read_arg_u32(mem, cpu, 2)? as u64;
                let b_hi = Self::read_arg_u32(mem, cpu, 3)? as u64;
                let a = (a_hi << 32) | a_lo;
                let b = (b_hi << 32) | b_lo;
                let r = if b == 0 { 0 } else { a % b };
                cpu.eax = r as u32;
                cpu.edx = (r >> 32) as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "__fixunsdfdi" | "_fixunsdfdi" => {
                let raw = Self::read_u64(mem, cpu.esp.wrapping_add(4))?;
                let value = f64::from_bits(raw);
                let out = if !value.is_finite() || value <= 0.0 {
                    0
                } else if value >= u64::MAX as f64 {
                    u64::MAX
                } else {
                    value.trunc() as u64
                };
                cpu.eax = out as u32;
                cpu.edx = (out >> 32) as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "__fixdfdi" | "_fixdfdi" => {
                let raw = Self::read_u64(mem, cpu.esp.wrapping_add(4))?;
                let value = f64::from_bits(raw);
                let out = if !value.is_finite() {
                    0i64
                } else if value >= i64::MAX as f64 {
                    i64::MAX
                } else if value <= i64::MIN as f64 {
                    i64::MIN
                } else {
                    value.trunc() as i64
                };
                cpu.eax = out as u32;
                cpu.edx = ((out as u64) >> 32) as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "pthread_self" | "pthread_self$UNIX2003" => {
                cpu.eax = self.ensure_pthread_self_ptr(mem)?;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "pthread_getspecific" | "pthread_getspecific$UNIX2003" => {
                let key = Self::read_arg_u32(mem, cpu, 0)?;
                cpu.eax = self.pthread_key_values.get(&key).copied().unwrap_or(0);
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "pthread_setspecific" | "pthread_setspecific$UNIX2003" => {
                let key = Self::read_arg_u32(mem, cpu, 0)?;
                let value = Self::read_arg_u32(mem, cpu, 1)?;
                self.pthread_key_values.insert(key, value);
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "pthread_key_create" | "pthread_key_create$UNIX2003" => {
                let key_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let _destructor = Self::read_arg_u32(mem, cpu, 1)?;
                let key = self.pthread_next_key;
                self.pthread_next_key = self.pthread_next_key.wrapping_add(1).max(1);
                if mem.write(key_ptr, &key.to_le_bytes()).is_err() {
                    cpu.eax = ERRNO_EFAULT;
                } else {
                    self.pthread_key_values.entry(key).or_insert(0);
                    cpu.eax = 0;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "pthread_key_delete" | "pthread_key_delete$UNIX2003" => {
                let key = Self::read_arg_u32(mem, cpu, 0)?;
                self.pthread_key_values.remove(&key);
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "pthread_mutex_lock"
            | "pthread_mutex_unlock"
            | "pthread_mutex_trylock"
            | "pthread_mutex_init"
            | "pthread_mutex_destroy" => {
                self.clear_errno(mem)?;
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "pthread_cond_init"
            | "pthread_cond_init$UNIX2003"
            | "pthread_cond_destroy"
            | "pthread_cond_destroy$UNIX2003"
            | "pthread_cond_signal"
            | "pthread_cond_signal$UNIX2003"
            | "pthread_cond_broadcast"
            | "pthread_cond_broadcast$UNIX2003"
            | "pthread_cond_wait"
            | "pthread_cond_wait$UNIX2003"
            | "pthread_cond_timedwait"
            | "pthread_cond_timedwait$UNIX2003" => {
                self.clear_errno(mem)?;
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "pthread_mutexattr_init"
            | "pthread_mutexattr_init$UNIX2003"
            | "pthread_mutexattr_destroy"
            | "pthread_mutexattr_destroy$UNIX2003"
            | "pthread_mutexattr_settype"
            | "pthread_mutexattr_settype$UNIX2003" => {
                let attr_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                if attr_ptr == 0 {
                    cpu.eax = ERRNO_EINVAL;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                if symbol_norm.starts_with("pthread_mutexattr_init")
                    || symbol_norm.starts_with("pthread_mutexattr_settype")
                {
                    let ty = if symbol_norm.starts_with("pthread_mutexattr_settype") {
                        Self::read_arg_u32(mem, cpu, 1)?
                    } else {
                        0
                    };
                    let _ = mem.write(attr_ptr, &ty.to_le_bytes());
                }
                self.clear_errno(mem)?;
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "gettimeofday" | "gettimeofday$UNIX2003" => {
                let tv_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let _tz_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                if tv_ptr == 0 {
                    self.set_errno(mem, ERRNO_EFAULT)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                let (secs, usec) = Self::unix_time_parts();
                if Self::write_timeval32(mem, tv_ptr, secs, usec).is_err() {
                    self.set_errno(mem, ERRNO_EFAULT)?;
                    cpu.eax = u32::MAX;
                } else {
                    self.clear_errno(mem)?;
                    cpu.eax = 0;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "usleep" | "usleep$UNIX2003" => {
                let usec = Self::read_arg_u32(mem, cpu, 0)?;
                std::thread::sleep(Duration::from_micros(usec as u64));
                self.clear_errno(mem)?;
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "time" | "time$UNIX2003" => {
                let tloc_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let (secs, _) = Self::unix_time_parts();
                if tloc_ptr != 0 && mem.write(tloc_ptr, &(secs as i32).to_le_bytes()).is_err() {
                    self.set_errno(mem, ERRNO_EFAULT)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                self.clear_errno(mem)?;
                cpu.eax = secs;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "localtime" | "localtime$UNIX2003" => {
                let time_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                if time_ptr == 0 {
                    self.set_errno(mem, ERRNO_EFAULT)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                let secs = match Self::read_u32(mem, time_ptr) {
                    Ok(v) => v,
                    Err(_) => {
                        self.set_errno(mem, ERRNO_EFAULT)?;
                        cpu.eax = 0;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    }
                };
                let tm_ptr = self.ensure_localtime_tm_ptr(mem)?;
                if Self::write_tm32(mem, tm_ptr, secs).is_err() {
                    self.set_errno(mem, ERRNO_EFAULT)?;
                    cpu.eax = 0;
                } else {
                    self.clear_errno(mem)?;
                    cpu.eax = tm_ptr;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "localtime_r" | "localtime_r$UNIX2003" => {
                let time_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let result_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                if time_ptr == 0 || result_ptr == 0 {
                    self.set_errno(mem, ERRNO_EFAULT)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                let secs = match Self::read_u32(mem, time_ptr) {
                    Ok(v) => v,
                    Err(_) => {
                        self.set_errno(mem, ERRNO_EFAULT)?;
                        cpu.eax = 0;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    }
                };
                if Self::write_tm32(mem, result_ptr, secs).is_err() {
                    self.set_errno(mem, ERRNO_EFAULT)?;
                    cpu.eax = 0;
                } else {
                    self.clear_errno(mem)?;
                    cpu.eax = result_ptr;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "sysctl" | "sysctl$UNIX2003" => {
                let name_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let namelen = Self::read_arg_u32(mem, cpu, 1)?;
                let oldp = Self::read_arg_u32(mem, cpu, 2)?;
                let oldlenp = Self::read_arg_u32(mem, cpu, 3)?;
                let _newp = Self::read_arg_u32(mem, cpu, 4)?;
                let _newlen = Self::read_arg_u32(mem, cpu, 5)?;

                if name_ptr == 0 || namelen == 0 {
                    self.set_errno(mem, ERRNO_EINVAL)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }

                let mib0 = Self::read_u32(mem, name_ptr).unwrap_or(u32::MAX);
                let mib1 = if namelen >= 2 {
                    Self::read_u32(mem, name_ptr.wrapping_add(4)).unwrap_or(u32::MAX)
                } else {
                    u32::MAX
                };

                // 仅实现常见查询，避免主流程因系统信息探测失败中断。
                let value: Option<Vec<u8>> = match (mib0, mib1) {
                    // CTL_HW + HW_NCPU / HW_AVAILCPU
                    (6, 3) | (6, 25) => Some((8u32).to_le_bytes().to_vec()),
                    // CTL_KERN + KERN_OSRELEASE
                    (1, 2) => Some(b"23.0.0\0".to_vec()),
                    // CTL_KERN + KERN_OSTYPE
                    (1, 1) => Some(b"Darwin\0".to_vec()),
                    _ => None,
                };

                let Some(value) = value else {
                    self.set_errno(mem, ERRNO_EINVAL)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };

                let need = value.len() as u32;
                if oldlenp != 0 {
                    let _ = mem.write(oldlenp, &need.to_le_bytes());
                }
                if oldp != 0 {
                    let have = if oldlenp != 0 {
                        Self::read_u32(mem, oldlenp).unwrap_or(0)
                    } else {
                        need
                    };
                    if have < need {
                        self.set_errno(mem, ERRNO_ENOMEM)?;
                        cpu.eax = u32::MAX;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    }
                    if mem.write(oldp, &value).is_err() {
                        self.set_errno(mem, ERRNO_EFAULT)?;
                        cpu.eax = u32::MAX;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    }
                }

                self.clear_errno(mem)?;
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "getcwd" | "getcwd$UNIX2003" => {
                let buf_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let size = Self::read_arg_u32(mem, cpu, 1)?;
                if buf_ptr == 0 || size == 0 {
                    self.set_errno(mem, ERRNO_EINVAL)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                let cwd = match env::current_dir() {
                    Ok(v) => v,
                    Err(err) => {
                        self.set_errno(mem, Self::errno_from_io(&err))?;
                        cpu.eax = 0;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    }
                };
                let cwd_str = cwd.to_string_lossy();
                let need = cwd_str.len().saturating_add(1);
                if need > size as usize {
                    self.set_errno(mem, ERRNO_ERANGE)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                if Self::write_c_buffer(mem, buf_ptr, size, &cwd_str).is_err() {
                    self.set_errno(mem, ERRNO_EFAULT)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                self.clear_errno(mem)?;
                cpu.eax = buf_ptr;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "atof" => {
                let s_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let s = Self::read_c_string(mem, s_ptr, 4096).unwrap_or_default();
                let bytes = s.as_bytes();
                let mut pos = 0usize;
                Self::skip_ascii_space(bytes, &mut pos);
                let value = if let Some((v, _used)) = Self::scan_float_prefix(&bytes[pos..], None) {
                    v
                } else {
                    0.0
                };
                cpu.fpu_push(value);
                self.clear_errno(mem)?;
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "sscanf" | "sscanf$UNIX2003" => {
                let input_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let fmt_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                let input = Self::read_c_string(mem, input_ptr, 64 * 1024).unwrap_or_default();
                let fmt = Self::read_c_string(mem, fmt_ptr, 8 * 1024).unwrap_or_default();
                let input_b = input.as_bytes();
                let fmt_b = fmt.as_bytes();
                let mut ii = 0usize;
                let mut fi = 0usize;
                let mut assigned = 0u32;
                let mut arg_cursor = cpu.esp.wrapping_add(12);

                while fi < fmt_b.len() {
                    let fc = fmt_b[fi];
                    if Self::is_ascii_space(fc) {
                        while fi < fmt_b.len() && Self::is_ascii_space(fmt_b[fi]) {
                            fi += 1;
                        }
                        Self::skip_ascii_space(input_b, &mut ii);
                        continue;
                    }
                    if fc != b'%' {
                        if ii >= input_b.len() || input_b[ii] != fc {
                            break;
                        }
                        ii += 1;
                        fi += 1;
                        continue;
                    }

                    fi += 1;
                    if fi >= fmt_b.len() {
                        break;
                    }
                    if fmt_b[fi] == b'%' {
                        if ii >= input_b.len() || input_b[ii] != b'%' {
                            break;
                        }
                        ii += 1;
                        fi += 1;
                        continue;
                    }

                    let mut suppress = false;
                    if fi < fmt_b.len() && fmt_b[fi] == b'*' {
                        suppress = true;
                        fi += 1;
                    }

                    let width_start = fi;
                    while fi < fmt_b.len() && fmt_b[fi].is_ascii_digit() {
                        fi += 1;
                    }
                    let width = if fi > width_start {
                        String::from_utf8_lossy(&fmt_b[width_start..fi])
                            .parse::<usize>()
                            .ok()
                    } else {
                        None
                    };

                    let mut len_h = false;
                    let mut len_hh = false;
                    let mut len_l = false;
                    let mut len_ll = false;
                    if fi < fmt_b.len() {
                        match fmt_b[fi] as char {
                            'h' => {
                                len_h = true;
                                fi += 1;
                                if fi < fmt_b.len() && fmt_b[fi] == b'h' {
                                    len_hh = true;
                                    fi += 1;
                                }
                            }
                            'l' => {
                                len_l = true;
                                fi += 1;
                                if fi < fmt_b.len() && fmt_b[fi] == b'l' {
                                    len_ll = true;
                                    fi += 1;
                                }
                            }
                            'L' => {
                                len_l = true;
                                fi += 1;
                            }
                            _ => {}
                        }
                    }
                    if fi >= fmt_b.len() {
                        break;
                    }

                    let spec = fmt_b[fi] as char;
                    fi += 1;
                    if !matches!(spec, 'c' | 'n' | '[') {
                        Self::skip_ascii_space(input_b, &mut ii);
                    }
                    let remain = &input_b[ii..];

                    match spec {
                        'd' | 'i' | 'u' | 'x' | 'X' | 'o' | 'p' => {
                            let int_spec = if spec == 'p' { 'x' } else { spec };
                            let Some((ival, used)) = Self::scan_int_prefix(remain, width, int_spec) else {
                                break;
                            };
                            ii += used;
                            if !suppress {
                                let dst = Self::read_u32(mem, arg_cursor)?;
                                arg_cursor = arg_cursor.wrapping_add(4);
                                let ok = if spec == 'd' || spec == 'i' {
                                    if len_hh {
                                        mem.write(dst, &((ival as i8) as u8).to_le_bytes()).is_ok()
                                    } else if len_h {
                                        mem.write(dst, &(ival as i16).to_le_bytes()).is_ok()
                                    } else {
                                        mem.write(dst, &(ival as i32).to_le_bytes()).is_ok()
                                    }
                                } else {
                                    let uval = ival as u64;
                                    if len_hh {
                                        mem.write(dst, &((uval as u8).to_le_bytes())).is_ok()
                                    } else if len_h {
                                        mem.write(dst, &((uval as u16).to_le_bytes())).is_ok()
                                    } else {
                                        mem.write(dst, &((uval as u32).to_le_bytes())).is_ok()
                                    }
                                };
                                if !ok {
                                    self.set_errno(mem, ERRNO_EFAULT)?;
                                    cpu.eax = assigned;
                                    Self::import_return(cpu, mem)?;
                                    return Ok(());
                                }
                                assigned = assigned.wrapping_add(1);
                            }
                        }
                        'f' | 'F' | 'e' | 'E' | 'g' | 'G' | 'a' | 'A' => {
                            let Some((fval, used)) = Self::scan_float_prefix(remain, width) else {
                                break;
                            };
                            ii += used;
                            if !suppress {
                                let dst = Self::read_u32(mem, arg_cursor)?;
                                arg_cursor = arg_cursor.wrapping_add(4);
                                let ok = if len_l || len_ll {
                                    mem.write(dst, &fval.to_le_bytes()).is_ok()
                                } else {
                                    mem.write(dst, &(fval as f32).to_le_bytes()).is_ok()
                                };
                                if !ok {
                                    self.set_errno(mem, ERRNO_EFAULT)?;
                                    cpu.eax = assigned;
                                    Self::import_return(cpu, mem)?;
                                    return Ok(());
                                }
                                assigned = assigned.wrapping_add(1);
                            }
                        }
                        's' => {
                            let w = width.unwrap_or(usize::MAX);
                            let mut end = ii;
                            let limit = ii.saturating_add(w).min(input_b.len());
                            while end < limit && !Self::is_ascii_space(input_b[end]) {
                                end += 1;
                            }
                            if end == ii {
                                break;
                            }
                            if !suppress {
                                let dst = Self::read_u32(mem, arg_cursor)?;
                                arg_cursor = arg_cursor.wrapping_add(4);
                                if mem.write(dst, &input_b[ii..end]).is_err()
                                    || mem.write(dst.wrapping_add((end - ii) as u32), &[0]).is_err()
                                {
                                    self.set_errno(mem, ERRNO_EFAULT)?;
                                    cpu.eax = assigned;
                                    Self::import_return(cpu, mem)?;
                                    return Ok(());
                                }
                                assigned = assigned.wrapping_add(1);
                            }
                            ii = end;
                        }
                        'c' => {
                            let w = width.unwrap_or(1);
                            if ii.saturating_add(w) > input_b.len() {
                                break;
                            }
                            if !suppress {
                                let dst = Self::read_u32(mem, arg_cursor)?;
                                arg_cursor = arg_cursor.wrapping_add(4);
                                if mem.write(dst, &input_b[ii..ii + w]).is_err() {
                                    self.set_errno(mem, ERRNO_EFAULT)?;
                                    cpu.eax = assigned;
                                    Self::import_return(cpu, mem)?;
                                    return Ok(());
                                }
                                assigned = assigned.wrapping_add(1);
                            }
                            ii += w;
                        }
                        'n' => {
                            if !suppress {
                                let dst = Self::read_u32(mem, arg_cursor)?;
                                arg_cursor = arg_cursor.wrapping_add(4);
                                let ok = if len_hh {
                                    mem.write(dst, &((ii as i32 as i8) as u8).to_le_bytes()).is_ok()
                                } else if len_h {
                                    mem.write(dst, &(ii as i16).to_le_bytes()).is_ok()
                                } else {
                                    mem.write(dst, &(ii as i32).to_le_bytes()).is_ok()
                                };
                                if !ok {
                                    self.set_errno(mem, ERRNO_EFAULT)?;
                                    cpu.eax = assigned;
                                    Self::import_return(cpu, mem)?;
                                    return Ok(());
                                }
                            }
                        }
                        _ => break,
                    }
                }

                self.clear_errno(mem)?;
                cpu.eax = assigned;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "fprintf" | "fprintf$UNIX2003" => {
                let fmt_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                let fmt = Self::read_c_string(mem, fmt_ptr, 2048).unwrap_or_default();
                let mut vararg_cursor = cpu.esp.wrapping_add(12);
                let rendered = Self::format_variadic_printf(mem, &fmt, &mut vararg_cursor)
                    .unwrap_or_else(|_| fmt.clone());
                if !rendered.is_empty() {
                    eprint!("{}", rendered);
                }
                cpu.eax = rendered.len() as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "printf" | "printf$UNIX2003" => {
                let fmt_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let fmt = Self::read_c_string(mem, fmt_ptr, 2048).unwrap_or_default();
                let mut vararg_cursor = cpu.esp.wrapping_add(8);
                let rendered = Self::format_variadic_printf(mem, &fmt, &mut vararg_cursor)
                    .unwrap_or_else(|_| fmt.clone());
                if !rendered.is_empty() {
                    print!("{}", rendered);
                }
                cpu.eax = rendered.len() as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "vfprintf" | "vfprintf$UNIX2003" => {
                let fmt_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                let va_list = Self::read_arg_u32(mem, cpu, 2)?;
                let fmt = Self::read_c_string(mem, fmt_ptr, 2048).unwrap_or_default();
                let mut vararg_cursor = va_list;
                let rendered = Self::format_variadic_printf(mem, &fmt, &mut vararg_cursor)
                    .unwrap_or_else(|_| fmt.clone());
                if !rendered.is_empty() {
                    eprint!("{}", rendered);
                }
                cpu.eax = rendered.len() as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "vprintf" | "vprintf$UNIX2003" => {
                let fmt_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let va_list = Self::read_arg_u32(mem, cpu, 1)?;
                let fmt = Self::read_c_string(mem, fmt_ptr, 2048).unwrap_or_default();
                let mut vararg_cursor = va_list;
                let rendered = Self::format_variadic_printf(mem, &fmt, &mut vararg_cursor)
                    .unwrap_or_else(|_| fmt.clone());
                if !rendered.is_empty() {
                    print!("{}", rendered);
                }
                cpu.eax = rendered.len() as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "vsnprintf" | "vsnprintf$UNIX2003" => {
                let dst = Self::read_arg_u32(mem, cpu, 0)?;
                let size = Self::read_arg_u32(mem, cpu, 1)?;
                let fmt_ptr = Self::read_arg_u32(mem, cpu, 2)?;
                let va_list = Self::read_arg_u32(mem, cpu, 3)?;
                let fmt = Self::read_c_string(mem, fmt_ptr, 16 * 1024).unwrap_or_default();
                let mut vararg_cursor = va_list;
                let rendered = Self::format_variadic_printf(mem, &fmt, &mut vararg_cursor)
                    .unwrap_or_else(|_| fmt.clone());
                if size > 0 {
                    if dst == 0 {
                        self.set_errno(mem, ERRNO_EFAULT)?;
                        cpu.eax = u32::MAX;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    }
                    if Self::write_c_buffer(mem, dst, size, &rendered).is_err() {
                        self.set_errno(mem, ERRNO_EFAULT)?;
                        cpu.eax = u32::MAX;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    }
                }
                self.clear_errno(mem)?;
                cpu.eax = rendered.len() as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "snprintf" | "snprintf$UNIX2003" => {
                let dst = Self::read_arg_u32(mem, cpu, 0)?;
                let size = Self::read_arg_u32(mem, cpu, 1)?;
                let fmt_ptr = Self::read_arg_u32(mem, cpu, 2)?;
                let fmt = Self::read_c_string(mem, fmt_ptr, 16 * 1024).unwrap_or_default();
                let mut vararg_cursor = cpu.esp.wrapping_add(16);
                let rendered = Self::format_variadic_printf(mem, &fmt, &mut vararg_cursor)
                    .unwrap_or_else(|_| fmt.clone());
                if size > 0 {
                    if dst == 0 {
                        self.set_errno(mem, ERRNO_EFAULT)?;
                        cpu.eax = u32::MAX;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    }
                    if Self::write_c_buffer(mem, dst, size, &rendered).is_err() {
                        self.set_errno(mem, ERRNO_EFAULT)?;
                        cpu.eax = u32::MAX;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    }
                }
                self.clear_errno(mem)?;
                cpu.eax = rendered.len() as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "sprintf" | "sprintf$UNIX2003" => {
                let dst = Self::read_arg_u32(mem, cpu, 0)?;
                let fmt_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                let fmt = Self::read_c_string(mem, fmt_ptr, 16 * 1024).unwrap_or_default();
                let mut vararg_cursor = cpu.esp.wrapping_add(12);
                let rendered = Self::format_variadic_printf(mem, &fmt, &mut vararg_cursor)
                    .unwrap_or_else(|_| fmt.clone());
                if dst == 0 || Self::write_c_buffer(mem, dst, (rendered.len() + 1) as u32, &rendered).is_err()
                {
                    self.set_errno(mem, ERRNO_EFAULT)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                self.clear_errno(mem)?;
                cpu.eax = rendered.len() as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "puts" | "puts$UNIX2003" => {
                let s_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let s = Self::read_c_string(mem, s_ptr, 8192).unwrap_or_default();
                println!("{}", s);
                cpu.eax = (s.len() + 1) as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "fwrite" | "fwrite$UNIX2003" => {
                let ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let size = Self::read_arg_u32(mem, cpu, 1)?;
                let nmemb = Self::read_arg_u32(mem, cpu, 2)?;
                let stream = Self::read_arg_u32(mem, cpu, 3)?;
                let total = size.saturating_mul(nmemb).min(1024 * 1024);
                let mut buf = vec![0u8; total as usize];
                if total != 0 && mem.read(ptr, &mut buf).is_err() {
                    self.set_errno(mem, ERRNO_EFAULT)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }

                let fd = self.stream_fd(mem, stream).unwrap_or(2);
                let written = if fd == 1 {
                    if let Ok(text) = std::str::from_utf8(&buf) {
                        print!("{}", text);
                    } else {
                        print!("{}", String::from_utf8_lossy(&buf));
                    }
                    total as usize
                } else if fd == 2 {
                    if let Ok(text) = std::str::from_utf8(&buf) {
                        eprint!("{}", text);
                    } else {
                        eprint!("{}", String::from_utf8_lossy(&buf));
                    }
                    total as usize
                } else if let Some(file) = self.host_files.get_mut(&fd) {
                    match file.write(&buf) {
                        Ok(n) => n,
                        Err(err) => {
                            self.set_errno(mem, Self::errno_from_io(&err))?;
                            cpu.eax = 0;
                            Self::import_return(cpu, mem)?;
                            return Ok(());
                        }
                    }
                } else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };

                if size == 0 {
                    cpu.eax = 0;
                } else {
                    cpu.eax = (written as u32) / size;
                }
                self.clear_errno(mem)?;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "fopen" | "fopen$UNIX2003" => {
                let path_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let mode_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                let path = Self::read_c_string(mem, path_ptr, 4096).unwrap_or_default();
                let mode = Self::read_c_string(mem, mode_ptr, 32).unwrap_or_else(|_| "r".to_string());
                if path.is_empty() {
                    self.set_errno(mem, ERRNO_ENOENT)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                let candidates = self.resolve_host_path_candidates(&path);
                let mut opened = None;
                let mut last_err: Option<io::Error> = None;
                for cand in candidates {
                    match Self::open_options_from_mode(&mode).open(&cand) {
                        Ok(file) => {
                            opened = Some(file);
                            break;
                        }
                        Err(err) => {
                            last_err = Some(err);
                        }
                    }
                }
                match opened {
                    Some(file) => {
                        let fd = self.next_file_fd();
                        self.host_files.insert(fd, file);
                        let stream_ptr = self.alloc_stream_struct(mem, fd)?;
                        self.clear_errno(mem)?;
                        cpu.eax = stream_ptr;
                    }
                    None => {
                        let errno = last_err
                            .as_ref()
                            .map(Self::errno_from_io)
                            .unwrap_or(ERRNO_ENOENT);
                        self.set_errno(mem, errno)?;
                        cpu.eax = 0;
                    }
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "fclose" | "fclose$UNIX2003" => {
                let stream = Self::read_arg_u32(mem, cpu, 0)?;
                let Some(fd) = self.stream_fd(mem, stream) else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };
                if fd >= 3 {
                    self.host_files.remove(&fd);
                }
                self.host_streams.remove(&stream);
                self.clear_errno(mem)?;
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "fflush" | "fflush$UNIX2003" => {
                let stream = Self::read_arg_u32(mem, cpu, 0)?;
                if stream == 0 {
                    self.clear_errno(mem)?;
                    cpu.eax = 0;
                } else if let Some(fd) = self.stream_fd(mem, stream) {
                    if let Some(file) = self.host_files.get_mut(&fd) {
                        if let Err(err) = file.flush() {
                            self.set_errno(mem, Self::errno_from_io(&err))?;
                            cpu.eax = u32::MAX;
                            Self::import_return(cpu, mem)?;
                            return Ok(());
                        }
                    }
                    self.clear_errno(mem)?;
                    cpu.eax = 0;
                } else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "fread" | "fread$UNIX2003" => {
                let ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let size = Self::read_arg_u32(mem, cpu, 1)?;
                let nmemb = Self::read_arg_u32(mem, cpu, 2)?;
                let stream = Self::read_arg_u32(mem, cpu, 3)?;
                if size == 0 || nmemb == 0 {
                    cpu.eax = 0;
                    self.clear_errno(mem)?;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                let total = size.saturating_mul(nmemb).min(16 * 1024 * 1024);
                let Some(fd) = self.stream_fd(mem, stream) else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };
                if fd <= 2 {
                    self.clear_errno(mem)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                let Some(file) = self.host_files.get_mut(&fd) else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };
                let mut buf = vec![0u8; total as usize];
                match file.read(&mut buf) {
                    Ok(n) => {
                        if n != 0 && mem.write(ptr, &buf[..n]).is_err() {
                            self.set_errno(mem, ERRNO_EFAULT)?;
                            cpu.eax = 0;
                        } else {
                            self.clear_errno(mem)?;
                            cpu.eax = (n as u32) / size;
                        }
                    }
                    Err(err) => {
                        self.set_errno(mem, Self::errno_from_io(&err))?;
                        cpu.eax = 0;
                    }
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "fseek" | "fseek$UNIX2003" => {
                let stream = Self::read_arg_u32(mem, cpu, 0)?;
                let offset = Self::read_arg_u32(mem, cpu, 1)? as i32 as i64;
                let whence = Self::read_arg_u32(mem, cpu, 2)?;
                let Some(fd) = self.stream_fd(mem, stream) else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };
                if fd <= 2 {
                    self.clear_errno(mem)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                let pos = match whence {
                    0 => SeekFrom::Start(offset.max(0) as u64),
                    1 => SeekFrom::Current(offset),
                    2 => SeekFrom::End(offset),
                    _ => {
                        self.set_errno(mem, ERRNO_EINVAL)?;
                        cpu.eax = u32::MAX;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    }
                };
                let Some(file) = self.host_files.get_mut(&fd) else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };
                match file.seek(pos) {
                    Ok(_) => {
                        self.clear_errno(mem)?;
                        cpu.eax = 0;
                    }
                    Err(err) => {
                        self.set_errno(mem, Self::errno_from_io(&err))?;
                        cpu.eax = u32::MAX;
                    }
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "ftell" | "ftell$UNIX2003" => {
                let stream = Self::read_arg_u32(mem, cpu, 0)?;
                let Some(fd) = self.stream_fd(mem, stream) else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };
                if fd <= 2 {
                    self.clear_errno(mem)?;
                    cpu.eax = 0;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                let Some(file) = self.host_files.get_mut(&fd) else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };
                match file.seek(SeekFrom::Current(0)) {
                    Ok(pos) => {
                        self.clear_errno(mem)?;
                        cpu.eax = pos as u32;
                    }
                    Err(err) => {
                        self.set_errno(mem, Self::errno_from_io(&err))?;
                        cpu.eax = u32::MAX;
                    }
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "fgetc" | "fgetc$UNIX2003" => {
                let stream = Self::read_arg_u32(mem, cpu, 0)?;
                let Some(fd) = self.stream_fd(mem, stream) else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };
                if fd <= 2 {
                    self.clear_errno(mem)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                let Some(file) = self.host_files.get_mut(&fd) else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };
                let mut one = [0u8; 1];
                match file.read(&mut one) {
                    Ok(1) => {
                        self.clear_errno(mem)?;
                        cpu.eax = one[0] as u32;
                    }
                    Ok(_) => {
                        self.clear_errno(mem)?;
                        cpu.eax = u32::MAX;
                    }
                    Err(err) => {
                        self.set_errno(mem, Self::errno_from_io(&err))?;
                        cpu.eax = u32::MAX;
                    }
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "fchmod" | "fchmod$UNIX2003" => {
                let fd = Self::read_arg_u32(mem, cpu, 0)?;
                let _mode = Self::read_arg_u32(mem, cpu, 1)?;
                if fd <= 2 || self.host_files.contains_key(&fd) {
                    self.clear_errno(mem)?;
                    cpu.eax = 0;
                } else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "chdir" | "chdir$UNIX2003" => {
                let path_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let path = Self::read_c_string(mem, path_ptr, 4096).unwrap_or_default();
                if path.is_empty() {
                    self.set_errno(mem, ERRNO_ENOENT)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                let candidates = self.resolve_host_path_candidates(&path);
                let mut changed = false;
                let mut last_err: Option<io::Error> = None;
                for cand in candidates {
                    match env::set_current_dir(&cand) {
                        Ok(_) => {
                            changed = true;
                            break;
                        }
                        Err(err) => last_err = Some(err),
                    }
                }
                if changed {
                    self.clear_errno(mem)?;
                    cpu.eax = 0;
                } else {
                    let errno = last_err
                        .as_ref()
                        .map(Self::errno_from_io)
                        .unwrap_or(ERRNO_ENOENT);
                    self.set_errno(mem, errno)?;
                    cpu.eax = u32::MAX;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "SetFrontProcess" | "TransformProcessType" => {
                self.clear_errno(mem)?;
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "open" | "open$UNIX2003" => {
                const O_CREAT: u32 = 0x0200;
                let path_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let flags = Self::read_arg_u32(mem, cpu, 1)?;
                let _mode = Self::read_arg_u32(mem, cpu, 2)?;
                let path = Self::read_c_string(mem, path_ptr, 4096).unwrap_or_default();
                if path.is_empty() {
                    self.set_errno(mem, ERRNO_ENOENT)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }
                let candidates = self.resolve_host_path_candidates(&path);

                let mut opened = None;
                let mut last_err: Option<io::Error> = None;
                for cand in candidates {
                    if (flags & O_CREAT) != 0 {
                        if let Some(parent) = cand.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
                    }
                    match Self::open_options_from_flags(flags).open(&cand) {
                        Ok(file) => {
                            opened = Some(file);
                            if self.trace {
                                eprintln!(
                                    "[DYLD] open('{}', flags={:#x}) succeeded",
                                    cand.display(),
                                    flags
                                );
                            }
                            break;
                        }
                        Err(err) => {
                            if self.trace {
                                eprintln!(
                                    "[DYLD] open('{}', flags={:#x}) failed: {}",
                                    cand.display(),
                                    flags,
                                    err
                                );
                            }
                            last_err = Some(err);
                        }
                    }
                }

                match opened {
                    Some(file) => {
                        let fd = self.next_file_fd();
                        self.host_files.insert(fd, file);
                        self.clear_errno(mem)?;
                        cpu.eax = fd;
                    }
                    None => {
                        let errno = last_err
                            .as_ref()
                            .map(Self::errno_from_io)
                            .unwrap_or(ERRNO_ENOENT);
                        self.set_errno(mem, errno)?;
                        cpu.eax = u32::MAX;
                    }
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "read" | "read$UNIX2003" => {
                let fd = Self::read_arg_u32(mem, cpu, 0)?;
                let buf_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                let count = Self::read_arg_u32(mem, cpu, 2)?.min(16 * 1024 * 1024);
                let mut buf = vec![0u8; count as usize];
                let Some(file) = self.host_files.get_mut(&fd) else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };
                match file.read(&mut buf) {
                    Ok(read_n) => {
                        if read_n > 0 && mem.write(buf_ptr, &buf[..read_n]).is_err() {
                            self.set_errno(mem, ERRNO_EFAULT)?;
                            cpu.eax = u32::MAX;
                        } else {
                            self.clear_errno(mem)?;
                            cpu.eax = read_n as u32;
                        }
                    }
                    Err(err) => {
                        self.set_errno(mem, Self::errno_from_io(&err))?;
                        cpu.eax = u32::MAX;
                    }
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "pread" | "pread$UNIX2003" => {
                let fd = Self::read_arg_u32(mem, cpu, 0)?;
                let buf_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                let count = Self::read_arg_u32(mem, cpu, 2)?.min(16 * 1024 * 1024);
                let offset = Self::read_arg_u32(mem, cpu, 3)?;
                let mut buf = vec![0u8; count as usize];
                let Some(file) = self.host_files.get_mut(&fd) else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                };

                let saved_pos = file.seek(SeekFrom::Current(0));
                if file.seek(SeekFrom::Start(offset as u64)).is_err() {
                    self.set_errno(mem, ERRNO_EINVAL)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }

                let read_result = file.read(&mut buf);
                if let Ok(pos) = saved_pos {
                    let _ = file.seek(SeekFrom::Start(pos));
                }
                match read_result {
                    Ok(read_n) => {
                        if read_n > 0 && mem.write(buf_ptr, &buf[..read_n]).is_err() {
                            self.set_errno(mem, ERRNO_EFAULT)?;
                            cpu.eax = u32::MAX;
                        } else {
                            self.clear_errno(mem)?;
                            cpu.eax = read_n as u32;
                        }
                    }
                    Err(err) => {
                        self.set_errno(mem, Self::errno_from_io(&err))?;
                        cpu.eax = u32::MAX;
                    }
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "close" | "close$UNIX2003" => {
                let fd = Self::read_arg_u32(mem, cpu, 0)?;
                if self.host_files.remove(&fd).is_some() {
                    self.clear_errno(mem)?;
                    cpu.eax = 0;
                } else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "flock" | "flock$UNIX2003" => {
                let fd = Self::read_arg_u32(mem, cpu, 0)?;
                let _operation = Self::read_arg_u32(mem, cpu, 1)?;
                if self.host_files.contains_key(&fd) {
                    self.clear_errno(mem)?;
                    cpu.eax = 0;
                } else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "fcntl" | "fcntl$UNIX2003" => {
                let fd = Self::read_arg_u32(mem, cpu, 0)?;
                let cmd = Self::read_arg_u32(mem, cpu, 1)?;
                if !self.host_files.contains_key(&fd) {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }

                // 只实现最常见控制命令；锁相关命令直接视作成功，避免流程中断。
                // F_GETFD=1, F_SETFD=2, F_GETFL=3, F_SETFL=4, F_SETLK=8, F_SETLKW=9.
                cpu.eax = match cmd {
                    1 => 0,
                    2 => 0,
                    3 => 0,
                    4 => 0,
                    8 | 9 => 0,
                    _ => 0,
                };
                self.clear_errno(mem)?;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "lseek" | "lseek$UNIX2003" => {
                let fd = Self::read_arg_u32(mem, cpu, 0)?;
                let offset = Self::read_arg_u32(mem, cpu, 1)? as i32 as i64;
                let whence = Self::read_arg_u32(mem, cpu, 2)?;
                let pos = match whence {
                    0 => SeekFrom::Start(offset.max(0) as u64),
                    1 => SeekFrom::Current(offset),
                    2 => SeekFrom::End(offset),
                    _ => {
                        self.set_errno(mem, ERRNO_EINVAL)?;
                        cpu.eax = u32::MAX;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    }
                };
                if let Some(file) = self.host_files.get_mut(&fd) {
                    match file.seek(pos) {
                        Ok(v) => {
                            self.clear_errno(mem)?;
                            cpu.eax = v as u32;
                        }
                        Err(err) => {
                            self.set_errno(mem, Self::errno_from_io(&err))?;
                            cpu.eax = u32::MAX;
                        }
                    }
                } else {
                    self.set_errno(mem, ERRNO_EBADF)?;
                    cpu.eax = u32::MAX;
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "mmap" | "mmap$UNIX2003" => {
                const MAP_ANON: u32 = 0x1000;
                let req_addr = Self::read_arg_u32(mem, cpu, 0)?;
                let len = Self::read_arg_u32(mem, cpu, 1)?;
                let prot = Self::read_arg_u32(mem, cpu, 2)?;
                let flags = Self::read_arg_u32(mem, cpu, 3)?;
                let fd = Self::read_arg_u32(mem, cpu, 4)?;
                let offset = Self::read_arg_u32(mem, cpu, 5)?;
                if len == 0 {
                    self.set_errno(mem, ERRNO_EINVAL)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }

                let len_aligned = Self::align_up(len, 0x1000);
                let addr = if req_addr != 0 {
                    req_addr
                } else {
                    let next = Self::align_up(self.host_next_mmap, 0x1000);
                    self.host_next_mmap = next.wrapping_add(len_aligned).wrapping_add(0x1000);
                    next
                };

                let mut map_prot = Prot::READ;
                if (prot & 0x2) != 0 {
                    map_prot = map_prot | Prot::WRITE;
                }
                if (prot & 0x4) != 0 {
                    map_prot = map_prot | Prot::EXEC;
                }
                let load_prot = map_prot | Prot::WRITE;
                if mem.map(addr, len_aligned, load_prot).is_err() {
                    self.set_errno(mem, ERRNO_ENOMEM)?;
                    cpu.eax = u32::MAX;
                    Self::import_return(cpu, mem)?;
                    return Ok(());
                }

                if (flags & MAP_ANON) == 0 {
                    let Some(file) = self.host_files.get_mut(&fd) else {
                        self.set_errno(mem, ERRNO_EBADF)?;
                        cpu.eax = u32::MAX;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    };
                    if file.seek(SeekFrom::Start(offset as u64)).is_err() {
                        self.set_errno(mem, ERRNO_EINVAL)?;
                        cpu.eax = u32::MAX;
                        Self::import_return(cpu, mem)?;
                        return Ok(());
                    }
                    let mut file_buf = vec![0u8; len as usize];
                    match file.read(&mut file_buf) {
                        Ok(n) => {
                            if n > 0 && mem.write(addr, &file_buf[..n]).is_err() {
                                self.set_errno(mem, ERRNO_EFAULT)?;
                                cpu.eax = u32::MAX;
                                Self::import_return(cpu, mem)?;
                                return Ok(());
                            }
                        }
                        Err(err) => {
                            self.set_errno(mem, Self::errno_from_io(&err))?;
                            cpu.eax = u32::MAX;
                            Self::import_return(cpu, mem)?;
                            return Ok(());
                        }
                    }
                }

                self.clear_errno(mem)?;
                cpu.eax = addr;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "munmap" | "munmap$UNIX2003" => {
                self.clear_errno(mem)?;
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "strlen" => {
                let s_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                match Self::c_strlen(mem, s_ptr, 1 << 20) {
                    Ok(len) => {
                        self.clear_errno(mem)?;
                        cpu.eax = len as u32;
                    }
                    Err(_) => {
                        self.set_errno(mem, ERRNO_EFAULT)?;
                        cpu.eax = 0;
                    }
                }
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "strcmp" => {
                let a_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let b_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                let mut i = 0u32;
                let res = loop {
                    let mut a = [0u8; 1];
                    let mut b = [0u8; 1];
                    mem.read(a_ptr.wrapping_add(i), &mut a)?;
                    mem.read(b_ptr.wrapping_add(i), &mut b)?;
                    if a[0] != b[0] {
                        break (a[0] as i32) - (b[0] as i32);
                    }
                    if a[0] == 0 {
                        break 0;
                    }
                    i = i.wrapping_add(1);
                };
                cpu.eax = res as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "strncmp" => {
                let a_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let b_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                let n = Self::read_arg_u32(mem, cpu, 2)?;
                let mut i = 0u32;
                let res = loop {
                    if i >= n {
                        break 0;
                    }
                    let mut a = [0u8; 1];
                    let mut b = [0u8; 1];
                    mem.read(a_ptr.wrapping_add(i), &mut a)?;
                    mem.read(b_ptr.wrapping_add(i), &mut b)?;
                    if a[0] != b[0] {
                        break (a[0] as i32) - (b[0] as i32);
                    }
                    if a[0] == 0 {
                        break 0;
                    }
                    i = i.wrapping_add(1);
                };
                cpu.eax = res as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "memcmp" => {
                let a_ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let b_ptr = Self::read_arg_u32(mem, cpu, 1)?;
                let n = Self::read_arg_u32(mem, cpu, 2)?.min(1024 * 1024);
                let mut a = vec![0u8; n as usize];
                let mut b = vec![0u8; n as usize];
                mem.read(a_ptr, &mut a)?;
                mem.read(b_ptr, &mut b)?;
                let mut res = 0i32;
                for i in 0..(n as usize) {
                    if a[i] != b[i] {
                        res = (a[i] as i32) - (b[i] as i32);
                        break;
                    }
                }
                cpu.eax = res as u32;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "memcpy" | "memmove" => {
                let dst = Self::read_arg_u32(mem, cpu, 0)?;
                let src = Self::read_arg_u32(mem, cpu, 1)?;
                let n = Self::read_arg_u32(mem, cpu, 2)?.min(8 * 1024 * 1024);
                let mut tmp = vec![0u8; n as usize];
                mem.read(src, &mut tmp)?;
                mem.write(dst, &tmp)?;
                cpu.eax = dst;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "memset" => {
                let dst = Self::read_arg_u32(mem, cpu, 0)?;
                let byte = Self::read_arg_u32(mem, cpu, 1)? as u8;
                let n = Self::read_arg_u32(mem, cpu, 2)?.min(8 * 1024 * 1024);
                let buf = vec![byte; n as usize];
                mem.write(dst, &buf)?;
                cpu.eax = dst;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "bzero" => {
                let dst = Self::read_arg_u32(mem, cpu, 0)?;
                let n = Self::read_arg_u32(mem, cpu, 1)?.min(8 * 1024 * 1024);
                let buf = vec![0u8; n as usize];
                mem.write(dst, &buf)?;
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "strcpy" => {
                let dst = Self::read_arg_u32(mem, cpu, 0)?;
                let src = Self::read_arg_u32(mem, cpu, 1)?;
                let len = Self::c_strlen(mem, src, 1 << 20)? as u32;
                let mut tmp = vec![0u8; (len + 1) as usize];
                mem.read(src, &mut tmp)?;
                mem.write(dst, &tmp)?;
                cpu.eax = dst;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "strncpy" => {
                let dst = Self::read_arg_u32(mem, cpu, 0)?;
                let src = Self::read_arg_u32(mem, cpu, 1)?;
                let n = Self::read_arg_u32(mem, cpu, 2)?.min(8 * 1024 * 1024);
                let mut out = vec![0u8; n as usize];
                let mut i = 0u32;
                while i < n {
                    let mut b = [0u8; 1];
                    mem.read(src.wrapping_add(i), &mut b)?;
                    out[i as usize] = b[0];
                    i = i.wrapping_add(1);
                    if b[0] == 0 {
                        break;
                    }
                }
                mem.write(dst, &out)?;
                cpu.eax = dst;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "strcat" => {
                let dst = Self::read_arg_u32(mem, cpu, 0)?;
                let src = Self::read_arg_u32(mem, cpu, 1)?;
                let dst_len = Self::c_strlen(mem, dst, 1 << 20)? as u32;
                let src_len = Self::c_strlen(mem, src, 1 << 20)? as u32;
                let mut src_buf = vec![0u8; (src_len + 1) as usize];
                mem.read(src, &mut src_buf)?;
                mem.write(dst.wrapping_add(dst_len), &src_buf)?;
                cpu.eax = dst;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "strncat" => {
                let dst = Self::read_arg_u32(mem, cpu, 0)?;
                let src = Self::read_arg_u32(mem, cpu, 1)?;
                let n = Self::read_arg_u32(mem, cpu, 2)?.min(8 * 1024 * 1024);
                let dst_len = Self::c_strlen(mem, dst, 1 << 20)? as u32;

                let mut append = Vec::with_capacity(n as usize + 1);
                let mut i = 0u32;
                while i < n {
                    let mut b = [0u8; 1];
                    mem.read(src.wrapping_add(i), &mut b)?;
                    if b[0] == 0 {
                        break;
                    }
                    append.push(b[0]);
                    i = i.wrapping_add(1);
                }
                append.push(0);
                mem.write(dst.wrapping_add(dst_len), &append)?;
                cpu.eax = dst;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "malloc" => {
                let size = Self::read_arg_u32(mem, cpu, 0)?;
                cpu.eax = self.alloc_heap(mem, size, false)?;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "calloc" => {
                let nmemb = Self::read_arg_u32(mem, cpu, 0)?;
                let size = Self::read_arg_u32(mem, cpu, 1)?;
                cpu.eax = self.alloc_heap(mem, nmemb.saturating_mul(size), true)?;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "realloc" => {
                let ptr = Self::read_arg_u32(mem, cpu, 0)?;
                let size = Self::read_arg_u32(mem, cpu, 1)?;
                cpu.eax = self.realloc_heap(mem, ptr, size)?;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "free" => {
                let ptr = Self::read_arg_u32(mem, cpu, 0)?;
                self.free_heap(ptr);
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "malloc_zone_malloc" => {
                let size = Self::read_arg_u32(mem, cpu, 1)?;
                cpu.eax = self.alloc_heap(mem, size, false)?;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "malloc_zone_calloc" => {
                let nmemb = Self::read_arg_u32(mem, cpu, 1)?;
                let size = Self::read_arg_u32(mem, cpu, 2)?;
                cpu.eax = self.alloc_heap(mem, nmemb.saturating_mul(size), true)?;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "malloc_zone_realloc" => {
                let ptr = Self::read_arg_u32(mem, cpu, 1)?;
                let size = Self::read_arg_u32(mem, cpu, 2)?;
                cpu.eax = self.realloc_heap(mem, ptr, size)?;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "malloc_zone_free" => {
                let ptr = Self::read_arg_u32(mem, cpu, 1)?;
                self.free_heap(ptr);
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "atexit" | "__cxa_atexit" => {
                cpu.eax = 0;
                Self::import_return(cpu, mem)?;
                Ok(())
            }
            "abort" => Err(DyldError::GuestExit(134)),
            "exit" => {
                let status = Self::read_arg_u32(mem, cpu, 0)?;
                if self.trace {
                    eprintln!("[DYLD] import exit({}) called, forcing guest halt", status);
                }
                Err(DyldError::GuestExit(status))
            }
            _ => self.noop_import(cpu, mem, &binding),
        }
    }
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
    let mut dyld = DyldState::from_macho(&macho, trace);
    dyld.set_main_executable_path(path_ref);

    // 新建 guest 内存空间
    let mut mem = GuestMemory::new();

    // ------------------------------------------------------------
    // 2. 把 Mach-O 各段映射到 guest 内存
    // ------------------------------------------------------------
    for seg in &macho.segments {
        if seg.segname == "__PAGEZERO" {
            continue;
        }
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

        let fileoff_abs = macho.file_base_offset + seg.fileoff as u64;

        if trace {
            eprintln!(
                "[LOADER] map seg='{}' vmaddr={:#010x} vmsize={:#010x} fileoff={:#010x} fileoff_abs={:#010x} filesize={:#010x} initprot={:#x}",
                seg.segname,
                seg.vmaddr,
                seg.vmsize,
                seg.fileoff,
                fileoff_abs,
                seg.filesize,
                seg.initprot,
            );
        }

        // 先映射整段虚拟内存
        mem.map(seg.vmaddr, seg.vmsize, load_prot)?;

        // 再把文件中的实际数据读出来覆盖进去
        if seg.filesize > 0 {
            file.seek(SeekFrom::Start(fileoff_abs))?;

            let mut buf = vec![0u8; seg.filesize as usize];
            file.read_exact(&mut buf)?;

            mem.write(seg.vmaddr, &buf)?;
        }

        // 如果 vmsize > filesize，后面的 zero-fill 区域（例如 bss）
        // 会自然保持为 0，因为 map() 默认分配的是零初始化缓冲区。
    }

    dyld.init_runtime_memory(&mut mem)?;
    if trace {
        eprintln!(
            "[DYLD] parsed imports: total={}, dylibs={}",
            macho.imports.len(),
            macho.dylibs.len()
        );
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
    if let Some(jump_table) = &macho.import_jump_table {
        if jump_table.stub_size != 0 {
            cpu.set_import_jump_table(
                jump_table.addr,
                jump_table.size,
                jump_table.stub_size,
                jump_table.reserved1,
            );
            if trace {
                eprintln!(
                    "[LOADER] import jump table addr={:#010x} size={:#010x} stub_size={} reserved1={}",
                    jump_table.addr,
                    jump_table.size,
                    jump_table.stub_size,
                    jump_table.reserved1,
                );
            }
        } else if trace {
            eprintln!(
                "[LOADER] import jump table ignored because stub_size=0 (addr={:#010x}, size={:#010x})",
                jump_table.addr, jump_table.size
            );
        }
    }

    dyld.patch_import_stubs(&mut mem, &mut cpu)?;

    Ok(LoadedProgram { mem, cpu, dyld })
}
