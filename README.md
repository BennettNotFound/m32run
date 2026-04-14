# m32run

`m32run` 是一个在现代 macOS 上转译运行 32 位 Mach-O（i386）程序的实验性工具（PoC）。

项目通过“解析 Mach-O + 映射客体内存 + x86 指令解释执行 + syscall shim”这一链路，让部分旧的 32 位用户态程序可以在当前系统上继续运行或用于研究。

## 特性

- 支持解析 32 位 i386 Mach-O 头与关键加载命令（`LC_SEGMENT`、`LC_UNIXTHREAD`）。
- 将程序段映射到客体内存，按权限位设置读/写/执行属性。
- 构建基础栈帧（`argc/argv`）并初始化 `EIP/ESP`。
- 提供一套 32 位 x86 指令子集解释器（非完整实现）。
- 通过 `int 0x80` 进入 syscall shim，目前已支持：
  - `exit(1)`
  - `write(4)`（`fd=1/2` 输出到宿主标准输出/错误）
- 支持 `--trace` 指令级寄存器追踪输出。

## 适用场景

- 研究 32 位 Darwin ABI / Mach-O 加载流程。
- 教学演示 x86 指令解释执行器实现。
- 迁移前的旧二进制行为观察与调试。

不建议作为生产级兼容层使用。

## 快速开始

### 1. 环境要求

- macOS（现代版本，Intel/Apple Silicon 均可）
- Rust 工具链（推荐 stable，含 `cargo`）

### 2. 编译

```bash
cargo build --release -p m32run
```

生成物路径：

```text
target/release/m32run
```

### 3. 运行

```bash
./target/release/m32run [--trace] <macho32-file> [args...]
```

也可以直接：

```bash
cargo run -p m32run -- [--trace] <macho32-file> [args...]
```

程序无参数时会输出：

```text
Usage: m32run [--trace] <macho32-file> [args...]
```

### 4. 输入文件要求

建议先用 `file` 验证目标二进制：

```bash
file <macho32-file>
```

目标应为 32 位 i386 Mach-O（例如包含 `Mach-O ... i386`）。

## 工作原理（简述）

1. `macho32` 解析 Mach-O，提取段信息与入口地址。
2. `loader` 将段映射到 `guestmem`，拷贝文件内容。
3. `abi` 构造初始栈并设置 `ESP`。
4. `x86core` 从入口地址开始解释执行指令。
5. 遇到 `int 0x80` 时，交给 `shim` 处理已实现 syscall。

## 当前限制

- 仅支持 32 位 i386 Mach-O，不支持 x86_64/FAT 通用二进制。
- 当前只解析关键加载命令，`LC_MAIN` 等路径未完整覆盖。
- syscall 目前仅实现 `exit(1)` 和 `write(4)`。
- 栈初始化仅覆盖 `argc/argv`，未完整注入 `envp/auxv`。
- 非完整 x86 指令集，遇到未实现 opcode 会中止并提示地址。
- 目前有固定执行上限（默认单次运行最多 100000 条指令）。

## 项目结构

```text
crates/
  macho32   # 32-bit Mach-O 解析
  guestmem  # 客体内存模型（区域映射 + 权限）
  abi       # 32-bit Darwin ABI 栈初始化
  x86core   # x86 指令解码与执行核心
  shim      # syscall shim（int 0x80 分发）
  loader    # 装载器：串联解析、映射、CPU 初始化
  host      # 宿主接口预留（当前为空）
  m32run    # CLI 入口
```

## 说明

这是一个以可读性和可扩展性为目标的实验项目。如果你准备扩展 syscall、完善指令覆盖、补齐 ABI/动态链接器支持，建议从 `shim`、`x86core`、`loader` 三个 crate 开始。

