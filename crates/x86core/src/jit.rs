use crate::cpu::Cpu;
use crate::exec::ExecError;
use guestmem::GuestMemory;
use std::cell::RefCell;
use std::collections::HashMap;
use std::env;

#[derive(Debug, Clone)]
struct CompiledBlock {
    pcs: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct JitConfig {
    enabled: bool,
    hot_threshold: u32,
    max_block_len: usize,
}

#[derive(Debug, Default)]
struct JitRuntime {
    hits: HashMap<u32, u32>,
    blocks: HashMap<u32, CompiledBlock>,
}

thread_local! {
    static JIT_RUNTIME: RefCell<JitRuntime> = RefCell::new(JitRuntime::default());
}

fn config_from_env() -> JitConfig {
    let enabled = env::var("M32RUN_JIT")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true);
    let hot_threshold = env::var("M32RUN_JIT_HOT")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(12);
    let max_block_len = env::var("M32RUN_JIT_BLOCK_LEN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(64);

    JitConfig {
        enabled,
        hot_threshold,
        max_block_len,
    }
}

fn run_interpreter_steps(
    cpu: &mut Cpu,
    mem: &mut GuestMemory,
    mut budget: usize,
) -> Result<usize, ExecError> {
    let mut ran = 0usize;
    while budget != 0 {
        cpu.step(mem)?;
        budget -= 1;
        ran += 1;
    }
    Ok(ran)
}

fn run_cached_block(
    cpu: &mut Cpu,
    mem: &mut GuestMemory,
    block: &CompiledBlock,
    remaining: usize,
) -> Result<usize, ExecError> {
    let mut ran = 0usize;
    for &expected_pc in &block.pcs {
        if ran >= remaining {
            break;
        }
        if cpu.eip != expected_pc {
            break;
        }
        cpu.step(mem)?;
        ran += 1;
    }
    Ok(ran)
}

fn build_hot_block(
    cpu: &mut Cpu,
    mem: &mut GuestMemory,
    max_len: usize,
) -> Result<(CompiledBlock, usize), ExecError> {
    let mut pcs = Vec::with_capacity(max_len.min(64));
    let mut ran = 0usize;

    while ran < max_len {
        let cur = cpu.eip;
        pcs.push(cur);
        cpu.step(mem)?;
        ran += 1;

        // 终止条件：发生明显控制流变化，避免把大范围路径粘在一个块里。
        if cpu.eip <= cur {
            break;
        }
        if cur.wrapping_sub(pcs[0]) > 0x4000 {
            break;
        }
    }

    Ok((CompiledBlock { pcs }, ran))
}

pub fn run_with_jit(
    cpu: &mut Cpu,
    mem: &mut GuestMemory,
    max_instructions: usize,
) -> Result<(), ExecError> {
    let cfg = config_from_env();
    if !cfg.enabled {
        run_interpreter_steps(cpu, mem, max_instructions)?;
        return Ok(());
    }

    let mut ran_total = 0usize;
    while ran_total < max_instructions {
        let pc = cpu.eip;
        let remaining = max_instructions - ran_total;

        // 先尝试命中缓存块。
        let block = JIT_RUNTIME.with(|rt| rt.borrow().blocks.get(&pc).cloned());
        if let Some(block) = block {
            let ran = run_cached_block(cpu, mem, &block, remaining)?;
            if ran == 0 {
                // 缓存块失配，清理后回退一步解释。
                JIT_RUNTIME.with(|rt| {
                    rt.borrow_mut().blocks.remove(&pc);
                });
                ran_total += run_interpreter_steps(cpu, mem, 1)?;
            } else {
                ran_total += ran;
            }
            continue;
        }

        // 热度统计：到阈值后，直接在本次执行中构建并缓存 trace block。
        let hits = JIT_RUNTIME.with(|rt| {
            let mut rt = rt.borrow_mut();
            let entry = rt.hits.entry(pc).or_insert(0);
            *entry = entry.saturating_add(1);
            *entry
        });

        if hits >= cfg.hot_threshold {
            let block_len = cfg.max_block_len.min(remaining);
            let (block, ran) = build_hot_block(cpu, mem, block_len)?;
            if !block.pcs.is_empty() {
                JIT_RUNTIME.with(|rt| {
                    rt.borrow_mut().blocks.insert(pc, block);
                });
            }
            ran_total += ran;
        } else {
            ran_total += run_interpreter_steps(cpu, mem, 1)?;
        }
    }

    Ok(())
}
