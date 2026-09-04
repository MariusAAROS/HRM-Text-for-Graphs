from typing import Any, Optional
from collections import defaultdict
import os
import pydantic
import json
import yaml
from omegaconf import OmegaConf

from utils.functions import load_model_class


class BenchmarkConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    generation_config: dict[str, Any] = {}


class EvaluationConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    run_only: Optional[list[str]] = None
    engine: str
    generation_config: dict[str, Any] = {}
    benchmarks: list[BenchmarkConfig]

    # Reporting (mirrors scripts/eval_graphqa.py so ablation runs are comparable).
    out: Optional[str] = None
    samples_out: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[list[str]] = None

    @pydantic.model_validator(mode='after')
    def check_run_only_against_benchmarks(self):
        if self.run_only is not None:
            assert self.run_only, "run_only cannot be empty."

            valid_set = {b_cfg.name for b_cfg in self.benchmarks}
            for b_name in self.run_only:
                if b_name not in valid_set:
                    raise ValueError(f"Unknown benchmark name in run_only: {b_name}")

        return self


def _jsonable(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def collect_samples(b_name: str, prompts: list[str], generations: list[str], benchmark) -> list[dict]:
    """Pair every prompt with what the model produced, for manual inspection."""
    truths = getattr(benchmark, "ground_truths", None) or []
    # Only some benchmarks (e.g. GSM8k) expose the parser used to score a generation.
    extract = getattr(benchmark, "_extract_answer", None)

    rows = []
    for i, (prompt, generation) in enumerate(zip(prompts, generations)):
        truth = _jsonable(truths[i]) if i < len(truths) else None
        parsed = _jsonable(extract(generation)) if extract is not None else None
        rows.append({
            "benchmark": b_name,
            "index": i,
            "prompt": prompt,
            "generation": generation,
            "parsed": parsed,
            "ground_truth": truth,
            "correct": (parsed == truth) if extract is not None and truth is not None else None,
        })
    return rows


def read_config_cycles(ckpt_path: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    if ckpt_path is None:
        return None, None
    cfg_file = os.path.join(ckpt_path, "all_config.yaml")
    if not os.path.exists(cfg_file):
        return None, None
    with open(cfg_file, "r") as f:
        arch = (yaml.safe_load(f) or {}).get("arch", {})
    return arch.get("H_cycles"), arch.get("L_cycles")


def main():
    # 1. Load and Override Config
    cli_conf = OmegaConf.from_cli()
    config_path = cli_conf.pop("config", "evaluation/config/hrm_benchmarking.yaml")
    # Merge YAML config with CLI overrides
    base_conf = OmegaConf.load(config_path)
    cfg = EvaluationConfig(**OmegaConf.to_container(OmegaConf.merge(base_conf, cli_conf), resolve=True))  # type: ignore

    # 2. Initialize Engine
    print(f"Initializing Engine: {cfg.engine}...")
    engine_cls = load_model_class(f"engines@{cfg.engine}", prefix="evaluation.")
    engine = engine_cls(**(cfg.__pydantic_extra__ or {}))

    # 3. Group Benchmarks by Generation Config
    # To minimize bubbles, we group benchmarks sharing the exact same generation kwargs
    print("Preparing and grouping benchmarks...")

    # grouped_tasks maps a hashed config tuple -> {"prompts": [...], "benchmarks": [...]}
    grouped_tasks = defaultdict(lambda: {"prompts": [], "benchmarks": []})
    for b_cfg in cfg.benchmarks:
        b_name = b_cfg.name
        if cfg.run_only is not None and b_name not in cfg.run_only:
            continue

        # Instantiate benchmark
        bench_cls = load_model_class(f"benchmarks@{b_name}", prefix="evaluation.")
        benchmark = bench_cls(**(b_cfg.__pydantic_extra__ or {}))

        # Resolve final generation config: Base -> Benchmark Specific -> Benchmark Overrides
        gen_cfg = cfg.generation_config | benchmark.generation_overrides | b_cfg.generation_config
        # Hash key for grouping identical ones
        gen_key = json.dumps(gen_cfg, sort_keys=True)
        
        # Track offsets to map flattened generations back to their source benchmarks
        start_idx = len(grouped_tasks[gen_key]["prompts"])
        grouped_tasks[gen_key]["prompts"].extend(benchmark.prompts)
        end_idx = len(grouped_tasks[gen_key]["prompts"])
        
        grouped_tasks[gen_key]["benchmarks"].append((b_name, benchmark, start_idx, end_idx))

    # 4. Generate and Evaluate per Group
    all_results = {}
    all_samples: list[dict] = []
    
    for gen_key, group in grouped_tasks.items():
        gen_kwargs = json.loads(gen_key)
        prompt_template = gen_kwargs.pop("prompt_template", "{prompt}")
        # Apply prompt templates
        prompts = [prompt_template.format(prompt=s) for s in group["prompts"]]
        
        print("\n" + "="*50)
        print(f"Running generation batch (Size: {len(prompts)}) with config:")
        for k, v in gen_kwargs.items():
            print(f"  {k}: {v}")
        print("="*50)

        # Generate all prompts for this config group
        generations = engine.generate(prompts, **gen_kwargs)

        # Dispatch results back to individual benchmarks
        for b_name, benchmark, start_idx, end_idx in group["benchmarks"]:
            b_generations = generations[start_idx:end_idx]
            metrics = benchmark.compute_metrics(b_generations)
            all_results[b_name] = metrics

            if cfg.samples_out is not None:
                # prompts (not group["prompts"]) is what the model actually saw, post-template.
                all_samples.extend(collect_samples(b_name, prompts[start_idx:end_idx], b_generations, benchmark))

    # 5. Summary Report
    print("\n" + "#"*50 + "\nEVALUATION SUMMARY\n" + "#"*50)
    for b_name, metrics in all_results.items():
        print(f"\n--- {b_name} ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k:.<25}: {v:.4f}")
            else:
                print(f"{k:.<25}: {v}")

    # 6. Persist (opt-in, for the L/H ablation sweeps)
    ckpt_path = (cfg.__pydantic_extra__ or {}).get("ckpt_path")
    H_cycles, L_cycles = read_config_cycles(ckpt_path)

    if cfg.out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
        with open(cfg.out, "w") as f:
            json.dump({
                "ckpt_path": ckpt_path,
                "H_cycles": H_cycles,
                "L_cycles": L_cycles,
                "ratio_L_over_H": (L_cycles / H_cycles) if (H_cycles and L_cycles) else None,
                "results": all_results,
            }, f, indent=2)
        print(f"\nWrote {cfg.out}")

    if cfg.samples_out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.samples_out)), exist_ok=True)
        with open(cfg.samples_out, "w", encoding="utf-8") as f:
            for row in all_samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(all_samples)} samples to {cfg.samples_out}")

    if cfg.wandb_project is not None:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name or (os.path.basename(os.path.normpath(ckpt_path)) if ckpt_path else None),
            tags=cfg.wandb_tags,
            config={
                "ckpt_path": ckpt_path,
                "engine": cfg.engine,
                "generation_config": cfg.generation_config,
                "H_cycles": H_cycles,
                "L_cycles": L_cycles,
                "ratio_L_over_H": (L_cycles / H_cycles) if (H_cycles and L_cycles) else None,
            },
        )
        wandb.log({
            f"eval/{b_name}/{k}": v
            for b_name, metrics in all_results.items()
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        })
        wandb.finish()


if __name__ == "__main__":
    main()
