# -*- coding: utf-8 -*-
"""Reproducible FP32 ONNX resource benchmark for a Raspberry Pi.

Power is measured by an external meter. This script creates a stable continuous
inference workload and can calculate energy after the measured active power is
provided.
"""
import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from test_subject06 import DEFAULT_DATA, DEFAULT_MODEL, load_data


HERE = Path(__file__).resolve().parent


def create_session(model_path, threads):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("请安装依赖: python3 -m pip install onnxruntime numpy scipy") from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def check_model_io(session):
    inp = session.get_inputs()[0]
    if inp.type != "tensor(float)":
        raise ValueError(f"本测试要求 FP32 ONNX，模型输入类型实际为 {inp.type}")
    shape = inp.shape
    if len(shape) != 4 or shape[1:] != [1, 30, 384]:
        raise ValueError(f"期望 ONNX 输入 [N,1,30,384]，实际为 {shape}")
    return inp.name, session.get_outputs()[0].name


def warm_up(session, input_name, output_name, samples, runs):
    for i in range(runs):
        session.run([output_name], {input_name: samples[i % len(samples):i % len(samples) + 1]})


def percentile_nearest(values, q):
    # Compatible with older NumPy versions commonly used on Raspberry Pi.
    try:
        return float(np.percentile(values, q, method="linear"))
    except TypeError:
        return float(np.percentile(values, q, interpolation="linear"))


def latency_benchmark(session, input_name, output_name, samples, rounds):
    timings_ns = np.empty(len(samples) * rounds, dtype=np.int64)
    pos = 0
    total_start = time.perf_counter_ns()
    for _ in range(rounds):
        for i in range(len(samples)):
            start = time.perf_counter_ns()
            session.run([output_name], {input_name: samples[i:i + 1]})
            timings_ns[pos] = time.perf_counter_ns() - start
            pos += 1
    wall_ns = time.perf_counter_ns() - total_start

    ms = timings_ns.astype(np.float64) / 1e6
    return {
        "inferences": int(len(ms)),
        "mean_ms": float(np.mean(ms)),
        "std_ms": float(np.std(ms)),
        "median_ms": percentile_nearest(ms, 50),
        "p95_ms": percentile_nearest(ms, 95),
        "min_ms": float(np.min(ms)),
        "max_ms": float(np.max(ms)),
        "wall_time_s": float(wall_ns / 1e9),
        "throughput_samples_s": float(len(ms) / (wall_ns / 1e9)),
    }


def power_workload(session, input_name, output_name, samples, duration_s):
    print(f"\n功率负载开始：请读取外部功率计，持续 {duration_s:.1f} 秒……", flush=True)
    count = 0
    index = 0
    start = time.perf_counter()
    deadline = start + duration_s
    while time.perf_counter() < deadline:
        session.run([output_name], {input_name: samples[index:index + 1]})
        count += 1
        index = (index + 1) % len(samples)
    elapsed = time.perf_counter() - start
    print("功率负载结束。", flush=True)
    return {
        "duration_s": float(elapsed),
        "inferences": count,
        "throughput_samples_s": float(count / elapsed),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Raspberry Pi FP32 ONNX论文资源测试")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=20,
                        help="延迟测试遍历全部样本的轮数，默认20")
    parser.add_argument("--mode", choices=("latency", "power", "both"), default="latency")
    parser.add_argument("--power-duration", type=float, default=60.0,
                        help="功率模式持续秒数，默认60")
    parser.add_argument("--power-w", type=float, default=None,
                        help="外部功率计测得的平均Active Power；用于计算Energy")
    parser.add_argument("--output", type=Path, default=HERE / "resource_results.json")
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 0 or args.rounds < 1 or args.power_duration <= 0:
        parser.error("threads/rounds/power-duration must be positive number，warmup cannot be negtive number")
    if args.power_w is not None and args.power_w <= 0:
        parser.error("--power-w 必须大于0")
    return args


def main():
    args = parse_args()
    x, _ = load_data(args.data)
    session = create_session(args.model, args.threads)
    input_name, output_name = check_model_io(session)

    print("===== Testing Configuration =====")
    print(f"model={args.model.resolve()}")
    print(f"data={args.data.resolve()}, samples={len(x)}")
    print(f"precision=FP32, batch_size=1, threads={args.threads}")
    print(f"warmup={args.warmup}, rounds={args.rounds}, mode={args.mode}")
    warm_up(session, input_name, output_name, x, args.warmup)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "samples": len(x),
        "precision": "FP32",
        "batch_size": 1,
        "threads": args.threads,
        "warmup_runs": args.warmup,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

    if args.mode in ("latency", "both"):
        latency = latency_benchmark(session, input_name, output_name, x, args.rounds)
        result["latency"] = latency
        print("\n===== Latency（Only session.run）=====")
        print(f"inferences={latency['inferences']} ({len(x)} samples x {args.rounds} rounds)")
        print(f"mean={latency['mean_ms']:.4f} ms, std={latency['std_ms']:.4f} ms")
        print(f"median={latency['median_ms']:.4f} ms, p95={latency['p95_ms']:.4f} ms")
        print(f"throughput={latency['throughput_samples_s']:.2f} samples/s")

    if args.mode in ("power", "both"):
        result["power_workload"] = power_workload(
            session, input_name, output_name, x, args.power_duration
        )

    if args.power_w is not None:
        result["active_power_w"] = args.power_w
        if "latency" in result:
            energy = args.power_w * result["latency"]["mean_ms"]
            result["energy_mj_per_inference"] = energy
            print(f"\nActive Power={args.power_w:.4f} W")
            print(f"Energy={args.power_w:.4f} x {result['latency']['mean_ms']:.4f}"
                  f" = {energy:.4f} mJ/inference")
        else:
            print("Power Record；Compute Energy，Please use --mode latency or --mode both。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResult saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
