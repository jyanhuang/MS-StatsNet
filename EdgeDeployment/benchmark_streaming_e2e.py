# -*- coding: utf-8 -*-
"""Raspberry Pi streaming end-to-end benchmark for the Ours ONNX model.

The input MAT file is loaded into RAM as *raw* EEG. Each 3-second window is
then replayed sample-by-sample at 128 Hz into a 384-sample buffer. Once full,
the script executes the deployment path:

    raw stream -> buffer -> z-score preprocessing -> ONNX -> softmax/threshold

Latency mode reports each stage, compute-pipeline latency, and true latency
from the first streamed sample to the classification output. Power mode loops
over the exact same full pipeline for a requested duration so an external
power meter measures representative deployment load (including acquisition
wait time when real-time pacing is enabled).

for i in $(seq 1 5); do
  python3 benchmark_streaming_e2e.py \
    --mode latency \
    --windows 20 \
    --warmup 20 \
    --threads 1 \
    --output "streaming_e2e_latency_run${i}.json"
done

for i in $(seq 1 5); do
  python3 benchmark_streaming_e2e.py \
    --mode power \
    --power-duration 60 \
    --warmup 20 \
    --threads 1 \
    --output "streaming_e2e_power_run${i}.json"
done
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io as sio


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "best_model.onnx"
DEFAULT_DATA = HERE.parent / "subject_06.mat"
FS_HZ = 128
WINDOW_SAMPLES = 384
CHANNELS = 30


def create_session(model_path, threads):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "Install dependencies with: python3 -m pip install onnxruntime numpy scipy"
        ) from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def check_model_io(session):
    model_input = session.get_inputs()[0]
    shape = model_input.shape
    if model_input.type != "tensor(float)":
        raise ValueError(f"Expected FP32 ONNX input, got {model_input.type}")
    if len(shape) != 4 or shape[1:] != [1, CHANNELS, WINDOW_SAMPLES]:
        raise ValueError(
            f"Expected ONNX input [N,1,{CHANNELS},{WINDOW_SAMPLES}], got {shape}"
        )
    return model_input.name, session.get_outputs()[0].name


def load_raw_eeg(path):
    """Load raw windows into RAM without performing model preprocessing."""
    mat = sio.loadmat(str(path))
    if "EEGsample" not in mat:
        raise KeyError("MAT file must contain EEGsample")
    eeg = np.asarray(mat["EEGsample"])
    if eeg.ndim != 3 or eeg.shape[1:] != (CHANNELS, WINDOW_SAMPLES):
        raise ValueError(
            f"Expected EEGsample [N,{CHANNELS},{WINDOW_SAMPLES}], got {eeg.shape}"
        )
    labels = None
    if "substate" in mat:
        labels = np.asarray(mat["substate"]).squeeze().astype(np.int64)
        if len(labels) != len(eeg):
            raise ValueError("EEGsample and substate lengths differ")
        keep = np.isin(labels, (0, 1))
        eeg, labels = eeg[keep], labels[keep]
    return np.ascontiguousarray(eeg, dtype=np.float32), labels


def preprocess(raw_buffer):
    """Match training preprocessing: per-channel temporal z-score."""
    x = raw_buffer[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=True)
    mean = np.mean(x, axis=3, keepdims=True)
    std = np.std(x, axis=3, keepdims=True)
    x = (x - mean) / (std + 1e-8)
    return np.ascontiguousarray(x, dtype=np.float32)


def softmax_positive_probability(logits):
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return float((exp / np.sum(exp, axis=1, keepdims=True))[0, 1])


def percentile(values, q):
    try:
        return float(np.percentile(values, q, method="linear"))
    except TypeError:
        return float(np.percentile(values, q, interpolation="linear"))


def summarize_ms(values_ns):
    values = np.asarray(values_ns, dtype=np.float64) / 1e6
    if len(values) == 0:
        return {key: None for key in ("mean_ms", "std_ms", "median_ms", "p95_ms", "min_ms", "max_ms")}
    return {
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values, ddof=0)),
        "median_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


class RealtimePacer:
    """Deadline-based pacer that avoids accumulating per-sample sleep drift."""

    def __init__(self, fs_hz, realtime):
        self.period_ns = int(round(1e9 / fs_hz))
        self.realtime = realtime

    def replay_window(self, raw_window, destination):
        first_sample_ns = time.perf_counter_ns()
        next_deadline_ns = first_sample_ns
        for sample_index in range(WINDOW_SAMPLES):
            if self.realtime and sample_index:
                next_deadline_ns += self.period_ns
                remaining_ns = next_deadline_ns - time.perf_counter_ns()
                if remaining_ns > 0:
                    time.sleep(remaining_ns / 1e9)
            destination[:, sample_index] = raw_window[:, sample_index]
        buffer_full_ns = time.perf_counter_ns()
        return first_sample_ns, buffer_full_ns


def execute_one_window(
    session,
    input_name,
    output_name,
    raw_window,
    stream_buffer,
    pacer,
    threshold,
):
    first_sample_ns, buffer_full_ns = pacer.replay_window(raw_window, stream_buffer)

    pre_start_ns = time.perf_counter_ns()
    model_input = preprocess(stream_buffer)
    pre_end_ns = time.perf_counter_ns()

    inference_start_ns = time.perf_counter_ns()
    logits = session.run([output_name], {input_name: model_input})[0]
    inference_end_ns = time.perf_counter_ns()

    post_start_ns = time.perf_counter_ns()
    probability = softmax_positive_probability(logits)
    prediction = int(probability >= threshold)
    output_ns = time.perf_counter_ns()

    return {
        "prediction": prediction,
        "probability": probability,
        "buffer_fill_ns": buffer_full_ns - first_sample_ns,
        "preprocessing_ns": pre_end_ns - pre_start_ns,
        "inference_ns": inference_end_ns - inference_start_ns,
        "postprocessing_ns": output_ns - post_start_ns,
        "compute_pipeline_ns": output_ns - buffer_full_ns,
        "end_to_end_ns": output_ns - first_sample_ns,
    }


def warm_up(session, input_name, output_name, raw_windows, runs, threshold):
    buffer = np.empty((CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
    accelerated = RealtimePacer(FS_HZ, realtime=False)
    for index in range(runs):
        execute_one_window(
            session,
            input_name,
            output_name,
            raw_windows[index % len(raw_windows)],
            buffer,
            accelerated,
            threshold,
        )


def latency_benchmark(
    session, input_name, output_name, raw_windows, labels, windows, realtime, threshold
):
    count = min(windows, len(raw_windows))
    pacer = RealtimePacer(FS_HZ, realtime=realtime)
    buffer = np.empty((CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
    stage_names = (
        "buffer_fill_ns",
        "preprocessing_ns",
        "inference_ns",
        "postprocessing_ns",
        "compute_pipeline_ns",
        "end_to_end_ns",
    )
    timings = {name: [] for name in stage_names}
    predictions = []
    probabilities = []

    wall_start_ns = time.perf_counter_ns()
    for index in range(count):
        record = execute_one_window(
            session,
            input_name,
            output_name,
            raw_windows[index],
            buffer,
            pacer,
            threshold,
        )
        for name in stage_names:
            timings[name].append(record[name])
        predictions.append(record["prediction"])
        probabilities.append(record["probability"])
    wall_ns = time.perf_counter_ns() - wall_start_ns

    result = {
        "windows": count,
        "realtime_pacing": realtime,
        "nominal_sample_rate_hz": FS_HZ,
        "nominal_window_s": WINDOW_SAMPLES / FS_HZ,
        "wall_time_s": float(wall_ns / 1e9),
        "outputs_per_s": float(count / (wall_ns / 1e9)),
        "stages": {name.removesuffix("_ns"): summarize_ms(values) for name, values in timings.items()},
        "predictions": predictions,
        "positive_probabilities": probabilities,
    }
    if labels is not None:
        truth = labels[:count]
        pred = np.asarray(predictions, dtype=np.int64)
        result["accuracy"] = float(np.mean(pred == truth))
    return result


def power_workload(
    session,
    input_name,
    output_name,
    raw_windows,
    duration_s,
    realtime,
    threshold,
):
    print(
        f"\nPower workload started for {duration_s:.1f} s "
        f"(realtime_pacing={realtime}). Read the external power meter now.",
        flush=True,
    )
    pacer = RealtimePacer(FS_HZ, realtime=realtime)
    buffer = np.empty((CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
    stage_totals_ns = {
        "buffer_fill": 0,
        "preprocessing": 0,
        "inference": 0,
        "postprocessing": 0,
        "compute_pipeline": 0,
        "end_to_end": 0,
    }
    completed = 0
    index = 0
    start_ns = time.perf_counter_ns()
    deadline_ns = start_ns + int(duration_s * 1e9)
    # A window already started is completed, so actual duration may exceed the
    # request by at most one 3-second acquisition interval in real-time mode.
    while time.perf_counter_ns() < deadline_ns:
        record = execute_one_window(
            session,
            input_name,
            output_name,
            raw_windows[index],
            buffer,
            pacer,
            threshold,
        )
        for name in stage_totals_ns:
            stage_totals_ns[name] += record[f"{name}_ns"]
        completed += 1
        index = (index + 1) % len(raw_windows)
    elapsed_s = (time.perf_counter_ns() - start_ns) / 1e9
    print("Power workload finished.", flush=True)
    return {
        "requested_duration_s": float(duration_s),
        "actual_duration_s": float(elapsed_s),
        "realtime_pacing": realtime,
        "completed_windows": completed,
        "outputs_per_s": float(completed / elapsed_s),
        "mean_stage_ms": {
            name: float(total / completed / 1e6) for name, total in stage_totals_ns.items()
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Streaming raw-EEG -> preprocessing -> ONNX -> output benchmark"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.540)
    parser.add_argument("--warmup", type=int, default=20,
                        help="Accelerated full-pipeline warm-up windows")
    parser.add_argument("--windows", type=int, default=10,
                        help="Number of windows in latency mode")
    parser.add_argument("--mode", choices=("latency", "power", "both"), default="latency")
    parser.add_argument("--power-duration", type=float, default=60.0)
    parser.add_argument("--power-w", type=float, default=None,
                        help="Externally measured average power in watts")
    parser.add_argument(
        "--accelerated",
        action="store_true",
        help="Do not sleep at 128 Hz. Useful for maximum-load stress testing; default is real time.",
    )
    parser.add_argument("--output", type=Path, default=HERE / "streaming_e2e_results.json")
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 0 or args.windows < 1 or args.power_duration <= 0:
        parser.error("threads/windows/power-duration must be positive; warmup cannot be negative")
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0, 1]")
    if args.power_w is not None and args.power_w <= 0:
        parser.error("power-w must be positive")
    return args


def main():
    args = parse_args()
    raw_windows, labels = load_raw_eeg(args.data)
    session = create_session(args.model, args.threads)
    input_name, output_name = check_model_io(session)
    realtime = not args.accelerated

    print("===== Streaming End-to-End Configuration =====")
    print(f"model={args.model.resolve()}")
    print(f"data={args.data.resolve()}, RAM windows={len(raw_windows)}")
    print(f"stream={FS_HZ} Hz, buffer={WINDOW_SAMPLES} samples (3 s), realtime={realtime}")
    print(f"precision=FP32, batch=1, threads={args.threads}, threshold={args.threshold:.3f}")
    warm_up(session, input_name, output_name, raw_windows, args.warmup, args.threshold)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "raw_windows_in_ram": int(len(raw_windows)),
        "sample_rate_hz": FS_HZ,
        "channels": CHANNELS,
        "buffer_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SAMPLES / FS_HZ,
        "precision": "FP32",
        "batch_size": 1,
        "threads": args.threads,
        "threshold": args.threshold,
        "warmup_windows": args.warmup,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

    if args.mode in ("latency", "both"):
        latency = latency_benchmark(
            session,
            input_name,
            output_name,
            raw_windows,
            labels,
            args.windows,
            realtime,
            args.threshold,
        )
        result["latency"] = latency
        print("\n===== Stage Latency =====")
        for name, stats in latency["stages"].items():
            print(
                f"{name:18s}: mean={stats['mean_ms']:.4f} ms, "
                f"std={stats['std_ms']:.4f}, p95={stats['p95_ms']:.4f}"
            )
        print(f"outputs={latency['windows']}, wall={latency['wall_time_s']:.3f} s")

    if args.mode in ("power", "both"):
        result["power_workload"] = power_workload(
            session,
            input_name,
            output_name,
            raw_windows,
            args.power_duration,
            realtime,
            args.threshold,
        )

    if args.power_w is not None:
        result["measured_average_power_w"] = args.power_w
        if "power_workload" in result:
            workload = result["power_workload"]
            workload["total_energy_j"] = float(args.power_w * workload["actual_duration_s"])
            workload["energy_j_per_output"] = float(
                workload["total_energy_j"] / workload["completed_windows"]
            )
        if "latency" in result:
            compute_ms = result["latency"]["stages"]["compute_pipeline"]["mean_ms"]
            result["estimated_compute_energy_mj_per_output"] = float(args.power_w * compute_ms)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
