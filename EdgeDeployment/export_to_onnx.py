# -*- coding: utf-8 -*-
"""Export the subject-06 PyTorch checkpoint to ONNX and verify its numerics."""
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from net_imp import FDNetCNN_CrossSubject  # noqa: E402


CHECKPOINT = HERE.parent / "best_model.pth"
OUTPUT = HERE / "best_model.onnx"


def main():
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "缺少导出依赖，请执行: python -m pip install onnx onnxruntime"
        ) from exc

    model = FDNetCNN_CrossSubject(
        classes=2,
        sampleChannel=30,
        sampleLength=384,
        fs=128,
        stem_channels=16,
        branch_channels=24,
        emb_dim=64,
        dropout=0.35,
        return_log_probs=False,
        active_branches=("s", "m", "l"),
        use_temporal_block=True,
        pool_type="mean_std",
    ).float()

    state = torch.load(str(CHECKPOINT), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    # Fixed channel/time dimensions, dynamic batch dimension.
    dummy = torch.randn(2, 1, 30, 384, dtype=torch.float32)
    with torch.no_grad():
        torch_logits = model(dummy).numpy()

    torch.onnx.export(
        model,
        dummy,
        str(OUTPUT),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["eeg"],
        output_names=["logits"],
        dynamic_axes={"eeg": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    graph = onnx.load(str(OUTPUT))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(OUTPUT), providers=["CPUExecutionProvider"])
    ort_logits = session.run(["logits"], {"eeg": dummy.numpy()})[0]
    diff = float(np.max(np.abs(torch_logits - ort_logits)))
    if not np.allclose(torch_logits, ort_logits, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"PyTorch/ONNX 输出不一致，max_abs_diff={diff:.8g}")

    print(f"已导出: {OUTPUT}")
    print("输入: eeg float32 [N, 1, 30, 384]")
    print("输出: logits float32 [N, 2]")
    print(f"PyTorch/ONNX 最大绝对误差: {diff:.8g}")


if __name__ == "__main__":
    main()
