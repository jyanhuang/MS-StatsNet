
import os
import time
import copy
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.nn as nn
from datetime import datetime
from net_imp import MSStatsNet_CrossSubject


torch.manual_seed(42)
np.random.seed(42)
torch.cuda.empty_cache()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



SAVE_DIR = "checkpoints" 
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_PATH = os.path.join(SAVE_DIR, "loso_results.txt")


def log_print(msg, log_file=None):
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()



def datanorm(x):
    x = x.copy().astype(np.float32)
    mean = np.mean(x, axis=3, keepdims=True)  
    std = np.std(x, axis=3, keepdims=True)     
    x = (x - mean) / (std + 1e-8)
    return x


# Metrics
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = ((y_pred == 1) & (y_true == 1)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()

    acc = (y_pred == y_true).mean()
    pre = TP / (TP + FP + 1e-6)
    rec = TP / (TP + FN + 1e-6)
    spec = TN / (TN + FP + 1e-6)
    f1 = 2 * pre * rec / (pre + rec + 1e-6)

    return {
        "acc": float(acc),
        "pre": float(pre),
        "recall": float(rec),
        "f1": float(f1),
        "spec": float(spec),
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
    }


def predict_probs(net, X_tensor, batch_size=128):
    net.eval()
    probs_all = []

    loader = DataLoader(
        Data.TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE).float()
            logits = net(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            probs_all.append(probs.cpu().numpy())

    return np.concatenate(probs_all, axis=0)


def eval_model(net, X_test, y_test, threshold=0.5):
    probs = predict_probs(net, X_test)
    pred = (probs >= threshold).astype(np.int64)
    y_true = y_test.cpu().numpy()
    return compute_metrics(y_true, pred), probs


def find_best_threshold(y_true, probs, metric="f1"):
    y_true = np.asarray(y_true).astype(np.int64)
    probs = np.asarray(probs)

    best_t = 0.5
    best_score = -1.0

    for t in np.linspace(0.1, 0.9, 81):
        pred = (probs >= t).astype(np.int64)
        m = compute_metrics(y_true, pred)

        if metric == "balanced_acc":
            score = 0.5 * (m["recall"] + m["spec"])
        else:
            score = m["f1"]

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score


def timed_inference_per_sample(net, X_test, warmup=20, repeat=50):
    net.eval()
    n = X_test.shape[0]
    if n == 0:
        return float("nan")

    x1 = X_test[0:1].to(DEVICE).float()

    with torch.no_grad():
        for _ in range(warmup):
            _ = net(x1)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(repeat):
            _ = net(x1)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

    latency_ms = (t1 - t0) * 1000.0 / repeat
    return float(latency_ms)

# FLOPs / Params
def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def estimate_flops(model, input_size=(1, 1, 30, 384), device=DEVICE):
    flops_dict = {"total": 0}

    def conv_hook(module, inp, out):
        if not isinstance(module, nn.Conv2d):
            return
        batch_size = out.shape[0]
        out_channels = out.shape[1]
        out_h = out.shape[2]
        out_w = out.shape[3]

        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups

        macs_per_output = (in_channels // groups) * kernel_h * kernel_w
        total_macs = batch_size * out_channels * out_h * out_w * macs_per_output
        total_flops = 2 * total_macs
        flops_dict["total"] += total_flops

    def linear_hook(module, inp, out):
        if not isinstance(module, nn.Linear):
            return
        batch_size = inp[0].shape[0]
        in_features = module.in_features
        out_features = module.out_features
        total_flops = 2 * batch_size * in_features * out_features
        flops_dict["total"] += total_flops

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_size).to(device)
        _ = model(dummy)

    for h in hooks:
        h.remove()

    mflops = flops_dict["total"] / 1e6
    return float(mflops)



# Train Utils
def make_class_weights(y_np):
    y_np = np.asarray(y_np).astype(np.int64)
    counts = np.bincount(y_np, minlength=2).astype(np.float32)
    weights = counts.sum() / (2.0 * (counts + 1e-8))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def split_train_val_by_subject(Xtr, ytr, Str, seed=42, strategy="random"):
    subs = np.unique(Str)

    if len(subs) < 2:
        raise ValueError("训练被试数不足，无法再划出一个独立 validation subject。")

    if strategy == "largest":
        counts = [(s, np.sum(Str == s)) for s in subs]
        counts = sorted(counts, key=lambda z: z[1], reverse=True)
        val_sub = counts[0][0]
    elif strategy == "random":
        rng = np.random.RandomState(seed)
        val_sub = rng.choice(subs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    val_mask = (Str == val_sub)
    tr_mask = ~val_mask

    return Xtr[tr_mask], ytr[tr_mask], Xtr[val_mask], ytr[val_mask], int(val_sub)


# Train one LOSO fold
def train_one_fold(
    X_train_np,
    y_train_np,
    S_train_np,
    X_test_np,
    y_test_np,
    cfg,
    epochs=40,
    batch_size=32,
    patience=8
):
    Xtr_np, ytr_np, Xval_np, yval_np, val_sub = split_train_val_by_subject(
        X_train_np, y_train_np, S_train_np, seed=42, strategy="largest"
    )

    Xtr_aug_np = Xtr_np.astype(np.float32)
    ytr_aug_np = ytr_np.astype(np.int64)

    X_train = torch.from_numpy(Xtr_aug_np.astype(np.float32))
    y_train = torch.from_numpy(ytr_aug_np.astype(np.int64)).long()

    X_val = torch.from_numpy(Xval_np.astype(np.float32))
    y_val = torch.from_numpy(yval_np.astype(np.int64)).long()

    X_test = torch.from_numpy(X_test_np.astype(np.float32))
    y_test = torch.from_numpy(y_test_np.astype(np.int64)).long()

    train_loader = DataLoader(
        Data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    dropout = cfg.get("dropout", 0.35 if cfg["tag"] == "cross_subject_dropout035" else 0.30)

    net = FDNetCNN_CrossSubject(
        classes=2,
        sampleChannel=30,
        sampleLength=384,
        fs=128,
        stem_channels=16,
        branch_channels=24,
        emb_dim=64,
        dropout=dropout,
        return_log_probs=False,
        active_branches=cfg.get("active_branches", ("s", "m", "l")),
        use_temporal_block=cfg.get("use_temporal_block", True),
        pool_type=cfg.get("pool_type", "mean_std")
    ).float().to(DEVICE)

    with torch.no_grad():
        _ = net(torch.zeros(1, 1, 30, 384, device=DEVICE))

    total_params = count_parameters(net)
    mflops = estimate_flops(net, input_size=(1, 1, 30, 384), device=DEVICE)

    class_weights = make_class_weights(ytr_np)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    ).to(DEVICE)

    optimizer = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = None
    best_val_f1 = -1.0
    best_threshold = 0.5
    wait = 0

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        val_probs = predict_probs(net, X_val)
        val_threshold, _ = find_best_threshold(y_val.cpu().numpy(), val_probs, metric="f1")
        val_pred = (val_probs >= val_threshold).astype(np.int64)
        val_metrics = compute_metrics(y_val.cpu().numpy(), val_pred)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_threshold = val_threshold
            best_state = copy.deepcopy(net.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    test_metrics, _ = eval_model(net, X_test, y_test, threshold=best_threshold)
    latency_ms = timed_inference_per_sample(net, X_test)

    extra_info = {
        "val_subject": int(val_sub),
        "best_val_f1": float(best_val_f1),
        "best_threshold": float(best_threshold),
        "num_train_samples": int(len(Xtr_aug_np)),
        "num_val_samples": int(len(Xval_np)),
    }

    return net, test_metrics, mflops, latency_ms, total_params, extra_info


# Main: LOSO
def run():
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_print("\n" + "=" * 70, log_file)
        log_print(f"NEW RUN START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
        log_print("=" * 70, log_file)
        log_print("========== LOSO START ==========", log_file)
        log_print(f"DEVICE: {DEVICE}", log_file)
        log_print(f"SAVE_DIR: {SAVE_DIR}", log_file)

        data_T = sio.loadmat("dataset.mat")

        eeg = data_T["EEGsample"]
        y = data_T["substate"].squeeze().astype(np.int64)

        if "subindex" in data_T:
            subindex = data_T["subindex"].squeeze()
        elif "subIndex" in data_T:
            subindex = data_T["subIndex"].squeeze()
        elif "subject_index" in data_T:
            subindex = data_T["subject_index"].squeeze()
        else:
            raise KeyError("dataset.mat no find subindex / subIndex / subject_index.")

        subindex = np.asarray(subindex).squeeze()

        idx = np.concatenate([np.where(y == 0)[0], np.where(y == 1)[0]])
        X = eeg[idx]
        Y = y[idx]
        S = subindex[idx]

        X = np.expand_dims(X, axis=1).astype(np.float32)
        X = datanorm(X)

        unique_subjects = np.unique(S)
        unique_subjects = np.sort(unique_subjects)

        if len(unique_subjects) < 11:
            raise ValueError(f"唯一被试数只有 {len(unique_subjects)}，少于 11。")
        test_subjects = unique_subjects[:11]

        log_print(f"All unique subjects: {unique_subjects.tolist()}", log_file)
        log_print(f"Selected LOSO subjects (11): {test_subjects.tolist()}", log_file)
        log_print(f"Total samples after label filter: {len(X)}", log_file)
        log_print(f"Input shape after expand: {X.shape}", log_file)

        candidates = [
            {"tag": "cross_subject_dropout035", "dropout": 0.35},
            # Ablation examples:
            # {"tag": "only_s", "dropout": 0.35, "active_branches": ("s",)},
            # {"tag": "only_m", "dropout": 0.35, "active_branches": ("m",)},
            # {"tag": "only_l", "dropout": 0.35, "active_branches": ("l",)},
            # {"tag": "no_temporal_block", "dropout": 0.35, "use_temporal_block": False},
            # {"tag": "mean_pool", "dropout": 0.35, "pool_type": "mean"},
            # {"tag": "mean_max_pool", "dropout": 0.35, "pool_type": "mean_max"},
        ]
        # no B1
        # candidates = [
        #     {
        #         "tag": "no_temporal_block",
        #         "dropout": 0.35,
        #         "active_branches": ("s", "m", "l"),
        #         "use_temporal_block": False,  # or False
        #         "pool_type": "mean_std",  # or mean
        #     },
        # ]
        # mean only
        # candidates = [
        #     {
        #         "tag": "mean_pool",
        #         "dropout": 0.35,
        #         "active_branches": ("s", "m", "l"),
        #         "use_temporal_block": True,
        #         "pool_type": "mean",
        #     },
        # ]

        overall_best = {
            "score": -1e18,
            "cfg": None,
            "subject": None,
            "path": None
        }

        for cfg in candidates:
            log_print("\n" + "=" * 70, log_file)
            log_print(f"Config: {cfg['tag']}", log_file)
            log_print("=" * 70, log_file)

            all_results = []

            for fold_id, test_sub in enumerate(test_subjects, start=1):
                test_mask = (S == test_sub)
                train_mask = ~test_mask

                Xtr, Xte = X[train_mask], X[test_mask]
                ytr, yte = Y[train_mask], Y[test_mask]
                Str = S[train_mask]

                log_print(
                    f"\n[LOSO Fold {fold_id:02d}] Test Subject = {int(test_sub)} | "
                    f"Train={len(Xtr)} | Test={len(Xte)}",
                    log_file
                )

                net, metrics, mflops, latency_ms, params, extra_info = train_one_fold(
                    Xtr, ytr, Str, Xte, yte, cfg,
                    epochs=40,
                    batch_size=32,
                    patience=8
                )

                model_path = os.path.join(
                    SAVE_DIR,
                    f"{cfg['tag']}_subject_{int(test_sub)}.pth"
                )
                torch.save(net.state_dict(), model_path)

                result = {
                    "fold": fold_id,
                    "subject": int(test_sub),
                    "val_subject": extra_info["val_subject"],
                    "num_val_samples": extra_info["num_val_samples"],
                    "num_train_samples": extra_info["num_train_samples"],
                    "best_threshold": extra_info["best_threshold"],
                    "best_val_f1": extra_info["best_val_f1"],
                    "acc": metrics["acc"],
                    "pre": metrics["pre"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "spec": metrics["spec"],
                    "MFLOPs": mflops,
                    "latency_ms": latency_ms,
                    "parameters": params,
                    "model_path": model_path
                }
                all_results.append(result)

                log_print(
                    (
                        f"[Fold {fold_id:02d} | Subject {int(test_sub)}] "
                        f"val_sub={result['val_subject']}, "
                        f"train_n={result['num_train_samples']}, "
                        f"val_n={result['num_val_samples']}, "
                        f"thr={result['best_threshold']:.3f}, "
                        f"val_f1={result['best_val_f1']:.4f}, "
                        f"acc={result['acc']:.4f}, "
                        f"pre={result['pre']:.4f}, "
                        f"recall={result['recall']:.4f}, "
                        f"f1={result['f1']:.4f}, "
                        f"spec={result['spec']:.4f}, "
                        f"MFLOPs={result['MFLOPs']:.4f}, "
                        f"latency={result['latency_ms']:.4f} ms/sample, "
                        f"parameters={result['parameters']}"
                    ),
                    log_file
                )
                log_print(f"Saved model: {model_path}", log_file)

            mean_acc = np.mean([r["acc"] for r in all_results])
            mean_pre = np.mean([r["pre"] for r in all_results])
            mean_recall = np.mean([r["recall"] for r in all_results])
            mean_f1 = np.mean([r["f1"] for r in all_results])
            mean_spec = np.mean([r["spec"] for r in all_results])
            mean_mflops = np.mean([r["MFLOPs"] for r in all_results])
            mean_latency = np.mean([r["latency_ms"] for r in all_results])
            mean_params = np.mean([r["parameters"] for r in all_results])

            std_acc = np.std([r["acc"] for r in all_results])
            std_pre = np.std([r["pre"] for r in all_results])
            std_recall = np.std([r["recall"] for r in all_results])
            std_f1 = np.std([r["f1"] for r in all_results])
            std_spec = np.std([r["spec"] for r in all_results])
            std_mflops = np.std([r["MFLOPs"] for r in all_results])
            std_latency = np.std([r["latency_ms"] for r in all_results])
            std_params = np.std([r["parameters"] for r in all_results])

            log_print("\n" + "-" * 70, log_file)
            log_print(f"[SUMMARY] Config: {cfg['tag']}", log_file)
            log_print(f"acc       = {mean_acc:.4f} ± {std_acc:.4f}", log_file)
            log_print(f"pre       = {mean_pre:.4f} ± {std_pre:.4f}", log_file)
            log_print(f"recall    = {mean_recall:.4f} ± {std_recall:.4f}", log_file)
            log_print(f"f1        = {mean_f1:.4f} ± {std_f1:.4f}", log_file)
            log_print(f"spec      = {mean_spec:.4f} ± {std_spec:.4f}", log_file)
            log_print(f"MFLOPs    = {mean_mflops:.4f} ± {std_mflops:.4f}", log_file)
            log_print(f"latency   = {mean_latency:.4f} ± {std_latency:.4f} ms/sample", log_file)
            log_print(f"parameters= {mean_params:.2f} ± {std_params:.2f}", log_file)
            log_print("-" * 70, log_file)

            score = mean_f1 - 0.001 * mean_latency

            if score > overall_best["score"]:
                overall_best["score"] = score
                overall_best["cfg"] = cfg
                best_fold_result = sorted(all_results, key=lambda x: x["f1"], reverse=True)[0]
                overall_best["subject"] = best_fold_result["subject"]
                overall_best["path"] = best_fold_result["model_path"]

        if overall_best["path"] is not None:
            best_model_path = os.path.join(SAVE_DIR, "best_model.pth")
            state_dict = torch.load(overall_best["path"], map_location=DEVICE)
            torch.save(state_dict, best_model_path)

            log_print("\n========== BEST MODEL ==========", log_file)
            log_print(f"Best Config : {overall_best['cfg']}", log_file)
            log_print(f"Best Subject: {overall_best['subject']}", log_file)
            log_print(f"Source Path : {overall_best['path']}", log_file)
            log_print(f"Saved Best  : {best_model_path}", log_file)

        log_print("\n========== LOSO END ==========", log_file)
        log_print(f"All logs saved to: {LOG_PATH}", log_file)


if __name__ == "__main__":
    run()
