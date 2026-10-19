# Raspberry Pi ONNX validation

`best_model.pth` 是 LOSO 的 subject 6 fold，不是面向所有被试重新训练的通用最终模型。

在开发机导出并校验：

```bash
python -m pip install onnx onnxruntime
python 0915checkpoints/Raspberry/export_to_onnx.py
python 0915checkpoints/Raspberry/test_subject06.py
```

训练日志对应阈值为 `0.540`。测试脚本默认读取仓库中的
`datasets/subject_06.mat`，并复现训练时逐样本、逐通道、沿时间轴的 z-score。

复制 `best_model.onnx`、`test_subject06.py`、`requirements.txt` 和测试 MAT 文件到树莓派后：

```bash
python3 -m pip install -r requirements.txt
python3 test_subject06.py --data subject_06.mat --threads 1
```

树莓派部署时应输入与训练一致的 30 通道、128 Hz、384 点（3 秒）数据，并保持相同通道顺序和预处理。

## 论文资源测试

延迟测试（预热 50 次，166 条样本逐条推理 20 轮）：

```bash
python3 benchmark_resources.py --mode latency
```

让模型连续推理 60 秒，以便使用外部功率计读取平均 Active Power：

```bash
python3 benchmark_resources.py --mode power --power-duration 60
```

已获得平均功率后，同时测试延迟并自动计算 Energy。例如功率为 5.74 W：

```bash
python3 benchmark_resources.py --mode latency --power-w 5.74
```

一条命令先测延迟、再提供 60 秒功率负载：

```bash
python3 benchmark_resources.py --mode both --power-duration 60 --power-w 5.74
```

默认结果保存在 `resource_results.json`。功率必须来自外部测量设备；脚本不会把软件估算值当作实测功率。
