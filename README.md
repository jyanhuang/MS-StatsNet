# MS-StatsNet: Driver Drowsiness EEG Cross-Subject Classification

A PyTorch implementation of a driver drowsiness EEG detection model evaluated using Leave-One-Subject-Out (LOSO) cross-subject validation.

## Project Structure

```text
.
├── EdgeDployment/       # Raspberry Pi deployment files
├── main_imp.py          # Training, validation, and evaluation
├── net_imp.py           # Neural network architecture
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

The dataset is not included in this repository. To run the code, place `dataset.mat` in the project root:

```text
.
├── dataset.mat
├── main_imp.py
└── net_imp.py
```

## Requirements

The following environment is recommended:

- Python 3.9 or later
- PyTorch 2.0 or later
- An NVIDIA GPU with CUDA support (optional)

Install the required packages with:

```bash
pip install -r requirements.txt
```

PyTorch installation may depend on your operating system and CUDA version. For GPU support, install the appropriate PyTorch build from the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Dataset Format

The training script reads the following file from the project root:

```text
dataset.mat
```

The MATLAB file must contain these fields:

- `EEGsample`: EEG samples
- `substate`: class labels
- `subindex`: subject identifiers

The subject identifier field may alternatively be named:

- `subIndex`
- `subject_index`

The expected data shapes are:

```text
EEGsample: (N, 30, 384)
substate:  (N,) or (1, N)
subindex:  (N,) or (1, N)
```

where:

- `N` is the number of samples.
- `30` is the number of EEG channels.
- `384` is the number of time points per sample.
- `substate` contains binary labels `0` and `1`.

The script automatically expands the input data to:

```text
(N, 1, 30, 384)
```

Each EEG channel is standardized independently along the temporal dimension.

## Usage

Run the experiment from the project root:

```bash
python main_imp.py
```

The script automatically uses a CUDA GPU when available and otherwise falls back to the CPU:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## Experimental Protocol

The code uses Leave-One-Subject-Out cross-subject evaluation:

1. One subject is selected as the test set.
2. The remaining subjects are used for training and validation.
3. One subject from the training set is selected as the validation set.
4. The validation set is used for model selection and classification-threshold selection.
5. The experiment is repeated for the first 11 sorted subject identifiers.

The default training configuration is:

```text
Epochs:                  40
Batch size:              32
Early-stopping patience: 8
Dropout:                 0.35
Random seed:             42
```

These settings can be changed in `main_imp.py`.

## Output

Experiment outputs are saved to:

```text
checkpoints/
```

The output directory contains the experiment log, fold-specific model weights, and the selected best model:

```text
checkpoints/
├── loso_results.txt
├── cross_subject_dropout035_subject_1.pth
├── cross_subject_dropout035_subject_2.pth
├── ...
└── best_model.pth
```

The following evaluation statistics are reported:

- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- MFLOPs
- Inference latency per sample
- Number of model parameters

The final log reports the mean and standard deviation of each metric across the LOSO folds.

## Reproducibility

The random seeds for NumPy and PyTorch are set to `42`:

```python
torch.manual_seed(42)
np.random.seed(42)
```

Exact results may still vary depending on the PyTorch version, CUDA version, GPU hardware, and nondeterministic CUDA operations.

## Data Availability

The dataset is not included because it may be subject to licensing, privacy, or redistribution restrictions. It may also exceed GitHub's standard file-size limit.

Users should obtain the dataset through the appropriate authorized source and convert it to the format described above.

## Notes

- Do not commit `dataset.mat` unless redistribution is permitted.
- Model checkpoint files are excluded from Git by default.
- CPU execution is supported, but full LOSO training may take considerably longer.
- Inference latency may vary across different devices.
- Before running the code, remove the unused `attention` imports from `net_imp.py` if `attention.py` is not included.

## Requirements File

The suggested contents of `requirements.txt` are:

```text
numpy>=1.23,<3.0
scipy>=1.9,<2.0
torch>=2.0,<3.0
```

## License

Add an appropriate open-source license before distributing or publishing the project.
