# Model Card: Cats vs Dogs SimpleCNN

## Intended use
Binary cat/dog image classification reference baseline.

## Dataset
Kaggle Cats vs Dogs dataset downloaded by `scripts/download_dataset.py`.

## Input format
Single image (RGB/grayscale/RGBA accepted), preprocessed to 224x224 RGB tensor.

## Classes
`cat`, `dog`

## Architecture
SimpleCNN: Conv-BN-ReLU-MaxPool x3 -> AdaptiveAvgPool -> Dropout -> Linear(2).

## Training
CrossEntropyLoss + Adam using deterministic seed 42.

## Evaluation
Accuracy, precision, recall, F1, confusion matrix.

## Limitations
Baseline model; not safety-critical; metrics depend on dataset quality and drift.

## Risks
Class/domain shift, poor calibration on out-of-distribution samples.

## Retraining
Run `dvc repro` after data/code/params changes.
