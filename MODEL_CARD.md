# Model Card: Cats vs Dogs SimpleCNN

## Intended use
Binary cat/dog image classification reference baseline.

## Dataset
Kaggle Cats vs Dogs dataset downloaded by `scripts/download_dataset.py`.

24,998 valid images after corrupt-file filtering: 19,998 train, 2,500 validation, and 2,500 test.
The split is deterministic, stratified, and versioned by DVC.

## Input format
Single image (RGB/grayscale/RGBA accepted), preprocessed to 224x224 RGB tensor.

## Classes
`cat`, `dog`

## Architecture
SimpleCNN: Conv-BN-ReLU-MaxPool x3 -> AdaptiveAvgPool -> Dropout -> Linear(2).

## Training
CrossEntropyLoss + Adam using seed 42 for ten epochs. Training augmentation uses horizontal flip,
rotation, and color jitter; validation/test transforms are deterministic.

## Evaluation
Accuracy, precision, recall, F1, confusion matrix.

Held-out test results: accuracy 0.7548, precision 0.7613, recall 0.7424, F1 0.7517. Confusion
matrix: `[[959, 291], [322, 928]]` for class order `[cat, dog]`.

Promoted checkpoint SHA-256:
`9697741dfc20759f7cf04f829b3abf6c3b1a50bc93bfca3ae45891890fa191b7`.

## Limitations
Baseline model with moderate accuracy; not safety-critical. It is not a pet identity, breed,
medical, or adoptability model. Do not use low-confidence or out-of-distribution predictions as an
automatic adoption decision.

## Risks
Class/domain shift, spurious background correlations, poor calibration, and unknown behavior for
non-cat/dog images.

## Retraining
Run `dvc repro` after data/code/params changes.
