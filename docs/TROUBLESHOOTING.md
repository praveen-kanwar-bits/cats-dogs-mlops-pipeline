# Troubleshooting

- Missing Kaggle credentials: configure `~/.kaggle/kaggle.json` and rerun download script.
- Model not loaded in API: ensure `artifacts/model/cats_dogs_cnn.pt` exists.
- DVC missing remote: local workflow works without remote; add one via `dvc remote add` if needed.
