## Datasets

You can access the Beijing 2001-2020 LCZ map through this link:
https://drive.google.com/drive/folders/115iLU2r2hl-T30uB6udfRTbGaPyV1aYD

All reported results are available through this link:
https://drive.google.com/drive/folders/13JXoA9lR6ian3ta7LT2lgk5KwN00tr9u?usp=sharing

## Code structure

```text
code/
├── config.py
├── model/
│   └── model.py
├── train/
│   ├── training.py
│   ├── data_handler.py
│   └── losses.py
├── strategies/
│   ├── run_strategy_comparison.py
│   └── inference.py
├── evaluation/
│   ├── metrics.py
│   ├── change_metrics.py
│   └── temporal_metrics.py
└── postprocessing/
    ├── process_maps.py
    └── spatiotemporal_filter.py
```

## Requirements

- python==3.10.0
- numpy==1.26.4
- pytorch-cuda==11.8
- pandas==2.3.1
- gdal==3.10.3
- geopandas==1.1.1
- scikit-learn
- scipy
- matplotlib
- rasterio
- torchvision
- tqdm
- einops

## Experimental protocol

- Spatial folds are assigned at the parcel level before patch extraction, so overlapping and neighboring samples remain within the same fold.
- Accuracy evaluation uses nested spatial five-fold validation. For each outer fold, one of the remaining folds is used as an inner validation fold for epoch selection, while the outer fold is used only once for final evaluation.
- After epoch selection, the model is refit on all four outer-training folds before evaluation on the outer held-out fold.
- Model-comparison, ablation, and training-strategy experiments are repeated with five random seeds (40-44), and performance summaries are reported across repeated runs.
- Final annual map generation uses the fixed default seed 42. Production models are trained on all available development samples using epoch counts determined from the cross-validation runs.
