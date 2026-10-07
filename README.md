## Datasets

You can access the Beijing 2001-2020 LCZ map through this link:https://drive.google.com/drive/folders/115iLU2r2hl-T30uB6udfRTbGaPyV1aYD
And all the results can be accessed through this linkhttps://drive.google.com/drive/folders/13JXoA9lR6ian3ta7LT2lgk5KwN00tr9u?usp=sharing

## The Code

```textcode/
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
README.md
```

## Requirements

- python==3.10.0
- numpy==1.26.4
- pytorch-cuda==11.8
- pandas==2.3.1
- gdal==3.10.3
- geopandas==1.1.1

## Experimental protocol

- Landsat imagery is handled at its native 30 m spatial resolution.
- Patch extraction uses an 8 x 8-pixel sliding window with 50% overlap for all years from 2001 to 2020.
- Spatial folds are assigned at the parcel level before patch extraction, so overlapping patches remain within the same fold.
- Model-comparison, ablation, and training-strategy experiments are repeated with five random seeds (40-44).
- The complete annual map series used for map-based temporal analysis is generated with the fixed default seed 42.
- The supervised contrastive loss uses temperature = 0.1 and base_temperature = 0.1.

