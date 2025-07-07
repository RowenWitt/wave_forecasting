# Experiment Report: production_spatial_multiyear_highres_20250629_171332

## Overview
- **Experiment Name**: production_spatial_multiyear_highres
- **Timestamp**: 20250629_171332
- **Status**: completed
- **Duration**: 3:24:34.414803

## Configuration
### Model Architecture
- **Hidden Dimension**: 256
- **Spatial Layers**: 12
- **Temporal Layers**: 4

### Training Parameters
- **Epochs**: 100
- **Batch Size**: 4
- **Learning Rate**: 0.0005
- **Weight Decay**: 0.0001

### Data Configuration
- **Sequence Length**: 4
- **Forecast Horizon**: 24h
- **Geographic Bounds**: (10.0, 60.0) × (120.0, 240.0)

## Training Results
- **Best Training Loss**: 0.111568
- **Final Training Loss**: 0.111685
- **Improvement**: 28.5%
- **Total Epochs Completed**: 58

## Final Metrics
- **total**: 0.111685
- **mse**: 0.021657
- **physics**: 0.600187
- **val_total**: 0.117554
- **val_mse**: 0.028597
- **val_physics**: 0.593047
- **val_swh_mae**: 0.166456
- **val_swh_rmse**: 0.211860
- **val_swh_skill**: 0.964401
- **val_swh_corr**: 0.985004
- **val_mwd_mae**: 0.033657
- **val_mwd_rmse**: 0.139074
- **val_mwd_skill**: 0.980721
- **val_mwd_corr**: 0.990330
- **val_mwp_mae**: 0.108018
- **val_mwp_rmse**: 0.146130
- **val_mwp_skill**: 0.982999
- **val_mwp_corr**: 0.994700
- **best_val_loss**: 0.117254

## System Information
- **Platform**: macOS-15.5-arm64-arm-64bit
- **Python Version**: 3.12.10
- **PyTorch Version**: 2.7.1
- **CUDA Available**: False
- **Memory**: 128.0 GB

## Files Generated
- **Config**: `config.json`
- **Training History**: `results.json`
- **Summary Plot**: `plots/training_summary.png`
- **Checkpoints**: `checkpoints/`

## Notes
- **2025-06-29T17:13:33.160235**: Production experiment with 1170 mesh nodes
- **2025-06-29T17:13:33.160358**: Multi-year training: 2020-2022
- **2025-06-29T17:13:33.160445**: High-resolution mesh: level 5
- **2025-06-29T17:17:01.381240**: New best validation loss: 0.127948 at epoch 1
- **2025-06-29T17:20:31.456445**: New best validation loss: 0.126946 at epoch 2
- **2025-06-29T17:24:05.214045**: New best validation loss: 0.124905 at epoch 3
- **2025-06-29T17:27:39.456663**: New best validation loss: 0.123012 at epoch 4
- **2025-06-29T17:31:07.165553**: New best validation loss: 0.122191 at epoch 5
- **2025-06-29T17:38:05.624398**: New best validation loss: 0.121325 at epoch 7
- **2025-06-29T17:41:32.119285**: New best validation loss: 0.120536 at epoch 8
- **2025-06-29T17:48:37.204479**: New best validation loss: 0.120290 at epoch 10
- **2025-06-29T17:52:12.105786**: New best validation loss: 0.119664 at epoch 11
- **2025-06-29T17:59:27.371430**: New best validation loss: 0.119039 at epoch 13
- **2025-06-29T18:06:30.608703**: New best validation loss: 0.118970 at epoch 15
- **2025-06-29T18:20:34.033368**: New best validation loss: 0.118660 at epoch 19
- **2025-06-29T18:24:04.923167**: New best validation loss: 0.118441 at epoch 20
- **2025-06-29T18:27:37.379079**: New best validation loss: 0.117912 at epoch 21
- **2025-06-29T18:41:46.736260**: New best validation loss: 0.117753 at epoch 25
- **2025-06-29T18:52:20.872334**: New best validation loss: 0.117694 at epoch 28
- **2025-06-29T19:10:15.141939**: New best validation loss: 0.117604 at epoch 33
- **2025-06-29T19:13:47.672994**: New best validation loss: 0.117433 at epoch 34
- **2025-06-29T19:38:35.870016**: New best validation loss: 0.117280 at epoch 41
- **2025-06-29T19:45:31.241819**: New best validation loss: 0.117254 at epoch 43

## Reproducibility
- **Git Commit**: `91490dacd81719fa96333ea91764a605df75cfc4`
