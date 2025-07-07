# Experiment Report: quick_production_test_20250629_165919

## Overview
- **Experiment Name**: quick_production_test
- **Timestamp**: 20250629_165919
- **Status**: completed
- **Duration**: 0:00:07.852164

## Configuration
### Model Architecture
- **Hidden Dimension**: 128
- **Spatial Layers**: 6
- **Temporal Layers**: 4

### Training Parameters
- **Epochs**: 20
- **Batch Size**: 8
- **Learning Rate**: 0.001
- **Weight Decay**: 0.0001

### Data Configuration
- **Sequence Length**: 4
- **Forecast Horizon**: 24h
- **Geographic Bounds**: (10.0, 60.0) × (120.0, 240.0)

## Training Results
- **Best Training Loss**: 0.106944
- **Final Training Loss**: 0.106964
- **Improvement**: 86.2%
- **Total Epochs Completed**: 20

## Final Metrics
- **total**: 0.106964
- **mse**: 0.041322
- **physics**: 0.656425
- **val_total**: 0.105959
- **val_mse**: 0.041500
- **val_physics**: 0.644589
- **val_swh_mae**: 0.203600
- **val_swh_rmse**: 0.274737
- **val_swh_skill**: 0.933174
- **val_swh_corr**: 0.967920
- **val_mwd_mae**: 0.051890
- **val_mwd_rmse**: 0.173215
- **val_mwd_skill**: 0.969912
- **val_mwd_corr**: 0.984841
- **val_mwp_mae**: 0.108601
- **val_mwp_rmse**: 0.152382
- **val_mwp_skill**: 0.976924
- **val_mwp_corr**: 0.989853
- **best_val_loss**: 0.105597

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
- **2025-06-29T16:59:19.210609**: Quick production test with validation
- **2025-06-29T16:59:19.624115**: New best validation loss: 0.335857 at epoch 1
- **2025-06-29T16:59:20.404878**: New best validation loss: 0.213832 at epoch 3
- **2025-06-29T16:59:20.803838**: New best validation loss: 0.155357 at epoch 4
- **2025-06-29T16:59:21.194213**: New best validation loss: 0.138347 at epoch 5
- **2025-06-29T16:59:21.603649**: New best validation loss: 0.133031 at epoch 6
- **2025-06-29T16:59:22.395087**: New best validation loss: 0.124973 at epoch 8
- **2025-06-29T16:59:22.790860**: New best validation loss: 0.120430 at epoch 9
- **2025-06-29T16:59:23.188072**: New best validation loss: 0.116807 at epoch 10
- **2025-06-29T16:59:23.588510**: New best validation loss: 0.113750 at epoch 11
- **2025-06-29T16:59:23.993756**: New best validation loss: 0.112409 at epoch 12
- **2025-06-29T16:59:24.370270**: New best validation loss: 0.110768 at epoch 13
- **2025-06-29T16:59:24.749588**: New best validation loss: 0.109339 at epoch 14
- **2025-06-29T16:59:25.123314**: New best validation loss: 0.108404 at epoch 15
- **2025-06-29T16:59:25.500022**: New best validation loss: 0.107401 at epoch 16
- **2025-06-29T16:59:25.912785**: New best validation loss: 0.106564 at epoch 17
- **2025-06-29T16:59:26.296715**: New best validation loss: 0.106103 at epoch 18
- **2025-06-29T16:59:26.674565**: New best validation loss: 0.105597 at epoch 19

## Reproducibility
- **Git Commit**: `91490dacd81719fa96333ea91764a605df75cfc4`
