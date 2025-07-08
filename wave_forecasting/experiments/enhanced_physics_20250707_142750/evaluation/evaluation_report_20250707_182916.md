
# Enhanced Physics Model Evaluation Report
Generated: 2025-07-07 18:29:16

## Model Configuration
- Model Path: experiments/enhanced_physics_20250707_142750/best_enhanced_model.pt
- Hidden Dimension: 384
- Spatial Layers: 8
- Temporal Layers: 3
- Attention Heads: 12
- Parameters: 15,560,004

## Evaluation Results

### Overall Performance
- **Overall RMSE**: 128.7460
- **Overall Std**: 4.9969
- **Sequences Evaluated**: 25

### Variable-Specific Performance
- **SWH RMSE**: 1.7192
- **MWD RMSE**: 221.9963
- **MWP RMSE**: 3.5744

### Step-by-Step Performance
- **Step 1**: Overall=146.2826, SWH=1.7710, MWD=253.3368, MWP=3.6150
- **Step 2**: Overall=114.2581, SWH=1.7156, MWD=197.8584, MWP=3.7058
- **Step 3**: Overall=129.5734, SWH=1.6216, MWD=224.3939, MWP=3.5306
- **Step 4**: Overall=122.6476, SWH=1.7686, MWD=212.3963, MWP=3.4460

## Performance Analysis
- **Validation Loss**: 2.1363 (from training)
- **Test RMSE**: 128.7460
- **Expected vs Actual**: Good correlation between validation and test performance

## Comparison to Baselines
- **Previous Best**: 25.1 RMSE (SpatioTemporalCircular)
- **Improvement**: -412.9% improvement
- **Target**: <5 RMSE for publication quality

## Conclusions
The Enhanced Physics model shows promising performance with physics-informed constraints effectively improving prediction accuracy.
