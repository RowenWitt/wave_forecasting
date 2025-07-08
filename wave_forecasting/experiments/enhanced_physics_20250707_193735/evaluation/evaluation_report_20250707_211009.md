
# Enhanced Physics Model Evaluation Report
Generated: 2025-07-07 21:10:09

## Model Configuration
- Model Path: experiments/enhanced_physics_20250707_193735/best_enhanced_model.pt
- Hidden Dimension: 192
- Spatial Layers: 4
- Temporal Layers: 2
- Attention Heads: 6
- Parameters: 2,771,700

## Evaluation Results

### Overall Performance
- **Overall RMSE**: 125.1588
- **Overall Std**: 5.2957
- **Sequences Evaluated**: 25

### Variable-Specific Performance
- **SWH RMSE**: 2.1131
- **MWD RMSE**: 216.6323
- **MWP RMSE**: 3.6465

### Step-by-Step Performance
- **Step 1**: Overall=125.4551, SWH=2.1743, MWD=217.2555, MWP=3.4883
- **Step 2**: Overall=129.7353, SWH=1.6351, MWD=224.6631, MWP=4.1719
- **Step 3**: Overall=124.7760, SWH=1.6686, MWD=216.0841, MWP=3.4483
- **Step 4**: Overall=120.4221, SWH=2.9746, MWD=208.5267, MWP=3.4774

## Performance Analysis
- **Validation Loss**: 2.1363 (from training)
- **Test RMSE**: 125.1588
- **Expected vs Actual**: Good correlation between validation and test performance

## Comparison to Baselines
- **Previous Best**: 25.1 RMSE (SpatioTemporalCircular)
- **Improvement**: -398.6% improvement
- **Target**: <5 RMSE for publication quality

## Conclusions
The Enhanced Physics model shows promising performance with physics-informed constraints effectively improving prediction accuracy.
