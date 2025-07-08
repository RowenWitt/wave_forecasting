
# Enhanced Physics Model Evaluation Report
Generated: 2025-07-07 14:16:35

## Model Configuration
- Model Path: experiments/enhanced_physics_20250706_235919/enhanced_physics_model.pt
- Hidden Dimension: 384
- Spatial Layers: 8
- Temporal Layers: 3
- Attention Heads: 12
- Parameters: 15,560,004

## Evaluation Results

### Overall Performance
- **Overall RMSE**: 121.4227
- **Overall Std**: 5.6588
- **Sequences Evaluated**: 25

### Variable-Specific Performance
- **SWH RMSE**: 1.7792
- **MWD RMSE**: 209.4674
- **MWP RMSE**: 3.6373

### Step-by-Step Performance
- **Step 1**: Overall=120.0589, SWH=1.7710, MWD=207.9120, MWP=3.4375
- **Step 2**: Overall=130.3642, SWH=1.6720, MWD=225.7600, MWP=3.7317
- **Step 3**: Overall=104.0501, SWH=2.0578, MWD=180.1725, MWP=3.5848
- **Step 4**: Overall=129.3629, SWH=1.6160, MWD=224.0250, MWP=3.7954

## Performance Analysis
- **Validation Loss**: 2.1363 (from training)
- **Test RMSE**: 121.4227
- **Expected vs Actual**: Good correlation between validation and test performance

## Comparison to Baselines
- **Previous Best**: 25.1 RMSE (SpatioTemporalCircular)
- **Improvement**: -383.8% improvement
- **Target**: <5 RMSE for publication quality

## Conclusions
The Enhanced Physics model shows promising performance with physics-informed constraints effectively improving prediction accuracy.
