History:
Trained prod model, took 3 hours, is documented, stats looked very promising

spent a ton of time working to setup autoregressive prediction and evaluation.  

RMSE has been >7000m for wave height, now down to 2.x rmse, still an order of magnitude off validation report of .2x.

🎯 FULL 7-DAY EVALUATION RESULTS:
========================================
Horizon  | RMSE     | Bias     | Correlation | Assessment
------------------------------------------------------------
 6h      | 2.6126m | -2.411m |       0.132 | ❌ Poor
1d00h    | 2.9754m | -2.633m |       0.138 | ❌ Poor
2d00h    | 3.0318m | -2.730m |       0.140 | ❌ Poor
3d00h    | 3.1534m | -2.855m |       0.086 | ❌ Poor
5d00h    | 3.3110m | -2.999m |       0.158 | ❌ Poor
7d00h    | 2.6013m | -2.441m |       0.015 | ❌ Poor

Working of ensuring evaluation doesn't have issues

Next steps will likely be train a larger model, and potentially modify model architecture

In modifying model architecture, I risk breaking everything, so will need to be an addition

SPATIAL MODEL PROD 0 Diagnostics

🔍 TESTING MODEL WITH CORRECT DIMENSIONS
=============================================
📁 Found 48 atmospheric files
📁 Found 48 wave files
Loading processed GEBCO data...
Building icosahedral mesh (level 5)...
  Level 1: 42 vertices, 80 faces
  Level 2: 162 vertices, 320 faces
  Level 3: 642 vertices, 1280 faces
  Level 4: 2562 vertices, 5120 faces
  Level 5: 10242 vertices, 20480 faces
✅ Mesh complete: 10242 vertices, 30720 edges
Setting up multi-resolution interpolators...
✅ Interpolators ready!
Mesh data loader ready: 1170 regional nodes
Computing edges for 1170 nodes...
✅ Edge computation complete: 6744 edges
🔧 WavePredictor initialized:
   Model: production_spatial_multiyear_highres_20250629_171332
   Parameters: 5,533,443
   Features: 11
   Wave indices: [3, 4, 5]
📊 Model dimensions:
   Nodes: 1170
   Features: 11
   Edges: 6744
   Node index range: 0 to 1169

🧪 TEST 1: Input Sensitivity (Correct Dimensions)
--------------------------------------------------
   ✅ All predictions successful!
   Input 1 (zeros): Output range 0.252159 to 0.458519
   Input 2 (ones):  Output range 0.252169 to 0.458532
   Input 3 (rand1): Output range 0.252165 to 0.458512
   Input 4 (rand2): Output range 0.252220 to 0.458423
   Difference 1-2: 0.000010
   Difference 1-3: 0.000007
   Difference 3-4: 0.000036
   ✅ Model responds to different inputs
📁 Found 48 atmospheric files
📁 Found 48 wave files
Loading processed GEBCO data...
Building icosahedral mesh (level 5)...
Setting up multi-resolution interpolators...
✅ Interpolators ready!
Mesh data loader ready: 1170 regional nodes
Computing edges for 1170 nodes...
✅ Edge computation complete: 6744 edges
🔧 WavePredictor initialized:
   Model: production_spatial_multiyear_highres_20250629_171332
   Parameters: 5,533,443
   Features: 11
   Wave indices: [3, 4, 5]

🧪 TEST 2: Individual Feature Sensitivity
----------------------------------------
   Baseline prediction range: 0.257646 to 0.525488
   u10            : Change = 0.000021
   v10            : Change = 0.000019
   msl            : Change = 0.000023
   swh            : Change = 0.000017
   mwd            : Change = 0.000034

   Top 5 most sensitive features:
   1. deep_water_mask: 0.000036
   2. mwd            : 0.000034
   3. mwp            : 0.000027
   4. shallow_water_mask: 0.000023
   5. msl            : 0.000023
   ✅ Model is sensitive to feature changes

🧪 TEST 3: Real Data vs Training Data Performance
--------------------------------------------------
📁 Found 48 atmospheric files
📁 Found 48 wave files
Loading processed GEBCO data...
Building icosahedral mesh (level 5)...
Setting up multi-resolution interpolators...
✅ Interpolators ready!
Mesh data loader ready: 1170 regional nodes
   Testing direct mesh loader features...
     Real data prediction: 0.257594 to 0.525229
     Real data shape: torch.Size([1170, 11])
   Testing training dataset format...
Creating spatial training data from 5 timesteps...
✅ Created 4 spatial training samples
     Training format prediction: 0.257575 to 0.525160
     Training format RMSE: 1.047377
     Training targets: -2.254177 to 2.358721
     Real vs Training prediction diff: 0.000005
   ⚠️  High RMSE: Model may not be well-trained

✅ MODEL FUNCTIONALITY CONFIRMED
   Model is working and responding to inputs
   The ~1.0m RMSE issue is likely due to:
   - Model genuinely has ~1.0m performance (not 0.21m)
   - Training reported performance was optimistic
   - Different evaluation methodology
   - Model needs more training or better architecture