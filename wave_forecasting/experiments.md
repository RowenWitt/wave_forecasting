Model Performance


1. **Spatiotemporal GNN (Original)**: 43.95 overall RMSE
2. **Physics-Informed NN (PINN)**: 11,701 overall RMSE (catastrophic failure)
3. **Spatiotemporal GNN + Circular MWD**: **25.1 overall RMSE** ✨

Circular SpatioTemporal on `02/2022`
- **Results**: 34% improvement across ALL metrics:
  - SWH: 1.328m → 0.798m (40% improvement)
  - MWD: 110.3° → 72.6° (34% improvement) 
  - MWP: 2.765s → 1.942s (30% improvement)
  - **Overall: 38.1 → 25.1 RMSE (34% improvement)**
Circular SpatioTemporal on `12/2021`
📊 COMPARISON RESULTS
========================================
Metric          Original        Circular        Improvement    
------------------------------------------------------------
SWH RMSE (m)    1.320           1.169           11.4           %
MWD RMSE (°)    117.3           39.9            66.0           %
MWP RMSE (s)    2.838           1.139           59.9           %
Overall RMSE    40.494          14.053          65.3           %

Enhanced Physics 1
- Path: experiments/enhanced_physics_20250706_235919/enhanced_physics_model.pt
- 25 Epochs
- 2019 - 2021 Data
- Stats on `12/2021`
📊 EVALUATION RESULTS:
   Overall RMSE: 121.4227
   SWH RMSE: 1.7792
   MWD RMSE: 209.4674
   MWP RMSE: 3.6373

- Training
📈 Enhanced single-step training (25 epochs)
Epoch  1/25: Total=1698.3987, Val=7.7234, Physics=25.7688, Energy=8391.6631, LR=7.99e-05, Time=1764.9s
Epoch  2/25: Total=17.8598, Val=7.3919, Physics=9.6065, Energy=17.7834, LR=7.95e-05, Time=1762.5s
Epoch  3/25: Total=10.0289, Val=9.7042, Physics=6.1880, Energy=2.1380, LR=7.89e-05, Time=1722.3s
Epoch  4/25: Total=10.4430, Val=4.1855, Physics=3.6016, Energy=1.0223, LR=7.80e-05, Time=1716.7s
Epoch  5/25: Total=3.5320, Val=2.4312, Physics=1.6742, Energy=0.1261, LR=7.70e-05, Time=1710.4s
Epoch  6/25: Total=2.4377, Val=2.2747, Physics=1.6612, Energy=0.0404, LR=7.56e-05, Time=1706.1s
Epoch  7/25: Total=2.3437, Val=2.2258, Physics=1.6447, Energy=0.0201, LR=7.41e-05, Time=1708.9s
Epoch  8/25: Total=2.3039, Val=2.1937, Physics=1.6380, Energy=0.0148, LR=7.24e-05, Time=1706.2s
Epoch  9/25: Total=2.2844, Val=2.1981, Physics=1.6345, Energy=0.0131, LR=7.04e-05, Time=1703.7s
Epoch 10/25: Total=2.2690, Val=2.1681, Physics=1.6367, Energy=0.0124, LR=6.83e-05, Time=1699.8s
Epoch 11/25: Total=2.2536, Val=2.1719, Physics=1.6344, Energy=0.0118, LR=6.60e-05, Time=1702.3s
Epoch 12/25: Total=2.2375, Val=2.2432, Physics=1.6367, Energy=0.0110, LR=6.35e-05, Time=1697.9s
Epoch 13/25: Total=2.2324, Val=2.1765, Physics=1.6338, Energy=0.0106, LR=6.09e-05, Time=1697.0s
Epoch 14/25: Total=2.2279, Val=2.1562, Physics=1.6318, Energy=0.0100, LR=5.82e-05, Time=1703.9s
Epoch 15/25: Total=2.2195, Val=2.1468, Physics=1.6317, Energy=0.0100, LR=5.53e-05, Time=1701.7s
Epoch 16/25: Total=2.2092, Val=2.1772, Physics=1.6316, Energy=0.0096, LR=5.24e-05, Time=1739.8s
Epoch 17/25: Total=2.2087, Val=2.1549, Physics=1.6320, Energy=0.0095, LR=4.93e-05, Time=1761.3s
Epoch 18/25: Total=2.2066, Val=2.1395, Physics=1.6300, Energy=0.0093, LR=4.63e-05, Time=1725.5s
Epoch 19/25: Total=2.2033, Val=2.1363, Physics=1.6323, Energy=0.0092, LR=4.31e-05, Time=1749.1s
Epoch 20/25: Total=2.1937, Val=2.1658, Physics=1.6262, Energy=0.0091, LR=4.00e-05, Time=1761.1s
Epoch 21/25: Total=2.1970, Val=2.1613, Physics=1.6268, Energy=0.0092, LR=3.69e-05, Time=1843.2s
Epoch 22/25: Total=2.1916, Val=2.1475, Physics=1.6246, Energy=0.0089, LR=3.37e-05, Time=1893.9s
Epoch 23/25: Total=2.1871, Val=2.1597, Physics=1.6267, Energy=0.0089, LR=3.07e-05, Time=1889.8s
Epoch 24/25: Total=2.1843, Val=2.1608, Physics=1.6249, Energy=0.0088, LR=2.76e-05, Time=1897.2s
Epoch 25/25: Total=2.1847, Val=2.1661, Physics=1.6310, Energy=0.0090, LR=2.47e-05, Time=1904.4s

Enhanced Physics 2
- Path: experiments/enhanced_physics_20250707_142750/best_enhanced_model.pt
- 7 Epochs
- 2019 - 2021 Data
- Stats on 12/2021: 
📊 EVALUATION RESULTS:
   Overall RMSE: 128.7460
   SWH RMSE: 1.7192
   MWD RMSE: 221.9963
   MWP RMSE: 3.5744

- Training 
Epoch  1/25: Total=723.8319, Val=8.5940, Physics=26.7130, Energy=3517.4650, LR=6.00e-05, Time=1874.7s
Epoch  2/25: Total=14.0223, Val=6.3360, Physics=6.3383, Energy=17.1581, LR=2.00e-05, Time=1961.1s
Epoch  3/25: Total=8.3256, Val=8.9426, Physics=4.5164, Energy=0.8393, LR=0.00e+00, Time=1965.0s
Epoch  4/25: Total=10.8041, Val=8.9426, Physics=8.0851, Energy=1.9813, LR=2.00e-05, Time=1983.5s
Epoch  5/25: Total=14.9561, Val=4.5680, Physics=12.1340, Energy=6.5882, LR=6.00e-05, Time=1981.8s
Epoch  6/25: Total=7.1661, Val=4.3971, Physics=3.1373, Energy=0.3079, LR=8.00e-05, Time=1975.7s
Epoch  7/25: Total=3.4180, Val=2.7840, Physics=1.0376, Energy=0.1503, LR=6.00e-05, Time=1977.5s

Enhanced Physics Small 1
- Path:
- Epochs
- 2019 - 2021 Data
- Stats on 12/2021
📊 EVALUATION RESULTS:
   Overall RMSE: 125.1588
   SWH RMSE: 2.1131
   MWD RMSE: 216.6323
   MWP RMSE: 3.6465

- Training
Epoch  1/25: Total=216.7124, Val=8.3992, Physics=147.9109, Energy=17309.0345, LR=7.80e-05, Time=862.7s
Epoch  2/25: Total=48.4918, Val=7.6053, Physics=66.5996, Energy=2685.0462, LR=7.24e-05, Time=847.4s
Epoch  3/25: Total=18.3190, Val=7.5937, Physics=18.6097, Energy=508.1217, LR=6.35e-05, Time=844.5s
Epoch  4/25: Total=20.6751, Val=7.6816, Physics=33.9268, Energy=247.9518, LR=5.24e-05, Time=866.2s


SpatioTemporal Circular - 80 epochs & 5 years of data
Epoch 80/80: Total=0.9319, SWH=0.0722, MWD_circ=0.3384, MWD_ang=0.1479, MWP=0.0521, Time=40.6s
✅ Month 1: Overall RMSE = 13.190 ± 0.665
✅ Month 2: Overall RMSE = 12.696 ± 0.766
✅ Month 3: Overall RMSE = 12.745 ± 0.775

SpatioTemporal Circular - 20 epochs & 3 years of data
📊 Performance Statistics:
   Overall RMSE: 15.829 ± 1.418
   SWH RMSE: 0.800 ± 0.109 m
   MWD RMSE: 44.7 ± 4.2°
   MWP RMSE: 1.948 ± 0.123 s

📈 Performance Ranges:
   Overall RMSE: [12.030, 20.253]
   MWD RMSE: [33.3°, 57.8°]

🌍 Seasonal Performance (Overall RMSE):
   Winter: 16.035 ± 0.255
   Spring: 15.596 ± 0.280
   Summer: 14.929 ± 1.171
   Fall: 16.757 ± 0.283

🏆 Monthly Performance:
   Best Month: 7 (RMSE: 13.659)
   Worst Month: 9 (RMSE: 17.070)
   Seasonal Variation: 3.411

SpatioTemporal Circular - 40 epochs & 3 years of data
📊 Performance Statistics:
   Overall RMSE: 12.405 ± 1.164
   SWH RMSE: 1.019 ± 0.139 m
   MWD RMSE: 34.8 ± 3.5°
   MWP RMSE: 1.376 ± 0.100 s

📈 Performance Ranges:
   Overall RMSE: [9.129, 16.322]
   MWD RMSE: [25.1°, 46.5°]

🌍 Seasonal Performance (Overall RMSE):
   Winter: 11.918 ± 0.490
   Spring: 12.123 ± 0.304
   Summer: 12.370 ± 0.554
   Fall: 13.207 ± 0.477

🏆 Monthly Performance:
   Best Month: 2 (RMSE: 11.437)
   Worst Month: 9 (RMSE: 13.783)
   Seasonal Variation: 2.347

SpatioTemporal Circular - 40 epochs & 5 years of data
Epoch 40/40: Total=1.0993, SWH=0.0965, MWD_circ=0.4061, MWD_ang=0.2006, MWP=0.0657, Time=45.5s
- No annual eval