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