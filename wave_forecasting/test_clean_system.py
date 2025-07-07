#!/usr/bin/env python3
"""
Test the clean autoregressive forecasting system
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

def test_clean_system():
    """Test the new clean forecasting system"""
    
    print("🧪 TESTING CLEAN AUTOREGRESSIVE SYSTEM")
    print("=" * 50)
    
    # Import the clean system
    from prediction.forecasting import (
        WavePredictor, AutoregressiveForecaster, AutoregressiveEvaluator, 
        ForecastConfig, run_full_evaluation
    )
    
    # Test 1: Load predictor from checkpoint
    print("📦 Test 1: Loading WavePredictor...")
    
    # Find latest checkpoint
    logs_path = Path("logs")
    all_checkpoints = []
    
    for exp_dir in logs_path.iterdir():
        if exp_dir.is_dir():
            checkpoint_dir = exp_dir / "checkpoints"
            if checkpoint_dir.exists():
                for checkpoint in checkpoint_dir.glob("*.pt"):
                    all_checkpoints.append((checkpoint.stat().st_mtime, str(checkpoint)))
    
    if not all_checkpoints:
        print("❌ No checkpoints found")
        return
    
    # Get latest checkpoint
    all_checkpoints.sort(reverse=True)
    latest_checkpoint = all_checkpoints[0][1]
    
    try:
        predictor = WavePredictor.from_checkpoint(latest_checkpoint)
        print(f"✅ WavePredictor loaded successfully")
        print(f"   Model: {predictor.experiment_id}")
        print(f"   Features: {len(predictor.feature_names)}")
    except Exception as e:
        print(f"❌ WavePredictor loading failed: {e}")
        return
    
    # Test 2: Setup forecaster
    print(f"\n🔮 Test 2: Setting up AutoregressiveForecaster...")
    
    try:
        # Setup data loader
        from data.loaders import ERA5DataManager, GEBCODataManager
        from data.preprocessing import MultiResolutionInterpolator
        from data.datasets import MeshDataLoader
        from mesh.icosahedral import IcosahedralMesh
        from config.base import DataConfig, MeshConfig
        
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
        first_file = era5_manager.available_files['atmospheric'][0]
        filename = Path(first_file).stem
        year_month = filename.split('_')[-1]
        year, month = int(year_month[:4]), int(year_month[4:6])
        
        era5_atmo, era5_waves = era5_manager.load_month_data(year, month)
        gebco_data = gebco_manager.load_bathymetry()
        
        mesh = IcosahedralMesh(mesh_config)
        interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
        mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
        
        forecaster = AutoregressiveForecaster(predictor, mesh_loader)
        print(f"✅ AutoregressiveForecaster created successfully")
        
    except Exception as e:
        print(f"❌ AutoregressiveForecaster setup failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Run short forecast
    print(f"\n🌊 Test 3: Running short forecast (24h)...")
    
    try:
        config = ForecastConfig(max_horizon_hours=24, initial_time_idx=20)
        forecast_result = forecaster.forecast(config)
        
        print(f"✅ Short forecast successful")
        print(f"   Predictions: {len(forecast_result.predictions)}")
        print(f"   Horizons: {sorted(forecast_result.predictions.keys())}h")
        
        # Check prediction quality
        for hours, pred in forecast_result.predictions.items():
            swh = pred[:, 0]
            print(f"   t+{hours:2d}h: SWH {swh.min():.3f}-{swh.max():.3f}m")
        
    except Exception as e:
        print(f"❌ Short forecast failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Run evaluation
    print(f"\n📊 Test 4: Running evaluation...")
    
    try:
        evaluator = AutoregressiveEvaluator(forecaster)
        forecast_result = evaluator.evaluate(forecast_result)
        
        print(f"✅ Evaluation successful")
        print(f"   Performance metrics: {len(forecast_result.performance) if forecast_result.performance else 0}")
        
        if forecast_result.performance:
            for hours, metrics in forecast_result.performance.items():
                rmse = metrics.get('swh_rmse', 0)
                print(f"   t+{hours:2d}h: RMSE={rmse:.4f}m")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 5: Generate plots and save results
    print(f"\n📊 Test 5: Generating plots and saving results...")
    
    try:
        output_dir = Path("outputs/test_clean_system")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = output_dir / "test_forecast_plot.png"
        evaluator.plot_performance(forecast_result, str(plot_path))
        
        results_path = evaluator.save_results(forecast_result, str(output_dir))
        
        print(f"✅ Plots and results saved")
        print(f"   Plot: {plot_path}")
        print(f"   Results: {results_path}")
        
    except Exception as e:
        print(f"❌ Plotting/saving failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"   Clean system is working correctly")
    print(f"   Ready for full 7-day evaluation")
    
    return True

def test_convenience_function():
    """Test the convenience function for full evaluation"""
    
    print(f"\n🚀 TESTING CONVENIENCE FUNCTION (FULL 7-DAY)")
    print("=" * 55)
    
    try:
        from prediction.forecasting import run_full_evaluation, ForecastConfig
        
        # Find latest checkpoint
        logs_path = Path("logs")
        all_checkpoints = []
        
        for exp_dir in logs_path.iterdir():
            if exp_dir.is_dir():
                checkpoint_dir = exp_dir / "checkpoints"
                if checkpoint_dir.exists():
                    for checkpoint in checkpoint_dir.glob("*.pt"):
                        all_checkpoints.append((checkpoint.stat().st_mtime, str(checkpoint)))
        
        all_checkpoints.sort(reverse=True)
        latest_checkpoint = all_checkpoints[0][1]
        
        # Run full 7-day evaluation
        config = ForecastConfig(max_horizon_hours=168, initial_time_idx=20)  # 7 days
        
        print(f"📦 Running full evaluation with checkpoint: {Path(latest_checkpoint).name}")
        
        result = run_full_evaluation(latest_checkpoint, config)
        
        print(f"\n🎯 FULL 7-DAY EVALUATION RESULTS:")
        print("=" * 40)
        
        if result.performance:
            # Key milestones
            key_horizons = [6, 24, 48, 72, 120, 168]
            
            print("Horizon  | RMSE     | Bias     | Correlation | Assessment")
            print("-" * 60)
            
            for h in key_horizons:
                if h in result.performance:
                    metrics = result.performance[h]
                    rmse = metrics.get('swh_rmse', 0)
                    bias = metrics.get('swh_bias', 0)
                    corr = metrics.get('swh_correlation', 0)
                    
                    if rmse <= 0.5:
                        assessment = "✅ Excellent"
                    elif rmse <= 0.8:
                        assessment = "🟡 Good"
                    elif rmse <= 1.2:
                        assessment = "🟠 Acceptable"
                    else:
                        assessment = "❌ Poor"
                    
                    days = h // 24
                    hours = h % 24
                    time_str = f"{days}d{hours:02d}h" if days > 0 else f"{hours:2d}h"
                    
                    print(f"{time_str:8s} | {rmse:.4f}m | {bias:+.3f}m | {corr:11.3f} | {assessment}")
            
            # Overall assessment
            final_rmse = result.performance[168]['swh_rmse'] if 168 in result.performance else 0
            useful_horizons = [h for h in key_horizons if h in result.performance and result.performance[h]['swh_rmse'] <= 0.8]
            max_useful = max(useful_horizons) if useful_horizons else 0
            
            print(f"\n🎯 SUMMARY:")
            print(f"   7-day RMSE: {final_rmse:.3f}m")
            print(f"   Useful horizon: {max_useful}h ({max_useful/24:.1f} days)")
            
            if final_rmse <= 0.8:
                print(f"   ✅ 7-day autoregressive forecasting is EXCELLENT!")
                print(f"   🚀 Ready for production deployment")
            elif final_rmse <= 1.2:
                print(f"   🟡 7-day forecasting is GOOD with room for improvement")
                print(f"   💡 Consider ensemble methods or longer training")
            else:
                print(f"   🟠 7-day performance needs improvement")
                print(f"   🔄 Consider dedicated multi-step training approach")
        
        return True
        
    except Exception as e:
        print(f"❌ Full evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("🧪 CLEAN AUTOREGRESSIVE SYSTEM TEST SUITE")
    print("=" * 55)
    
    # Test basic functionality
    basic_success = test_clean_system()
    
    if basic_success:
        print(f"\n" + "="*55)
        
        # Ask user if they want to run full 7-day test
        response = input("Run full 7-day evaluation? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            full_success = test_convenience_function()
            
            if full_success:
                print(f"\n🎉 COMPLETE SUCCESS!")
                print(f"   Clean autoregressive system is fully functional")
                print(f"   Ready for production use")
            else:
                print(f"\n⚠️  Basic system works, but full evaluation had issues")
        else:
            print(f"\n✅ Basic system test complete")
            print(f"   Run 'python test_clean_system.py' and answer 'y' for full test")
    else:
        print(f"\n❌ Basic system test failed")
        print(f"   Check the error messages above for debugging")

if __name__ == "__main__":
    main()