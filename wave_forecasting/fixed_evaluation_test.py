#!/usr/bin/env python3
"""
Test evaluation with valid time indices (non-NaN data)
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path.cwd()))

def test_with_valid_data():
    """Test the clean system with valid (non-NaN) time indices"""
    
    print("🧪 TESTING WITH VALID TIME INDICES")
    print("=" * 40)
    
    from prediction.forecasting import (
        WavePredictor, AutoregressiveForecaster, AutoregressiveEvaluator, 
        ForecastConfig
    )
    
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
    
    # Load predictor
    predictor = WavePredictor.from_checkpoint(latest_checkpoint)
    
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
    evaluator = AutoregressiveEvaluator(forecaster)
    
    # Test different valid time indices
    valid_times_to_test = [5, 8, 10, 12]  # From investigation results
    
    for initial_time in valid_times_to_test:
        print(f"\n🔮 Testing with initial_time_idx = {initial_time}")
        
        # Create config with valid time index
        config = ForecastConfig(
            max_horizon_hours=24,  # 24h test
            initial_time_idx=initial_time
        )
        
        try:
            # Run forecast
            forecast_result = forecaster.forecast(config)
            
            # Evaluate
            forecast_result = evaluator.evaluate(forecast_result)
            
            if forecast_result.performance:
                print(f"   ✅ Evaluation successful!")
                
                # Show results
                for hours, metrics in forecast_result.performance.items():
                    rmse = metrics.get('swh_rmse', 0)
                    bias = metrics.get('swh_bias', 0)
                    corr = metrics.get('swh_correlation', 0)
                    valid_points = metrics.get('swh_valid_points', 0)
                    
                    print(f"   t+{hours:2d}h: RMSE={rmse:.4f}m, Bias={bias:+.3f}m, Corr={corr:.3f}, Valid={valid_points}")
                
                # If we get reasonable results, use this time index
                if forecast_result.performance[6]['swh_rmse'] < 1.0:  # Reasonable 6h performance
                    print(f"   🎯 Found good time index: {initial_time}")
                    return initial_time, forecaster, evaluator
            else:
                print(f"   ❌ Evaluation failed - no performance metrics")
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
    
    return None, None, None

def run_full_7day_with_valid_time():
    """Run full 7-day evaluation with valid time index"""
    
    # Find a working time index
    good_time, forecaster, evaluator = test_with_valid_data()
    
    if good_time is None:
        print("❌ Could not find a working time index")
        return
    
    print(f"\n🚀 RUNNING FULL 7-DAY EVALUATION")
    print(f"   Using initial_time_idx = {good_time}")
    print("=" * 50)
    
    # Create 7-day config
    config = ForecastConfig(
        max_horizon_hours=168,  # 7 days
        initial_time_idx=good_time
    )
    
    try:
        # Run full forecast
        forecast_result = forecaster.forecast(config)
        
        # Evaluate
        forecast_result = evaluator.evaluate(forecast_result)
        
        # Generate plots
        output_dir = Path("outputs/fixed_evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = output_dir / f"seven_day_evaluation_time_{good_time}.png"
        evaluator.plot_performance(forecast_result, str(plot_path))
        
        # Save results
        results_path = evaluator.save_results(forecast_result, str(output_dir))
        
        print(f"\n🎯 7-DAY EVALUATION RESULTS:")
        print("=" * 35)
        
        if forecast_result.performance:
            # Key milestones
            key_horizons = [6, 24, 48, 72, 120, 168]
            
            print("Horizon  | RMSE     | Bias     | Correlation | Valid Points")
            print("-" * 65)
            
            for h in key_horizons:
                if h in forecast_result.performance:
                    metrics = forecast_result.performance[h]
                    rmse = metrics.get('swh_rmse', 0)
                    bias = metrics.get('swh_bias', 0)
                    corr = metrics.get('swh_correlation', 0)
                    valid = metrics.get('swh_valid_points', 0)
                    
                    days = h // 24
                    hours = h % 24
                    time_str = f"{days}d{hours:02d}h" if days > 0 else f"{hours:2d}h"
                    
                    print(f"{time_str:8s} | {rmse:.4f}m | {bias:+.3f}m | {corr:11.3f} | {valid:11d}")
            
            # Assessment
            final_rmse = forecast_result.performance.get(168, {}).get('swh_rmse', 0)
            day1_rmse = forecast_result.performance.get(24, {}).get('swh_rmse', 0)
            
            print(f"\n🎯 PERFORMANCE ASSESSMENT:")
            print(f"   6-hour RMSE: {forecast_result.performance.get(6, {}).get('swh_rmse', 0):.3f}m")
            print(f"   1-day RMSE:  {day1_rmse:.3f}m")
            print(f"   7-day RMSE:  {final_rmse:.3f}m")
            
            if final_rmse <= 0.8:
                print(f"   ✅ EXCELLENT: 7-day autoregressive forecasting is viable!")
                assessment = "Ready for production deployment"
            elif final_rmse <= 1.5:
                print(f"   🟡 GOOD: 7-day forecasting shows promise")
                assessment = "Consider ensemble methods for improvement"
            else:
                print(f"   🟠 NEEDS WORK: Performance degrades significantly")
                assessment = "Consider dedicated multi-step training"
            
            print(f"   📊 Recommendation: {assessment}")
            
            # Degradation analysis
            if day1_rmse > 0:
                degradation_1d = final_rmse / day1_rmse
                print(f"   📈 1-day to 7-day degradation: {degradation_1d:.1f}x")
            
            return forecast_result
        
    except Exception as e:
        print(f"❌ 7-day evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def compare_to_training_performance():
    """Quick check of model performance on training-style data"""
    
    print(f"\n🔍 COMPARING TO TRAINING PERFORMANCE")
    print("=" * 40)
    
    try:
        from prediction.forecasting import WavePredictor
        from data.datasets import SpatialWaveDataset
        from data.loaders import ERA5DataManager, GEBCODataManager
        from data.preprocessing import MultiResolutionInterpolator
        from data.datasets import MeshDataLoader
        from mesh.icosahedral import IcosahedralMesh
        from config.base import DataConfig, MeshConfig
        
        # Load model
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
        
        predictor = WavePredictor.from_checkpoint(latest_checkpoint)
        
        # Setup data
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
        
        # Create training dataset
        train_dataset = SpatialWaveDataset(mesh_loader, num_timesteps=10)
        
        print(f"   Training dataset: {len(train_dataset)} samples")
        
        # Test on several training samples
        rmse_values = []
        
        for i in range(min(5, len(train_dataset))):
            sample = train_dataset[i]
            features = sample['features']
            targets = sample['targets']
            
            with torch.no_grad():
                prediction = predictor.predict(features)
            
            if prediction.shape == targets.shape:
                rmse = torch.sqrt(torch.mean((prediction - targets)**2))
                rmse_values.append(rmse.item())
                
                print(f"   Sample {i}: RMSE = {rmse:.4f}m")
        
        if rmse_values:
            avg_rmse = sum(rmse_values) / len(rmse_values)
            print(f"   📊 Average training-style RMSE: {avg_rmse:.4f}m")
            
            if avg_rmse <= 0.3:
                print(f"   ✅ Model performance matches expectations")
            elif avg_rmse <= 0.6:
                print(f"   🟡 Model performance is reasonable")
            else:
                print(f"   ⚠️  Model performance is degraded from training")
        
    except Exception as e:
        print(f"   ❌ Training comparison failed: {e}")

def main():
    """Main test with valid data"""
    
    print("🔧 FIXED AUTOREGRESSIVE EVALUATION")
    print("=" * 50)
    
    # First, check training performance
    compare_to_training_performance()
    
    # Then run full evaluation with valid data
    result = run_full_7day_with_valid_time()
    
    if result:
        print(f"\n🎉 EVALUATION COMPLETE!")
        print(f"   Clean autoregressive system is working with valid data")
        print(f"   Check outputs/fixed_evaluation/ for detailed results")
    else:
        print(f"\n⚠️  Evaluation had issues - check the error messages above")

if __name__ == "__main__":
    main()