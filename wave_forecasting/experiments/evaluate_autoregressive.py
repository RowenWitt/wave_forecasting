import argparse
from prediction.autoregressive import AutoregressiveWavePredictor
from prediction.evaluators import AutoregressiveEvaluator
from prediction.utils import list_available_models, setup_prediction_environment
import xarray as xr

def main():
    """Main autoregressive evaluation script"""
    
    parser = argparse.ArgumentParser(description='Evaluate autoregressive wave forecasting')
    parser.add_argument('--experiment-id', type=str, help='Experiment ID to load')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--test-type', type=str, choices=['quick', 'comprehensive'], 
                       default='quick', help='Type of test to run')
    parser.add_argument('--output-dir', type=str, default='outputs/predictions', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 AUTOREGRESSIVE WAVE FORECASTING EVALUATION")
    print("=" * 60)
    
    # Setup prediction environment
    if args.experiment_id or args.checkpoint:
        print("📦 Loading specified model...")
        pred_env = setup_prediction_environment(
            experiment_id=args.experiment_id,
            checkpoint_path=args.checkpoint,
            log_dir=args.log_dir
        )
    else:
        print("🔍 Auto-detecting models...")
        pred_env = setup_prediction_environment(log_dir=args.log_dir)
    
    # Create predictor and evaluator
    predictor = AutoregressiveWavePredictor(
        spatial_model=pred_env.model,
        mesh_loader=pred_env.mesh_loader,
        edge_index=pred_env.edge_index,
        edge_attr=pred_env.edge_attr,
        feature_names=pred_env.feature_names
    )
    
    evaluator = AutoregressiveEvaluator(
        mesh_loader=pred_env.mesh_loader,
        feature_names=pred_env.feature_names
    )
    
    # Define test scenarios
    if args.test_type == 'quick':
        test_times = [10, 20, 30]
        forecast_steps = [1, 2, 4, 8]  # 6h, 12h, 24h, 48h
        print("🧪 Running quick test (3 forecasts, up to 48h)")
    else:
        test_times = list(range(20, 100, 10))
        forecast_steps = [1, 2, 3, 4, 6, 8, 12]  # Up to 72h
        print("🔬 Running comprehensive test (8 forecasts, up to 72h)")
    
    # Run evaluation
    results = evaluator.evaluate_multiple_forecasts(
        predictor=predictor,
        test_time_indices=test_times,
        forecast_steps=forecast_steps
    )
    
    # Save results
    import json
    results_file = output_dir / f"autoregressive_{args.test_type}_{pred_env.experiment_id}.json"
    
    # Convert numpy types for JSON serialization
    json_results = {}
    for horizon, metrics in results.items():
        json_results[str(horizon)] = {k: float(v) if isinstance(v, (int, float)) else v 
                                     for k, v in metrics.items()}
    
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_id': pred_env.experiment_id,
            'test_type': args.test_type,
            'results': json_results
        }, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    
    # Generate plots
    from prediction.autoregressive import plot_autoregressive_performance
    plot_file = output_dir / f"autoregressive_{args.test_type}_{pred_env.experiment_id}.png"
    plot_autoregressive_performance(results, save_path=str(plot_file))
    print(f"📊 Plots saved to: {plot_file}")
    
    # Print key results
    print(f"\n🎯 KEY RESULTS:")
    for horizon in [24, 48, 72]:
        if horizon in results:
            rmse = results[horizon].get('swh_rmse_mean', 0)
            print(f"   {horizon}h forecast: {rmse:.4f}m RMSE")

if __name__ == "__main__":
    main()