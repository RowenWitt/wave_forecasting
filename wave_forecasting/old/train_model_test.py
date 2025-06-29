from load_transform_interpolate import load_and_inspect_datasets, process_gebco_for_waves, MultiResolutionInterpolator, MeshDataLoader
from build_dataset import WaveDataset,FixedWaveDataset, create_training_data
from build_mesh import IcosahedralMesh
from model import compute_edge_features, SimpleWaveGNN, train_wave_model_fixed
from train_model import train_wave_model, evaluate_model, visualize_predictions
from loss import WavePhysicsLoss
from torch.utils.data import Dataset, DataLoader
from train_model import visualize_predictions

def test():
    era5_atmo, era5_waves, gebco = load_and_inspect_datasets()
    processed_gebco = process_gebco_for_waves(gebco)
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, processed_gebco)
    mesh = IcosahedralMesh(refinement_level=4)  # Start with level 4 for POC
    mesh_loader = MeshDataLoader(mesh, interpolator)
    # create model
    test_features = mesh_loader.load_mesh_features(time_idx=0)
    n_features = len(test_features['feature_names'])
    edge_index, edge_attr = compute_edge_features(mesh, test_features['node_indices'])
    model = SimpleWaveGNN(input_features=n_features, hidden_dim=64, num_layers=4)

    # Create training data
    train_inputs, train_targets = create_training_data(mesh_loader, num_timesteps=50)
    print(edge_index.shape)
    # Create dataset and dataloader
    train_dataset = FixedWaveDataset(train_inputs, train_targets, edge_index, edge_attr)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # return train_loader
    # Create loss function
    criterion = WavePhysicsLoss(mse_weight=1.0, physics_weight=0.1)
    # # Train the model!
    # trained_model = train_wave_model(model, train_loader, criterion, num_epochs=20, lr=0.001)
    trained_model = train_wave_model_fixed(model, train_loader, criterion, num_epochs=20, lr=0.001)
    # # Evaluate
    test_loss = evaluate_model(trained_model, mesh_loader, criterion, edge_index, edge_attr)
    # Visualize predictions
    visualize_predictions(trained_model, mesh_loader, edge_index, edge_attr, time_idx=0)

test()
