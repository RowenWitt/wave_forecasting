from model import clean_features_for_training
import torch
import numpy as np
import matplotlib.pyplot as plt

def train_wave_model(model, train_loader, criterion, num_epochs=20, lr=0.001):
    """Simple training loop"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    
    print(f"🚀 Starting training: {num_epochs} epochs, lr={lr}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        epoch_losses = {'total': [], 'mse': [], 'physics': []}

        edge_index = train_loader.dataset.edge_index
        edge_attr = train_loader.dataset.edge_attr
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            features = batch['features']
            targets = batch['targets']
  
            
            # Model prediction
            predictions = model(features, edge_index, edge_attr)
            
            # Compute loss
            loss_dict = criterion(predictions, targets, features)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            epoch_losses['total'].append(loss_dict['total_loss'].item())
            epoch_losses['mse'].append(loss_dict['mse_loss'].item())
            epoch_losses['physics'].append(loss_dict['physics_loss'].item())
        
        # Print epoch summary
        avg_total = np.mean(epoch_losses['total'])
        avg_mse = np.mean(epoch_losses['mse'])
        avg_physics = np.mean(epoch_losses['physics'])
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Total={avg_total:.4f}, MSE={avg_mse:.4f}, Physics={avg_physics:.4f}")
        
        # Simple learning rate decay
        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"   Reduced LR to {optimizer.param_groups[0]['lr']:.6f}")
    
    print("✅ Training complete!")
    return model


def evaluate_model(model, mesh_loader, criterion, edge_index, edge_attr):
    """Quick evaluation on a few test samples"""
    
    model.eval()
    
    print("🔍 Evaluating trained model...")
    
    # Test on a few timesteps
    test_losses = []
    predictions_list = []
    targets_list = []
    
    with torch.no_grad():
        for t in range(5):  # Test on 5 timesteps
            # Load test data
            input_data = mesh_loader.load_mesh_features(time_idx=t)
            features = clean_features_for_training(
                torch.tensor(input_data['features'], dtype=torch.float32)
            )
            
            target_data = mesh_loader.load_mesh_features(time_idx=t+1)
            target_features_raw = torch.tensor(target_data['features'], dtype=torch.float32)
            
            # Get wave targets
            feature_names = input_data['feature_names']
            wave_indices = [i for i, name in enumerate(feature_names) 
                           if name in ['swh', 'mwd', 'mwp']]
            targets = clean_features_for_training(target_features_raw[:, wave_indices])
            
            # Predict
            predictions = model(features, edge_index, edge_attr)
            
            # Compute loss
            loss_dict = criterion(predictions.unsqueeze(0), targets.unsqueeze(0))
            test_losses.append(loss_dict['total_loss'].item())
            
            predictions_list.append(predictions)
            targets_list.append(targets)
    
    avg_test_loss = np.mean(test_losses)
    print(f"📊 Average test loss: {avg_test_loss:.4f}")
    
    # Show some predictions vs targets
    print("\nSample predictions vs targets (first 5 nodes):")
    pred_sample = predictions_list[0][:5]  # First 5 nodes from first timestep
    target_sample = targets_list[0][:5]
    
    print("      SWH      MWD      MWP")
    print("Pred:", pred_sample.numpy())
    print("True:", target_sample.numpy())
    
    return avg_test_loss

def visualize_predictions(model, mesh_loader, edge_index, edge_attr, time_idx=0):
    """Visualize predictions on the mesh"""
    
    model.eval()
    
    # Get data
    input_data = mesh_loader.load_mesh_features(time_idx=time_idx)
    features = clean_features_for_training(
        torch.tensor(input_data['features'], dtype=torch.float32)
    )
    coordinates = input_data['coordinates']
    
    # Predict
    with torch.no_grad():
        predictions = model(features, edge_index, edge_attr)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    wave_names = ['Significant Wave Height', 'Mean Wave Direction', 'Mean Wave Period']
    
    for i in range(3):
        scatter = axes[i].scatter(
            coordinates[:, 1],  # longitude
            coordinates[:, 0],  # latitude
            c=predictions[:, i].numpy(),
            s=20, alpha=0.7, cmap='viridis'
        )
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        axes[i].set_title(f'Predicted {wave_names[i]}')
        axes[i].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[i])
    
    plt.tight_layout()
    plt.show()

