from torch.utils.data import Dataset, DataLoader
from model import clean_features_for_training
import torch

# Quick fix - don't include edge data in dataset
class FixedWaveDataset(Dataset):
    def __init__(self, inputs, targets, edge_index, edge_attr):
        self.inputs = inputs
        self.targets = targets
        self.edge_index = edge_index  # Store but don't return in __getitem__
        self.edge_attr = edge_attr
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'features': self.inputs[idx],
            'targets': self.targets[idx],
        }

class WaveDataset(Dataset):
    """PyTorch dataset for wave prediction"""
    
    def __init__(self, inputs, targets, edge_index, edge_attr):
        self.inputs = inputs
        self.targets = targets
        self.edge_index = edge_index
        self.edge_attr = edge_attr
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'features': self.inputs[idx],
            'targets': self.targets[idx],
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr
        }


# print(f"✅ Dataset ready: {len(train_dataset)} samples, batch size 8")

def create_training_data(mesh_loader, num_timesteps=50):
    """Create training data: predict wave state at t+1 from all features at t"""
    
    print(f"Creating training data from {num_timesteps} timesteps...")
    
    inputs, targets = [], []
    
    for t in range(num_timesteps - 1):
        try:
            # Input: all features at time t
            input_data = mesh_loader.load_mesh_features(time_idx=t)
            input_features = clean_features_for_training(
                torch.tensor(input_data['features'], dtype=torch.float32)
            )
            
            # Target: wave variables at time t+1
            target_data = mesh_loader.load_mesh_features(time_idx=t+1)
            target_features_raw = torch.tensor(target_data['features'], dtype=torch.float32)
            
            # Extract just wave variables as targets [swh, mwd, mwp]
            feature_names = input_data['feature_names']
            wave_indices = [i for i, name in enumerate(feature_names) 
                           if name in ['swh', 'mwd', 'mwp']]
            
            target_waves = target_features_raw[:, wave_indices]
            target_waves_clean = clean_features_for_training(target_waves)
            
            inputs.append(input_features)
            targets.append(target_waves_clean)
            
        except Exception as e:
            print(f"⚠️  Skipping timestep {t}: {e}")
            continue
    
    print(f"✅ Created {len(inputs)} training samples")
    print(f"   Input shape: {inputs[0].shape}")
    print(f"   Target shape: {targets[0].shape}")
    
    return inputs, targets
