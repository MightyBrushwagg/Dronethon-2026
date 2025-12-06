"""
Training script for the drone transition model.

This script loads collected transition data and trains a neural network
to predict the next state given the current state and action.

Usage:
    python train_transition_model.py [--mode both] [--epochs 100] [--batch_size 256] [--lr 0.001]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class TransitionDataset(Dataset):
    """Dataset for loading transition data."""
    
    def __init__(self, transitions, state_dim=12, action_dim=4, normalize=True):
        """
        Initialize dataset from transition array.
        
        Args:
            transitions: numpy array of shape (N, state_dim + action_dim + state_dim)
                         containing [state, action, next_state] concatenated
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector
            normalize: Whether to normalize the data
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Split transitions into states, actions, next_states
        self.states = transitions[:, :state_dim].astype(np.float32)
        self.actions = transitions[:, state_dim:state_dim + action_dim].astype(np.float32)
        self.next_states = transitions[:, state_dim + action_dim:].astype(np.float32)
        
        # Compute normalization statistics
        if normalize:
            self.state_mean = torch.tensor(np.mean(self.states, axis=0), dtype=torch.float32)
            self.state_std = torch.tensor(np.std(self.states, axis=0) + 1e-8, dtype=torch.float32)
            self.action_mean = torch.tensor(np.mean(self.actions, axis=0), dtype=torch.float32)
            self.action_std = torch.tensor(np.std(self.actions, axis=0) + 1e-8, dtype=torch.float32)
            
            # Normalize
            self.states = (self.states - self.state_mean.numpy()) / self.state_std.numpy()
            self.actions = (self.actions - self.action_mean.numpy()) / self.action_std.numpy()
            self.next_states = (self.next_states - self.state_mean.numpy()) / self.state_std.numpy()
        else:
            self.state_mean = torch.zeros(state_dim, dtype=torch.float32)
            self.state_std = torch.ones(state_dim, dtype=torch.float32)
            self.action_mean = torch.zeros(action_dim, dtype=torch.float32)
            self.action_std = torch.ones(action_dim, dtype=torch.float32)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        next_state = torch.tensor(self.next_states[idx], dtype=torch.float32)
        return state, action, next_state


class TransitionModel(nn.Module):
    """Neural network model for predicting next state from current state and action."""
    
    def __init__(self, state_dim=12, action_dim=4, hidden_dim=64, num_layers=2):
        """
        Initialize transition model.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector
            hidden_dim: Number of hidden units per layer
            num_layers: Number of hidden layers
        """
        super(TransitionModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim
        
        # Build layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, action):
        """
        Predict next state from current state and action.
        
        Args:
            state: Current state tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
        
        Returns:
            Predicted next state [batch_size, state_dim]
        """
        x = torch.cat([state, action], dim=1)
        return self.network(x)


def load_transition_data(data_path):
    """Load transition data from numpy file and metadata."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load transitions
    transitions = np.load(data_path)
    print(f"Loaded transitions from {data_path}")
    print(f"  Shape: {transitions.shape}")
    
    # Load metadata
    metadata_path = data_path.parent / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {metadata_path}")
        print(f"  Num samples: {metadata.get('num_samples', 'unknown')}")
        print(f"  State dim: {metadata.get('state_dim', 'unknown')}")
        print(f"  Action dim: {metadata.get('action_dim', 'unknown')}")
    
    return transitions, metadata


def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device=device):
    """Train the transition model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for state, action, next_state in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            state = state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)
            
            optimizer.zero_grad()
            predicted_next_state = model(state, action)
            loss = criterion(predicted_next_state, next_state)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_count += 1
        
        avg_train_loss = train_loss / train_count
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for state, action, next_state in val_loader:
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
                
                predicted_next_state = model(state, action)
                loss = criterion(predicted_next_state, next_state)
                
                val_loss += loss.item()
                val_count += 1
        
        avg_val_loss = val_loss / val_count
        val_losses.append(avg_val_loss)
        
        # Get learning rate before scheduler step
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        if new_lr < old_lr:
            print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  ✓ New best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses, best_val_loss


def save_checkpoint(model, dataset, checkpoint_dir, best_val_loss):
    """Save model checkpoint and normalization statistics."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model checkpoint
    checkpoint_path = checkpoint_dir / "transition_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_dim': model.state_dim,
        'action_dim': model.action_dim,
        'best_val_loss': best_val_loss,
    }, checkpoint_path)
    print(f"\nSaved model checkpoint to {checkpoint_path}")
    
    # Save normalization statistics
    norm_stats = {
        'state_mean': dataset.state_mean.tolist(),
        'state_std': dataset.state_std.tolist(),
        'action_mean': dataset.action_mean.tolist(),
        'action_std': dataset.action_std.tolist(),
    }
    
    norm_path = checkpoint_dir / "normalization_stats.json"
    with open(norm_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved normalization statistics to {norm_path}")


def main():
    parser = argparse.ArgumentParser(description='Train transition model')
    parser.add_argument('--data_path', type=str, 
                       default='controllers/data_collector_controller/data/transitions/transitions.npy',
                       help='Path to transitions.npy file')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='controllers/drone_controller/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (0-1)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Number of hidden units per layer')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of hidden layers')
    parser.add_argument('--no_normalize', action='store_true',
                       help='Disable data normalization')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRANSITION MODEL TRAINING")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Validation split: {args.val_split}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num layers: {args.num_layers}")
    print(f"Normalize: {not args.no_normalize}")
    print("=" * 60)
    print()
    
    # Load data
    try:
        transitions, metadata = load_transition_data(args.data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Get dimensions from metadata or infer from data
    state_dim = metadata.get('state_dim', 12)
    action_dim = metadata.get('action_dim', 4)
    
    # Verify data shape
    expected_dim = state_dim + action_dim + state_dim
    if transitions.shape[1] != expected_dim:
        print(f"Warning: Expected data shape (N, {expected_dim}), got {transitions.shape}")
        print("  Inferring dimensions from data shape...")
        # Infer: assume equal state and next_state dimensions
        total_dim = transitions.shape[1]
        action_dim = 4  # Known from metadata
        state_dim = (total_dim - action_dim) // 2
        if state_dim * 2 + action_dim != total_dim:
            raise ValueError(f"Cannot infer dimensions from shape {transitions.shape}")
        print(f"  Inferred: state_dim={state_dim}, action_dim={action_dim}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = TransitionDataset(
        transitions, 
        state_dim=state_dim, 
        action_dim=action_dim,
        normalize=not args.no_normalize
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\nCreating model...")
    model = TransitionModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses, best_val_loss = train_model(
        model, 
        train_loader, 
        val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    
    # Save checkpoint
    print("\nSaving checkpoint...")
    save_checkpoint(model, dataset, args.checkpoint_dir, best_val_loss)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to: {Path(args.checkpoint_dir) / 'transition_model.pth'}")
    print(f"Normalization stats saved to: {Path(args.checkpoint_dir) / 'normalization_stats.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

