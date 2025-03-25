import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
import os

# Define the PolicyNetwork here to avoid circular imports
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_action_types=5, num_raise_classes=100, num_discard_classes=3):
        """
        The network uses a shared base and three heads:
          - Action type head: outputs logits over 5 actions (FOLD, RAISE, CHECK, CALL, DISCARD)
          - Raise head: outputs logits over 100 values (to be shifted to [1, 100])
          - Discard head: outputs logits over 3 values (mapped to [-1, 0, 1])
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_type_head = nn.Linear(hidden_dim, num_action_types)
        self.raise_head = nn.Linear(hidden_dim, num_raise_classes)
        self.discard_head = nn.Linear(hidden_dim, num_discard_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_type_logits = self.action_type_head(x)
        raise_logits = self.raise_head(x)
        discard_logits = self.discard_head(x)
        return action_type_logits, raise_logits, discard_logits

class SupervisedTrainer:
    def __init__(self, input_dim, hidden_dim=128, lr=1e-4):
        self.policy_net = PolicyNetwork(input_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Loss functions for each head
        self.action_type_loss_fn = nn.CrossEntropyLoss()
        self.raise_amount_loss_fn = nn.CrossEntropyLoss()
        self.discard_loss_fn = nn.CrossEntropyLoss()
        
    def train_step(self, state, action_type, raise_amount, card_to_discard, valid_actions=None, min_raise=None, max_raise=None):
        """
        Perform one training step with the given batch.
        """
        # Convert numpy arrays to tensors if necessary
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if isinstance(action_type, np.ndarray):
            action_type = torch.tensor(action_type, dtype=torch.long)
        if isinstance(raise_amount, np.ndarray):
            raise_amount = torch.tensor(raise_amount, dtype=torch.float32)  # Changed to float for percentages
        if isinstance(card_to_discard, np.ndarray):
            # Map card_to_discard from [-1, 0, 1] to [0, 1, 2] for CrossEntropyLoss
            card_to_discard = torch.tensor(card_to_discard + 1, dtype=torch.long)
        if isinstance(min_raise, np.ndarray):
            min_raise = torch.tensor(min_raise, dtype=torch.float32)
        if isinstance(max_raise, np.ndarray):
            max_raise = torch.tensor(max_raise, dtype=torch.float32)
        
        # Get policy outputs
        action_type_logits, raise_logits, discard_logits = self.policy_net(state)
        
        # If valid_actions is provided, mask the invalid actions
        if valid_actions is not None:
            mask = (valid_actions == 0)
            action_type_logits = action_type_logits.clone()
            action_type_logits[mask] = -1e9
        
        # Calculate losses
        action_type_loss = self.action_type_loss_fn(action_type_logits, action_type)
        
        # Only calculate raise loss for samples where action_type is RAISE (index 1)
        raise_mask = (action_type == 1)
        raise_loss = torch.tensor(0.0, device=state.device, requires_grad=True)
        if raise_mask.any():
            # Convert actual raise amounts to percentages between min and max raise
            # Formula: percentage = (raise_amount - min_raise) / (max_raise - min_raise)
            # Then scale to [0, 99] for the 100-class classification
            raise_range = max_raise[raise_mask] - min_raise[raise_mask]
            # Handle case where min_raise equals max_raise (only one possible value)
            valid_range_mask = (raise_range > 0)
            
            # Initialize raise targets with zeros (defaults to min_raise)
            raise_targets = torch.zeros_like(raise_amount[raise_mask], dtype=torch.long)
            
            # Only compute percentages where we have a valid range
            if valid_range_mask.any():
                # Calculate normalized position within the range [0, 1]
                norm_position = (raise_amount[raise_mask][valid_range_mask] - 
                                min_raise[raise_mask][valid_range_mask]) / raise_range[valid_range_mask]
                
                # Scale to [0, 99] and convert to integer class
                raise_targets[valid_range_mask] = (norm_position * 99).clamp(0, 99).long()
            
            raise_loss = self.raise_amount_loss_fn(raise_logits[raise_mask], raise_targets)
        
        
        # Only calculate discard loss for samples where action_type is DISCARD (index 4)
        discard_mask = (action_type == 4)
        discard_loss = torch.tensor(0.0, device=state.device, requires_grad=True)
        if discard_mask.any():
            discard_targets = card_to_discard[discard_mask]
            discard_loss = self.discard_loss_fn(discard_logits[discard_mask], discard_targets)
            
        
        # Combine losses
        # reweight them?
        loss = action_type_loss + raise_loss + discard_loss
        
        
        if self.policy_net.training:
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return {
            "total_loss": loss.item(),
            "action_type_loss": action_type_loss.item(),
            "raise_loss": raise_loss.item(),
            "discard_loss": discard_loss.item()
        }
    
    def train_epoch(self, dataloader, device):
        """
        Train for one epoch on the given dataloader.
        """
        self.policy_net.train()
        self.policy_net.to(device)
        
        total_loss = 0
        total_samples = 0
        
        for batch in dataloader:
            states = batch["state"].to(device)
            action_types = batch["action_type"].to(device)
            raise_amounts = batch["raise_amount"].to(device)
            card_to_discards = batch["card_to_discard"].to(device)
            valid_actions = batch["valid_actions"].to(device)
            min_raises = batch["min_raise"].to(device)
            max_raises = batch["max_raise"].to(device)
            
            batch_size = states.size(0)
            loss_dict = self.train_step(
                states, action_types, raise_amounts, card_to_discards, 
                valid_actions, min_raises, max_raises
            )
            
            total_loss += loss_dict["total_loss"] * batch_size
            total_samples += batch_size
        
        return total_loss / total_samples

    def save_weights(self, path):
        """Save model weights."""
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_weights(self, path):
        """Load model weights."""
        self.policy_net.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


def prepare_dataset(data_path, batch_size=64, train_ratio=0.9):
    """
    Load the dataset and prepare DataLoader objects for training and validation.
    """
    # Load the collected data
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Convert to appropriate formats
    states = np.array([sample["state"] for sample in dataset])
    action_types = np.array([sample["action_type"] for sample in dataset])
    raise_amounts = np.array([sample["raise_amount"] for sample in dataset])
    card_to_discards = np.array([sample["card_to_discard"] for sample in dataset])
    valid_actions = np.array([sample["valid_actions"] for sample in dataset])
    min_raises = np.array([sample["min_raise"] for sample in dataset])
    max_raises = np.array([sample["max_raise"] for sample in dataset])
    
    # Split into training and validation sets
    num_samples = len(dataset)
    num_train = int(train_ratio * num_samples)
    
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # Create PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(states[train_indices], dtype=torch.float32),
        torch.tensor(action_types[train_indices], dtype=torch.long),
        torch.tensor(raise_amounts[train_indices], dtype=torch.float32),  # Changed to float
        torch.tensor(card_to_discards[train_indices], dtype=torch.long),
        torch.tensor(valid_actions[train_indices], dtype=torch.float32),
        torch.tensor(min_raises[train_indices], dtype=torch.float32),
        torch.tensor(max_raises[train_indices], dtype=torch.float32)
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(states[val_indices], dtype=torch.float32),
        torch.tensor(action_types[val_indices], dtype=torch.long),
        torch.tensor(raise_amounts[val_indices], dtype=torch.float32),  # Changed to float
        torch.tensor(card_to_discards[val_indices], dtype=torch.long),
        torch.tensor(valid_actions[val_indices], dtype=torch.float32),
        torch.tensor(min_raises[val_indices], dtype=torch.float32),
        torch.tensor(max_raises[val_indices], dtype=torch.float32)
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    print(f"Dataset loaded: {num_train} training samples, {len(val_indices)} validation samples")
    
    # Wrap data in a dictionary format for the trainer
    train_loader = [{
        "state": batch[0],
        "action_type": batch[1],
        "raise_amount": batch[2],
        "card_to_discard": batch[3],
        "valid_actions": batch[4],
        "min_raise": batch[5],
        "max_raise": batch[6]
    } for batch in train_loader]
    
    val_loader = [{
        "state": batch[0],
        "action_type": batch[1],
        "raise_amount": batch[2],
        "card_to_discard": batch[3],
        "valid_actions": batch[4],
        "min_raise": batch[5],
        "max_raise": batch[6]
    } for batch in val_loader]
    
    return train_loader, val_loader

def plot_losses(losses):
    """
    Plot the training losses.
    """
    import matplotlib.pyplot as plt
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.savefig("sft_losses.png")

def run_supervised_training(data_path="mc_training_data.pkl", 
                            num_epochs=500, 
                            save_path="sft_weights.pth"):
    """
    Run the supervised fine-tuning process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare dataset
    train_loader, val_loader = prepare_dataset(data_path)
    
    # Create trainer (input dimension is 13 as in the original code)
    trainer = SupervisedTrainer(input_dim=13)
    
    losses = []
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(num_epochs)):
        # Train for one epoch
        train_loss = trainer.train_epoch(train_loader, device)
        
        losses.append(train_loss)
        
        # Validate
        trainer.policy_net.eval()
        val_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                states = batch["state"].to(device)
                action_types = batch["action_type"].to(device)
                raise_amounts = batch["raise_amount"].to(device)
                card_to_discards = batch["card_to_discard"].to(device)
                valid_actions = batch["valid_actions"].to(device)
                min_raises = batch["min_raise"].to(device)
                max_raises = batch["max_raise"].to(device)
                
                batch_size = states.size(0)
                loss_dict = trainer.train_step(
                    states, action_types, raise_amounts, card_to_discards, 
                    valid_actions, min_raises, max_raises
                )
                
                val_loss += loss_dict["total_loss"] * batch_size
                total_samples += batch_size
        
        val_loss /= total_samples
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_weights(save_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
            
    plot_losses(losses)
    
    print(f"Supervised training complete. Best model saved to {save_path}")
    return save_path

if __name__ == "__main__":
    run_supervised_training()