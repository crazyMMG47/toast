import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import numpy as np
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from utils.phiseg_unet import * 
from utils.debug import * 

def add_specific_debug_hooks(model):
    """Add debug hooks to problematic layers"""
    hooks = []
    
    # Hook the problematic z_ups_convs in prior_net
    if hasattr(model, 'prior_net') and hasattr(model.prior_net, 'z_ups_convs'):
        print(f"Adding debug hooks to {len(model.prior_net.z_ups_convs)} z_ups_convs layers")
        
        for name, conv in model.prior_net.z_ups_convs.items():
            def make_hook(layer_name):
                def hook(module, input, output):
                    print(f"=== {layer_name} Debug ===")
                    
                    # Debug input
                    if isinstance(input, tuple) and len(input) > 0:
                        inp = input[0]
                        print(f"{layer_name} input: shape={inp.shape}, dtype={inp.dtype}")
                        print(f"{layer_name} input channels: {inp.shape[1]}")
                    
                    # Debug module
                    if hasattr(module, 'in_channels'):
                        print(f"{layer_name} expects: {module.in_channels} channels")
                        print(f"{layer_name} produces: {module.out_channels} channels")
                    
                    # Debug output (if successful)
                    if isinstance(output, torch.Tensor):
                        print(f"{layer_name} output: shape={output.shape}")
                    
                    print(f"=== End {layer_name} ===")
                
                return hook
            
            hook = conv.register_forward_hook(make_hook(name))
            hooks.append(hook)
    
    # Optionally hook other problematic areas
    if hasattr(model, 'prior_net') and hasattr(model.prior_net, 'encoder_blocks'):
        print(f"Adding debug hooks to {len(model.prior_net.encoder_blocks)} encoder blocks")
        
        for i, block in enumerate(model.prior_net.encoder_blocks):
            def make_encoder_hook(block_idx):
                def hook(module, input, output):
                    if isinstance(input, tuple) and len(input) > 0:
                        inp = input[0]
                        print(f"Encoder[{block_idx}] input: {inp.shape}")
                    if isinstance(output, torch.Tensor):
                        print(f"Encoder[{block_idx}] output: {output.shape}")
                return hook
            
            hook = block.register_forward_hook(make_encoder_hook(i))
            hooks.append(hook)
    
    print(f"Total debug hooks added: {len(hooks)}")
    return hooks

class PHISegTrainer:
    """
    Training pipeline for PHI-Seg model
    """
    
    def __init__(
        self,
        model: 'CompletePHISegModel',
        device: torch.device,
        learning_rate: float = 1e-3,
        beta_kl: float = 1.0,
        beta_schedule: Optional[str] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.beta_kl = beta_kl
        self.beta_schedule = beta_schedule
        self.current_epoch = 0
        
        # Loss weights for different components
        self.loss_weights = loss_weights or {
            'reconstruction': 1.0,
            'kl': 1.0,
            'multi_scale': [1.0, 0.8, 0.6, 0.4, 0.2]  # Weights for each scale
        }
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss functions
        self.dice_loss = DiceLoss(softmax=True, to_onehot_y=True)
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
    
    def get_beta_kl(self, epoch: int) -> float:
        """
        Get KL divergence weight with optional scheduling
        """
        if self.beta_schedule == "linear":
            # Linear annealing from 0 to beta_kl over first 100 epochs
            return min(self.beta_kl, self.beta_kl * epoch / 100)
        elif self.beta_schedule == "cyclical":
            # Cyclical annealing
            cycle_length = 100
            cycle_position = epoch % cycle_length
            return self.beta_kl * (cycle_position / cycle_length)
        else:
            return self.beta_kl
    
    def compute_reconstruction_loss(
        self,
        predictions: list,
        targets: torch.Tensor,
        use_dice: bool = True
    ) -> torch.Tensor:
        """
        Compute multi-scale reconstruction loss
        """
        total_loss = 0.0
        scale_weights = self.loss_weights['multi_scale']
        
        # Ensure we have enough weights
        if len(scale_weights) < len(predictions):
            scale_weights = scale_weights + [scale_weights[-1]] * (len(predictions) - len(scale_weights))
        
        # Normalize weights
        weight_sum = sum(scale_weights[:len(predictions)])
        scale_weights = [w / weight_sum for w in scale_weights[:len(predictions)]]
        
        for i, (pred, weight) in enumerate(zip(predictions, scale_weights)):
            # Resize target to match prediction if needed
            target_resized = targets
            if pred.shape[2:] != targets.shape[2:]:
                target_resized = torch.nn.functional.interpolate(
                    targets.float(),
                    size=pred.shape[2:],
                    mode='nearest'
                ).long()
            
            if use_dice:
                loss = self.dice_loss(pred, target_resized)
            else:
                loss = self.ce_loss(pred, target_resized.squeeze(1))
            
            total_loss += weight * loss
        
        return total_loss
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single training step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        images = batch['image'].to(self.device)
        masks = batch['label'].to(self.device)
        
        # Forward pass
        predictions, kl_loss = self.model(
            images, 
            masks, 
            target_size=images.shape[2:],
            use_posterior=True
        )
        
        # Compute losses
        reconstruction_loss = self.compute_reconstruction_loss(predictions, masks)
        
        # Get current KL weight
        current_beta = self.get_beta_kl(self.current_epoch)
        
        # Total loss
        total_loss = (
            self.loss_weights['reconstruction'] * reconstruction_loss +
            current_beta * self.loss_weights['kl'] * kl_loss
        )
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item(),
            'beta_kl': current_beta
        }
    
    def validate_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single validation step
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass (using posterior for validation)
            predictions, kl_loss = self.model(
                images,
                masks,
                target_size=images.shape[2:],
                use_posterior=True
            )
            
            # Compute losses
            reconstruction_loss = self.compute_reconstruction_loss(predictions, masks)
            total_loss = reconstruction_loss + self.beta_kl * kl_loss
            
            # Compute Dice score on final prediction (highest resolution)
            final_pred = predictions[0]  # Finest scale prediction
            pred_classes = torch.argmax(final_pred, dim=1, keepdim=True)
            dice_metric = DiceMetric(include_background=False, reduction="mean")
            
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item(),
            'dice_score': dice_metric.mean().item()
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch
        """
        self.current_epoch = epoch
        epoch_losses = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'beta_kl': []
        }
        
        for batch_idx, batch in enumerate(train_loader):
            losses = self.train_step(batch)
            
            for key, value in losses.items():
                epoch_losses[key].append(value)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss = {losses['total_loss']:.4f}, "
                      f"Recon = {losses['reconstruction_loss']:.4f}, "
                      f"KL = {losses['kl_loss']:.4f}")
        
        # Average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        self.train_losses.append(avg_losses)
        
        return avg_losses
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate for one epoch
        """
        epoch_metrics = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'dice_score': []
        }
        
        for batch in val_loader:
            metrics = self.validate_step(batch)
            
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        self.val_losses.append(avg_metrics)
        
        print(f"Validation Epoch {epoch}: "
              f"Loss = {avg_metrics['total_loss']:.4f}, "
              f"Dice = {avg_metrics['dice_score']:.4f}")
        
        return avg_metrics
    
    def generate_samples(
        self,
        test_image: torch.Tensor,
        n_samples: int = 5,
        save_path: Optional[str] = None
    ) -> list:
        """
        Generate diverse segmentation samples for a test image
        """
        self.model.eval()
        
        with torch.no_grad():
            test_image = test_image.to(self.device)
            samples = self.model.sample_diverse_predictions(
                test_image,
                n_samples=n_samples,
                target_size=test_image.shape[2:]
            )
        
        # Convert to numpy for visualization
        samples_np = []
        for sample in samples:
            # Take finest scale prediction
            pred = torch.softmax(sample[0], dim=1)
            pred_classes = torch.argmax(pred, dim=1)
            samples_np.append(pred_classes.cpu().numpy())
        
        if save_path:
            np.save(save_path, np.array(samples_np))
        
        return samples_np


def create_phiseg_model(config: Dict[str, Any]) -> 'CompletePHISegModel':
    """
    Factory function to create PHI-Seg model from config
    """
    # from phiseg_decoder import CompletePHISegModel
    
    return CompletePHISegModel(
        image_channels=config['image_channels'],
        mask_channels=config['mask_channels'],
        latent_dim=config['latent_dim'],
        n_classes=config['n_classes'],
        spatial_dims=config.get('spatial_dims', 3),
        feature_channels=config.get('feature_channels', (32, 64, 128, 256, 256, 256)),
        latent_levels=config.get('latent_levels', 5),
        resolution_levels=config.get('resolution_levels', 6),
        act=config.get('activation', 'PRELU'),
        norm=config.get('normalization', 'INSTANCE'),
        dropout=config.get('dropout', 0.1)
    )


# Example usage and training loop
def main():
    """
    Example training script
    """
    # Configuration
    config = {
        'image_channels': 1,
        'mask_channels': 1,
        'latent_dim': 16,
        'n_classes': 4,  # Background + 3 classes
        'spatial_dims': 3,
        'feature_channels': (32, 64, 128, 256, 256, 256),
        'latent_levels': 5,
        'resolution_levels': 6,
        'activation': 'PRELU',
        'normalization': 'INSTANCE',
        'dropout': 0.1
    }
    
    # Create model
    model = create_phiseg_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create trainer
    trainer = PHISegTrainer(
        model=model,
        device=device,
        learning_rate=1e-4,
        beta_kl=1.0,
        beta_schedule="linear",
        loss_weights={
            'reconstruction': 1.0,
            'kl': 1.0,
            'multi_scale': [1.0, 0.8, 0.6, 0.4, 0.2]
        }
    )
    
    # Training loop (pseudo-code - you'll need actual data loaders)
    num_epochs = 200
    
    for epoch in range(num_epochs):
        # Training
        # train_losses = trainer.train_epoch(train_loader, epoch)
        
        # Validation
        # val_metrics = trainer.validate_epoch(val_loader, epoch)
        
        # Save checkpoint
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
            }, f'phiseg_checkpoint_epoch_{epoch}.pth')
    
    print("Training completed!")


if __name__ == "__main__":
    main()