import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict

class TabularGenerator(nn.Module):
    """
    Generator for Tabular Data (NSL-KDD) in a Federated Setting.
    Takes random noise + conditional label vector and generates synthetic samples.
    """
    def __init__(self, latent_dim: int, output_dim: int, num_classes: int = 2):
        super(TabularGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Input: Noise + One-hot label
        self.input_dim = latent_dim + num_classes
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, output_dim),
            # No activation here; data will be normalized post-generation
        )
        
    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Concatenate noise and one-hot encoded labels
        one_hot_labels = torch.zeros(noise.size(0), self.num_classes, device=noise.device)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
        
        gen_input = torch.cat([noise, one_hot_labels], dim=1)
        output = self.model(gen_input)
        return output

class TabularDiscriminator(nn.Module):
    """
    Discriminator for Tabular Data.
    Outputs a scalar score (critic score) indicating real vs fake.
    """
    def __init__(self, input_dim: int, num_classes: int = 2):
        super(TabularDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Input: Data + One-hot label
        self.model_input_dim = input_dim + num_classes
        
        self.model = nn.Sequential(
            nn.Linear(self.model_input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1) # Scalar score (Wasserstein distance)
        )
        
    def forward(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        one_hot_labels = torch.zeros(data.size(0), self.num_classes, device=data.device)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
        
        disc_input = torch.cat([data, one_hot_labels], dim=1)
        validity = self.model(disc_input)
        return validity

class FederatedGANClient:
    """
    Client logic for training a WGAN-GP locally.
    Uses Gradient Penalty for stability.
    """
    def __init__(self, data_dim: int, latent_dim: int = 100, lr: float = 0.0002, device: torch.device = None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        
        self.generator = TabularGenerator(latent_dim, data_dim).to(self.device)
        self.discriminator = TabularDiscriminator(data_dim).to(self.device)
        
        self.opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        self.lambda_gp = 10.0 # Gradient penalty coefficient
        
    def compute_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = torch.rand(real_samples.size(0), 1, device=self.device)
        alpha = alpha.expand_as(real_samples)
        
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates, labels)
        fake = torch.ones_like(d_interpolates, device=self.device)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, real_data: torch.Tensor, real_labels: torch.Tensor, n_critic: int = 5) -> Tuple[float, float]:
        """
        Performs one step of GAN training (n_critic steps for D, 1 for G).
        Returns average D loss and G loss.
        """
        self.generator.train()
        self.discriminator.train()
        
        batch_size = real_data.size(0)
        d_loss_total = 0.0
        g_loss_total = 0.0
        
        # Train Discriminator n_critic times
        for _ in range(n_critic):
            # Sample noise
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            # Use real labels for conditional generation
            gen_fake = self.generator(z, real_labels)
            
            # Real samples
            real_validity = self.discriminator(real_data, real_labels)
            # Fake samples
            fake_validity = self.discriminator(gen_fake.detach(), real_labels)
            
            # Gradient Penalty
            gp = self.compute_gradient_penalty(real_data.data, gen_fake.data, real_labels)
            
            # WGAN-GP Loss
            d_loss = -(torch.mean(real_validity) - torch.mean(fake_validity)) + self.lambda_gp * gp
            
            self.opt_d.zero_grad()
            d_loss.backward()
            self.opt_d.step()
            
            d_loss_total += d_loss.item()
            
        # Train Generator
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        gen_fake = self.generator(z, real_labels)
        fake_validity = self.discriminator(gen_fake, real_labels)
        
        g_loss = -torch.mean(fake_validity)
        
        self.opt_g.zero_grad()
        g_loss.backward()
        self.opt_g.step()
        
        g_loss_total += g_loss.item()
        
        return d_loss_total / n_critic, g_loss_total

    def get_generator_weights(self) -> dict:
        return self.generator.state_dict()
    
    def set_generator_weights(self, weights: dict):
        self.generator.load_state_dict(weights)
        
    def generate_synthetic_data(self, num_samples: int, target_class: int) -> torch.Tensor:
        """Generate synthetic samples for a specific class"""
        self.generator.eval()
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        labels = torch.full((num_samples,), target_class, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            synthetic = self.generator(z, labels)
        
        return synthetic.cpu()
