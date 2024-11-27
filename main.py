import torch
import pandas as pd
from torch import nn, autograd, optim
from torch.utils.data import DataLoader, TensorDataset
from gan import Discriminator, Generator, compute_gradient_penalty

# Define your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device: ", device)

# Define the path to your CSV file
csv_file_path = 'worms.csv'

# Load the CSV file using pandas
df = pd.read_csv(csv_file_path)

# If necessary, convert any non-numeric columns to numeric (if preprocessing was done beforehand, this step may not be needed)
# df = df.apply(pd.to_numeric, errors='coerce')

# Assuming that each row in the CSV represents a sample and each column represents a feature
# Convert the dataframe into a PyTorch tensor
data = torch.tensor(df.values, dtype=torch.float32)

# Add a channel dimension (e.g., if you have a 1D signal)
data = data.unsqueeze(1)  # Add a "1" channel dimension for 1D Conv layers

# Define batch size
batch_size = 64

# Now the dataloader can be used in the training loop

# Initialize the Discriminator (Critic) and Generator models
nz = 100  # Latent vector size for the Generator
discriminator = Discriminator().to(device)
generator = Generator(nz).to(device)

# Set optimizers for the discriminator and generator
lr = 0.0002  # learning rate
beta1, beta2 = 0.5, 0.999
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

# Define WGAN-GP hyperparameters
n_critic = 5  # Number of training steps for discriminator per generator step
lambda_gp = 10  # Gradient penalty lambda

batch_size = 64
dataloader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i, (real_samples,) in enumerate(dataloader):
        
        # Move real samples to device
        real_samples = real_samples.to(device)
        print("real samples",real_samples.shape)
        batch_size = real_samples.size(0)
        # print("batch_size",batch_size)
        
        # Training the Discriminator
        optimizer_D.zero_grad()
        
        # Generate noise and create fake samples with the generator
        noise = torch.randn(batch_size, nz, 1).to(device)
        fake_samples = generator(noise)
        
        # Calculate discriminator loss on real and fake samples
        real_validity = discriminator(real_samples)
        fake_validity = discriminator(fake_samples.detach())
        print(real_validity.shape)
        print(fake_validity.shape)
        
        # Compute WGAN-GP loss and gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_samples, fake_samples, lambda_gp)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
        
        # Backward and optimizer step for the discriminator
        d_loss.backward()
        optimizer_D.step()
        
        # Train the generator every n_critic steps
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            
            # Generate fake samples and calculate generator loss
            fake_samples = generator(noise)
            fake_validity = discriminator(fake_samples)
            g_loss = -torch.mean(fake_validity)
            
            # Backward and optimizer step for the generator
            g_loss.backward()
            optimizer_G.step()

    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# After training, use the generator to create synthetic samples
with torch.no_grad():
    noise = torch.randn(batch_size, nz, 1).to(device)
    synthetic_data = generator(noise)
