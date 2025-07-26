import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# Import our custom modules
from hifi_lora import HiFiLoraLayer
from wem_loss import WEMLoss


# --- 1. Define a Simple U-Net Model (as a stand-in for SAM) ---
# This allows us to demonstrate the concept without heavy dependencies.
# The key is that it contains nn.Linear layers that we can replace.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # This linear layer simulates a component we might want to adapt,
        # e.g., a projection layer in a more complex attention block.
        self.linear_projection = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        # Reshape for linear layer -> (B, C, H*W) -> (B, H*W, C)
        b, c, h, w = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_reshaped = self.linear_projection(x_reshaped)
        # Reshape back -> (B, H*W, C) -> (B, C, H, W)
        x = x_reshaped.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = DoubleConv(128, 256)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = DoubleConv(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))

        bottle = self.bottleneck(self.pool(d2))

        u1 = self.upconv1(bottle)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up1(u1)

        u2 = self.upconv2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up2(u2)

        return self.final_conv(u2)


# --- 2. Helper function to apply HiFi-LoRA ---

def apply_hifi_lora_to_model(model: nn.Module, r: int, lora_alpha: float):
    """
    Recursively finds all nn.Linear layers and replaces them with HiFiLoraLayer.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            print(f"Applying HiFi-LoRA to: {name}")
            setattr(model, name, HiFiLoraLayer(module, r=r, lora_alpha=lora_alpha))
        elif len(list(module.children())) > 0:
            apply_hifi_lora_to_model(module, r, lora_alpha)


# --- 3. Dice Loss for Segmentation ---

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = torch.sum(probs * targets)
        union = torch.sum(probs) + torch.sum(targets)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice


# --- 4. Main Training Script ---

if __name__ == '__main__':
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LORA_R = 8
    LORA_ALPHA = 16
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    EPOCHS = 10
    LAMBDA_WEM = 0.1  # Weight for the WEM regularizer

    # --- Model Setup ---
    print("Initializing model...")
    model = SimpleUNet(in_channels=1, out_channels=1).to(DEVICE)

    # Apply HiFi-LoRA to all linear layers in the model
    apply_hifi_lora_to_model(model, r=LORA_R, lora_alpha=LORA_ALPHA)

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    # Print trainable parameters to verify
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params}")

    # --- Data Generation (Dummy Data for Demonstration) ---
    print("Generating dummy data...")


    def generate_dummy_data(num_samples, size=128):
        images = torch.rand(num_samples, 1, size, size)
        masks = torch.zeros(num_samples, 1, size, size)
        for i in range(num_samples):
            # Create a mask with some sharp, irregular shapes
            x1, y1 = np.random.randint(20, 40, 2)
            x2, y2 = np.random.randint(60, 100, 2)
            masks[i, :, y1:y2, x1:x2] = 1
            # Add some noise/holes to simulate complex boundaries
            masks[i, :, y1 + 5:y2 - 5, x1 + 5:x2 - 5] = 0
            x3, y3 = np.random.randint(y1 + 10, y2 - 20, 2)
            masks[i, :, y3:y3 + 10, x3:x3 + 10] = 1
        return images, masks


    train_images, train_masks = generate_dummy_data(100)
    train_dataset = TensorDataset(train_images, train_masks)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Training Setup ---
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    seg_loss_fn = DiceLoss().to(DEVICE)
    wem_loss_fn = WEMLoss(device=DEVICE)

    # --- Training Loop ---
    print(f"Starting training on {DEVICE}...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss_epoch = 0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            pred_logits = model(images)
            pred_probs = torch.sigmoid(pred_logits)

            # Calculate losses
            seg_loss = seg_loss_fn(pred_logits, masks)
            wem_loss = wem_loss_fn(pred_probs)

            # Combine losses (as per the paper's final objective)
            total_loss = seg_loss + LAMBDA_WEM * wem_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {total_loss_epoch / len(train_loader):.4f}, "
              f"Seg Loss: {seg_loss.item():.4f}, WEM Loss: {wem_loss.item():.4f}")

    # --- Visualization of Results ---
    print("Visualizing results...")
    model.eval()
    with torch.no_grad():
        test_images, test_masks = generate_dummy_data(1, size=256)
        test_images, test_masks = test_images.to(DEVICE), test_masks.to(DEVICE)

        pred_logits = model(test_images)
        pred_probs = torch.sigmoid(pred_logits)
        pred_masks_binary = (pred_probs > 0.5).float()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(test_images.cpu().numpy(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(test_masks.cpu().numpy(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask (HiFi-LoRA + WEM)")
        plt.imshow(pred_masks_binary.cpu().numpy(), cmap='gray')
        plt.axis('off')

        plt.suptitle("Deep Seek Boundary - Model Output Example")
        plt.show()