import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from scene import GaussianModel

def initialize_weights(model):
    """
    Initialize weights of the model.
    Args:
    - model (torch.nn.Module): Model to initialize weights.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class MultiModalEncoder(nn.Module):
    """
    Encoder that combines RGB image, time t, and motion parameters.
    """

    def __init__(self, image_size=64, latent_dim=64, motion_dim=4, time_dim=1, hidden_dim=256):
        super(MultiModalEncoder, self).__init__()

        # Convolutional layers for RGB image encoding
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # (image_size/2 x image_size/2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # (image_size/4 x image_size/4)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # (image_size/8 x image_size/8)
        self.conv4 = nn.Conv2d(128, hidden_dim, kernel_size=4, stride=2, padding=1)  # (image_size/16 x image_size/16)

        # Flatten and project image features
        self.image_fc = nn.Linear(hidden_dim * (image_size // 16) ** 2, latent_dim)

        # Fully connected layers for time and motion parameters
        self.motion_fc = nn.Linear(motion_dim + time_dim, latent_dim)

        # Fusion layer
        self.fusion_fc = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, rgb_image, motion_params, t):
        """
        Forward pass for the encoder.

        Args:
        - rgb_image (torch.Tensor): Input RGB image (batch_size, 3, image_size, image_size).
        - motion_params (torch.Tensor): Motion parameters (batch_size, motion_dim).
        - t (torch.Tensor): Time step (batch_size, 1).

        Returns:
        - torch.Tensor: Latent vector (batch_size, latent_dim).
        """
        # Encode RGB image
        x = F.relu(self.conv1(rgb_image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten
        z_image = self.image_fc(x)  # (batch_size, latent_dim)

        # Encode time and motion parameters
        z_motion = self.motion_fc(torch.cat([motion_params, t], dim=1))  # (batch_size, latent_dim)

        # Combine features
        z_combined = torch.cat([z_image, z_motion], dim=1)  # (batch_size, latent_dim * 2)
        z_params = self.fusion_fc(z_combined)
        mu, logvar = z_params.chunk(2, dim=1)
        #z_latent = F.relu(self.fusion_fc(z_combined))  # (batch_size, latent_dim)

        return mu, logvar

def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)  # Standard deviation
    eps = torch.randn_like(std)   # Sample from standard normal distribution
    return mu + eps * std

class MultiModalDecoder(nn.Module):
    """
    Decoder that takes latent vector, time t, and motion parameters to reconstruct RGB image.
    """

    def __init__(self, image_size=64, latent_dim=64, motion_dim=4, time_dim=1, hidden_dim=256):
        super(MultiModalDecoder, self).__init__()

        # Fusion layer for latent vector, time, and motion parameters
        self.fusion_fc = nn.Linear(latent_dim + motion_dim + time_dim, hidden_dim)

        # Fully connected layer to project to convolutional input shape
        self.fc = nn.Linear(hidden_dim, hidden_dim * (image_size // 16) ** 2)

        # Transposed convolutional layers for upsampling
        self.deconv1 = nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)  # Output RGB image

    def forward(self, latent_vector, motion_params, t):
        """
        Forward pass for the decoder.

        Args:
        - latent_vector (torch.Tensor): Latent vector (batch_size, latent_dim).
        - motion_params (torch.Tensor): Motion parameters (batch_size, motion_dim).
        - t (torch.Tensor): Time step (batch_size, 1).

        Returns:
        - torch.Tensor: Reconstructed RGB image (batch_size, 3, image_size, image_size).
        """
        # Combine latent vector, motion parameters, and time
        z_combined = torch.cat([latent_vector, motion_params, t],
                               dim=1)  # (batch_size, latent_dim + motion_dim + time_dim)
        x = F.relu(self.fusion_fc(z_combined))

        # Project to convolutional feature map
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), -1, 4, 4)  # Reshape to (batch_size, hidden_dim, 4, 4)

        # Upsample using transposed convolutions
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))  # Output normalized to [0, 1]

        return x


def loss_function(reconstructed_images, target_images, latent_vectors, time_steps, motion_params):
    """
    Compute the loss for the model.

    Args:
    - reconstructed_images (torch.Tensor): Reconstructed RGB images (batch_size, 3, image_size, image_size).
    - target_images (torch.Tensor): Target RGB images (batch_size, 3, image_size, image_size).
    - latent_vectors (torch.Tensor): Latent vectors (batch_size, latent_dim).
    - time_steps (torch.Tensor): Time steps (batch_size, 1).
    - motion_params (torch.Tensor): Motion parameters (batch_size, motion_dim).

    Returns:
    - total_loss (torch.Tensor): Total loss value.
    """
    # Reconstruction loss (MSE or L1 loss)
    reconstruction_loss = F.mse_loss(reconstructed_images, target_images)

    # Motion smoothness regularization (assumes motion_params include velocity or displacement)
    motion_diff = motion_params[:, 1:] - motion_params[:, :-1]
    smoothness_loss = torch.mean(motion_diff ** 2)  # Encourage smooth transitions in motion

    # Latent space smoothness regularization (assume latent_vectors are sequential w.r.t time_steps)
    latent_diff = latent_vectors[:, 1:] - latent_vectors[:, :-1]
    time_diff = time_steps[:, 1:] - time_steps[:, :-1]
    latent_smoothness_loss = torch.mean((latent_diff / (time_diff + 1e-6)) ** 2)

    # Total loss
    total_loss = reconstruction_loss + 0.1 * smoothness_loss + 0.1 * latent_smoothness_loss
    return total_loss


def vae_train_setting(revolute: bool, learning_rate = 1e-4):
    if revolute:
        motion_dim = 4
    else:
        motion_dim = 3
    learning_rate = 1e-4

    # Initialize encoder and decoder
    encoder = MultiModalEncoder(image_size=64, latent_dim=64, time_dim=1, motion_dim=motion_dim)
    decoder = MultiModalDecoder(image_size=64, latent_dim=32, time_dim=1, motion_dim=motion_dim)
    initialize_weights(encoder)
    initialize_weights(decoder)
    # Define optimizer (shared for encoder and decoder)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    return encoder.cuda(), decoder.cuda(), optimizer




def train_vae(iteration, warm_up, end, render_fun, encoder, decoder, fid, gt_image, viewpoint_cam,
              gaussians: GaussianModel, render_args, samplp_steps=3):
    t = torch.tensor([fid])
    crop_gt_img = crop_img(gt_image)
    label_imgs = torch.nn.functional.interpolate(crop_gt_img[None], size=(64, 64), mode="bilinear", align_corners=True)
    # save_image = ToPILImage()(crop_gt_img)
    # save_image.save(f"vae/vae_{self.iteration}.png")
    if iteration < warm_up:
        t_inputs = t.unsqueeze(dim=0).cuda()
        for t_input in t_inputs[1:]:
            with torch.no_grad():
                unseen_img = render_fun(viewpoint_cam, gaussians, *render_args, fid=t_input.item())["render"][:3, :, :]
                crop_unseen_img = crop_img(unseen_img)
                unseen_img = torch.nn.functional.interpolate(crop_unseen_img[None], size=(64, 64), mode="bilinear", align_corners=True)
                # unseen_imgs = torch.cat((unseen_imgs, unseen_img), dim=0)
                label_imgs = torch.cat((label_imgs, unseen_img), dim=0)
    elif iteration < 20000:
        t_inputs = torch.cat((t, torch.linspace(0.2, 0.8, steps=samplp_steps)), dim=0).cuda()
        for t_input in t_inputs[1:]:
            with torch.no_grad():
                unseen_img = render_fun(viewpoint_cam, gaussians, *render_args, fid=t_input.item())["render"][:3, :, :]
                crop_unseen_img = crop_img(unseen_img)
                unseen_img = torch.nn.functional.interpolate(crop_unseen_img[None], size=(64, 64), mode="bilinear", align_corners=True)
                # unseen_imgs = torch.cat((unseen_imgs, unseen_img), dim=0)
                label_imgs = torch.cat((label_imgs, unseen_img), dim=0)

    elif iteration < end:
        t_inputs = torch.cat((t, torch.tensor([0.5])), dim=0).cuda()
        for t_input in t_inputs[1:]:
            unseen_img = render_fun(viewpoint_cam, gaussians, *render_args, fid=t_input.item())["render"][:3, :, :]
            crop_unseen_img = crop_img(unseen_img)
            unseen_img = torch.nn.functional.interpolate(crop_unseen_img[None], size=(64, 64), mode="bilinear", align_corners=True)
            # unseen_imgs = torch.cat((unseen_imgs, unseen_img), dim=0)
            label_imgs = torch.cat((label_imgs, unseen_img), dim=0)
    else:
        return 0.0
    t_inputs = t_inputs.view(-1, 1)
    motion_param = gaussians.quaternions.detach().unsqueeze(dim=0).repeat(label_imgs.shape[0], 1)

    #resize_images = torch.nn.functional.interpolate(label_imgs, size=(64, 64), mode="bilinear", align_corners=True)
    mu, logvar = encoder(label_imgs, motion_param, t_inputs)
    latent_vector = reparameterize(mu, logvar )
    reconstructed_images = decoder(latent_vector, motion_param, t_inputs)
    kl_loss  = kl_divergence_loss(mu, logvar)
    vae_loss = (reconstructed_images - label_imgs).abs().mean() + 0.1*kl_loss
    if iteration % 1000 == 0:
        save_image = ToPILImage()(reconstructed_images[-1])
        save_image.save(f"vae/vae_{iteration}.png")
        print(f"vae_loss:{vae_loss}")
    return vae_loss

def crop_img(image_tensor: torch.Tensor, threshold=0.):
    assert len(image_tensor.shape) == 3 and image_tensor.shape[0] == 3, "Input must be [3, H, W] tensor."

    # Compute a mask for non-black pixels (any channel > threshold)
    non_black_mask = (image_tensor > threshold).any(dim=0)

    # Find the bounding box of non-black pixels
    rows = torch.where(non_black_mask.any(dim=1))[0]
    cols = torch.where(non_black_mask.any(dim=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("The entire image is black.")

    top, bottom = rows[0], rows[-1] + 1  # Include the last row
    left, right = cols[0], cols[-1] + 1  # Include the last column

    # Crop the image tensor
    cropped_image = image_tensor[:, top:bottom, left:right]
    return cropped_image


# Example usage
if __name__ == "__main__":
    batch_size = 8
    image_size = 64
    latent_dim = 64
    motion_dim = 6

    # Simulated inputs
    rgb_images = torch.randn(batch_size, 3, image_size, image_size)  # Input RGB images
    motion_params = torch.randn(batch_size, motion_dim)  # Random motion parameters
    t = torch.rand(batch_size, 1)  # Random time step

    # Initialize encoder and decoder
    encoder = MultiModalEncoder(image_size=image_size, latent_dim=latent_dim, motion_dim=motion_dim)
    decoder = MultiModalDecoder(image_size=image_size, latent_dim=latent_dim, motion_dim=motion_dim)

    # Forward pass
    latent_vector = encoder(rgb_images, motion_params, t)
    reconstructed_images = decoder(latent_vector, motion_params, t)

    print(f"Latent vector shape: {latent_vector.shape}")  # (batch_size, latent_dim)
    print(f"Reconstructed images shape: {reconstructed_images.shape}")  # (batch_size, 3, image_size, image_size)
