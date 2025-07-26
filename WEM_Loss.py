import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class WEMLoss(nn.Module):
    """
    Implements the Wavelet Energy Modulation (WEM) regularizer.

    This loss function encourages the model to generate boundaries with a
    complexity that is proportional to the object's scale. It calculates
    the loss based on Equations (11) and (12) from the paper.
    """

    def __init__(self, wavelet: str = 'haar', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            wavelet (str): The type of wavelet to use for the DWT. 'haar' is
                           a good default as suggested by related literature.
            device (str): The device to run the DWT on.
        """
        super().__init__()
        # DWT is a differentiable operation.
        self.dwt = DWTForward(J=1, mode='symmetric', wave=wavelet).to(device)
        self.device = device
        self.eps = 1e-8  # Epsilon to prevent log(0)

    def forward(self, pred_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the WEM loss for a batch of predicted masks.

        Args:
            pred_mask (torch.Tensor): The predicted probability mask from the
                                      model, with values in .
                                      Shape: (B, 1, H, W).

        Returns:
            torch.Tensor: A scalar tensor representing the mean WEM loss for the batch.
        """
        pred_mask = pred_mask.to(self.device)

        # 1. Decompose the mask using 2D DWT
        # Yl: Approximation (low-frequency) sub-band
        # Yh: Detail (high-frequency) sub-bands list
        Yl, Yh = self.dwt(pred_mask)
        P_lh, P_hl, P_hh = Yh[:, :, 0, :, :], Yh[:, :, 1, :, :], Yh[:, :, 2, :, :]

        # 2. Calculate Normalized Wavelet Energy 'E' (Eq. 11)
        # Energy is the sum of squared coefficients
        high_freq_energy = torch.sum(P_lh ** 2 + P_hl ** 2 + P_hh ** 2, dim=[1, 2, 3])
        total_energy = torch.sum(pred_mask ** 2, dim=[1, 2, 3])

        # Add epsilon to avoid division by zero for empty masks
        E = high_freq_energy / (total_energy + self.eps)

        # 3. Calculate Mask Area Ratio 'gamma'
        gamma = torch.mean(pred_mask, dim=[1, 2, 3])

        # 4. Calculate the WEM regularizer 'R_WEM' (Eq. 12)
        # We clamp E to avoid log(0) or log(negative) due to numerical instability.
        E_clamped = torch.clamp(E, self.eps, 1.0 - self.eps)

        term1 = gamma * (1 - gamma) * torch.log(E_clamped)
        term2 = torch.abs(1 - 2 * gamma) * torch.log(1 - E_clamped)

        R_wem = term1 - term2

        # Return the mean loss for the batch
        return R_wem.mean()