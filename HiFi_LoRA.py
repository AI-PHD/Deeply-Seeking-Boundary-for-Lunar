import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fft import dctn, idctn
import math


def create_high_pass_filter(d: int, cutoff_freq_ratio: float = 0.25) -> np.ndarray:
    """
    Creates a d x d spatial-domain high-pass filter H.

    This function implements the filter creation process described in the
    HiFi-LoRA paper, Section 2.1. It addresses the ambiguity in the paper
    by correctly implementing a 2D DCT-based filter synthesis.

    Args:
        d (int): The dimension of the square filter. This should match the
                 feature dimension of the target nn.Linear layer.
        cutoff_freq_ratio (float): The ratio of low frequencies to cut off.
                                   A value of 0.25 means the central 25%
                                   of frequencies are considered "low" and
                                   will be set to zero.

    Returns:
        np.ndarray: A d x d numpy array representing the spatial-domain
                    high-pass filter H.
    """
    if not (0 < cutoff_freq_ratio < 1):
        raise ValueError("cutoff_freq_ratio must be between 0 and 1.")

    # 1. Create the binary frequency mask matrix M (Eq. 4 in the paper)
    mask = np.ones((d, d), dtype=np.float32)
    cutoff_pixels = int(d * cutoff_freq_ratio)
    start = (d - cutoff_pixels) // 2
    end = start + cutoff_pixels
    # Set the central low-frequency area to 0
    mask[start:end, start:end] = 0

    # 2. Synthesize the high-pass filter H by transforming the mask M
    #    from the frequency domain back to the spatial domain using the
    #    inverse 2D Discrete Cosine Transform (IDCT).
    #    H = C^T * M * C (as per Eq. 4) is equivalent to idctn(M).
    #    We use 'ortho' normalization to match the paper's definition of C.
    H = idctn(mask, norm='ortho')

    return H


def get_hifi_lora_initialization(H: np.ndarray, r: int) -> tuple:
    """
    Performs SVD on the high-pass filter H and returns initial weights
    for LoRA matrices A and B.

    This function implements the core initialization logic of HiFi-LoRA,
    following Equations (5), (6), and (7) from the paper.

    Args:
        H (np.ndarray): The spatial-domain high-pass filter.
        r (int): The rank of the LoRA decomposition.

    Returns:
        tuple: A tuple containing the
                                           initialized LoRA A and B matrices
                                           (A_init, B_init).
    """
    # 1. Decompose the high-pass filter H using SVD (Eq. 5)
    #    H = U * Sigma * V^T
    U, S, Vt = np.linalg.svd(H, full_matrices=False)

    # 2. Select the top 'r' principal components
    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]

    # Ensure Sigma_r is a diagonal matrix for sqrt operation
    Sigma_r_diag = np.diag(S_r)

    # 3. Initialize LoRA matrices A and B (Eq. 6 and 7)
    #    We use the square root of singular values to balance the norms.
    sqrt_Sigma_r = np.sqrt(Sigma_r_diag)

    # B_init = U_r @ sqrt(Sigma_r)
    B_init = torch.from_numpy(U_r @ sqrt_Sigma_r).float()

    # A_init = sqrt(Sigma_r) @ V_r^T
    A_init = torch.from_numpy(sqrt_Sigma_r @ Vt_r).float()

    return A_init, B_init


class HiFiLoraLayer(nn.Module):
    """
    A LoRA layer with High-Frequency Initialization (HiFi-LoRA).

    This layer replaces a standard nn.Linear layer and injects a trainable,
    low-rank update matrix that is initialized with a high-frequency prior.
    It implements the forward pass described in Equation (8) of the paper.
    """

    def __init__(
            self,
            original_layer: nn.Linear,
            r: int,
            lora_alpha: float = 1.0,
            cutoff_freq_ratio: float = 0.25,
    ):
        """
        Args:
            original_layer (nn.Linear): The linear layer to apply LoRA to.
            r (int): The rank of the LoRA decomposition.
            lora_alpha (float): The alpha parameter for scaling, equivalent to 's'
                                in the paper's Equation (8).
            cutoff_freq_ratio (float): The cutoff ratio for the high-pass filter.
        """
        super().__init__()
        self.original_layer = original_layer

        in_features, out_features = original_layer.in_features, original_layer.out_features

        # Freeze the original weights
        self.original_layer.weight.requires_grad = False
        if original_layer.bias is not None:
            original_layer.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        # Create and initialize LoRA matrices A and B
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # --- HiFi-LoRA Initialization ---
        # Note: This initialization assumes the layer is square (in_features == out_features),
        # which is common for attention projection layers in Transformers (like SAM).
        # If the layer is not square, a choice must be made on which dimension (in or out)
        # to base the filter on. The paper simplifies this by assuming d x d.
        if in_features != out_features:
            print(f"Warning: HiFiLoraLayer is applied to a non-square layer ({in_features}x{out_features}). "
                  f"Initializing based on the smaller dimension ({min(in_features, out_features)}).")

        d = min(in_features, out_features)

        # Create the high-pass filter and get initial weights
        H = create_high_pass_filter(d, cutoff_freq_ratio)
        A_init, B_init = get_hifi_lora_initialization(H, r)

        # Assign the initialized weights to the parameters
        # We handle non-square layers by padding/slicing.
        self.lora_A.data[:, :d] = A_init
        self.lora_B.data[:d, :] = B_init
        # -----------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: h = W_0*x + s * B*A*x
        """
        original_output = self.original_layer(x)
        lora_update = (self.lora_B @ self.lora_A) * self.scaling
        lora_output = F.linear(x, lora_update)

        return original_output + lora_output

    def __repr__(self):
        return (f"{self.__class__.__name__}(r={self.r}, alpha={self.lora_alpha}, "
                f"original_layer={self.original_layer})")