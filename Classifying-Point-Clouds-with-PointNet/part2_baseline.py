from typing import Tuple

import torch
from torch import nn

class Baseline(nn.Module):
    '''
    A simple baseline that counts points per voxel in the point cloud
    and then uses a linear classifier to make a prediction.
    '''

    def __init__(self,
                 classes: int,
                 voxel_resolution=4,
                 mode="count") -> None:
        '''
        Constructor for Baseline to define layers.

        Args:
        -   classes: Number of output classes
        -   voxel_resolution: Number of positions per dimension to count
        -   mode: "count" or "occupancy"
        '''
        assert mode in ["count", "occupancy"]

        super().__init__()
        self.voxel_resolution = voxel_resolution
        self.mode = mode

        # The feature dimension is voxel_resolution^3
        in_dim = voxel_resolution ** 3
        self.classifier = nn.Linear(in_dim, classes)

    def count_points(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Create the feature as input to the linear classifier by counting
        the number of points per voxel.

        Hint:
        1) Use torch.histogramdd to compute a 3D histogram.
        2) The point clouds are in a 4x4x4 region, likely centered at zero.
           We'll assume coordinates range from -2 to 2 in each axis.
        
        Args:
        -   x: tensor of shape (B, N, 3)

        Output:
        -   counts: tensor of shape (B, voxel_resolution**3) containing normalized counts
        '''

        B, N, _ = x.shape
        # Define voxel edges
        edges = torch.linspace(-2, 2, self.voxel_resolution + 1, device=x.device, dtype=x.dtype)

        # We'll compute histogram per example in a loop since torch.histogramdd doesn't batch
        counts_list = []
        for i in range(B):
            sample_points = x[i]  # (N, 3)
            hist = torch.histogramdd(sample_points, bins=(edges, edges, edges))
            # hist.hist is of shape (voxel_resolution, voxel_resolution, voxel_resolution)
            # Normalize so that sum is 1 (percentage of points in each voxel)
            voxel_counts = hist.hist / (N if N > 0 else 1.0)  # Avoid division by zero
            voxel_counts = voxel_counts.flatten()  # (voxel_resolution^3,)
            counts_list.append(voxel_counts)

        counts = torch.stack(counts_list, dim=0)  # (B, voxel_resolution^3)
        return counts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the Baseline model.

        If mode == "occupancy", threshold the counts at >0 to produce a binary occupancy grid.
        If mode == "count", use the normalized counts as is.

        Args:
        -   x: tensor of shape (B, N, 3)

        Output:
        -   class_outputs: tensor of shape (B, classes)
        -   None: dummy return to keep consistency with other models
        '''

        counts = self.count_points(x)
        if self.mode == "occupancy":
            # Convert to binary occupancy
            counts = (counts > 0).float()

        class_outputs = self.classifier(counts)
        return class_outputs, None
