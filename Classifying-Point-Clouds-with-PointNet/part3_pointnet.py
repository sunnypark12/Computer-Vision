from typing import Tuple

import torch
from torch import nn


class PointNet(nn.Module):
    '''
    A simplified version of PointNet (https://arxiv.org/abs/1612.00593)
    Ignoring the transforms and segmentation head.
    '''
    def __init__(self,
        classes: int,
        in_dim: int=3,
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        classifier_dims: Tuple[int, int]=(512, 256),
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for PointNet to define layers.

        Hint: See the modified PointNet architecture diagram from the pdf.
        You will need to repeat the first hidden dim (see mlp(64, 64) in the diagram).
        Furthermore, you will want to include a BatchNorm1d after each layer in the encoder
        except for the final layer for easier training.

        Args:
        -   classes: Number of output classes
        -   in_dim: Input dimensionality for points. This parameter is 3 by default for
                    for the basic PointNet.
        -   hidden_dims: The dimensions of the encoding MLPs.
        -   classifier_dims: The dimensions of classifier MLPs.
        -   pts_per_obj: The number of points that each point cloud is padded to
        '''
        super().__init__()

        self.encoder_head = None
        self.classifier_head = None

        ############################################################################
        # Student code begin
        ############################################################################

        ## ENCODER MLP
        enc = []
        enc.append(nn.Linear(3, 64))
        enc.append(nn.BatchNorm1d(64))
        enc.append(nn.ReLU())

        enc.append(nn.Linear(64, 64))
        enc.append(nn.BatchNorm1d(64))
        enc.append(nn.ReLU())

        enc.append(nn.Linear(64, 128))
        enc.append(nn.BatchNorm1d(128))
        enc.append(nn.ReLU())

        enc.append(nn.Linear(128, 1024))
        self.encoder_head = nn.ModuleList(enc)

        ## Classifier MLP

        clf = []
        clf.append(nn.Linear(1024, 512))
        clf.append(nn.ReLU())
        clf.append(nn.Linear(512, 256))
        clf.append(nn.ReLU())
        clf.append(nn.Linear(256, classes))

        self.classifier_head = nn.Sequential(*clf)

        ############################################################################
        # Student code end
        ############################################################################


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the PointNet model.

        Args:
            x: tensor of shape (B, N, in_dim), where B is the batch size, N is the number of points per
               point cloud, and in_dim is the input point dimension

        Output:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   encodings: tensor of shape (B, N, hidden_dims[-1]), the final vector for each input point
                       before global maximization. This will be used later for analysis.
        '''

        B, N, in_dim = x.shape

        ############################################################################
        # Student code begin
        ############################################################################

        out = x

        i = 0
        while i < len(self.encoder_head):
            layer = self.encoder_head[i]
            if isinstance(layer, nn.Linear):
                out = out.reshape(B * N, -1)
                out = layer(out)
                out = out.reshape(B, N, -1)
                i += 1
            elif isinstance(layer, nn.BatchNorm1d):
                out = out.reshape(B * N, -1)
                out = layer(out)
                out = out.reshape(B, N, -1)
                i += 1
            elif isinstance(layer, nn.ReLU):
                out = layer(out)
                i += 1 
            else:
                i += 1
            
        encodings = out
        g, _ = torch.max(encodings, dim = 1)
        class_outputs = self.classifier_head(g)

        ############################################################################
        # Student code end
        ############################################################################

        return class_outputs, encodings
