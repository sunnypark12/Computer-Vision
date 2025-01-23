import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable

class NerfModel(nn.Module):
    
    def __init__(self, in_channels: int, filter_size: int=256):
        """This network will have a total of 8 fully connected layers. The activation function will be ReLU

        The number of input features to layer 5 will be a bit different. Refer to the docstring for the forward pass.
        Do not include an activation after layer 8 in the Sequential block. Layer 8's should output 4 features.

        Args
        ---
        in_channels (int): the number of input features from 
            the data
        filter_size (int): the number of in/out features for all layers. Layers 1 (because of in_channels), 5, and 8 are
            a bit different.
        """
        super().__init__()

        self.fc_layers_group1: nn.Sequential = None  # For layers 1-3
        self.layer_4: nn.Linear = None
        self.fc_layers_group2: nn.Sequential = None  # For layers 5-8
        self.loss_criterion = None

        ##########################################################################
        # Student code begins here
        ##########################################################################
        
        self.fc_layers_group1 = nn.Sequential(
            nn.Linear(in_channels, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU()
        )
        self.layer_4 = nn.Linear(filter_size, filter_size)

        self.fc_layers_group2 = nn.Sequential(
            nn.Linear(filter_size * 2, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, 4)
        )

        self.loss_criterion = nn.MSELoss()

        ##########################################################################
        # Student code ends here
        ##########################################################################
  
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass of the model. 
        
        NOTE: The input to layer 5 should be the concatenation of post-activation values from layer 4 with 
        post-activation values from layer 3. Therefore, be extra careful about how self.layer_4 is used and what 
        the specified input size to layer 5 should be. The output from layer 5 and the dimensions thereafter should be
        filter_size.
        
        Args
        ---
        x (torch.Tensor): input of shape 
            (batch_size, in_channels)
        
        Returns
        ---
        rgb (torch.Tensor): The predicted rgb values with 
            shape (batch_size, 3)
        sigma (torch.Tensor): The predicted density values with shape (batch_size)
        """
        rgb = None
        sigma = None

        ##########################################################################
        # Student code begins here
        #########################################################################

        out = self.fc_layers_group1(x)

        out2 = F.relu(self.layer_4(out))

        input = torch.cat([out2, out], dim=-1)

        out3 = self.fc_layers_group2(input)

        rgb = torch.sigmoid(out3[..., :3])
        sigma = F.relu(out3[..., 3])
        ##########################################################################
        # Student code ends here
        ##########################################################################

        return rgb, sigma

def get_rays(height: int, width: int, intrinsics: torch.Tensor, tform_cam2world: torch.Tensor) \
    -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).
    
    Args
    ---
    height (int): 
        the height of an image.
    width (int): the width of an image.
    intrinsics (torch.Tensor): Camera intrinsics matrix of shape (3, 3).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
        transforms a 3D point from the camera coordinate space to the world frame coordinate space.
    
    Returns
    ---
    ray_origins (torch.Tensor): A tensor of shape :math:`(height, width, 3)` denoting the centers of
        each ray. Note that desipte that all ray share the same origin, 
        here we ask you to return the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape :math:`(height, width, 3)` denoting the
        direction of each ray.
    """
    device = tform_cam2world.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

    ##########################################################################
    # Student code begins here
    ##########################################################################
    
    dtype = tform_cam2world.dtype

    i = torch.arange(height, device=device, dtype=dtype)
    j = torch.arange(width, device=device, dtype=dtype)
    i, j = torch.meshgrid(i, j, indexing='ij')

    ones = torch.ones_like(i)
    pixel = torch.stack([j, i, ones], dim=-1)
    
    inv = torch.inverse(intrinsics)
    pixel_flat = pixel.reshape(-1, 3)
    dir_flat = pixel_flat @ inv.T
    dir = dir_flat.reshape(height, width, 3)
    
    R = tform_cam2world[:3, :3]
    t = tform_cam2world[:3, 3]

    directions_world = dir @ R.transpose(0, 1)
    ray_directions[:] = directions_world
    ray_origins[:] = t


    ##########################################################################
    # Student code ends here
    ##########################################################################

    return ray_origins, ray_directions

def sample_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize:bool = True
) -> tuple[torch.tensor, torch.tensor]:
    """Sample 3D points on the given rays. The near_thresh and far_thresh
    variables indicate the bounds of sampling range.
    
    Args
    ---
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
        `get_rays` method (shape: :math:`(height, width, 3)`).
    ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
        `get_rays` method (shape: :math:`(height, width, 3)`).
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
    num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray (i.e., the lower bound of each bin).
    
    Returns
    ---
    query_points (torch.Tensor): Query 3D points along each ray
        (shape: :math:`(height, width, num_samples, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(height, width, num_samples)`).
    """
    device = ray_origins.device
    height, width = ray_origins.shape[:2]
    
    
    ##########################################################################
    # Student code begins here
    ##########################################################################
    val = torch.linspace(near_thresh, far_thresh, num_samples + 1, device=device)
    val = val.unsqueeze(0).unsqueeze(0)
    val = val.expand(height, width, num_samples + 1)

    if randomize:
        rand = torch.rand(height, width, num_samples, device=device)
        depth_values = val[..., :-1] + (val[..., 1:] - val[..., :-1]) * rand
    else:
        depth_values = val[..., :-1]

    query_points = ray_origins.unsqueeze(-2) + ray_directions.unsqueeze(-2) * depth_values.unsqueeze(-1)

    

    ##########################################################################
    # Student code ends here
    ##########################################################################
    
    return query_points, depth_values

def cumprod_exclusive(x: torch.tensor) -> torch.tensor:
    """ Helper function that computes the cumulative product of the input tensor, excluding the current element
    Example:
    > cumprod_exclusive(torch.tensor([1,2,3,4,5]))
    > tensor([ 1,  1,  2,  6, 24])
    
    Args:
    -   x: Tensor of length N
    
    Returns:
    -   cumprod: Tensor of length N containing the cumulative product of the tensor
    """

    cumprod = torch.cumprod(x, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.
    return cumprod

def compute_compositing_weights(sigma: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:
    """This function will compute the compositing weight for each query point.

    Args
    ---
    sigma (torch.Tensor): Volume density at each query location (X, Y, Z)
        (shape: :math:`(height, width, num_samples)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(height, width, num_samples)`).
    
    Returns:
    weights (torch.Tensor): Rendered compositing weight of each sampled point 
        (shape: :math:`(height, width, num_samples)`).
    """

    # device = depth_values.device
    # weights = torch.ones_like(sigma, device=device) # placeholder
    device = sigma.device
    ##########################################################################
    # Student code begins here
    ##########################################################################
    
    d = depth_values[..., 1:] - depth_values[..., :-1]
    l = torch.tensor(1e9, device=device).expand(*d.shape[:-1], 1)
    d = torch.cat([d, l], dim=-1)
    a = 1.0 - torch.exp(-sigma * d)
    T = cumprod_exclusive(1.0 - a)
    weights = T * a

    ##########################################################################
    # Student code ends here
    ##########################################################################

    return weights

def get_minibatches(inputs: torch.Tensor, chunksize: int = 1024 * 32) -> list[torch.Tensor]:
    """Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def render_image_nerf(height: int, width: int, intrinsics: torch.tensor, tform_cam2world: torch.tensor,
                      near_thresh: float, far_thresh: float, depth_samples_per_ray: int,
                      encoding_function: Callable, model:NerfModel, rand:bool=False) \
                      -> tuple[torch.Tensor, torch.Tensor]:
    """ This function will utilize all the other rendering functions that have been implemented in order to sample rays,
    pass those rays to the NeRF model to get color and density predictions, and then use volume rendering to create
    an image of this view. 

    Hints: 
    ---
    It is a good idea to "flatten" the height/width dimensions of the data when passing to the NeRF (maintain the color
    channel dimension) and then "unflatten" the outputs. 
    To avoid running into memory limits, it's recommended to use the given get_minibatches() helper function to 
    divide up the input into chunks. For each minibatch, supply them to the model and then concatenate the corresponding
    output vectors from each minibatch to form the complete outpute vectors. 
    
    Args
    ---
    height (int): 
        the pixel height of an image.
    width (int): the pixel width of an image.
    intrinsics (torch.tensor): Camera intrinsics matrix of shape (3, 3).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
        transforms a 3D point from the camera coordinate space to the world frame coordinate space.
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
    depth_samples_per_ray (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
    encoding_function (Callable): The function used to encode the query points (e.g. positional encoding)
    model (NerfModel): The NeRF model that will be used to render this image
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray (i.e., the lower bound of each bin).
    
    Returns
    ---
    rgb_predicted (torch.tensor): 
        A tensor of shape (height, width, num_channels) with the color info at each pixel.
    depth_predicted (torch.tensor): A tensor of shape (height, width) containing the depth from the camera at each pixel.
    """

    rgb_predicted, depth_predicted = None, None

    ##########################################################################
    # Student code begins here
    ##########################################################################
    
    device = tform_cam2world.device
    intrinsics = intrinsics.to(device=device, dtype=tform_cam2world.dtype)
    ray_origins, ray_directions = get_rays(height, width, intrinsics, tform_cam2world)
    query_points, depth_values = sample_points_from_rays(ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray, randomize=rand)

    query_points_flat = query_points.reshape(-1, 3)
    encoded = encoding_function(query_points_flat)

    minibatches = get_minibatches(encoded, chunksize=1024 * 32)

    num_rays = height * width
    rgbs_all = torch.empty(num_rays * depth_samples_per_ray, 3, device=device)
    sigmas_all = torch.empty(num_rays * depth_samples_per_ray, device=device)

    start = 0
    for m in minibatches:
        rgb_batch, sigma_batch = model(m)
        end = start + rgb_batch.shape[0]
        rgbs_all[start:end] = rgb_batch
        sigmas_all[start:end] = sigma_batch
        start = end

    rgbs = rgbs_all.view(height, width, depth_samples_per_ray, 3)
    sigmas = sigmas_all.view(height, width, depth_samples_per_ray)
    weights = compute_compositing_weights(sigmas, depth_values)
    rgb_predicted = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)
    depth_predicted = torch.sum(weights * depth_values, dim=-1)


    ##########################################################################
    # Student code ends here
    ##########################################################################

    return rgb_predicted, depth_predicted