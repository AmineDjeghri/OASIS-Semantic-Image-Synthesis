import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

class DotDict(dict):
    """dot.notation access to dictionary attributes (Thomas Robert)"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_yaml(path):
    with open(path, 'r') as stream:
        opt = yaml.load(stream,Loader=yaml.Loader)
    opt = DotDict(opt)
    for key in opt:
        if isinstance(opt[key], dict):
            opt[key] = DotDict(opt[key])
    return DotDict(opt)

def one_hot_encode(tensor, N):
    """
    tensor: torch.LongTensor, (bs, H, W) shape
    output: binary torch.LongTensor, (bs, N, H, W) shape
    """
    # actions = actions.view(-1).to(torch.long)
    bs, h, w = tensor.shape
    one_hot = torch.zeros((bs, N, h, w), dtype=torch.long, device=tensor.device)
    one_hot[torch.arange(bs).repeat_interleave(h*w),
            tensor.flatten(),
            torch.arange(h).repeat_interleave(w).repeat(bs),
            torch.arange(w).repeat(h).repeat(bs)] = 1
    return one_hot

def one_hot_decode(tensor):
    """
    tensor: binary torch.LongTensor, (bs, N, H, W) shape
    output: torch.LongTensor, (bs, H, W) shape
    """

    bs, N, h, w = tensor.shape
    indices = torch.arange(N, device=tensor.device)[None, :, None, None].expand((bs, N, h,w))
    decoded = (indices * tensor).sum(1)
    return decoded

def get_n_params(Network, dimensions=True):
    """
    Returns the number of parameters in an instance of nn.Module
    """
    n_params = 0
    sizes = {}
    for param_tensor in Network.state_dict():
        sizes[param_tensor] = Network.state_dict()[param_tensor].size()
        n_params += Network.state_dict()[param_tensor].numel()
    if dimensions:
        return n_params, sizes
    return n_params

def compute_class_weights(loader, C, H, W, path_to_class_weights=None):
    """
    loader provifing (_, targets) of shape: (N, h, w)
    C: the number of classes
    """
    freqs = torch.zeros((C, H*W))
    for _, target in loader:
        freq_i = torch.stack(
            [(target.flatten(start_dim=1).t() == c).to(torch.float).mean(-1) for c in range(C)]
        )
        freqs += freq_i
    freqs = torch.true_divide(freqs, len(loader))

    numel = H * W
    return numel / freqs.sum(-1)

def get_weights(class_weights, device, opt=None):
    if isinstance(class_weights, str):
        class_weights = torch.load(open(opt.data.class_weights, "rb"))
    elif not isinstance(class_weights, torch.Tensor):
        raise TypeError(f"`class_weights` must be a path towards saved tensor, or a tensor")
    # Add 0 weight for the new fake class
    return torch.cat([class_weights, torch.tensor([0])], 0).to(device)

def labelmix(x1, x2, mask):
    mask = mask[:, None, :, :].expand(x1.shape)
    return x1 * mask + x2 * (~mask)

def sample_mask(labelmap):
    """
    labelmap: size (bs, C, h, w)
    """
    classes = torch.unique(labelmap)
    flips = torch.rand(classes.size(0)) > 0.5
    pos_classes = classes[flips]
    mask = torch.zeros(labelmap.shape, device=labelmap.device).to(bool)
    for c in pos_classes:
        mask = torch.logical_or(mask, labelmap == c)
    return mask

def color_map(labelmap):
    random_colorized = torch.cat([
        (torch.tanh(labelmap.to(float)) + 1) / 2 * 255,
        (torch.sin(labelmap.to(float)) + 1) / 2 * 255,
        (torch.cos(labelmap.to(float)) + 1) / 2 * 255
    ], 1).to(int)
    return random_colorized

def visualize_generation(examples, annot, img, i=None, tb=None):
    examples_grid = torchvision.utils.make_grid((examples + 1) / 2, nrow=examples.size(0)) # 3 images per row
    img_grid = torchvision.utils.make_grid((img + 1 ) / 2, nrow=examples.size(0)) # 3 images per row
    labels_grid = torchvision.utils.make_grid(color_map(annot), nrow=examples.size(0)) # 10 images per row
    
    fig, ax = plt.subplots(nrows=3, figsize=(15,15), gridspec_kw={"wspace":0.2}) 
    ax[0].imshow(np.transpose(examples_grid, (1,2,0)))
    ax[1].imshow(np.transpose(labels_grid, (1,2,0)))
    ax[2].imshow(np.transpose(img_grid, (1,2,0)))
    if tb:
        tb.add_figure('Generated', fig, i)
    else:
        plt.show()

class ExponentialMovingAverage():
    def __init__(self, parameters, decay=0.9999):
        self.parameters = parameters
        self.decay = decay

    def update(self, module):
        with torch.no_grad():
            for ptarget, pold in zip(module.parameters(), self.parameters):
                if ptarget.requires_grad:
                    new_val = self.decay * pold.data + (1 - self.decay) * ptarget.data
                    ptarget.copy_(new_val)