from torch.nn import Module, Parameter, functional
from torch import fft
import torch
from torchvision.transforms.functional import crop


# ---- Activation functions ---- #


class Phase_ReLU(Module):
  def __init__(self):
    super().__init__() # init the base class

  def forward(self, input):
    phase = input.angle()
    functional.relu(phase, inplace=True)
    input = input.abs()
    input = input.type(torch.cfloat)
    return input.mul_(torch.complex(real=torch.sin(phase),
                                      imag=torch.cos(phase)))


# ---- Layers ---- #

class RealFFT2D(Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    fft_output = fft.rfft2(x, dim=(-2, -1))

    return fft_output


class RealIFFT2D(Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    ifft_output = fft.irfft2(x, dim=(-2, -1))

    return ifft_output


class FourierConv2D(Module):
  def __init__(self, in_channels: int, num_filters: int, height: int):
    super().__init__()
    self.num_filters = num_filters
    self.half_height = height//2

    # Initialize the trainable kernel and offset as parameters
    self.kernel = Parameter(torch.randn(height, self.half_height+1, in_channels, num_filters, dtype=torch.cfloat))

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    x.unsqueeze_(-2)  # [b,w,h,1,c]

    x = x.type(torch.cfloat)
    output = x @ self.kernel
    output.squeeze_(-2)

    output = output.permute(0, 3, 1, 2)

    return output


class FourierPooling(Module):  # Only keep a quarter of the image (half the dimensions)
  def __init__(self, image_size):
    super().__init__()
    self.crop_size = image_size//2
    self.half_crop_size = image_size//4

  # Only keep the center left part of the image
  def forward(self, x):
    return crop(x, top=self.half_crop_size,
                   left=0,
                   height=self.crop_size,
                   width=self.half_crop_size)
