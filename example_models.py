from torch.nn import Conv2d, Sequential, Flatten, Linear, Dropout, ReLU
import torch
from real_layers import FourierConv2D, FourierPooling, PhaseReLU


class SpatialModel(Sequential):
    def __init__(self):
        super().__init__(
          Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
	  ReLU(),
          Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
          Flatten(),
          Linear(in_features=18432, out_features=128),
          Dropout(),
          Linear(in_features=128, out_features=2),
        )


class FourierModel(Sequential):
    def __init__(self):
        super().__init__(
          FourierConv2D(in_channels=1, num_filters=32, height=28),
	  PhaseReLU(),
          FourierConv2D(in_channels=32, num_filters=32, height=28),
          FourierPooling(image_size=28),
          Flatten(),
          Linear(in_features=3136, out_features=128, dtype=torch.cfloat),
          Dropout(),
          Linear(in_features=128, out_features=2, dtype=torch.cfloat),
        )