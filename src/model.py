import torch
import torch.nn as nn


class ConvRegressionModel(nn.Module):

  def __init__(self, window_size=10, num_features=8):
    """1D-CNN regression model for geometric deviation prediction in roll forming.

    Args:
        window_size (int): Temporal window length (treated as input channels in
          Conv1d).
        num_features (int): Number of sensory features/channels per time step.
    """
    super().__init__()
    self.window_size = window_size
    self.num_features = num_features

    # Convolutional blocks
    self.conv1 = nn.Conv1d(window_size, 288, kernel_size=5, padding=2)
    self.pool1 = nn.MaxPool1d(2)
    self.conv2 = nn.Conv1d(288, 32, kernel_size=5, padding=2)
    self.pool2 = nn.MaxPool1d(2)
    self.conv3 = nn.Conv1d(32, 448, kernel_size=5, padding=2)
    self.pool3 = nn.MaxPool1d(2)

    # Dynamic calculation of flattened dimension to prevent shape mismatches
    with torch.no_grad():
      dummy = torch.zeros(1, window_size, num_features)
      d = self.pool1(self.conv1(dummy))
      d = self.pool2(self.conv2(d))
      d = self.pool3(self.conv3(d))
      in_fc = d.view(1, -1).size(1)

    # Dense layers (MLP trunk)
    self.fc1 = nn.Linear(in_fc, 256)
    self.fc2 = nn.Linear(256, 512)
    self.fc3 = nn.Linear(512, 160)
    self.fc4 = nn.Linear(160, 64)

    # Independent regression heads
    self.head_x = nn.Linear(64, 1)
    self.head_y = nn.Linear(64, 1)
    self.head_phi = nn.Linear(64, 1)
    self.relu = nn.ReLU()

  def forward(self, x):
    # Feature extraction via 1D convolutions and max-pooling
    x = self.relu(self.conv1(x))
    x = self.pool1(x)
    x = self.relu(self.conv2(x))
    x = self.pool2(x)
    x = self.relu(self.conv3(x))
    x = self.pool3(x)

    # Flatten representation
    x = x.view(x.size(0), -1)

    # Multi-layer perceptron
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.relu(self.fc4(x))

    # Multi-head output concatenation: [Delta_X, Delta_Y, Delta_Phi]
    out_x = self.head_x(x)
    out_y = self.head_y(x)
    out_phi = self.head_phi(x)
    return torch.cat((out_x, out_y, out_phi), dim=1)