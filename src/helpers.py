import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from model import ConvRegressionModel


# -------------------------------------------------------------
# 1. Data Ingestion & Sliding Window Extraction
# -------------------------------------------------------------
def load_rollforming_df(file_path, delimiter=None, decimal=None):
  """Loads the dataset while automatically detecting delimiters and decimal formats."""
  if delimiter is None or decimal is None:
    with open(file_path, 'r', encoding='utf-8') as f:
      header = f.readline()
    delimiter = ';' if ';' in header else ','
    decimal = ',' if delimiter == ';' else '.'

  df = pd.read_csv(file_path, delimiter=delimiter, decimal=decimal)
  df = df.apply(pd.to_numeric, errors='coerce').dropna()
  return df


def generate_F_Y_matrices(
    file_path,
    delimiter=None,
    decimal=None,
    window_size=10,
    num_label_columns=3,
    class_column='id',
    only_forces=False,
):
  """Extracts sliding window tensors: (N_samples, window_size, features)."""
  df = load_rollforming_df(file_path, delimiter=delimiter, decimal=decimal)

  # Fallback check for profile grouping column
  if class_column not in df.columns:
    if 'id' in df.columns:
      class_column = 'id'
    elif 'class' in df.columns:
      class_column = 'class'
    else:
      class_column = df.columns[-1]

  all_cols = list(df.columns)
  label_cols = [c for c in all_cols if c != class_column][-num_label_columns:]

  if only_forces:
    feature_cols = [c for c in all_cols if 'kraft' in c.lower()]
  else:
    feature_cols = [
        c for c in all_cols if c not in label_cols and c != class_column
    ]

  X_windows, Y_windows, C_windows = [], [], []
  for cls, group in df.groupby(class_column):
    features = group[feature_cols].values
    labels = group[label_cols].values
    n = len(group)
    if n <= window_size:
      continue
    for i in range(n - window_size):
      X_windows.append(features[i : i + window_size])
      Y_windows.append(labels[i + window_size - 1])
      C_windows.append(cls)

  F_matrix = np.stack(X_windows, axis=0).astype(np.float32)
  Y_matrix = (
      np.stack(Y_windows, axis=0)
      .astype(np.float32)
      .reshape(-1, num_label_columns)
  )
  C_array = np.array(C_windows)
  return F_matrix, Y_matrix, C_array


# -------------------------------------------------------------
# 2. Statewise Split (Zero-Shot Profile ID Partitioning)
# -------------------------------------------------------------
def split_data_statewise(
    file_path,
    window_size=10,
    test_ratio=0.2,
    val_ratio=0.1,
    seed=42,
    class_column='id',
    only_forces=False,
):
  """Performs an ID-level statewise split to ensure zero data leakage between splits.

  No time steps from test profiles will ever be seen during training or
  validation.
  """
  df = load_rollforming_df(file_path)
  if class_column not in df.columns:
    class_column = (
        'id'
        if 'id' in df.columns
        else ('class' if 'class' in df.columns else df.columns[-1])
    )

  unique_ids = df[class_column].unique()

  # Partition unique profile IDs
  train_val_ids, test_ids = train_test_split(
      unique_ids, test_size=test_ratio, random_state=seed
  )
  val_rel_size = val_ratio / (1.0 - test_ratio)
  train_ids, val_ids = train_test_split(
      train_val_ids, test_size=val_rel_size, random_state=seed
  )

  label_cols = ['X_opt - X_ist', 'Y_opt - Y_ist', 'Phi_opt - Phi_ist']
  if only_forces:
    feature_cols = [c for c in df.columns if 'kraft' in c.lower()]
  else:
    feature_cols = [
        c for c in df.columns if c not in label_cols and c != class_column
    ]

  def get_windows_from_ids(id_list):
    X_w, Y_w, C_w = [], [], []
    for pid in id_list:
      sub = df[df[class_column] == pid]
      feats = sub[feature_cols].values
      labs = sub[label_cols].values
      n = len(sub)
      if n <= window_size:
        continue
      for i in range(n - window_size):
        X_w.append(feats[i : i + window_size])
        Y_w.append(labs[i + window_size - 1])
        C_w.append(pid)
    return np.array(X_w, dtype=np.float32), np.array(
        Y_w, dtype=np.float32
    ), np.array(C_w)

  X_train, Y_train, C_train = get_windows_from_ids(train_ids)
  X_val, Y_val, C_val = get_windows_from_ids(val_ids)
  X_test, Y_test, C_test = get_windows_from_ids(test_ids)

  return (
      (X_train, Y_train, C_train),
      (X_val, Y_val, C_val),
      (X_test, Y_test, C_test),
  )


# -------------------------------------------------------------
# 3. Training Loop with Train-Only Scaler Fitting & Validation
# -------------------------------------------------------------
def set_seeds(seed):
  """Sets random seeds across libraries for strict reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def train_model_full(
    F_train,
    Y_train,
    seed,
    F_val=None,
    Y_val=None,
    epochs=25,
    batch_size=32,
    lr=0.001,
):
  """Trains the 1D-CNN regression model.

  Fits StandardScaler strictly on training samples and checkpoints model
  weights based on best validation loss.
  """
  set_seeds(seed)

  n_train = F_train.shape[0]
  num_features = F_train.shape[2]
  window_size = F_train.shape[1]

  # Standard scalers fitted strictly on training data
  scaler_X = StandardScaler().fit(F_train.reshape(n_train, -1))
  scaler_y = StandardScaler().fit(Y_train.reshape(n_train, -1))

  F_train_sc = scaler_X.transform(F_train.reshape(n_train, -1)).reshape(
      n_train, window_size, num_features
  )
  Y_train_sc = scaler_y.transform(Y_train.reshape(n_train, -1)).reshape(
      n_train, 3
  )

  train_dataset = TensorDataset(
      torch.tensor(F_train_sc, dtype=torch.float32),
      torch.tensor(Y_train_sc, dtype=torch.float32),
  )
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  # Validation loader setup if validation split is provided
  val_loader = None
  if F_val is not None and Y_val is not None:
    n_val = F_val.shape[0]
    F_val_sc = scaler_X.transform(F_val.reshape(n_val, -1)).reshape(
        n_val, window_size, num_features
    )
    Y_val_sc = scaler_y.transform(Y_val.reshape(n_val, -1)).reshape(n_val, 3)
    val_dataset = TensorDataset(
        torch.tensor(F_val_sc, dtype=torch.float32),
        torch.tensor(Y_val_sc, dtype=torch.float32),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = ConvRegressionModel(
      window_size=window_size, num_features=num_features
  ).to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
  criterion = nn.SmoothL1Loss()

  best_metric = float('inf')
  best_wts = None

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for Xb, Yb in train_loader:
      Xb, Yb = Xb.to(device), Yb.to(device)
      optimizer.zero_grad()
      preds = model(Xb)
      loss = criterion(preds, Yb)
      loss.backward()
      optimizer.step()
      running_loss += loss.item() * Xb.size(0)
    scheduler.step()

    # Track validation loss if available, otherwise track training loss
    if val_loader is not None:
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
        for Xb, Yb in val_loader:
          Xb, Yb = Xb.to(device), Yb.to(device)
          val_loss += criterion(model(Xb), Yb).item() * Xb.size(0)
      current_loss = val_loss / len(val_loader.dataset)
    else:
      current_loss = running_loss / len(train_loader.dataset)

    # Save best checkpoint
    if current_loss < best_metric:
      best_metric = current_loss
      best_wts = model.state_dict().copy()

  model.load_state_dict(best_wts)
  return model, scaler_X, scaler_y, device


# -------------------------------------------------------------
# 4. Model Evaluation & Inversion to Physical Units
# -------------------------------------------------------------
def evaluate_model(model, scaler_X, scaler_y, F_data, Y_data, device):
  """Evaluates the model on unseen profiles and computes MAE in physical units (mm, degrees)."""
  n = F_data.shape[0]
  window_size = F_data.shape[1]
  num_features = F_data.shape[2]

  F_scaled = scaler_X.transform(F_data.reshape(n, -1)).reshape(
      n, window_size, num_features
  )
  Y_scaled = scaler_y.transform(Y_data.reshape(n, -1)).reshape(n, 3)

  model.eval()
  with torch.no_grad():
    out = model(torch.tensor(F_scaled, dtype=torch.float32).to(device))

  # Invert scaling back to physical domain
  preds = scaler_y.inverse_transform(out.cpu().numpy())
  trues = scaler_y.inverse_transform(Y_scaled)
  mae = np.mean(np.abs(trues - preds), axis=0)
  return mae, trues, preds