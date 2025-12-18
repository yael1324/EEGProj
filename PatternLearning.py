# -*- coding: utf-8 -*-
"""
Stage 3 – Pattern Learning (Autoencoder Training)
Goal:
Train an Autoencoder model to learn EEG feature patterns and compute
the reconstruction error per window. This error will later serve
as a proxy for "understanding level" (low error = likely understanding).
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# input/output
INPUT_CSV = r"C:\_Davidson\projectFiles\eeg files\results\1009_1009_2023-04-21_07-40-21_band_features.csv"
OUTPUT_DIR = r"C:\_Davidson\projectFiles\eeg files\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# model parameters
EPOCHS = 100              # maximum training epochs (passes through all data a hundred times)
BATCH_SIZE = 32           # number of samples per batch. the model improves itself after each batch
LEARNING_RATE = 1e-3      # learning rate for the optimizer
PATIENCE = 10             # early stopping patience (epochs without improvement)


# Load and prepare the data
def load_features(csv_path):
    """
    Load EEG feature CSV and prepare the matrix for Pattern Learning (Autoencoder).

    n_samples = number of EEG windows (epochs after preprocessing)
    n_features = number of selected feature columns (Engagement index, alpha, etc.)

    Returns:
        X_train, X_validation, feature_names, scaler
        (X is a matrix)
    """
    df = pd.read_csv(csv_path)

    # Select numeric columns only (mynautoencoder cant handle different types of data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove columns that are not actual features (like time for example)
    feature_cols = [c for c in numeric_cols if c not in ["time_s"]]

    X = df[feature_cols].values
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")

    # Standardize features: mean 0, std 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split 80% train / 20% validation
    # 80% gives the model enough data to learn patterns,
    # 20% is held aside to evaluate progress and apply early stopping.
    X_train, X_validation = train_test_split(X_scaled, test_size=0.2, random_state=42, shuffle=True)
    return X_train, X_validation, feature_cols, scaler


# ====== 2. Define the Autoencoder structure ======
class PatternLearner(nn.Module):
    def __init__(self, input_dim):
        super(PatternLearner, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)  # latent space (compressed pattern representation)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ====== 3. Train the model with Early Stopping ======
def train_pattern_learner(X_train, X_validation, input_dim):
    """
    Train the Autoencoder (Pattern Learner) and apply early stopping
    based on validation loss to avoid overfitting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PatternLearner(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_validation_t = torch.tensor(X_validation, dtype=torch.float32).to(device)

    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, X_train_t)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_validation_t)
            val_loss = criterion(val_outputs, X_validation_t)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

        # Early stopping logic
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping activated (no validation improvement).")
                break

    # Restore the best model version
    model.load_state_dict(best_model_state)
    return model


# ====== 4. Compute reconstruction errors ======
def compute_reconstruction_error(model, X, scaler):
    """
    Compute reconstruction error for each EEG window (sample).

    Returns:
        numpy array with reconstruction errors [n_samples]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        X_reconstructed = model(X_t).cpu().numpy()

    # Mean squared error per sample
    errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    return errors


# ====== 5. Main function ======
def main():
    X_train, X_validation, feature_cols, scaler = load_features(INPUT_CSV)
    input_dim = len(feature_cols)

    model = train_pattern_learner(X_train, X_validation, input_dim)

    # Compute reconstruction errors for the full dataset
    df = pd.read_csv(INPUT_CSV)
    X_full = df[feature_cols].values
    errors = compute_reconstruction_error(model, X_full, scaler)

    df["reconstruction_error"] = errors
    out_path = os.path.join(OUTPUT_DIR, "pattern_learning_with_errors.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nSaved file with reconstruction errors → {out_path}")


if __name__ == "__main__":
    main()