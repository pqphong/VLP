import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset_multitask import VLPMultiHeadDataset
from models_multitask import MultiHeadMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = VLPMultiHeadDataset("X_vlp.npy", "y_vlp.npy")
loader = DataLoader(dataset, batch_size=64)

model = MultiHeadMLP(dataset.X.shape[1]).to(device)
model.load_state_dict(torch.load("multitask_model.pth"))
model.eval()

pred_xy, true_xy = [], []
pred_h, true_h = [], []
pred_r, true_r = [], []

with torch.no_grad():
    for X, y_xy, y_h, y_r in loader:
        X = X.to(device)
        p_xy, p_h, p_r = model(X)

        pred_xy.append(p_xy.cpu().numpy())
        true_xy.append(y_xy.numpy())

        pred_h.append(p_h.cpu().numpy())
        true_h.append(y_h.numpy())

        pred_r.append(p_r.cpu().numpy())
        true_r.append(y_r.numpy())

pred_xy = np.vstack(pred_xy)
true_xy = np.vstack(true_xy)

pred_h = np.vstack(pred_h)
true_h = np.vstack(true_h)

pred_r = np.vstack(pred_r)
true_r = np.vstack(true_r)

# ================= ERROR HISTOGRAM =================
def plot_error_hist(pred, true, title):
    error = (pred - true).flatten()
    plt.figure(figsize=(6,4))
    plt.hist(error, bins=30, edgecolor='black')
    plt.title(title)
    plt.xlabel("Prediction Error")
    plt.ylabel("Samples")
    plt.grid(True)
    plt.show()

plot_error_hist(pred_h, true_h, "Height Estimation Error Histogram")
plot_error_hist(pred_r, true_r, "Radius Estimation Error Histogram")

# ================= REGRESSION PLOT =================
def plot_regression(pred, true, title):
    plt.figure(figsize=(5,5))
    plt.scatter(true, pred, s=5)
    plt.plot([0,1], [0,1], 'r--')
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_regression(pred_h, true_h, "Height Regression Plot")
plot_regression(pred_r, true_r, "Radius Regression Plot")
