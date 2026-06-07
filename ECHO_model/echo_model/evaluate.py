import math

import numpy as np
import torch


def calc_nse(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    denom = np.sum((o - np.mean(o)) ** 2)
    if denom <= 0:
        return np.nan
    return 1.0 - np.sum((s - o) ** 2) / denom


def calc_kge(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    if np.std(o) <= 0 or np.mean(o) == 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1] if np.std(s) > 0 else np.nan
    alpha = np.std(s) / np.std(o) if np.std(o) > 0 else np.nan
    beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
    return 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def calc_r2(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    if np.std(o) == 0 or np.std(s) == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1] ** 2)


@torch.no_grad()
def rollout_prediction(model, x_eval, z_eval, device):
    model.eval()
    x = torch.from_numpy(np.swapaxes(x_eval, 1, 0)).float().to(device)
    z = torch.from_numpy(np.swapaxes(z_eval, 1, 0)).float().to(device)
    pred = model(x, z).detach().cpu().numpy()[:, :, 0].T
    return pred


def evaluate_model(model, dataset, device):
    pred_full = rollout_prediction(model, dataset["x_eval"], dataset["z_eval"], device)
    obs_test = dataset["obs_test"]
    pred_test = pred_full[:, -obs_test.shape[1] :]
    rows = []
    for i, basin_id in enumerate(dataset["basin_ids"]):
        obs = obs_test[i]
        sim = pred_test[i]
        rows.append(
            {
                "basin_id": str(basin_id),
                "NSE": calc_nse(obs, sim),
                "KGE": calc_kge(obs, sim),
                "R2": calc_r2(obs, sim),
            }
        )
    return rows, pred_test
