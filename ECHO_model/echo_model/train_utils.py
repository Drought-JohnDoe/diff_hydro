import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_index(ngrid: int, nt: int, batch_size: int, rho: int, bufftime: int = 0):
    i_grid = np.random.randint(0, ngrid, [batch_size])
    i_t = np.random.randint(bufftime, max(bufftime + 1, nt - rho), [batch_size])
    return i_grid, i_t


def select_subset(x, i_grid, i_t, rho, *, c=None, bufftime: int = 0, device=None):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(i_grid):
        i_grid = np.arange(0, len(i_grid))
    if nt <= rho:
        i_t.fill(0)
    batch_size = i_grid.shape[0]
    x_tensor = torch.zeros([rho + bufftime, batch_size, nx], dtype=torch.float32)
    for k in range(batch_size):
        temp = x[i_grid[k] : i_grid[k] + 1, np.arange(i_t[k] - bufftime, i_t[k] + rho), :]
        x_tensor[:, k : k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(np.reshape(c[i_grid, :], [batch_size, 1, nc]), rho + bufftime, axis=1)
        c_tensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()
        x_tensor = torch.cat((x_tensor, c_tensor), 2)
    if device is not None:
        x_tensor = x_tensor.to(device)
    return x_tensor


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, payload) -> None:
    Path(path).write_text(json.dumps(payload, indent=2))


def get_device(use_gpu: bool = True, gpu_id: int = 0) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")
