import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import evaluate_model
from .train_utils import ensure_dir, get_device, random_index, seed_everything, select_subset


def train_model(
    model,
    dataset,
    loss_fun,
    *,
    epochs=10,
    batch_size=16,
    rho=365,
    bufftime=365,
    max_iter_ep=20,
    save_every=1,
    out_dir="outputs/train_run",
    seed=111111,
    use_gpu=True,
    gpu_id=0,
    adadelta_rho=0.9,
    learning_rate=1.0,
):
    out_dir = ensure_dir(out_dir)
    seed_everything(seed)
    device = get_device(use_gpu=use_gpu, gpu_id=gpu_id)
    model = model.to(device)
    loss_fun = loss_fun.to(device)
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    z_train = dataset["z_train"]
    attrs = dataset["attrs"]
    ngrid, nt, _ = x_train.shape
    batch_size = min(batch_size, ngrid)
    optimizer = torch.optim.Adadelta(model.parameters(), rho=adadelta_rho, lr=learning_rate)
    run_path = out_dir / "run.csv"
    with run_path.open("w") as fp:
        fp.write("epoch,loss\n")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for step in range(max_iter_ep):
            i_grid, i_t = random_index(ngrid, nt, batch_size, rho, bufftime=bufftime)
            x_batch = select_subset(x_train, i_grid, i_t, rho, bufftime=bufftime, device=device)
            z_batch = select_subset(z_train, i_grid, i_t, rho, c=attrs, bufftime=bufftime, device=device)
            y_batch = select_subset(y_train, i_grid, i_t, rho, bufftime=0, device=device)
            optimizer.zero_grad()
            y_pred = model(x_batch, z_batch)
            loss = loss_fun(y_pred, y_batch)
            aux_loss = model.get_auxiliary_loss()
            if aux_loss is not None:
                loss = loss + aux_loss
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            print(f"epoch={epoch:03d} step={step + 1:03d}/{max_iter_ep:03d} loss={loss_value:.6f}")
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": mean_loss})
        with run_path.open("a") as fp:
            fp.write(f"{epoch},{mean_loss:.8f}\n")
        ckpt = out_dir / f"model_Ep{epoch}.pt"
        torch.save(model.state_dict(), ckpt)
        if epoch % save_every == 0:
            rows, _ = evaluate_model(model, dataset, device)
            metrics_path = out_dir / f"metrics_ep{epoch}.json"
            metrics_path.write_text(json.dumps(rows[: min(10, len(rows))], indent=2))
    summary_rows, pred = evaluate_model(model, dataset, device)
    (out_dir / "final_metrics.json").write_text(json.dumps(summary_rows, indent=2))
    np.save(out_dir / "final_test_prediction.npy", pred)
    return history
