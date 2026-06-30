import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn  # <-- AGGIUNTO

from .model import ConditionalDiffusionModel, ConditionalDropOutDiffusionModel
from .gaussian_diffusion import GaussianDiffusion
from .dataset import load_and_split_data, load_and_split_data_history


def train(
        zarr_path="trajectory_log.zarr",
        num_epochs=150,
        batch_size=1024,
        lr=1e-4,
        num_timesteps=500,
        device="cuda",
        save_path="diffusion_model.pt",
        train_ratio=0.8,
        val_ratio=0.1,
        log_dir="runs/diffusion",
        hidden_dim=512,
        time_dim=256,
        num_blocks=6,
        eta_min_factor=10,
        dropout_rate=0.1,
        retroaction=False,
        trial=None,
):
    print(f"Training on device: {device}")

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    if retroaction:
        train_dataset, val_dataset, test_dataset, num_steps, num_joints = (
            load_and_split_data_history(zarr_path, train_ratio=train_ratio, val_ratio=val_ratio)
        )
    else:
        train_dataset, val_dataset, test_dataset, num_steps, num_joints = (
            load_and_split_data(zarr_path, train_ratio=train_ratio, val_ratio=val_ratio)
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    delta_dim = train_dataset.delta.shape[1]
    if retroaction:
        condition_dim = delta_dim + (num_joints * 3) # 6 (goal XYZ RPY) - 12 (joint state) - 12 (previous joint state) - 12 (previous action)
    else:
        condition_dim = delta_dim + num_joints

    if dropout_rate > 0:
        model = ConditionalDropOutDiffusionModel(
            num_steps=num_steps,
            num_joints=num_joints,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate
        ).to(device)
    else:
        model = ConditionalDiffusionModel(
            num_steps=num_steps,
            num_joints=num_joints,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_blocks=num_blocks,
        ).to(device)

    ema_avg_fn = get_ema_multi_avg_fn(decay=0.999)
    ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)

    diffusion = GaussianDiffusion(num_timesteps=num_timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr / eta_min_factor,
    )
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    total_batches = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            delta = batch["delta"].to(device, non_blocking=True)
            current_joints = batch["current_joints"].to(device, non_blocking=True)
            trajectory = batch["trajectory"].to(device, non_blocking=True)

            if retroaction:
                prev_joints = batch["prev_joints"].to(device, non_blocking=True)
                prev_actions = batch["prev_actions"].to(device, non_blocking=True)
                condition = torch.cat([delta, current_joints, prev_joints, prev_actions], dim=-1)
            else:
                condition = torch.cat([delta, current_joints], dim=-1)
            t = torch.randint(0, num_timesteps, (trajectory.shape[0],), device=device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = diffusion.p_losses(model, trajectory, t, condition)

            if torch.isnan(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            ema_model.update_parameters(model)

            total_loss += loss.item()
            num_batches += 1

            pct = (batch_idx + 1) / total_batches * 100
            print(
                f"\rEpoch {epoch + 1}/{num_epochs} | "
                f"Batch {batch_idx + 1}/{total_batches} ({pct:.1f}%) | "
                f"Loss: {loss.item():.6f}",
                end="",
            )

        train_loss = total_loss / max(num_batches, 1)
        print()

        # Validation
        model.eval()
        ema_model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for batch in val_loader:
                delta = batch["delta"].to(device, non_blocking=True)
                current_joints = batch["current_joints"].to(device, non_blocking=True)
                trajectory = batch["trajectory"].to(device, non_blocking=True)
                if retroaction:
                    prev_joints = batch["prev_joints"].to(device, non_blocking=True)
                    prev_actions = batch["prev_actions"].to(device, non_blocking=True)
                    condition = torch.cat([delta, current_joints, prev_joints, prev_actions], dim=-1)
                else:
                    condition = torch.cat([delta, current_joints], dim=-1)
                t = torch.randint(0, num_timesteps, (trajectory.shape[0],), device=device)

                loss = diffusion.p_losses(ema_model, trajectory, t, condition)
                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / max(val_batches, 1)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                writer.close()
                raise __import__("optuna").TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": ema_model.module.state_dict(),
                "traj_mean": train_dataset.traj_mean,
                "traj_std": train_dataset.traj_std,
                "delta_mean": train_dataset.delta_mean,
                "delta_std": train_dataset.delta_std,
                "dropout": dropout_rate,
                "retroaction": retroaction,
                "num_steps": num_steps,
                "num_joints": num_joints,
                "num_timesteps": num_timesteps,
                "hidden_dim": hidden_dim,
                "time_dim": time_dim,
                "condition_dim": condition_dim,
                "num_blocks": num_blocks,
            }, save_path)

        scheduler.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    # Test evaluation
    ema_model.eval()
    test_loss = 0
    test_batches = 0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in test_loader:
            delta = batch["delta"].to(device, non_blocking=True)
            current_joints = batch["current_joints"].to(device, non_blocking=True)
            trajectory = batch["trajectory"].to(device, non_blocking=True)

            if retroaction:
                prev_joints = batch["prev_joints"].to(device, non_blocking=True)
                prev_actions = batch["prev_actions"].to(device, non_blocking=True)
                condition = torch.cat([delta, current_joints, prev_joints, prev_actions], dim=-1)
            else:
                condition = torch.cat([delta, current_joints], dim=-1)
            t = torch.randint(0, num_timesteps, (trajectory.shape[0],), device=device)

            loss = diffusion.p_losses(ema_model, trajectory, t, condition)
            test_loss += loss.item()
            test_batches += 1

    test_loss = test_loss / max(test_batches, 1)
    writer.add_scalar("Loss/test", test_loss, num_epochs)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Best model saved to {save_path}")

    writer.close()

    return ema_model.module, diffusion, train_dataset, best_val_loss