import os
import csv
import time
import argparse
import torch
from torch.utils.data import DataLoader
from einops import rearrange

from align.MAE_ViViT import MAE_ViViT
from data.CNS_data_utils import FNODatasetMultistep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,
                        default="data")
    parser.add_argument("--flnm", type=str,
                        default="2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5")
    parser.add_argument("--output-dir", type=str,
                        default="exps/mae")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--reduced-resolution", type=int, default=4)
    return parser.parse_args()


def plot_loss(csv_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs, losses = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, losses, linewidth=1.2)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("MAE ViViT Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Loss curve saved: {out_path}", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    train_dataset, _ = FNODatasetMultistep.get_train_test_datasets(
        args.flnm,
        reduced_resolution=args.reduced_resolution,
        reduced_resolution_t=1,
        reduced_batch=1,
        initial_step=0,
        saved_folder=args.data_path,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset size: {len(train_dataset)}, iters/epoch: {len(train_loader)}", flush=True)

    image_size = 512 // args.reduced_resolution
    model = MAE_ViViT(
        image_size=image_size,
        patch_size=8,
        emb_dim=256,
        encoder_layer=4,
        encoder_head=4,
        decoder_layer=2,
        decoder_head=4,
        mask_ratio=args.mask_ratio,
        in_out_channel=4,
        num_frames=4,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    csv_path = os.path.join(args.output_dir, "loss_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "lr"])

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t_epoch = time.time()

        for target, grid, raw_video in train_loader:
            raw_video = rearrange(raw_video, "B H W T C -> B T C H W").to(device)
            predicted_img, mask = model(raw_video)
            loss = ((predicted_img - raw_video) ** 2 * mask).sum() / (mask.sum() + 1e-8)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - t_epoch

        # log every epoch to CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.8f}", f"{current_lr:.2e}"])

        if epoch % 5 == 0:
            elapsed = time.time() - t_start
            remaining = elapsed / epoch * (args.epochs - epoch)
            print(f"Epoch [{epoch:4d}/{args.epochs}]  loss: {avg_loss:.6f}  "
                  f"time/epoch: {epoch_time:.1f}s  eta: {remaining/3600:.1f}h", flush=True)

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"vivit-M0.1-mask{args.mask_ratio}-mae_{epoch}.pt")
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}", flush=True)
            plot_loss(csv_path, os.path.join(args.output_dir, f"loss_curve_{epoch}.png"))

    final_path = os.path.join(args.output_dir, f"vivit-M0.1-mask{args.mask_ratio}-mae_final.pt")
    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs}, final_path)
    print(f"Done. Final checkpoint: {final_path}", flush=True)
    plot_loss(csv_path, os.path.join(args.output_dir, "loss_curve_final.png"))


if __name__ == "__main__":
    main()
