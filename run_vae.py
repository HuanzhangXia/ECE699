from torcheeg.datasets import SEEDIVDataset
from torcheeg import transforms
from torch.utils.data import random_split
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import  torch.nn.functional as F
import os
from  models import ConvVAE, ConvVAE_L

# def vae_loss(x, x_hat, mu, logvar):

#     recon_loss = F.mse_loss(x_hat, x, reduction='mean')
#     kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
#     return recon_loss + kl_div, recon_loss, kl_div
BETA_FINAL      = 4.0     # target β
WARMUP_STEPS    = 10_000  # steps to reach BETA_FINAL

def vae_loss(x, x_hat, mu, logvar, step):
    # print( (x_hat - x).abs().max())
    recon = F.mse_loss(x_hat, x, reduction='none')
    recon = recon.view(x.size(0), -1).mean(1)        # per-sample
    kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)

    beta_eff = BETA_FINAL * min(step / WARMUP_STEPS, 1.0)
    elbo = 100 * recon + beta_eff * kl
    return elbo.mean(), recon.mean(), kl.mean(), beta_eff


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs: int        = 10,
    device:  str       = "cuda",
    log_every: int     = 50,
    ckpt_dir: str      = "./checkpoints",   # where to save files
    ckpt_name: str     = "vae_final.pth",   # file name for last epoch
    save_best: bool    = True               # also keep best-val model
):
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = float("inf")

    history = {"train_loss": [], "val_loss": [], "train_recon": [], "val_recon": [], "train_kl": [], "val_kl": []}
    model.to(device)

    global_step = 0
    for epoch in range(1, epochs + 1):
        # ───────────── TRAIN ─────────────
        model.train()
        t_loss = t_recon = t_kl = 0.0

        for step, (x, label) in enumerate(train_loader, 1):
            # breakpoint()
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon, kl, beta  = vae_loss(x, x_hat, mu, logvar, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            t_loss  += loss.item()
            t_recon += recon.item()
            t_kl    += kl.item()

            if step % log_every == 0 or step == len(train_loader):
                print(f"[Train] Ep {epoch:02d} ({step:03d}/{len(train_loader)}) "
                      f"loss={t_loss/step:.4f}")

        train_loss  = t_loss  / len(train_loader)
        train_recon = t_recon / len(train_loader)
        train_kl    = t_kl    / len(train_loader)

        # ─────────── VALIDATION ───────────
        model.eval()
        v_loss = v_recon = v_kl = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, mu, logvar = model(x)
                loss, recon, kl, beta   = vae_loss(x, x_hat, mu, logvar, global_step)

                v_loss  += loss.item()
                v_recon += recon.item()
                v_kl    += kl.item()

        val_loss  = v_loss  / len(val_loader)
        val_recon = v_recon / len(val_loader)
        val_kl    = v_kl    / len(val_loader)

        # ──────────── log  &  save ────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_recon"].append(train_recon)
        history["val_recon"].append(val_recon)
        history["train_kl"].append(train_kl)
        history["val_kl"].append(val_kl)


        # record loss to a csv file
        with open(os.path.join(ckpt_dir, "loss_history.csv"), "a") as f:
            if epoch == 1:
                f.write("epoch,train_loss,val_loss,train_recon,val_recon,train_kl,val_kl\n")
            f.write(f"{epoch},{train_loss},{val_loss},{train_recon},{val_recon},{train_kl},{val_kl}\n")
        

        print(f"Epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"(recon {train_recon:.4f}/{val_recon:.4f}, kl {train_kl:.4f}/{val_kl:.4f})")

        # Save best model so far
        if save_best and val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val,
                },
                os.path.join(ckpt_dir, "vae_best.pth"),
            )
            print(f"  ✓ Saved new best model  (val_loss={best_val:.4f})")

    # ───── save final epoch weights ─────
    torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt_name))
    print(f"Finished training. Final weights saved to {os.path.join(ckpt_dir, ckpt_name)}")

    return history


if __name__ == '__main__':
    chunk_size = 50
    overlap = 0

    # dataset_train = SEEDIVDataset(
    #     root_path='/home/field/scratch/eeg_raw_data_train',
    #     io_path=f'/home/field/scratch/eeg_cache_train_{chunk_size}_{overlap}',  
    #     chunk_size=chunk_size,
    #     offline_transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.To2d()
    #     ]),
    #     overlap=overlap,
    #     online_transform=transforms.Compose([
    #         transforms.MeanStdNormalize(axis=(1, 2))
    #     ]),
    #     label_transform=transforms.Select('emotion'),
    #     num_worker=32
    # )

    # dataset_test = SEEDIVDataset(
    #     root_path='/home/field/scratch/eeg_raw_data_test',
    #     io_path=f'/home/field/scratch/eeg_cache_test_{chunk_size}_{overlap}',  
    #     chunk_size=chunk_size,
    #     offline_transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.To2d()
    #     ]),
    #     online_transform=transforms.Compose([
    #         transforms.MeanStdNormalize(axis=(1, 2))
    #     ]),
    #     label_transform=transforms.Select('emotion'),
    #     num_worker=32
    # )


    dataset = SEEDIVDataset(
        root_path='/home/field/scratch/eeg_raw_data',
        io_path=f'/home/field/scratch/eeg_cache_{chunk_size}_{overlap}',  
        chunk_size=chunk_size,
        offline_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.To2d()
        ]),
        online_transform=transforms.Compose([
            transforms.MeanStdNormalize(axis=(1, 2))
        ]),
        label_transform=transforms.Select('emotion'),
        num_worker=32
    )


    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    dataset_train, dataset_test = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    print("Number of training samples: {}".format(len(dataset_train)))
    print("Number of testing samples: {}".format(len(dataset_test)))

    batch_size=32
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_sample_shape = dataset_train[0][0].shape[1:]

    model = ConvVAE_L(input_shape=data_sample_shape, latent_dim=128).to(device)

    # ckpt = torch.load(f"./checkpoints_{chunk_size}_{overlap}/vae_best.pth", map_location=device)
    # model.load_state_dict(ckpt["model_state_dict"])
    # print('loaded')


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # total_loss=train(model, train_loader, test_loader, optimizer, epochs=5, device=device)

    hist = train(
        model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        epochs=10,
        device=device,
        ckpt_dir=f"./checkpoints_{chunk_size}_{overlap}",
        ckpt_name="vae_final.pth",
        save_best=True,
    )

    