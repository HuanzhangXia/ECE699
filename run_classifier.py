from torcheeg.datasets import SEEDIVDataset
from torcheeg import transforms
from torch.utils.data import random_split
import torch
from torch.utils.data import DataLoader
import os
from models import ConvVAE, VAEEncoderClassifier, ConvVAE_L
import torch.nn as nn
from tqdm import tqdm

if __name__ == '__main__':
    chunk_size = 50
    overlap = 0

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

   

    train_size = int(0.9 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size



    dataset_train, dataset_val, dataset_test = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    # dataset_train = dataset_val

    # breakpoint()

    print("Number of training samples: {}".format(len(dataset_train)))
    print("Number of validation samples: {}".format(len(dataset_val)))
    print("Number of testing samples: {}".format(len(dataset_test)))




    batch_size=32
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_sample_shape = dataset_train[0][0].shape[1:]

    vae = ConvVAE_L(input_shape=data_sample_shape, latent_dim=128).to(device)
    ckpt = torch.load(f"./checkpoints_{chunk_size}_{overlap}/vae_best.pth", map_location=device)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()

    num_classes = 4  # e.g. SEED-IV emotions
    model = VAEEncoderClassifier(vae, num_classes, freeze_encoder=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
    #                             lr=3e-4, weight_decay=1e-4)


    # freeze nothing
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.mu_layer.parameters():
        p.requires_grad = True

    enc_params = list(model.encoder.parameters()) + list(model.mu_layer.parameters())
    head_params = model.classifier.parameters()

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params,  "lr": 1e-4},
            {"params": head_params, "lr": 3e-4},
        ],
        weight_decay=1e-4
    )

    ckpt_dir       = f"./classifier_ckpts_{chunk_size}_{overlap}_test_added"
    last_ckpt_path = os.path.join(ckpt_dir, "model_last.pth")
    best_ckpt_path = os.path.join(ckpt_dir, "model_best.pth")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ─────────── RESUME (if last checkpoint exists) ───────────
    start_epoch = 1
    best_val_loss = float("inf")
    if os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch     = ckpt["epoch"] + 1
        best_val_loss   = ckpt["best_val"]
        print(f"▶ Resumed from epoch {ckpt['epoch']}  (best_val_loss={best_val_loss:.4f})")


    # ─────────── TRAIN / VALIDATE LOOP ───────────
    epochs = 100
    for epoch in range(start_epoch, epochs + 1):
        # -------- TRAIN --------
        
        model.train()
        tr_total = tr_correct = 0
        tr_loss_sum = 0.0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss_sum += loss.item() * y.size(0)
            preds        = logits.argmax(1)
            tr_total    += y.size(0)
            tr_correct  += (preds == y).sum().item()

        train_loss = tr_loss_sum / tr_total
        train_acc  = 100.0 * tr_correct / tr_total

        # -------- VALIDATION --------
        model.eval()
        val_total = val_correct = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss   = criterion(logits, y)

                val_loss_sum += loss.item() * y.size(0)
                preds         = logits.argmax(1)
                val_total    += y.size(0)
                val_correct  += (preds == y).sum().item()

        val_loss = val_loss_sum / val_total
        val_acc  = 100.0 * val_correct / val_total

        # -------- REPORT --------
        print(f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:.2f}%")

        

        # -------- CHECKPOINTS --------
        # save best‐val model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "best_val": best_val_loss,
                },
                best_ckpt_path,
            )
            print("   ✓ saved new BEST checkpoint")

            # always save last epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "best_val": best_val_loss,
                },
                last_ckpt_path,
            )
        
        test_total = test_correct = 0
        test_loss_sum = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss   = criterion(logits, y)
                test_loss_sum += loss.item() * y.size(0)
                preds         = logits.argmax(1)
                test_total    += y.size(0)
                test_correct  += (preds == y).sum().item()

        test_loss = test_loss_sum / test_total
        test_acc  = 100.0 * test_correct / test_total
        print(f"test loss {test_loss:.4f} acc {test_acc:.2f}%") 


        # log to a csv file
        with open(os.path.join(ckpt_dir, "loss_history.csv"), "a") as f:
            if epoch == start_epoch:
                f.write("epoch,train_loss,val_loss,test_loss,train_acc,val_acc,test_acc\n")
            f.write(f"{epoch},{train_loss},{val_loss},{test_loss},{train_acc},{val_acc},{test_acc}\n")


    # -------- TEST --------
    # load best val model
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_total = test_correct = 0
    test_loss_sum = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            test_loss_sum += loss.item() * y.size(0)
            preds         = logits.argmax(1)
            test_total    += y.size(0)
            test_correct  += (preds == y).sum().item()

    test_loss = test_loss_sum / test_total
    test_acc  = 100.0 * test_correct / test_total
    print(f"test loss {test_loss:.4f} acc {test_acc:.2f}%") 