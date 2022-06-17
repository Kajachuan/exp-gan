import argparse
import os
import torch
import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from dataset import MUSDB18Dataset
from model import Model

def train(network, train_loader, device, optimizer):
    batch_loss, count = 0, 0
    network.train()
    pbar = tqdm.tqdm(train_loader)
    for (x_real, x_imag), (y_real, y_imag) in pbar:
        pbar.set_description("Entrenando batch")
        x_real, x_imag = x_real.to(device, non_blocking=True), x_imag.to(device, non_blocking=True)
        y_real, y_imag = y_real.to(device, non_blocking=True), y_imag.to(device, non_blocking=True)
        
        optimizer.zero_grad()

        y_hat_real, y_hat_imag = network(x_real, x_imag)

        loss = mse_loss(y_hat_real, y_real) + mse_loss(y_hat_imag, y_imag)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item() * y_real.size(0)
        count += y_real.size(0)
    return batch_loss / count

def valid(network, valid_loader, device, stft):
    batch_loss, count = 0, 0
    network.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(valid_loader)
        for (x_real, x_imag), (y_real, y_imag) in pbar:
            pbar.set_description("Validando")
            x_real, x_imag = x_real.to(device, non_blocking=True), x_imag.to(device, non_blocking=True)
            y_real, y_imag = y_real.to(device, non_blocking=True), y_imag.to(device, non_blocking=True)

            y_hat_real, y_hat_imag = network(x_real, x_imag)

            loss = mse_loss(y_hat_real, y_real) + mse_loss(y_hat_imag, y_imag)
            batch_loss += loss.item() * y_real.size(0)
            count += y_real.size(0)
        return batch_loss / count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10, help="Tamaño del batch")
    parser.add_argument("--checkpoint", type=str, help="Directorio de los checkpoints")
    parser.add_argument("--duration", type=float, default=5.0, help="Duración de cada canción")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--layers", type=int, default=5, help="Número de capas")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Tasa de aprendizaje")
    parser.add_argument("--nfft", type=int, default=4096, help="Tamaño de la FFT del STFT")
    parser.add_argument("--output", type=str, help="Directorio de salida")
    parser.add_argument("--root", type=str, help="Ruta del dataset")
    parser.add_argument("--samples", type=int, default=1, help="Muestras por cancion")
    parser.add_argument("--weight-decay", type=float, default=0, help="Decaimiento de los pesos de Adam")
    parser.add_argument("--workers", type=int, default=0, help="Número de workers para cargar los datos")

    subparsers = parser.add_subparsers(help="Tipo de modelo", dest="model")

    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    use_cuda = torch.cuda.is_available()
    print("GPU disponible:", use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model_args = [args.layers, args.nfft // 2 + 1 ]
    network = Model(*model_args).to(device)

    train_dataset = MUSDB18Dataset(base_path=args.root, subset="train", split="train",
                                    duration=args.duration, samples=args.samples, random=True)
    valid_dataset = MUSDB18Dataset(base_path=args.root, subset="train", split="valid",
                                    duration=None, samples=1, random=False, partitions=args.partitions)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=args.workers, pin_memory=True)

    optimizer = Adam(network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)

    if args.checkpoint:
        state = torch.load(f"{args.checkpoint}/last_checkpoint.pt", map_location=device)
        network.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

        train_losses = state["train_losses"]
        valid_losses = state["valid_losses"]
        initial_epoch = state["epoch"] + 1
        best_loss = state["best_loss"]
    else:
        train_losses = []
        valid_losses = []
        initial_epoch = 1
        best_loss = float("inf")

    out_path = f"{args.output}"
    os.makedirs(out_path, exist_ok=True)

    t = tqdm.trange(initial_epoch, args.epochs + 1)
    for epoch in t:
        t.set_description("Entrenando iteración")
        train_loss = train(network, train_loader, device, optimizer)
        valid_loss = valid(network, valid_loader, device)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, valid_loss=valid_loss)

        state = {
            "args": model_args,
            "epoch": epoch,
            "best_loss": best_loss,
            "state_dict": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_losses": train_losses,
            "valid_losses": valid_losses
        }

        if valid_loss < best_loss:
            best_loss = valid_loss
            state["best_loss"] = best_loss
            torch.save(state, f"{out_path}/best_checkpoint.pt")
        torch.save(state, f"{out_path}/last_checkpoint.pt")

if __name__ == '__main__':
    main()