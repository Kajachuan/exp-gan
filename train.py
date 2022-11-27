import argparse, os, torch, tqdm
from accelerate import Accelerator
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from dataset import MUSDB18Dataset
from model import Model

def train(model, train_loader, optimizer, accelerator):
    model.train()
    pbar = tqdm.tqdm(train_loader)
    for x, y in pbar:
        pbar.set_description("Entrenando batch")
        
        optimizer.zero_grad()

        y_hat = model(x)

        loss = mse_loss(y_hat, y)
        accelerator.backward(loss)
        optimizer.step()

def valid(model, valid_loader, accelerator):
    batch_loss, count = 0, 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(valid_loader)
        for x, y in pbar:
            pbar.set_description("Validando")

            y_hat = model(x)

            y_hat, y = accelerator.gather_for_metrics((y_hat, y))

            loss = mse_loss(y_hat, y)
            accelerator.print(f"loss: ${loss}")
            batch_loss += loss.item() * y.size(0)
            count += y.size(0)
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
    parser.add_argument("--partitions", type=int, default=1, help="Número de partes de las canciones de validación")
    parser.add_argument("--patience", type=int, help="Cantidad máxima de iteraciones sin mejora")
    parser.add_argument("--root", type=str, help="Ruta del dataset")
    parser.add_argument("--samples", type=int, default=1, help="Muestras por cancion")
    parser.add_argument("--weight-decay", type=float, default=0, help="Decaimiento de los pesos de Adam")

    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    model_args = [args.layers, args.nfft]
    model = Model(*model_args).to(device)

    train_dataset = MUSDB18Dataset(root=args.root, is_wav=True, 
                                   subset="train", split="train", 
                                   duration=args.duration, nfft=args.nfft, 
                                   samples=args.samples, random=True)
    valid_dataset = MUSDB18Dataset(root=args.root, is_wav=True, 
                                   subset="train", split="valid", 
                                   duration=None, nfft=args.nfft, 
                                   samples=args.partitions, random=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience, verbose=True)

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, valid_loader, scheduler)
    accelerator.register_for_checkpointing(model, optimizer, scheduler)

    if args.checkpoint:
        accelerator.load_state(f"{args.checkpoint}/last_state")
        metadata = torch.load(f"{args.checkpoint}/metadata")
        valid_losses = metadata["valid_losses"]
        initial_epoch = metadata["epoch"] + 1
        best_loss = metadata["best_loss"]
    else:
        valid_losses = []
        initial_epoch = 1
        best_loss = float("inf")

    out_path = f"{args.output}"
    os.makedirs(out_path, exist_ok=True)

    t = tqdm.trange(initial_epoch, args.epochs + 1)
    for epoch in t:
        t.set_description("Entrenando iteración")
        
        train(model, train_loader, optimizer, accelerator)
        valid_loss = valid(model, valid_loader, accelerator)
        scheduler.step(valid_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(valid_loss=valid_loss)

        metadata = {
            "args": model_args,
            "epoch": epoch,
            "best_loss": best_loss,
            "valid_losses": valid_losses
        }

        accelerator.wait_for_everyone()

        if valid_loss < best_loss:
            best_loss = valid_loss
            metadata["best_loss"] = best_loss
            accelerator.save(accelerator.get_state_dict(model), f"{out_path}/model.pt")

        accelerator.save_state(f"{out_path}/last_checkpoint.pt")
        accelerator.save(metadata, f"{out_path}/metadata")

if __name__ == '__main__':
    main()