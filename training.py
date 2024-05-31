import torch
import torchvision
import torch.nn as nn
from configs import Config
from data_prep import train_data, valid_data
from unet_model import hikari_segmenter_model
from tqdm.auto import tqdm # type: ignore
from pathlib import Path


optimizer = torch.optim.Adam(hikari_segmenter_model.parameters(), lr=Config.lr)

def get_weights_file_path(epoch: str):
    model_folder = 'model_weights'
    model_file = Config.model_name
    model_filename = f'{model_file}_{epoch}.pth'
    
    return str(Path('.')/model_folder/model_filename)

# Training loop

def training_loop(config, model, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    initial_epoch = 0
    step = 0

    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(initial_epoch, config.epochs)):

        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}")

        print(f"Training epoch {epoch}....")

        for batch in batch_iterator:
            image, mask = batch

            image = image.to(device)
            mask = mask.to(device)
            
            loss = criterion()
            
            batch_iterator.set_postfix({f"loss": f"{loss.item():.2f}"})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

        print(f"Epoch {epoch} complete ✅")

        checkpoint_name = get_weights_file_path(f'{epoch:02d}')
        
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "step": step,
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_name
        )

    torch.save(model.state_dict(), Config.model_output_path)
    

training_loop(Config, hikari_segmenter_model, train_data)