import os
import math
import torch
from tqdm import tqdm

def train(model, optim, start_step, save_steps, start_epoch, epochs, train_loader, val_loader, prev_val_loss, save_dir, device):
    train_loss, val_loss = [], []
    for e in range(start_epoch, epochs):
        train_loss.append(train_step(model, train_loader, val_loader, optim, e, 
                                    start_step if e == start_epoch else 0, device, save_steps, save_dir))
        val_loss = val_step(model, val_loader, e, device)
        if val_loss < prev_val_loss:
            save_model(save_dir, model, optim, e + 1, 0, val_loss)
            prev_val_loss = val_loss


def train_step(model, dataloader, val_loader, optimizer, epoch, step, device, save_steps, save_dir):
    model.train()
    running_loss = 0.0
    with tqdm(desc='Train: Epoch %d' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, x in enumerate(dataloader):
            if it < step:
                continue
            optimizer.zero_grad()
            loss, _ = model(x['input_ids'].to(device), masked_lm_labels=x['masked_lm_labels'].to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            if (it + 1) % save_steps == 0:
                val_loss = val_step(model, val_loader, epoch, device)
                save_model(save_dir, model, optimizer, epoch, it, val_loss)
                model.train()
    return running_loss / len(dataloader)

def val_step(model, dataloader, epoch, device):
    model.eval()
    running_loss = 0.0
    with tqdm(desc='Val: Epoch %d' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, x in enumerate(dataloader):
            x = x.to(device)
            loss, _ = model(x['input_ids'].to(device), masked_lm_labels=x['masked_lm_labels'].to(device))
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
    return running_loss / len(dataloader)

def save_model(dir, model, optim, epoch, step, val_loss):
    to_save = {
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optim.state_dict(),
        'val_loss': val_loss,
        'epoch': epoch,
        'step': step
    }
    torch.save(to_save, os.path.join(dir, f'tabformer_epoch{epoch}_step{step}.pth'))

def load_model(path, model, optim):
    if os.path.exists(path):
        saved = torch.load(path)
        model.load_state_dict(saved['model_state_dict'])
        optim.load_state_dict(saved['optim_state_dict'])
        val_loss = saved['val_loss']
        epoch = saved['epoch']
        step = saved['step']
    else:
        epoch = 0
        step = 0
        val_loss = math.inf
    return model, optim, epoch, step, val_loss