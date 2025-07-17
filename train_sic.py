# @author Tom Nuno Wolf, Technical University of Munich
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import argparse
import torch
import numpy as np
from tqdm import tqdm
from dogs_dataset import get_dogs_dataloader
from functools import partial
from sic import SIC
from bcos import resnet50_long, BcosEncoderWrapper
from torch.nn.functional import one_hot
from pathlib import Path

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def default_to_one_hot(x, num_classes):
    targets = one_hot(x.long(), num_classes)
    return targets.float()

def get_optimizer(params, lr, epochs, n_iters_per_epoch):
    optimizer = torch.optim.AdamW(
        params=params, lr=lr, weight_decay=0.
    )
    n_iters_warmup = int(2 * n_iters_per_epoch)
    start_decay = int(0.4 * n_iters_per_epoch * epochs)
    step_counts = int(0.1 * n_iters_per_epoch * epochs)
    milestones = [n_iters_warmup, start_decay]

    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_iters_warmup)
    scheduler2 = torch.optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=start_decay,
    )
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_counts, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2, scheduler3],
                                                        milestones=milestones, verbose=True)
    print("Milestones....", milestones)
    return optimizer, scheduler

def default_step(model, x, y, to_one_hot, criterion):
    logits = model(x, y)
    if logits.size(-1) == 1:
        logits = logits.squeeze()
    preds = (logits > 0).float()
    targets = to_one_hot(y)

    loss = criterion(logits, targets)

    return loss, preds, targets

def default_val_step(model, x, y, to_one_hot, criterion):
    assert model.training == False
    logits = model.predict(x)  # must use model.predict() for inference!!!
    if logits.size(-1) == 1:
        logits = logits.squeeze()
    preds = torch.argmax(logits, dim=1)

    loss = criterion(logits, to_one_hot(y))

    return loss, preds, y

def main(args=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--n_way", type=int, default=None, help="Reduces number of classes sampled for training on small GPU memory to n_way classes")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save the results into")
    args = parser.parse_args(args=args)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    n_classes = 120
    train_loader, val_loader, support_loader = get_dogs_dataloader(args.data_dir, batch_size=args.batch_size, num_workers=8, is_bcos=True)

    # initialize featurizer
    featurizer = BcosEncoderWrapper(resnet50_long(pretrained=True))

    # initialize model
    sic = SIC(
        featurizer=featurizer,
        n_classes=n_classes,
        proj_dim=128,
        n_way=args.n_way,  # use if number of classes is too large to sample from all classes during training
        n_shot=3,
        temperature=10,
        support_loader=support_loader,
        device=device,
    )
    # sanity check
    sic.eval()
    sic.precompute()

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=None)  # add pos_weight if required
    to_one_hot = partial(default_to_one_hot, num_classes=n_classes)
    optimizer, scheduler = get_optimizer([p for p in sic.parameters() if p.requires_grad], lr=0.001, epochs=args.epochs, n_iters_per_epoch=len(train_loader))

    for epoch in range(args.epochs):
    
        # TRAIN
        total_loss = 0
        sic.train()
        for (x, y) in tqdm(train_loader):
            optimizer.zero_grad()
            loss, preds, targets = default_step(sic, x.to(device), y.to(device), to_one_hot, criterion)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} train loss: {total_loss / len(train_loader)}")

        # VALIDATE
        total_loss = 0

        ##########
        # validation routine.
        #
        # Must first set to eval, followed by precomputing support vectors.
        # Then is ready for evaluation
        ##########
        sic.eval()
        sic.precompute()
        all_preds = []
        all_targets = []
        for (x, y) in tqdm(val_loader):
            loss, preds, targets = default_val_step(sic, x.to(device), y.to(device), to_one_hot, criterion)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            total_loss += loss.item()
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        # compute foreground classes accuracy only
        accuracy = (all_preds == all_targets).mean() * 100
        print(f"Epoch {epoch} val loss: {total_loss / len(val_loader)}, val accuracy: {accuracy:.2f}%")

    torch.save(sic.state_dict(), Path(args.results_dir) / "sic_dogs.pth")


if __name__ == "__main__":
    main()
