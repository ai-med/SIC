# @author Tom Nuno Wolf, Technical University of Munich
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

from pathlib import Path
import pandas as pd
import yaml
from typing import Optional, List
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt
import torch
import sys
from tqdm import tqdm
import numpy as np
import math
from dogs_dataset import get_dogs_dataloader
from bcos import resnet50_long, BcosEncoderWrapper
from sic import SIC

def plot_prediction(model, test_img, pred, correct_clf, percentile=99.9, smooth=9):
    fig_size_x = (model.n_shot + 2)
    fig, axs = plt.subplots(2, fig_size_x, figsize=(fig_size_x, 2), constrained_layout=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.01, right=0.99, top=0.95, bottom=0.05)

    axs[0, 0].imshow(test_img.permute(1,2,0)[:,:,:3])
    axs[0, 0].get_xaxis().set_visible(False)
    axs[0, 0].get_yaxis().set_visible(False)
    axs[1, 0].get_xaxis().set_visible(False)
    axs[1, 0].get_yaxis().set_visible(False)
    model.plot_calibration_scores(test_img.unsqueeze(0).to(model.device), pred, axs[0, 1], color="tab:blue" if correct_clf else "tab:red", y_probs=False)

    for i in range(2, model.n_shot+2):
        pidx = model.n_shot * pred + i - 2
        expl_out = model.explain_prototype(pidx, None, smooth=smooth, alpha_percentile=percentile)
        axs[1, i].imshow(expl_out["explanation"])
        axs[1, i].get_xaxis().set_visible(False)
        axs[1, i].get_yaxis().set_visible(False)
        axs[0, i].imshow(expl_out["image"][:3].transpose(1, 2, 0))
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].get_yaxis().set_visible(False)
    expl_out = model.explain_prediction(test_img.unsqueeze(0).to(model.device), pred, alpha_percentile=percentile, smooth=smooth)
    axs[1, 1].imshow(expl_out["explanation"])
    axs[1, 1].get_xaxis().set_visible(False)
    axs[1, 1].get_yaxis().set_visible(False)
    for ax in axs.flat:
        ax.set_box_aspect(1)
    plt.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.2)
    return fig, axs


def visualize_prototypes(model, percentile: float = 99.9, smooth: int = 9):
    model.eval()
    n_protos = model.n_shot * model.n_classes
    protos_per_row = 15
    rows_per_proto = 3
    n_rows = math.ceil(n_protos / protos_per_row) * rows_per_proto
    n_cols = min(n_protos, protos_per_row)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), constrained_layout=True)
    if n_rows == 1 and n_cols == 1:
        axs = [[axs]]
    elif n_rows == 1:
        axs = [axs]
    elif n_cols == 1:
        axs = [[ax] for ax in axs]
    used_subplots = 0
    for pidx in range(n_protos):
        row_base = (pidx // protos_per_row) * rows_per_proto
        col = pidx % protos_per_row
        expl_out = model.explain_prototype(pidx, None, smooth=smooth, alpha_percentile=percentile)
        model.plot_contribution_map(
            expl_out["contribution_map"].cpu().detach().numpy().squeeze(),
            axs[row_base + 1][col],
            percentile=None
        )
        axs[row_base + 2][col].get_xaxis().set_visible(False)
        axs[row_base + 2][col].get_yaxis().set_visible(False)
        axs[row_base + 2][col].imshow(expl_out["explanation"])
        used_subplots += 1
        axs[row_base + 1][col].get_xaxis().set_visible(False)
        axs[row_base + 1][col].get_yaxis().set_visible(False)
        axs[row_base][col].imshow(expl_out["image"][:3].transpose(1, 2, 0))
        axs[row_base][col].get_xaxis().set_visible(False)
        axs[row_base][col].get_yaxis().set_visible(False)
        used_subplots += 2
    total_subplots = n_rows * n_cols
    for idx in range(used_subplots, total_subplots):
        row = idx // n_cols
        col = idx % n_cols
        axs[row][col].set_visible(False)
    plt.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.05)
    return fig, axs

    
def main(args=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--outdir", type=str, default="visualizations", help="Path to store the visualizations in experiment directory.")
    args = parser.parse_args(args=args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    n_classes = 120
    train_loader, val_loader, support_loader = get_dogs_dataloader(args.data_dir, batch_size=args.batch_size, num_workers=8, is_bcos=True)

    # initialize featurizer
    featurizer = BcosEncoderWrapper(
        resnet50_long(pretrained=True))

    # initialize model
    sic = SIC(
        featurizer=featurizer,
        n_classes=n_classes,
        proj_dim=128,
        n_way=None,
        n_shot=3,
        temperature=10,
        support_loader=support_loader,
        device=device,
    )
    sic.load_state_dict(torch.load(args.checkpoint, map_location=device))
    sic.eval()
    sic.precompute(support_loader)


    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # Visualize Prototypes
    fig, _ = visualize_prototypes(sic, smooth=9)
    fig.savefig(outdir / "prototypes.pdf", dpi=300)
    plt.close()

  
    # Load some sample from validation set
    x, y = val_loader.dataset.__getitem__(0)
    x = x.unsqueeze(0).to(sic.device)

    with torch.no_grad():
        logits = sic.predict(x)
    pred = logits.argmax(dim=1).item()
    correct_clf = y == pred

    print(f"Predicted class: {pred}, Correct class: {y}, Correct: {correct_clf}")
    fig, _ = plot_prediction(sic, x.squeeze(0).cpu().detach(), pred, correct_clf=correct_clf, \
                            percentile=99.9, smooth=9)

    fig.savefig(outdir / "example_prediction.pdf", dpi=300)
    plt.close()


if __name__=='__main__':
    main()
