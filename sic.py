# @author Tom Nuno Wolf, Technical University of Munich
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
import warnings

from bcos import BcosUtilMixin, BcosLinear, BcosReLU, LogitLayer, DetachableModule
from nw import SupportSetTrain, compute_clusters


class BCosSimilarity(DetachableModule):

    def __init__(self, b: int=2):
        super(BCosSimilarity, self).__init__()
        self.b = b

    def forward(self, x, y):
        w = y / (LA.vector_norm(y, dim=-1, keepdim=True) + 1e-12)
        out = torch.bmm(x, w.transpose(-2, -1))
        if self.b == 1:
            return out

        norm = LA.vector_norm(x, dim=-1, keepdim=True) + 1e-12

        maybe_detached_out = out
        if self.detach:
            maybe_detached_out = out.detach()
            norm = norm.detach()

        if self.b == 2:
            dynamic_scaling = maybe_detached_out.abs() / norm
        else:
            abs_cos = (maybe_detached_out / norm).abs() + 1e-6
            dynamic_scaling = abs_cos.pow(self.b - 1)

        out = dynamic_scaling * out  # |cos|^(B-1) (ŵ·x)
        return out


class SIC(BcosUtilMixin, nn.Module):
    def __init__(self,
                featurizer,
                n_classes,
                support_loader=None,
                feat_dim=None,
                proj_dim=0,
                n_way=None,
                n_shot=1,
                temperature=None,
                device=None,
                **args):
        super().__init__()
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_shot = n_shot
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.device = device

        # Initialize featurizer
        if proj_dim > 0:
            featurizer = nn.Sequential(featurizer,
                BcosLinear(featurizer.n_filters_out if ((feat_dim is None) and (proj_dim > 0)) else feat_dim, proj_dim, bias=False, b=2, max_out=1),
                BcosReLU())
        else:
            featurizer = nn.Sequential(featurizer, BcosReLU())
        self.featurizer = featurizer.to(self.device)
        # Initialize similarity measure
        self.kernel = BCosSimilarity(b=2)
        # Initialize head
        self.nwhead = NWHead(kernel=self.kernel, n_classes=n_classes)
        # Initialize logit layer
        self.logit_layer = LogitLayer(logit_temperature=self.temperature, logit_bias=-math.log(max(self.n_classes - 1, 2)))

        # Initialize support set
        self.support_train = SupportSetTrain(support_loader.dataset, self.n_classes, "random", self.n_shot, n_way=self.n_way)  # used for random sampling
        self.supports = support_loader  # used for computing support vectors for inference


    @torch.no_grad()
    def precompute(self, dataset=None):
        assert not self.featurizer.training
        self._compute_all_support_feats(dataset=dataset)


    def predict(self, x, return_support_scores=False):
        qfeat = self.featurizer(x)
        sfeat, sy = self.cluster_feat, self.cluster_y
        sfeat, sy = sfeat.to(x.device), sy.to(x.device)

        if return_support_scores:
            logits, support_scores = self.nwhead(qfeat, sfeat, sy, return_support_scores=return_support_scores)
            logits = self.logit_layer(logits)
            return logits, support_scores
        else:
            logits = self.nwhead(qfeat, sfeat, sy, return_support_scores=return_support_scores)
            logits = self.logit_layer(logits)
            return logits
    

    def compute_log_contribution(self, odds, support_odds):
        total_logodds = odds
        divisor = self.logit_layer.logit_temperature if self.logit_layer.logit_temperature is not None else 1
        support_logodds = [x / divisor for x in support_odds]
        return total_logodds, support_logodds, self.logit_layer.logit_bias


    def plot_calibration_scores(self, x, cls_idx, ax, color="tab:blue", y_probs=False, font_size=6):

        odds, support_probs = self.predict(x, return_support_scores=True)
        odds, support_probs = odds.squeeze(), support_probs.squeeze()
        odds, support_probs = odds[cls_idx].item(), support_probs[cls_idx * self.n_shot: (cls_idx + 1) * self.n_shot].tolist()
        odds, support_probs, bias_prob = self.compute_log_contribution(odds, support_probs)

        y_vals = [odds] + support_probs + [bias_prob]
        x_range = [r'$\mathbf{\mu}$'] + [str(int(x)) for x in range(cls_idx * self.n_shot, (cls_idx + 1) * self.n_shot)] + [r'$\mathbf{b}$']

        ax.bar(x_range, y_vals, color=color, width=0.7)
        if y_probs:
            ax.set_ylim(0, 1)
        ax.yaxis.tick_right()
        ax.tick_params(axis='y', direction='in', pad=-4, labelsize=font_size)
        ax.tick_params(axis='x', direction='in', pad=-15, labelsize=font_size, rotation=45)

        yticks = ax.get_yticks()[:-1]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{tick:.1f}" for tick in yticks], ha='right')

        y_min, y_max = ax.get_ylim()
        margin = 0.3 * (y_max - y_min)
        ax.set_ylim(y_min - margin, y_max)
        x_min, x_max = ax.get_xlim()
        margin = 0.35 * (x_max - x_min)
        ax.set_xlim(x_min, x_max + margin)
        ax.margins(y=0.0, x=0.0)
        return ax


    def forward(self, x, y, support_data=None):
        assert x.min() >= 0 and x.max() <= 1, "Input tensor must be normalized to [0, 1]"
        # param support_data: Only used during training
        if support_data is not None:
            sx, sy = support_data
        else:
            sx, sy = self.support_train.get_support(y)

        sx, sy = sx.to(x.device), sy.to(x.device)
    
        batch_size = len(x)
        inputs = torch.cat((x, sx), dim=0)
        feats = self.featurizer(inputs)
        query_feats, support_feats = feats[:batch_size], feats[batch_size:]

        logits = self.nwhead(query_feats, support_feats, sy)
        logits = self.logit_layer(logits)
        return logits


    def _compute_all_support_feats(self, dataset=None):
        feats = []
        targets = []
        for (img, label) in tqdm(self.supports, total=len(self.supports)):
            feat = self.featurizer(img.to(self.device))
            feats.append(feat.cpu().detach())
            targets.append(label.cpu().detach())

        feats = torch.cat(feats, dim=0)
        targets = torch.cat(targets, dim=0)
        self.cluster_feat, self.cluster_y, self.cluster_indices = \
            compute_clusters(feats, targets, self.n_shot)


    def sanity_checks_bcos(self, in_tensor=None):
        if in_tensor is not None:
            if in_tensor.ndim == 3:
                raise ValueError("Expected 4-dimensional input tensor")
            if in_tensor.shape[0] != 1:
                raise ValueError("Expected batch size of 1")
            if not in_tensor.requires_grad:
                warnings.warn(
                    "Input tensor did not require grad! Has been set automatically to True!"
                )
                in_tensor.requires_grad = True  # nonsense otherwise
        if self.training:
            warnings.warn(
                "Model is in training mode! "
                "This might lead to unexpected results! Use model.eval()!"
            )


    def explain_prototype(
            self,
            proto_idx,
            feature_idx=None,
            **grad2img_kwargs,
    ) -> Dict[str, Any]:
        """
        Generates an explanation for the given input tensor and a latent feature at index idx.

        :param proto_idx: Index of the prototype to explain
        :param feature_idx: Index of the feature to explain. If None, the similarity score is explained.
        grad2img_kwargs : Any
            Additional keyword arguments passed to `gradient_to_image` method
            for generating the explanation.

        """
        self.sanity_checks_bcos()

        result = dict()
        with torch.enable_grad(), self.explanation_mode():
            sfeat, sy, simg = self.cluster_feat[proto_idx], self.cluster_y[proto_idx], self.supports.dataset.__getitem__(self.cluster_indices[proto_idx])[0]
            sfeat, sy, simg = sfeat.unsqueeze(0).to(self.device), sy.unsqueeze(0).to(self.device), simg.unsqueeze(0).to(self.device)
            
            simg.requires_grad = True
            out = self.featurizer(simg)
            out, sfeat, sy = self.nwhead.forward_feats(out, sfeat, sy)
            assert out.ndim == 3
            assert out.shape[1] == 1 and out.shape[0] == 1

            result["proto_index"] = proto_idx

            if feature_idx is None:  # propagate until after similarity measure
                to_be_explained_feat = self.nwhead.kernel(out, sfeat)
            else:
                # select output (logit) to explain
                to_be_explained_feat = out[0, 0, feature_idx]
            to_be_explained_feat.backward(inputs=[simg])

        # get weights and contribution map
        result["dynamic_linear_weights"] = simg.grad
        result["contribution_map"] = (simg * simg.grad).sum(1)
        result["image"] = simg[0].cpu().detach().numpy()

        # generate (color) explanation
        result["explanation"] = self.gradient_to_image(
            simg[0], simg.grad[0], **grad2img_kwargs
        )
        return result


    def explain_feature(
        self,
        in_tensor,
        idx,
        after_kernel,
        **grad2img_kwargs,
    ) -> "Dict[str, Any]":
        """
        Generates an explanation for the given input tensor and a latent feature at index idx.

        after_kernel : bool
            If True, the explanation is generated after the kernel operation, explaining the similarity score.
            Results in one image for each prototype.
        grad2img_kwargs : Any
            Additional keyword arguments passed to `gradient_to_image` method
            for generating the explanation.

        """
        self.sanity_checks_bcos()

        result = dict()
        with torch.enable_grad(), self.explanation_mode():
            # fwd + prediction
            out = self.featurizer(in_tensor)
            sfeat, sy = self.cluster_feat, self.cluster_y
            sfeat, sy = sfeat.unsqueeze(0).to(self.device), sy.unsqueeze(0).to(self.device)
            out, sfeat, sy = self.nwhead.forward_feats(out, sfeat, sy)

            assert out.ndim == 3
            assert out.shape[1] == 1 and out.shape[0] == 1

            if after_kernel:  # propagate until after similarity measure
                to_be_explained_feat = self.nwhead.kernel(out, sfeat)
                # select feature in terms of prototype similarity score
                to_be_explained_feat = to_be_explained_feat[0, 0, idx]
            else:
                # select feature in terms of feature index in latent space
                to_be_explained_feat = out[0, 0, idx]
            to_be_explained_feat.backward(inputs=[in_tensor])

        # get weights and contribution map
        result["dynamic_linear_weights"] = in_tensor.grad
        result["contribution_map"] = (in_tensor * in_tensor.grad).sum(1)
        result["image"] = in_tensor[0].cpu().detach().numpy()

        # generate (color) explanation
        result["explanation"] = self.gradient_to_image(
            in_tensor[0], in_tensor.grad[0], **grad2img_kwargs
        )
        return result


    def explain_prediction(
        self,
        in_tensor,
        idx,
        **grad2img_kwargs,
    ) -> "Dict[str, Any]":
        """
        Generates an explanation for the given input tensor and the class with index idx.

        grad2img_kwargs : Any
            Additional keyword arguments passed to `gradient_to_image` method
            for generating the explanation.
        """
        self.sanity_checks_bcos(in_tensor)

        result = dict()
        with torch.enable_grad(), self.explanation_mode():
            out = self.predict(in_tensor)
            assert out.ndim == 2
            assert out.shape[0] == 1
            to_be_explained_out = out[0, idx]
            to_be_explained_out.backward(inputs=[in_tensor])

        # get weights and contribution map
        result["dynamic_linear_weights"] = in_tensor.grad
        result["contribution_map"] = (in_tensor * in_tensor.grad).sum(1)
        result["image"] = in_tensor[0].cpu().detach().numpy()

        # generate (color) explanation
        result["explanation"] = self.gradient_to_image(
            in_tensor[0], in_tensor.grad[0], **grad2img_kwargs
        )
        return result


class NWHead(BcosUtilMixin, nn.Module):
    def __init__(self, 
                 kernel, 
                 n_classes,):
        super(NWHead, self).__init__()
        self.kernel = kernel
        self.n_classes = n_classes

    def forward_feats(self, x, sx, sy):
        batch_size = len(x)
        sy = F.one_hot(sy, self.n_classes).float()
        if len(sx.shape) == len(x.shape):
            sx = sx[None].expand(batch_size, *sx.shape)
            sy = sy[None].expand(batch_size, *sy.shape)

        x = x.unsqueeze(1)
        return x, sx, sy

    def forward(self, x, sx, sy, return_support_scores=False):
        x, sx, sy = self.forward_feats(x, sx, sy)
        scores = self.kernel(x, sx)
 
        # (B, num_queries, num_keys) x (B, num_keys=num_vals, num_classes) -> (B, num_queries, num_classes)
        output = torch.bmm(scores, sy)
        output = output.squeeze(1)
        if return_support_scores:
            return output, scores
        else:
            return output
