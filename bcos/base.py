# Copyright 2023 Moritz Böhle, Max-Planck-Gesellschaft
# Copyright 2025 Tom Nuno Wolf, Technical University of Munich
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original: https://github.com/B-cos/B-cos-v2 
# Modified by: Tom Nuno Wolf 2025


from abc import ABCMeta, abstractmethod
import torch.nn as nn
from torch import Tensor
from typing import Sequence

# TODO: this is fine for bcos models. For GOAT models, I'll have to override the forward method!
class BCosBase(nn.Module, metaclass=ABCMeta):
    """Abstract base class for models that can be executed by
    :class:`daft.training.train_and_eval.ModelRunner`.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_base(x)
        return self.forward_clf(x)

    @abstractmethod
    def forward_base(self, x: Tensor) -> Tensor:
        """return features extracted by a feature extractor"""
        raise NotImplemented

    @abstractmethod
    def forward_clf(self, x: Tensor) -> Tensor:
        """transforms features into logits for classification"""
        raise NotImplemented