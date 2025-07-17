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

from typing import Optional

import torch.nn as nn
from torch import Tensor

__all__ = [
    "LogitLayer",
]


class LogitLayer(nn.Module):
    def __init__(
        self,
        logit_temperature: Optional[float] = None,
        logit_bias: Optional[float] = None,
    ):
        # note: T=None => T=1 and b=None => b=0
        super().__init__()
        self.logit_bias = logit_bias
        self.logit_temperature = logit_temperature

    def forward(self, in_tensor: Tensor) -> Tensor:
        if self.logit_temperature is not None:
            in_tensor = in_tensor / self.logit_temperature
        if self.logit_bias is not None:
            in_tensor = in_tensor + self.logit_bias
        return in_tensor

    def extra_repr(self) -> str:
        ret = ""
        if self.logit_temperature is not None:
            ret += f"logit_temperature={self.logit_temperature}, "
        if self.logit_bias is not None:
            ret += f"logit_bias={self.logit_bias}, "
        ret = ret[:-2]
        return ret
