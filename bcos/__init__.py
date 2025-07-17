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

from bcos.bcoslinear import BcosLinear, BcosReLU
from bcos.logitlayer import LogitLayer
from bcos.utils import BcosUtilMixin, DetachableModule, BcosUtilMixin
from bcos.convnext import *
from bcos.densenet import *
from bcos.resnet import *
from bcos.vgg import *
from bcos.pretrained_imagenet import *
