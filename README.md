# SIC: Similarity-Based Interpretable Image Classification

[![Conference Paper](https://img.shields.io/static/v1?label=DOI&message=ToBeReleasedAtICCV25&color=3a7ebb)](https://doi.org/)
[![Preprint](https://img.shields.io/badge/arXiv-2501.17328-b31b1b)](https://arxiv.org/abs/2501.17328)
[![License](https://img.shields.io/badge/license-Apache%20License%202.0-blue)](LICENSE)


This is the official implementation of SIC for interpretable image classification with neural networks.

If you use this code, please cite:

```
@article{wolf2025sic,
  title={SIC: Similarity-Based Interpretable Image Classification with Neural Networks},
  author={Wolf, Tom Nuno and Kavak, Emre and Bongratz, Fabian and Wachinger, Christian},
  journal={arXiv preprint arXiv:2501.17328},
  year={2025}
}
```

## Setup Instructions

### Prerequisites

Before getting started, ensure you have conda installed and initialized. If not, follow the installation instructions at:
- [Miniconda Installation Guide](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)

### Installation and Data Preparation

1. **Install dependencies**
   ```bash
   bash setup/setup.sh

   conda activate sic
   ```
   This will install the conda environment and all required pip dependencies.

2. **Download the dataset**
   
   Follow the instructions at [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) to download the `Images` and `Lists` directories. Place them in any directory of your choice.

3. **Prepare the dataset**
   ```bash
   python dogs_dataset.py --data_dir=PATH/TO/YOUR/DIR
   ```

## Training

To train a model, run:
```bash
python train_sic.py --data_dir=PATH/TO/YOUR/DIR
```

For a complete list of available arguments and options:
```bash
python train_sic.py --help
```

## Visualization

For exemplary visualization of prototypes and the forward pass of a test sample, run (see `--help` for more arguments):
```bash
python vis_sic.py --data_dir=PATH/TO/YOUR/DIR --checkpoint=PATH/TO/YOUR/.PTH
```

## License and Acknowledgments

This project is licensed under the Apache License 2.0. See the LICENSE file for complete details.

This project includes code from multiple sources:
- (Modified) code from the [B-cos-v2](https://github.com/B-cos/B-cos-v2) repository (Apache License 2.0)
- (Modified) code from the [Nadaraya-Watson Head](https://github.com/alanqrwang/nwhead) repository (no license specified)
