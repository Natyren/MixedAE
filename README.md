# MixedAE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

## Overview

MixedAE is a implementation of the Mixed Autoencoder model, designed to push the boundaries of self-supervised representation learning for Vision Transformers (ViT). Combining the strengths of autoencoding and transformer-based architectures, MixedAE is built with both research and practical applications in mind.

This repository empowers you to explore new ideas and leverage state-of-the-art techniques in computer vision and deep learning.

**Featured Paper:** [MixedAE: A Novel Approach to Autoencoding with Vision Transformers](https://arxiv.org/pdf/2303.17152.pdf)

## Getting Started

### Installation

Ensure you have Python 3.7+ installed. To install MixedAE, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Natyren/MixedAE.git
cd MixedAE
pip install -r requirements.txt
```

Alternatively, install the dependencies manually:

```bash
pip install torch>=2.0.0 numpy>=1.26.1
```

### Quick Usage

Here's a simple example to get you started:

```python
from mixedae import MixedAutoencoderViT

model = MixedAutoencoderViT()
# Add your model training and inference code here.
```

## Current Features

- [x] Core model implementation

## Roadmap

We are continuously working to enhance MixedAE. Future improvements include:

- Expanding and optimizing the pretraining module.
- Integrating advanced fine-tuning techniques.
- Performance benchmarking and extensive experimentation.
- Enhanced documentation and tutorials.
- Fostering community contributions and collaborations.

## Contributing

Contributions are welcome! Please fork the repository and submit your pull requests. For major changes, open an issue first to discuss your ideas.

For detailed guidelines, refer to our contributing guide (coming soon).

## Citation

If you use MixedAE in your research, please cite the following paper:

MixedAE: [ArXiv](https://arxiv.org/pdf/2303.17152.pdf)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to contact:

George Bredis - georgy.bredis@gmail.com

Enjoy exploring MixedAE!

```python
from mixedae import MixedAutoencoderViT
model = MixedAutoencoderViT()
```

## Current status

- [x] model code

