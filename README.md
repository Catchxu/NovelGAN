# NovelGAN: Novel cell type detection and integration of multiple datasets via generate adversarial networks
We propose a Generative Adversarial Networks called NovelGAN, which can detect novel cell type (or malignant cells) from
known cells, and remove the batch effects among multiple batches on single-cell datasets.

## Dependencies
- anndata==0.8.0
- numpy==1.21.6
- scanpy==1.9.3
- loguru==0.6.0
- scipy==1.9.3
- scikit-learn==1.2.0
- pandas==1.5.2
- torch==1.13.1

## Tested environment
- CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
- CPU Memory: 256 GB
- GPU: NVIDIA GeForce RTX 3090
- GPU Memory: 24 GB
- System: Ubuntu 20.04.5 LTS
- Python: 3.9.15

## Notes
This repository has been abandoned. Please go to https://github.com/Catchxu/ODBC-GAN. ODBC-GAN is based on NovelGAN, but
it's more powerful. 
