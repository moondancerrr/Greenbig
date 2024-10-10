#  BigBro for Token Classification - Splicing prediction

## Note:

Due to GitHub's space limitations, the dataset used in this project could not be included in this repository. You can generate the dataset from Greenbigdata repo.

##  Overview

This project is designed to train and fine-tune a deep learning model, BigBro (a variant of BigBird with RoFormer positional encoding), for token classification tasks across 115 species of plants. The primary aim is to develop a machine learning model that can accurately classify each token of gene sequences based on species-specific features and gene annotations, using transformer-based architectures. The code utilizes PyTorch Lightning for efficient model training and management of experiments, along with Hugging Face Transformers to handle tokenization and the transformer model. It is a model that is aware of the evolutionary context and that predicts how RNA sequences are spliced with the assumption that they belong to a given genome.

##  Key Features:

-  Species-Specific Sequence Classification: A transformer-based model is trained to classify genomic sequences for a wide range of plant species, with token embeddings that correspond to specific species.
-  Large-Scale Transformer Model: Uses BigBird architecture for handling long sequences, modified with RoFormer positional encodings for improved efficiency.
-  Custom Tokenizer and Collation: Includes a tokenizer and data loader to process plant genomic data with species-specific information.
-  Efficient Training: Built with PyTorch Lightning to manage multi-GPU distributed training, checkpointing, and logging for large datasets.

##  Prerequisites

To use this code, you will need the following installed:

-  Python 3.x
-  PyTorch (with GPU support)
-  PyTorch Lightning
-  Hugging Face Transformers

Required datasets (gene sequences and calls) - prepared prior

##  Libraries

-  gzip: For working with compressed files.
-  torch: PyTorch framework for deep learning models.
-  transformers: Hugging Face library for the BigBird and tokenization process.
-  pytorch_lightning: For high-level training abstractions.
-  bigbro: Custom model definition for BigBro.
-  json, os, sys: Standard libraries for file operations and system interaction.
