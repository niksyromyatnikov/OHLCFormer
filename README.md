<p align="center">
    <a href="#">
        <img alt="Codacy grade" src="https://app.codacy.com/project/badge/Grade/c91af6e4013a4adba31c2a3c23b102a0">
    </a>
    <a href="#">
        <img alt="GitHub License" src="https://img.shields.io/github/license/niksyromyatnikov/OHLCFormer">
    </a>
    <a href="#">
       <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/niksyromyatnikov/OHLCFormer">
    </a>
    <a href="#">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/niksyromyatnikov/OHLCFormer?style=social">
    </a>
    <a href="#">
       <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/niksyromyatnikov/OHLCFormer?style=social">
    </a>
</p>

<h3 align="center">
    <p>Neural networks training and evaluation tool for open-high-low-close (OHLC) data forecasting.</p>
</h3>
OHLCFormer provides an easy-to-use API for model prototyping, training, and evaluation to perform open-high-low-close (OHLC) data forecasting tasks.

## Getting started
You can find here a list of the official notebooks.

<table>
  <tr>
    <td style="text-align: center;">Notebook</td>
    <td style="text-align: center;">Description</td>
    <td></td>
  </tr>
  <tr>
    <td><a href="https://github.com/niksyromyatnikov/OHLCFormer/blob/master/examples/notebooks/data_processing.ipynb">Data processing</a></td>
    <td>How to preprocess your data and build a dataset.</td>
    <td><a href="https://colab.research.google.com/github/niksyromyatnikov/OHLCFormer/blob/master/examples/notebooks/data_processing.ipynb"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/niksyromyatnikov/OHLCFormer/blob/master/examples/notebooks/train_model.ipynb">Model training</a></td>
    <td>How to set up and train a model on your dataset.</td>
    <td><a href="https://colab.research.google.com/github/niksyromyatnikov/OHLCFormer/blob/master/examples/notebooks/train_model.ipynb"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a></td>
  </tr>
<tr>
    <td><a href="https://github.com/niksyromyatnikov/OHLCFormer/blob/master/examples/notebooks/models_evaluation.ipynb">Models evaluation</a></td>
    <td>How to benchmark models with OHLCFormer.</td>
    <td><a href="https://colab.research.google.com/github/niksyromyatnikov/OHLCFormer/blob/master/examples/notebooks/models_evaluation.ipynb"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a></td>
  </tr>
</table>

## Model architectures

OHLCFormer currently provides the following architectures:
1. **FNet** (from Google Research) released with the paper [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824) by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon.
2. **BERT** (from Google) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.