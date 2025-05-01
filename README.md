# Saluki In Torch
[Saluki](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) is a foundational RNA sequence model. I'd argue that, as of March 2025, if you're building any RNA sequence based supervised model, you may want to benchmark with Saluki first. For a small model it's incredibly powerful and has performed comparably, and in some cases, beat the benchmarks of relatively large models (see [this](https://www.biorxiv.org/content/10.1101/2024.10.10.617658v1))

But unfortunately, [Saluki's original model](https://github.com/calico/basenji/tree/master/manuscripts/saluki)) is in Keras, which no shade for Keras users, but is kinda annoying to work with if you're using torch.

This repo has code to convert the keras version of [Saluki](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x)  to torch for inference as well as training.

# Getting Started

1. Download Saluki models from [here](https://zenodo.org/records/6326409)

2. Load keras weights into a dictionary
```commandline
from src.basenji_utils import get_weights

# model_file : path to model usually ends with .h5
# params_file: path to params file, usually params.json

layer_weights, params_model = get_weights(model_file, params_file)
```
3. Load Model
```commandline
model = SalukiFineTune(
         params=params_model,
         weights=layer_weights,
         exp_name="test_run",
         train_loader=None,
         val_loader=None,
         loss_fn="mse",
     )
```


