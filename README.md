# Saluki In Torch
[Saluki](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) is a foundational RNA sequence model. I'd argue that, as of March 2025, if you're building any RNA sequence based supervised model, you may want to benchmark with Saluki first. For a small model it's incredibly powerful and has performed comparably, and in some cases, beat the benchmarks of relatively large models (see [this](https://www.biorxiv.org/content/10.1101/2024.10.10.617658v1))

But unfortunately, [Saluki's original model](https://github.com/vagarwal87/saluki_paper) is in Keras, which no shade for Keras users, but is kinda annoying to work with if you're using torch.

This repo has code to convert the keras version of [Saluki](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x)  to torch for inference as well as training.



