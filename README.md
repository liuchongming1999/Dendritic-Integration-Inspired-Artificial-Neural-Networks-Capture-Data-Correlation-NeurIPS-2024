# Dendritic Integration Inspired Artificial Neural Networks Capture Data Correlation (NeurIPS 2024)

This repository is the official implementation of [Dendritic Integration Inspired Artificial Neural Networks Capture Data Correlation](https://arxiv.org/abs/2030.12345). 

![](/img/interpretation.jpeg)
**Figure:** **A.** Experiments confirmed the quadratic integration rule under general
cases, along with a comprehensive theoretical framework for single neuron computation (From [Li et al. 2023](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.22020)). **B.**
An illustration of the biological interpretation of our Dit-CNNs.

Our Dit-CNN is inspired by neural networks in the visual system. For example, different types of cone cells encode various color (channel) information, and retinal ganglion cells receive inputs from multiple types of cone cells, the responses can be modeled as having receptive fields (convolutional kernels) related to different color channels ( $w_1 * x_1, w_2 * x_2, w_3 * x_3$ ). When multiple channel inputs are present, traditional CNNs simply linearly sum the corresponding responses. In contrast, neurons integrate these inputs with an additional quadratic term based on the dendritic bilinear integration rule. This approach leads to the formulation of our Dit-CNN after simplification. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

### CIFAR
To train the models on CIFAR as described in the paper, run the following command:

```train
python cifar10.py --model dit_resnet20
```

### ImageNet-1K
>📋  For details on configuring data and training popular models on ImageNet-1K, refer [here](https://github.com/liuchongming1999/ImageNet-1K-training-and-validation).

After configuring the data, run the following commands to integrate dit_convnext into the timm library:
```train
mv quadratic.py .../env/lib/python3.10/site-packages/timm/layers
mv convnext.py .../env/lib/python3.10/site-packages/timm/models
```
Then train Dit-ConvNeXt using the following command (with multiple GPUs):
```train
torchrun --nproc_per_node=8 train.py data_path -b 64 --model convnext_tiny --amp --resplit --weight-decay 0.08 --sched cosine --lr 0.006 --epochs 300 --warmup-epochs 20 --opt adamw --aa rand-m9-mstd0.5 --mixup 0.8 --cutmix 1.0 --reprob 0.25 --drop-path 0.1 --model-ema --grad-accum-steps 8 --crop-pct 0.95
```

## Results on ImageNet-1K

Our model achieves the following performance:

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  |
| ------------------ |---------------- |
| Dit-ConvNeXt-T   |     82.6%         |
| Dit-ConvNeXt-S   |     83.6%         |
| Dit-ConvNeXt-B   |     84.2%         |


## License
This project is licensed under the MIT License.


