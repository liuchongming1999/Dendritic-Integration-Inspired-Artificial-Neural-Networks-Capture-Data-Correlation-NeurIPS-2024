# Dendritic Integration Inspired Artificial Neural Networks Capture Data Correlation (NeurIPS 2024)

This repository is the official implementation of [Dendritic Integration Inspired Artificial Neural Networks Capture Data Correlation](https://arxiv.org/abs/2030.12345). 

![](/img/interpretation.jpeg)
**Figure:** **A.** Experiments confirmed the quadratic integration rule under general
cases, along with a comprehensive theoretical framework for single neuron computation (From [Li et al. 2023](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.22020)). **C.**
An illustration of the biological interpretation of our Dit-CNNs.

Our Dit-CNN is inspired by neural networks in the visual system. For example, different types of cone cells encode various color (channel) information, and retinal ganglion cells receive inputs from multiple types of cone cells, the responses can be modeled as having receptive fields (convolutional kernels) related to different color channels ( $w_1 * x_1, w_2 * x_2, w_3 * x_3$ ). When multiple channel inputs are present, traditional CNNs simply linearly sum the corresponding responses. In contrast, neurons integrate these inputs with an additional quadratic term based on the dendritic bilinear integration rule. This approach leads to the formulation of our Dit-CNN after simplification. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
