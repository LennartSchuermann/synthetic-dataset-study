# The Synthetic Dilemma: Assessing the Impact of AI-Generated Images on the Accuracy of Convolutional Neural Networks, Illustrated through the Example of Emotion Classification

This study investigates the efficacy of incorporating AI-generated images as supplementary data in emotion-recognizing Convolutional Neural Networks. Emotion recognition is a critical task in various applications such as human-computer interaction and affective computing. However, limited labeled data availability often hinders the performance of deep learning models. To address this challenge, we explore the use of stable diffusion, a generative model capable of synthesizing high-quality images, to augment existing datasets.


[Study](Synthetic_Data.pdf)

## Install & Run

Clone the project

```bash
  git clone https://github.com/LennartSchuermann/synthetic-dataset-study.git
```

Next: get [PyTorch](https://pytorch.org/get-started/locally/) & [Cuda](https://developer.nvidia.com/cuda-downloads)

### Packages for Inference Only

```bash
  pip install torch, numpy, opencv-python, onnxruntime-gpu
```

### Packages for Image Generation

```bash
  pip install torch, numpy, opencv-python, diffusers
```

### Packages for Training

```bash
  pip install matplotlib, torch, numpy
```


## Author

- Lennart S. | [Github](https://github.com/LennartSchuermann)