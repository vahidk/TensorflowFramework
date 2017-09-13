# Tensorflow training framework

This repository includes an extendable framework for training neural network models using TensorFlow. The framework handles things like parallel data reading, batching, shuffling, saving/loading checkpoints, and logging. It also supports training on multiple GPUs. You can simply add your own dataset by following the example in dataset/mnist.py, and write your own classifier similar to the bundled CNN classifier in model/alexnet.py. The framework does most of the boilerplate code for you, letting you focus on developing the actual neural net model.

Also, make sure to checkout [Effective Tensorflow](https://github.com/vahidk/EffectiveTensorflow), which to some extents explains the code in this framework.

Pull requests with new datasets or models are welcome!

## Install dependencies
```
pip install tensorflow numpy pillow matplotlib six
```

## Training
To train an mnist classification model run:
```
python -m trainer --model=alexnet --dataset=mnist
```

To visualize the training logs on Tensorboard run:
```
tensorboard --logdir=output
```

Current included datasets are:
- mnist
- cifar10
- cifar100 

Bundled models include:
- alexnet
- allconv
- resnet

