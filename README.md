# Through the void: interpolating paths for autoencoder's parameters

This is a repository for the final project of Machine Learning course at Skoltech, term 3 '22.
Team: Anton Zubekhin, Polina Karpikova, Nikita Fedyashin
The study and code are based on a paper [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026)

by Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov and Andrew Gordon Wilson (NIPS 2018, Spotlight)
([github repo](https://github.com/timgaripov/dnn-mode-connectivity))

# Intro

Neural networks are used for many tasks (image and video classification, natural language processing, etc.). 
They are typically trained by minimizing some loss function, which is a function of the parameters of the network. 
Thus, a loss function is represented by a surface in a high dimensional space. 
The optimally trained networks are associated with local minima on the loss surface.
The original [paper](https://arxiv.org/abs/1802.10026) refers to connections of such local minima for 
convolutional neural networks which are trained to classify images. It's shown, that there are
low-loss paths between minima on loss surface, which may be used to construct an optimal neural network satisfying low loss
requirement.

We study another type of neural networks - autoencoders - that are designed to construct 
low-dimensional representations of high-dimensional data. We claim, that for the task of embedding high dimensional data into low dimensional and further 
reconstruction of original data from its embedding there are low-loss paths, connecting different optimal neural networks (i.e. local minima 
on some loss surface).

## Autoencoder architecture
We use a publically available autoencoder [architecture](https://github.com/iamalexkorotin/Wasserstein2GenerativeNetworks/blob/master/src/autoencoders.py)
with slight adjustments - instead of transposed convolutions in the decoder part, we use a combination of upsampling and usual convolution.

## Presented losses 
To train an autoencoder we stick to two losses: mean absolute reconstruction error 
and reconstruction loss based on Laplacian pyramid. For the latter we use [this](https://gist.github.com/alper111/b9c6d80e2dba1ee0bfac15eb7dad09c8)
implementation.

## Dataset
We use CelebA 64x64 dataset to train and evaluate the performance of the models. The
data is splitted as 4 to 1 for train set and test set.

# Dependencies 
The list of required packages is presented in `requirements.txt`.
Be sure to install `gdown==4.4.0` to be able to download the CelebA dataset.

To estimate the quality of reconstructions we use LPIPS metric (implementation is 
[here](https://github.com/S-aiueo32/lpips-pytorch.git)).

Simply run
```
pip install -r requirements.txt
```
to install all required packages. We highly suggest using GPU 
for both train and inference of the autoencoder.

# Usage 

## Training the endpoints of the curve

You need two trained autoencoders to build a curve connecting them. Run
```
python train.py --dir=<DIR> \
                --data_path=<PATH> \
                --device=<DEVICE> \
                --loss_function=<LOSS> [mae|laplacian] \
                --conv_init=<WEIGHT_INIT> [normal|kaiming_normal|kaiming_uniform] \ 
                --latent_dim=<DIM> \ 
                --epochs=<EPOCHS>
```
Main parameters:
* ```DIR``` &mdash; path to the directory to store checkpoints of training
* ```PATH``` &mdash; path to the dataset (if predownloaded, by default on the first run
  it will be downloaded automatically) 
* ```DEVICE``` &mdash; device to train and infere models on (we suggest using cuda)
* ```LOSS``` &mdash; type of reconstruction loss (MAE or Laplace pyramid loss, default MAE)
* ```WEIGHT_INIT``` &mdash; type of weights initialization in convolutional layers (default: normal)
* ```DIM``` &mdash; dimensionality of the embeddings (default: 128)
* ```EPOCHS``` &mdash; number of training epochs (default: 100)

You can also choose on optimizer (Adam or SGD) and their parameters. Run 
```python train.py -h``` to see all available options.

## Training the curves

Once you have to checkpoints, you can connect them with a curve of your choice (Bezier or PolyChain)
```
python train.py --dir=<DIR> \
                --data_path=<PATH> \
                --device=<DEVICE> \
                --loss_function=<LOSS> [mae|laplacian] \
                --conv_init=<WEIGHT_INIT> [normal|kaiming_normal|kaiming_uniform] \
                --curve=<CURVE> [Bezier|PolyChain] \ 
                --num_bends=<NBENDS> \ 
                --init_start=<START> \ 
                --init_end=<END> \  
                --latent_dim=<DIM> \ 
                --epochs=<EPOCHS>
                [--fix_start] \ 
                [--fix_end] \ 
```
Main parameters:
* ```CURVE``` &mdash; type of curve parametrization (Bezier or PolyChain)
* ```NBENDS``` &mdash; number of bends in the curve (default: 3)
* ```START, END``` &mdash; paths to the checkpoints of the endpoints in the curve

You may also use `--fix_start --fix_end` to fix the endpoints of the curve (otherwise they will also 
be trained)

## Evaluating the curves

If you have a checkpoint of the curve, you can start the evaluation procedure. You are able to 
track the value of the loss along the trained curve (and optionally LPIPS). By default 
you will also get the dynamics of the reconstructed images by the networks, initialized 
with weights on the trained curve. We use 4 images from the training set to track their dynamics.
We also leave an opportunity to connect endpoints with a straight line to compare it with the trained low-loss curve.
Run
```
python eval_curve.py  --dir=<DIR> \
                      --ckpt=<CKPT> \
                      --device=<DEVICE> \
                      --connect=<CONNECT> [CURVE|TRIVIAL]\
                      --curve=<CURVE> \
                      --num_bends=<NBENDS> 
                      --num_points=<NPOINTS> 
                      [--lpips]
```
Main parameters:
* ```CKPT``` &mdash; path to the checkpoint of the curve
* ```CONNECT``` &mdash; type of connection - trained low-loss curve or straight line in the hyperspace
* ```NPOINTS``` &mdash; number of points to evaluate the curve at (default: 10)

Use flag `--lpips` to track the LPIPS score of the reconstructions along the curve. 

# Repo structure

`models` folder stores architectures of autoencoder and special
network class, that embodies the low-loss curve;

`notebooks` folder has some additional `.ipynb` notebooks:

* `colab_training.ipynb` - notebook for training models on `GoogleColab;
* `check_outputs.ipynb` - notebook to check the outputs of autoencoders ,
  to verify that endpoints are in different positions in the loss space,
to check the size of the models;
 
`check_ends.py` is there for a sanity check to compare the endpoints;

`dataset.py` has a dataset class for CelebA dataset. By default, it is downloaded
upon initialization of the dataset in the main code;

`train.py` trains the base autoencoder or the curve network (detailed description above);

`eval_curve.py` evaluates the curve (description above);

`trainer.py` stores basic functions for training and testing of the models;


`utils.py` has additional utility functions inherited from original repo;

# Results 
Image dynamics along L1 trained low-loss curve:
![curve_mae_norm_kaimnorm](https://user-images.githubusercontent.com/62748704/159593734-8d9c8456-9cba-4328-8cfa-c04efc089b31.gif)

Along Laplacian pyramid trained curve:
![kaimnorm_kaimuni](https://user-images.githubusercontent.com/62748704/159594581-c0268302-a562-4f8e-8336-3bebdfcb200f.gif)

Along segment connection:
![seg_mae_norm_kaimnorm](https://user-images.githubusercontent.com/62748704/159594283-d378f126-d504-4d42-9a99-ee0c0b8f0432.gif)

And one more:
![triv_kaimnorm_kaimuni](https://user-images.githubusercontent.com/62748704/159594616-2136f0bf-349e-41bc-b056-5ce39782e612.gif)


Bad checkpoints for the curve:

Check other in `/media/gifs` folder.


# Credits

* [DNN Loss Connectivity](https://github.com/timgaripov/dnn-mode-connectivity) original implementation of low-loss curve finding algorithm
* [Autoencoder arcitecture](https://github.com/iamalexkorotin/Wasserstein2GenerativeNetworks/blob/master/src/autoencoders.py) for CelebA dataset
* [Laplacian pyramid loss](https://gist.github.com/alper111/b9c6d80e2dba1ee0bfac15eb7dad09c8) for models training
* [LPIPS](https://github.com/S-aiueo32/lpips-pytorch.git) score evaluation
* [CelebA](https://www.kaggle.com/jessicali9530/celeba-dataset) aligned 64x64 dataset
