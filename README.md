# Skoltech ML'22 course project
 
 ## 1. Autoencoder
 
 ### CelebA setup
 It is necessary to install `gdown=4.4.0` with `--no-cache-dir` parameter
 in order to download CelebA dataset properly
 
 ### Repo setup
 Run `git clone https://github.com/antonzub99/AEproject.git` 
 to clone repo to your local machine/colab VM
 
 ### Training
 Run `python main.py --output_path=YOUR_PATH --loss_function=YOUR_LOSS --conv_init=YOUR_INIT`

* `YOUR_PATH` - path to the main folder, where outputs of the model (weights, optimizer state, produced images) will be stored
* `YOUR_LOSS` - type of the reconstruction loss. Mean absolute error as `mae`
or Laplacian pyramid loss as `laplacian` are available
* `YOUR_INIT` - type of weights initialization in convolutional layers. Normal as `normal`,
Kaiming uniform as `kaiming_uniform` and Kaiming normal as `kaiming_normal` are available.

More information about arguments can be obtained by running `python main.py -h`

 to be updated;
 
 ### TBD
