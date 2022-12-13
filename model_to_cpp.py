import numpy as np
import torch
from dataset import print_hyperparameters, profiles_dataset
from NN import bVAE
from glob import glob


run_loaded = f'checkpoints/trained_model_eta_I_mapping_0.0_time_20221213-093459'
checkpoint = sorted(glob(f'{run_loaded}/trained_*.pth'))[-2]
# Load the checkpoint and initialize the model
print(f'Loading the model from {run_loaded}')
print(f'Loading the checkpoint {checkpoint[len(run_loaded):]}')
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

coefficient = checkpoint['hyperparameters']['coefficient']
archiquecture = checkpoint['hyperparameters']['archiquecture']
readir = checkpoint['hyperparameters']['dataset_dir']
readfile = checkpoint["hyperparameters"]["dataset_file"]
hyperparameters = checkpoint['hyperparameters']
savedir = run_loaded + '/'

# print the hyperparameters (tabulated correctly) and loss
print('\nHyperparameters')
print('-' * 80)
print_hyperparameters(hyperparameters)
print(f'{"training loss":<25}: {checkpoint["train_loss"]}')
print(f'{"validation loss":<25}: {checkpoint["valid_loss"]}')
print('-' * 80 + '\n')

print('Reading data from: ', readir + readfile.replace("5","1"))
# create the dataset to test
test_dataset = profiles_dataset(f'{readir}{readfile.replace("5","1")}', train=False)


print(f'Using {archiquecture}')
print('Loading the bvae parameters ...')
enc_size = hyperparameters['params']['bvae']['enc_size']
dec_size = hyperparameters['params']['bvae']['dec_size']
latent_size = hyperparameters['params']['bvae']['latent_size']
beta = hyperparameters['params']['bvae']['beta']

# load the model
model = bVAE(test_dataset.n_components, test_dataset.n_features, latent_size, enc_size, dec_size, beta)
model.load_state_dict(checkpoint['state_dict'])

""" Now we have the model we can prepare it and export it to use it in cpp code
First we need to convert the model to a torchscript model
https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html """

# Convert the model to a torchscript model
model.eval()

example = torch.rand(1, test_dataset.n_features)
traced_script_module = torch.jit.trace(model, example)

# Save the model
traced_script_module.save(f"{savedir}serialiced_model_cpp.pt")

""" Now we can use the model in cpp code
We need to create a cpp file with the following code
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>";
        return -1;
    }

    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

    assert(module != nullptr);
    std::cout << "ok";
}

We can compile it with the following command
g++ -std=c++14 -I $HOME/miniconda3/envs/Pytorch/lib/python3.8/site-packages/torch/include -I $HOME/miniconda3/envs/Pytorch/lib/python3.8/site-packages/torch/include/torch/csrc/api/include -L $HOME/miniconda3/envs/Pytorch/lib/python3.8/site-packages/torch/lib -lc10 -ltorch -ltorch_cpu -ltorch_python example-app.cpp -o example-app

and test it with
./example-app serialiced_model_cpp.pt
"""
