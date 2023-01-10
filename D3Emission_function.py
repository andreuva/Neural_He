import numpy as np
import torch
from NN import MLP



class D3inference:
    """
    Class to do the inference of the emision of the D3 line
    """

    def __init__(self, file_checkpoint_I, file_checkpoint_Q, file_checkpoint_U):
        """
        file_checkpoint_I: path to the checkpoint of the model for the Stokes I
        file_checkpoint_Q: path to the checkpoint of the model for the Stokes Q
        file_checkpoint_U: path to the checkpoint of the model for the Stokes U
        """

        # check if the GPU is available
        cuda = torch.cuda.is_available()
        # set the GPU to use (0 if there is only one)
        gpu = 0
        # select the device
        self.device = torch.device(f"cuda:{gpu}" if cuda else "cpu")
        # print the device configuration used
        if cuda:
            print('GPU is available')
            print('Using GPU {}'.format(gpu))
        else:
            print('GPU is not available')
            print('Using CPU')
            print(self.device)
        
        # load the checkpoint, instanciate the model and load the weigths of the model for each Stokes
        self.checkpoint_I = torch.load(file_checkpoint_I, map_location=lambda storage, loc: storage)
        self.model_I = MLP(7, 1, self.checkpoint_I['hyperparameters']['mlp_hidden_size']).to(self.device)
        self.model_I.load_state_dict(self.checkpoint_I['state_dict'])

        self.checkpoint_Q = torch.load(file_checkpoint_Q, map_location=lambda storage, loc: storage)
        self.model_Q = MLP(7, 1, self.checkpoint_Q['hyperparameters']['mlp_hidden_size']).to(self.device)
        self.model_Q.load_state_dict(self.checkpoint_Q['state_dict'])

        self.checkpoint_U = torch.load(file_checkpoint_U, map_location=lambda storage, loc: storage)
        self.model_U = MLP(7, 1, self.checkpoint_U['hyperparameters']['mlp_hidden_size']).to(self.device)
        self.model_U.load_state_dict(self.checkpoint_U['state_dict'])

    def __call__(self, b, x, Bx, By, Bz):
        """
        call to do the inference of the integrated emission of the D3 line

        Parameters:
        ------------------------------------------------------
        b: distance to the 90 deg LOS (-20, 20)*cts.R_sun
        x: radial distance to the surface (0, 10)*cts.R_sun
        Bx: B field in the x direction
        By: B field in the y direction
        Bz: B field in the z direction
        # the network is trained with profiles in the range:
        B = np.random.normal(10, 50)
        B_inc = np.random.uniform(0,180)
        B_az = np.random.uniform(0, 360)
        """
        
        B_mod = np.sqrt(Bx**2 + By**2 + Bz**2)
        B_inc = np.arccos(Bz/B_mod)
        B_az = np.arctan2(By, Bx)

        h = np.sqrt(x**2 + b**2)
        mu = x/h

        # construct the parameters list with the normalization
        params = np.array([B_mod, B_inc, B_az, x, b, h, mu], dtype=np.float32)
        params_min = np.array(
                     [5.02408413e-04,  5.16185230e-03,  4.31011672e-03, -1.39139561e+12,
                      1.77925802e+06,  3.02479808e+09, -1.00000000e+00], dtype=np.float32)
        params_max = np.array(
                     [2.41667130e+02, 1.79999725e+02, 3.59996481e+02, 1.39139222e+12,
                      6.95696304e+11, 1.55437942e+12, 1.00000000e+00], dtype=np.float32)

        params_norm = (params - params_min)/(params_max - params_min)

        epsI = self.model_I.forward(torch.tensor(params_norm).to(self.device))
        epsQ = self.model_Q.forward(torch.tensor(params_norm).to(self.device))
        epsU = self.model_U.forward(torch.tensor(params_norm).to(self.device))

        epsI = epsI.detach().cpu().numpy()
        epsQ = epsQ.detach().cpu().numpy()
        epsU = epsU.detach().cpu().numpy()

        return epsI, epsQ, epsU



if __name__ == "__main__":
    # intanciate the class to do the predictions
    file_checkpoint_I = 'checkpoints/trained_model_mlp_eps_Q_time_20230109-140242/trained_model_mlp_20230109-160317.pth'
    file_checkpoint_Q = 'checkpoints/trained_model_mlp_eps_Q_time_20230109-140242/trained_model_mlp_20230109-160317.pth'
    file_checkpoint_U = 'checkpoints/trained_model_mlp_eps_Q_time_20230109-140242/trained_model_mlp_20230109-160317.pth'

    D3 = D3inference(file_checkpoint_I, file_checkpoint_Q, file_checkpoint_U)

    R_sun = 6.957e10           # solar radius [cm]
    epsI, epsQ, epsU = D3(0*R_sun, 5*R_sun, 10, 0, 0)

    print(epsI, epsQ, epsU)
