import numpy as np
import torch
from NN import MLP



class D3inference:
    """
    Class to do the inference of the emision of the D3 line
    """

    def __init__(self, file_checkpoint_I, file_checkpoint_Q, file_checkpoint_U, file_checkpoint_V):
        """
        file_checkpoint_I: path to the checkpoint of the model for the Stokes I
        file_checkpoint_Q: path to the checkpoint of the model for the Stokes Q
        file_checkpoint_U: path to the checkpoint of the model for the Stokes U
        file_checkpoint_V: path to the checkpoint of the model for the Stokes V
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

        self.checkpoint_V = torch.load(file_checkpoint_V, map_location=lambda storage, loc: storage)
        self.model_V = MLP(7, 1, self.checkpoint_V['hyperparameters']['mlp_hidden_size']).to(self.device)
        self.model_V.load_state_dict(self.checkpoint_V['state_dict'])

        self.params_norm_coeffs = np.loadtxt('params_norm_coeffs.txt')
        self.params_min = self.params_norm_coeffs[0,:]
        self.params_max = self.params_norm_coeffs[1,:]
        self.params_mean = self.params_norm_coeffs[2,:]

        self.prof_norm_coeffs = np.loadtxt('prof_norm_coeffs.txt')
        self.eps_I_max = self.prof_norm_coeffs[0]
        self.eps_I_min = self.prof_norm_coeffs[1]
        self.eps_I_mean = self.prof_norm_coeffs[2]

    def __call__(self, b, x_r, Bx, By, Bz):
        """
        call to do the inference of the integrated emission of the D3 line

        Parameters:
        ------------------------------------------------------
        b: radial distance to the surface (0, 10)*cts.R_sun
        x_r: distance to the 90 deg LOS (-20, 20)*cts.R_sun
        Bx: B field in the x direction
        By: B field in the y direction
        Bz: B field in the z direction
        # the network is trained with profiles in the range:
        B = np.random.normal(10, 50)
        B_inc = np.random.uniform(0,180)
        B_az = np.random.uniform(0, 360)
        """

        # compute height over the surface and angle from the plane of the sky
        h = np.sqrt(b**2 + x_r**2)
        mu = b/h
        sin_delt = x_r/h
        cos_delt = mu

        # make a rotation of the magnetic field to to have the z in the solar radial direction
        # this is equivalent to rotate the reference frame using e_y an angle of delt
        Bx_new = Bx*cos_delt + Bz*sin_delt
        By_new = By
        Bz_new = -Bx*sin_delt + Bz*cos_delt

        B_mod = np.sqrt(Bx_new**2 + By_new**2 + Bz_new**2)
        B_inc = np.arccos(Bz_new/B_mod)
        B_az = np.arctan2(By_new, Bx_new)

        # construct the parameters list with the normalization
        params = np.array([B_mod, B_inc, B_az, b, x_r, h, mu])

        params_norm = (params - self.params_mean)/(self.params_max - self.params_min)
        params_norm = params_norm.astype('float32')

        epsI = self.model_I.forward(torch.tensor(params_norm).to(self.device))
        epsQ = self.model_Q.forward(torch.tensor(params_norm).to(self.device))
        epsU = self.model_U.forward(torch.tensor(params_norm).to(self.device))
        epsV = self.model_V.forward(torch.tensor(params_norm).to(self.device))

        epsI = epsI.detach().cpu().numpy()
        epsQ = epsQ.detach().cpu().numpy()
        epsU = epsU.detach().cpu().numpy()
        epsV = epsV.detach().cpu().numpy()

        epsI = epsI*(self.eps_I_max - self.eps_I_min) + self.eps_I_mean
        epsI = 10**(epsI)
        epsQ = epsQ*epsI
        epsU = epsU*epsI
        epsV = epsV*epsI

        return epsI, epsQ, epsU, epsV



if __name__ == "__main__":
    # intanciate the class to do the predictions
    file_checkpoint_I = 'weigths_eps_I.pth'
    file_checkpoint_Q = 'weigths_eps_Q.pth'
    file_checkpoint_U = 'weigths_eps_U.pth'
    file_checkpoint_V = 'weigths_eps_V.pth'

    D3 = D3inference(file_checkpoint_I, file_checkpoint_Q, file_checkpoint_U, file_checkpoint_V)

    R_sun = 6.957e10           # solar radius [cm]
    epsI, epsQ, epsU, epsV = D3(8*R_sun, 0.1*R_sun, 0.0000001, 0, 0)

    print(epsI, epsQ, epsU, epsV)

    # print the percentage of the integrated emission over intensity
    print(epsQ/epsI, epsU/epsI, epsV/epsI)