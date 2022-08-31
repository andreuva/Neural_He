from PIL import Image
import numpy as np
from siren import SirenNet
import torch
import torch.nn as nn
import matplotlib.pyplot as pl
import mlp

device = torch.device("cuda")

img = np.array(Image.open('obama.jpg')).astype('float32')

img = torch.tensor(img.reshape((600*480, 3)) / 255.0).to(device)

dim_in = 2
dim_hidden = 128
dim_out = 3
num_layers = 5

tmp = mlp.MLPMultiFourier(n_input=dim_in, n_output=dim_out, n_hidden=dim_hidden, n_hidden_layers=num_layers, sigma=[5], activation=nn.Tanh(), final_activation=nn.Sigmoid()).to(device) #, 0.1, 1.0])
tmp.weights_init(type='kaiming')

# tmp = SirenNet(dim_in = dim_in, dim_hidden = dim_hidden, dim_out = dim_out, num_layers = num_layers, final_activation=nn.Sigmoid()).to(device)

print(f'N. parameters : {sum(x.numel() for x in tmp.parameters())}')
print((dim_hidden * dim_hidden) * (num_layers - 1) + dim_hidden * (num_layers-1) + dim_hidden * dim_in + dim_hidden + dim_hidden * dim_out + dim_out)

x = np.linspace(-1, 1, 480)
y = np.linspace(-1, 1, 600)
X, Y = np.meshgrid(x, y)

optimizer = torch.optim.Adam(tmp.parameters(), lr=5e-3)
loss_L2 = nn.MSELoss().to(device)

xin = torch.tensor(np.vstack([X.flatten(), Y.flatten()]).T.astype('float32')).to(device)

for epoch in range(1000):
    optimizer.zero_grad()

    out = tmp(xin)
    
    loss = loss_L2(out, img)

    loss.backward()
    optimizer.step()

    print(loss.item())

fig, ax = pl.subplots(nrows=1, ncols=2)
ax[0].imshow(out.detach().cpu().numpy().reshape((600,480,3)))
ax[1].imshow(img.cpu().numpy().reshape((600,480,3)))
pl.show()