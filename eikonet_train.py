# %%
import torch
import pylab as plt
import pandas as pd
import json
import numpy as np
import math
from scipy import interpolate
from torch.utils.data import DataLoader
import pyproj

# %%
device = "cuda:6"
infile = '/home/zross/git/SPIDER/noto/eikonet.json'
#infile = '/home/zross/git/SPIDER/maunaloa/eikonet.json'
# infile = '/home/zross/git/SPIDER/tottori/eikonet.json'
# infile = '/home/zross/git/SPIDER/cahuilla/eikonet.json'
with open(infile) as f:
    params = json.load(f)
model_file = params["model_file"]
velmod = pd.read_csv(params["velmod_file"])
# stations = pd.read_csv(params["station_file"])
scale = params["scale"]

vp = interpolate.interp1d(velmod["depth"], velmod["vp"], kind='linear')
vs = interpolate.interp1d(velmod["depth"], velmod["vs"], kind='linear')
batch_size = params["batch_size"]
n_train = params["n_train"]
n_test = params["n_test"]

# %%
def linear_velmod(x):
    return torch.sin(2*np.pi*10.0*x[:,5]) + 3.0

def custom_1d_velmod(x, vp, vs):
#     return torch.tensor(vp(x[:,5] * model.scale)).float() ** -1
    v_full = torch.zeros(x.shape[0], 2)
    v_full[:,0] = torch.tensor(vp(x[:,5]))
    v_full[:,1] = torch.tensor(vs(x[:,5]))
    return (v_full[range(x.shape[0]), x[:,6].int()]) ** -1


# def custom_1d_velmod(x, vp, vs):
# #     return torch.tensor(vp(x[:,5] * model.scale)).float() ** -1
#     v_full = torch.zeros(x.shape[0], 2, 2)
#     v_full[:,0,0] = torch.tensor(vp(x[:,2]))
#     v_full[:,1,0] = torch.tensor(vs(x[:,2]))
#     v_full[:,0,1] = torch.tensor(vp(x[:,5]))
#     v_full[:,1,1] = torch.tensor(vs(x[:,5]))
#     return (v_full[range(x.shape[0]), x[:,6].int(),:]) ** -1

# %%
class EikoData(torch.utils.data.Dataset):
    def __init__(self,X,Y):
        self.X = X                           # set data
        self.Y = Y                           # set lables

    def __len__(self):
        return len(self.X)                   # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]


# %%
def check_bounds(pfile, stations):
    if stations["latitude"].min() < params["lat_min"]:
        raise(Exception("One or more stations is outside lat bounds"))
    if stations["latitude"].max() < params["lat_max"]:
        raise(Exception("One or more stations is outside lat bounds"))
    if stations["longitude"].min() < params["lon_min"]:
        raise(Exception("One or more stations is outside lon bounds"))
    if stations["longitude"].max() < params["lon_max"]:
        raise(Exception("One or more stations is outside lon bounds"))
    return
# check_bounds(params, stations)

# %%
def init_weights_eiko_sine(m):
    """
    init weights of eikonet with sine activation
    """
    if type(m) == torch.nn.Linear:
        stdv = (1. / math.sqrt(m.weight.size(1))/1.)
        m.weight.data.uniform_(-stdv,stdv)
        if m.bias.data is not None:
            m.bias.data.uniform_(-stdv,stdv)
        else:
            m.weight.data.fill_(1.0)

class EikoNet(torch.nn.Module):

    def __init__(self, scale, vs=3.3, vp=6.0):
        super(EikoNet, self).__init__()
        self.scale = scale
        self.vs = vs
        self.vp = vp
        self.activation = torch.nn.ELU()
        # self.sine_freq = 1.0
        self.n_hidden = 128

        self.linear1 = torch.nn.Linear(4, self.n_hidden)
        self.linear2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear3 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear4 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear5 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear6 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear7 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear8 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear9 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear10 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear11 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.linear_out = torch.nn.Linear(self.n_hidden, 1)

        # self.net = SirenNet(
        #     dim_in = 4,                        # input dimension, ex. 2d coor
        #     dim_hidden = self.n_hidden,                  # hidden dimension
        #     dim_out = 1,                       # output dimension, ex. rgb value
        #     num_layers = 10,                    # number of layers
        #     final_activation = torch.exp,   # activation of final layer (nn.Identity() for direct output)
        #     w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        # )

    # def sine_encoding(self, x):
    #     x_enc = torch.einsum("zi,zj->zij", self.ω, x[:,:4]).view(x.shape[0],-1).float()
    #     return torch.cat((torch.sin(2*np.pi*x_enc), torch.cos(2*np.pi*x_enc), x[:,-1,None]), dim=1).float()

    # def ff_gaussian_encoding(self, x):
    #     z = x[:,:-1].matmul(self.B.T)
    #     γ = torch.cat((torch.cos(2*np.pi*z), torch.sin(2*np.pi*z), x[:,-1,None]), dim=1)
    #     return γ

    def T0(self, x):
        scalar = torch.where(x[:,6] >= 0.5, self.vs, self.vp).unsqueeze(1)
        return torch.sqrt(((x[:,0:3]-x[:,3:6])**2).sum(dim=1)).unsqueeze(dim=1) / scalar

    # def activation(self, x):
    #     return torch.sin(self.sine_freq * x)

    def T1(self, x):
        r = torch.sqrt(((x[:,0:2] - x[:,3:5])**2).sum(dim=1)).unsqueeze(1)
        x = torch.cat((r, x[:,2,None], x[:,5:]), dim=1)
        x[:,:3] /= self.scale

        x = self.linear1(x)
        x = self.activation(x)

        x0 = x
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x) + x0

        x0 = x
        x = self.linear4(x)
        x = self.activation(x)
        x = self.linear5(x)
        x = self.activation(x) + x0

        x0 = x
        x = self.linear6(x)
        x = self.activation(x)
        x = self.linear7(x)
        x = self.activation(x) + x0

        x0 = x
        x = self.linear8(x)
        x = self.activation(x)
        x = self.linear9(x)
        x = self.activation(x) + x0

        x0 = x
        x = self.linear10(x)
        x = self.activation(x)
        x = self.linear11(x)
        x = self.activation(x) + x0

        x = self.linear_out(x)
        x = torch.abs(x)

        # x = self.net(x)

        return x

    def forward(self, x):
        return self.T0(x) * self.T1(x)

    def EikonalPDE(self, x):
        x.requires_grad_()
        T = self.forward(x)
        dT_dx = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
        # s_src = (dT_dx[:,0:3]**2).sum(dim=1).unsqueeze(1).sqrt()
        s_rec = (dT_dx[:,3:6]**2).sum(dim=1).sqrt()
        # s_pred = torch.cat((s_src, s_rec), dim=1)
        return s_rec

def build_dataset():
    n_dataset = n_train

    x = torch.tensor(np.random.uniform(0.0, scale, (n_dataset, 7))).float()
    x[:,2] = torch.tensor(np.random.uniform(params["z_min"], params["z_max"], n_dataset))

    sphere = torch.randn(n_dataset, 3)
    sphere /= (sphere**2).sum(dim=1).sqrt().unsqueeze(1) + 1e-8
    radius = torch.tensor(np.random.uniform(0.1, params["scale"], n_dataset)).unsqueeze(1)
    sphere *= radius

    x_rec = x[:,:3] + sphere
    x_rec[:,:2] = torch.clamp(x_rec[:,:2], min=0.0, max=params["scale"])
    x_rec[:,2] = torch.clamp(x_rec[:,2], min=params["z_min"], max=params["z_max"])
    x[:,3:6] = x_rec
    # x[:,5] = torch.tensor(np.random.uniform(params["z_min"], params["z_max"], n_dataset))
    x[:,6] = torch.tensor(np.random.randint(2, size=n_dataset))
    s_true = custom_1d_velmod(x, vp, vs).to(device)
    x = x.to(device)
    train_dataset = EikoData(x, s_true)

    n_dataset = n_test
    x = torch.tensor(np.random.uniform(0.0, scale, (n_dataset, 7))).float()
    x[:,2] = torch.tensor(np.random.uniform(params["z_min"], params["z_max"], n_dataset))
    sphere = torch.randn(n_dataset, 3)
    sphere /= (sphere**2).sum(dim=1).sqrt().unsqueeze(1) + 1e-8
    radius = torch.tensor(np.random.uniform(0.1, params["scale"], n_dataset)).unsqueeze(1)
    sphere *= radius

    x_rec = x[:,:3] + sphere
    x_rec[:,:2] = torch.clamp(x_rec[:,:2], min=0.0, max=params["scale"])
    x_rec[:,2] = torch.clamp(x_rec[:,2], min=params["z_min"], max=params["z_max"])
    x[:,3:6] = x_rec
    # x[:,5] = torch.tensor(np.random.uniform(params["z_min"], params["z_max"], n_dataset))
    x[:,6] = torch.tensor(np.random.randint(2, size=n_dataset))
    s_true = custom_1d_velmod(x, vp, vs).to(device)
    x = x.to(device)
    test_dataset = EikoData(x, s_true)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=20000, shuffle=True)
    return train_dataloader, test_dataloader

# %%
model = EikoNet(scale, vp=6.0, vs=3.2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100,200], gamma=0.1)
best_loss = torch.inf
train_dataloader, test_dataloader = build_dataset()

x = torch.tensor(np.random.uniform(0.0, scale, (1000, 7))).float()
x[:,2] = torch.tensor(np.random.uniform(params["z_min"], params["z_max"], 1000))
x[:,5] = torch.tensor(np.random.uniform(params["z_min"], params["z_max"], 1000))
x[:,6] = torch.tensor(np.random.randint(2, size=1000))
s_true = custom_1d_velmod(x, vp, vs).to(device)
x = x.to(device)
print(1.0/model.EikonalPDE(x).mean())

for epoch in range(params["n_epochs"]):
    # train_dataloader, test_dataloader = build_dataset()
    train_loss = 0.0
    count = 0
    for (x, s_true) in train_dataloader:
        opt.zero_grad()
        s_pred = model.EikonalPDE(x)
        pde_loss = torch.norm(s_pred-s_true) / torch.norm(s_true)
        # pde_loss = torch.nn.functional.mse_loss(s_pred, s_true)
        pde_loss.backward()
        with torch.no_grad():
            train_loss += torch.norm(s_pred-s_true) / torch.norm(s_true)
        opt.step()
        count += 1
    train_loss /= count
    scheduler.step()

    val_loss = 0.0
    count = 0
    for (x, s_true) in test_dataloader:
        s_pred = model.EikonalPDE(x)
        with torch.no_grad():
            val_loss += torch.norm(s_pred-s_true) / torch.norm(s_true)
            count += 1
    val_loss /= count
    print(epoch, train_loss.item(), val_loss.item(), best_loss)
    
    if val_loss < best_loss:
        best_loss = val_loss 
        # print("NOT SAVING FILE") 
        torch.save(model, model_file)
