from torch import nn
import torch
import torch.autograd as autograd
from torch.autograd import Variable

import numpy as np

"""
# Weights initializer
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def init_weights(Layer):
    name = Layer.__class__.__name__
    if name == 'Linear':
        torch.nn.init.normal_(Layer.weight, mean=0, std=0.02)
        if Layer.bias is not None:
            torch.nn.init.constant_(Layer.bias, 0)

class TransposedCBR(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.model(x)


class CRBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding, bias=True):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel, stride, padding, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_num = 64):
        super().__init__()
        self.model = nn.Sequential(
            CRBlock(1, hidden_num, (1, in_dim), (1, 1), (0, 0), bias=False),
            CRBlock(hidden_num, hidden_num * 2, (4, 1), (2, 1), (1, 0), bias=False),
            CRBlock(hidden_num * 2, hidden_num * 4, (8, 1), (1, 1), (1, 0), bias=False),
            CRBlock(hidden_num * 4, hidden_num * 8, (4, 1), (2, 1), (1, 0), bias=False),
            nn.Conv2d(hidden_num * 8, 100, (5, 1), (1, 1), (0, 0), bias=False),
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, in_dim, hidden_num = 64, out_dim = 6):
        super().__init__()
        self.model = nn.Sequential(
            TransposedCBR(in_dim, hidden_num * 8, (5, 1), (1, 1), (0, 0)),
            TransposedCBR(hidden_num * 8, hidden_num * 4, (4, 1), (2, 1), (1, 0)),
            TransposedCBR(hidden_num * 4, hidden_num * 2, (8, 1), (1, 1), (1, 0)),
            TransposedCBR(hidden_num * 2, hidden_num, (4, 1), (2, 1), (1, 0)),
            nn.ConvTranspose2d(hidden_num, 1, (1, out_dim), (1, 1), (0, 0), bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


"""
# Discriminator
"""
# Discriminator using CONV
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(1, 64, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf) x 32 x 32``
#             nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*2) x 16 x 16``
#             nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*4) x 8 x 8``
#             nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*8) x 4 x 4``
#             nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
#             nn.Linear((1,5,14),(1,128,1)),
#             nn.Sigmoid(),
#         )
#     def forward(self, X, z):
#         X = X.reshape(-1, 30*6) ## reshape single PC6 encoded peptide
#         z = z.reshape(-1, 100)
#         Xz = torch.cat([X, z], dim = 1)
#         Xz = Xz.unsqueeze(0)
#         Xz = Xz.unsqueeze(0)
#         print(Xz.shape)
#         return self.layers(Xz)

class JointCritic(nn.Module):
  def __init__(self, x_mapping, z_mapping, joint_mapping):
    """ A joint Wasserstein critic function.

    Args:
      x_mapping: An nn.Sequential module that processes x.
      z_mapping: An nn.Sequential module that processes z.
      joint_mapping: An nn.Sequential module that process the output of x_mapping and z_mapping.
    """
    super().__init__()

    self.x_net = x_mapping
    self.z_net = z_mapping
    self.joint_net = joint_mapping

  def forward(self, x, z):
    assert x.size(0) == z.size(0)
    x_out = self.x_net(x)
    z_out = self.z_net(z)
    joint_input = torch.cat((x_out, z_out), dim=1)
    output = self.joint_net(joint_input)
    return output

#  CRBlock(1, hidden_num, (1, in_dim), (1, 1), (0, 0), bias=False),
#             CRBlock(hidden_num, hidden_num * 2, (4, 1), (2, 1), (1, 0), bias=False),
#             CRBlock(hidden_num * 2, hidden_num * 4, (8, 1), (1, 1), (1, 0), bias=False),
#             CRBlock(hidden_num * 4, hidden_num * 8, (4, 1), (2, 1), (1, 0), bias=False),
#             nn.Conv2d(hidden_num * 8, 1, (5, 1), (1, 1), (0, 0), bias=False),
def create_critic():
    LEAK = 0.2
    x_mapping = nn.Sequential(
        nn.Conv2d(1, 64, (1,6), (1,1), (0,0)), nn.LeakyReLU(LEAK),
        nn.Conv2d(64, 64 * 2, (4,1), (2,1),(1,0)), nn.LeakyReLU(LEAK),
        nn.Conv2d(64 * 2, 64 * 4, (8,1), (1,1), (1,0)), nn.LeakyReLU(LEAK),
        nn.Conv2d(64 * 4, 64 * 8, (4,1), (2,1), (1,0)), nn.LeakyReLU(LEAK),
        nn.Conv2d(64 * 8, 64 * 8, (5,1), (1,1), (0,0)), nn.LeakyReLU(LEAK))

    z_mapping = nn.Sequential(
        nn.Conv2d(100, 512, 1, 1, 0), nn.LeakyReLU(LEAK),
        nn.Conv2d(512, 512, 1, 1, 0), nn.LeakyReLU(LEAK))

    joint_mapping = nn.Sequential(
        nn.Conv2d(64 * 8 + 512, 2048, 1, 1, 0), nn.LeakyReLU(LEAK),
        nn.Conv2d(2048, 2048, 1, 1, 0), nn.LeakyReLU(LEAK),
        nn.Conv2d(2048, 1, 1, 1, 0))

    return JointCritic(x_mapping, z_mapping, joint_mapping)


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(30*6 + 100, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.2),
#             nn.Linear(128,1)
#         )
#     def forward(self, X, z):
#         X = X.reshape(-1, 30*6) ## reshape single PC6 encoded peptide
#         z = z.reshape(-1, 100)
#         Xz = torch.cat([X, z], dim = 1)
#         return self.layers(Xz)

"""
# Discriminator loss
"""
def D_loss(DG, DE, eps=1e-6):
    loss = torch.log(DE + eps) + torch.log(1 - DG + eps)    
    return -torch.mean(loss)

"""
# Encoder - Generator loss
"""
def EG_loss(DG, DE, eps=1e-6):
    loss = torch.log(DG + eps) + torch.log(1 - DE + eps)
    return -torch.mean(loss)

"""
WGAN-GP loss
"""
class loss_GP():
    def __init__(self, C, x, z_hat, x_tilde, z):
        self.C = C
        self.x = x
        self.z_hat = z_hat
        self.x_tilde = x_tilde
        self.z = z
    
    def criticize(self):
      x, z_hat, x_tilde, z = self.x, self.z_hat,self.x_tilde,self.z   
      input_x = torch.cat((x, x_tilde), dim=0)
      input_z = torch.cat((z_hat, z), dim=0)
      output = self.C(input_x, input_z)
      data_preds, sample_preds = output[:x.size(0)], output[x.size(0):]
      return data_preds, sample_preds

    def calculate_grad_penalty(self):
      x, z_hat, x_tilde, z =  self.x, self.z_hat,self.x_tilde,self.z  

      bsize = x.size(0)
      eps = torch.rand(bsize, 1, 1, 1).to(x.device) # eps ~ Unif[0, 1]
      intp_x = Variable(eps * x + (1 - eps) * x_tilde)
      intp_z = Variable(eps * z_hat + (1 - eps) * z)
      # check following two lines
      intp_x.requires_grad = True
      intp_z.requires_grad = True
      C_intp_loss = self.C(intp_x, intp_z).sum()
      grads = autograd.grad(C_intp_loss, (intp_x, intp_z), retain_graph=True, create_graph=True)
      grads_x, grads_z = grads[0].view(bsize, -1), grads[1].view(bsize, -1)
      grads = torch.cat((grads_x, grads_z), dim=1)
      grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
      return grad_penalty
    
    def loss(self,lamb=10):
      data_preds, sample_preds = self.criticize()
      EG_loss = torch.mean(data_preds - sample_preds)
      C_loss = -EG_loss + lamb * self.calculate_grad_penalty()
      E_loss = torch.mean(data_preds)
      G_loss = torch.mean(sample_preds)
      return C_loss, EG_loss, E_loss, G_loss
    

def get_model_and_optimizer(latent_size = 100):
    encoded_num = 6
    hidden_size = 64
    generator = Generator(latent_size, hidden_size, encoded_num)
    encoder = Encoder(encoded_num, hidden_size)
    discriminator = create_critic()
    return generator, encoder, discriminator
