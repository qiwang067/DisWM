import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import tools
from torch import distributions as torchd


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H_Encoder(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H_Encoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ELU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ELU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ELU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ELU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ELU(True),
            View((-1, 256*1*1)),                 # B, 256
        )

        self.dis_encoder = nn.Sequential(
            nn.Linear(256, z_dim * 2),  # B, z_dim*2
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


    def forward(self, x, pred=False):
        if pred:
            x_img = x['image'][:6].reshape((-1,) + tuple(x['image'][:6].shape[-3:]))
            x_img = x_img.permute(0, 3, 1, 2)
        else:
            x_img = x['image'].reshape((-1,) + tuple(x['image'].shape[-3:]))
            x_img = x_img.permute(0, 3, 1, 2)

        z_recon = self._encode(x_img)

        distributions = self.dis_encoder(z_recon)

        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]

        z_vae = reparametrize(mu, logvar)

        if x['image'].ndim == 4:
            z_vae = z_vae.reshape(x['image'].shape[0], -1)
        else:
            if pred:
                z_vae = z_vae.reshape(x['image'][:6].shape[0], x['image'][:6].shape[1], self.z_dim)
            else:
                z_vae = z_vae.reshape(x['image'].shape[0], x['image'].shape[1], self.z_dim)

        return mu, logvar, z_vae

    def _encode(self, x):
        return self.encoder(x)


class BetaVAE_H_Decoder(nn.Module):
    def __init__(self, shape, feat_size, z_dim=10, nc=3):
        super(BetaVAE_H_Decoder, self).__init__()
        self._shape = shape
        self.z_dim = z_dim
        self.nc = nc


        self.feat_to_decoder = nn.Sequential(
            nn.Linear(feat_size, z_dim),
            nn.ELU(True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ELU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ELU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ELU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ELU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ELU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, feature):
        z = self.feat_to_decoder(feature)
        x_recon = self._decode(z)
        mean = x_recon.reshape(feature.shape[:-1] + self._shape)
        mean = mean.permute(0, 1, 3, 4, 2)  # 50, 50, 64, 64, 3
        return tools.ContDist(torchd.independent.Independent(
            torchd.normal.Normal(mean, 1), len(self._shape)))


    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    pass
