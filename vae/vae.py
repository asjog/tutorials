import torch
import torchvision
import os
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import time
import copy
from tqdm import tqdm
import subprocess
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib.pyplot import Line2D
from torch.utils.data import Subset

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        print(n)
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

# define the model
# vae has an encoder: image -> cnn -> \mu, \sigma, -> decoder -> image
class vae_encoder(nn.Module):

    def __init__(self, latent_dim=2, do_train=True):
        super(vae_encoder, self).__init__()
        self.latent_dim = latent_dim
        self.do_train = do_train
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1)
        self.conv12 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv21 = nn.Conv2d(in_channels=8*1, out_channels=8*2, kernel_size=(3, 3), padding=1)
        self.conv22 = nn.Conv2d(in_channels=8*2, out_channels=8*2, kernel_size=(3, 3), padding=1)

        # self.conv31 = nn.Conv2d(in_channels=8*2, out_channels=8*4, kernel_size=(3, 3), padding=1, bias=False,
        #                         padding_mode='zeros')
        # self.conv32 = nn.Conv2d(in_channels=8*4, out_channels=8*4, kernel_size=(3, 3), padding=1, bias=False,
        #                     padding_mode='zeros')

        self.flatten = nn.Flatten()
        self.mu = nn.Linear(in_features=8*2*8*8, out_features=latent_dim)
        self.log_sigma = nn.Linear(in_features=8*2*8*8, out_features=latent_dim)


    def forward(self, x):
        # x = F.relu(self.conv11(x))
        # x = F.relu((self.conv21(x)))
        x = F.relu(self.conv12(F.relu(self.conv11(x))))
        x = self.pool(x)
        x = F.relu(self.conv22(F.relu(self.conv21(x))))
        x = self.pool(x)
        # x = F.relu(self.conv32(F.relu(self.conv31(x))))
        # x = self.pool(x)
        # x = self.flatten(x)
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        if self.do_train:
            std_normal = torch.distributions.MultivariateNormal(loc=torch.zeros(size=(mu.shape[1],)),
                                                                covariance_matrix=torch.diag(torch.ones((mu.shape[1],))))
            epsilons = std_normal.sample(sample_shape=(mu.shape[0],))
            # z_i = u_i + exp(log_sigma_i)*epsilons
            # epsilons = 0
            z = mu + torch.exp(log_sigma)*epsilons
        else:
            z = mu
        return mu, log_sigma, z

class vae_decoder(nn.Module):

    def __init__(self, latent_dim=256):
        super(vae_decoder, self).__init__()
        self.latent_dim = latent_dim
        self.dense_layer = nn.Linear(in_features=latent_dim, out_features=8*2*8*8)
        self.conv11 = nn.Conv2d(in_channels=8*2, out_channels=8, kernel_size=(3, 3), padding=1)
        self.conv12 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1)
        self.conv12dt = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)

        self.conv21 = nn.Conv2d(in_channels=8, out_channels=8*2, kernel_size=(3, 3), padding=1)
        self.conv22 = nn.Conv2d(in_channels=8*2, out_channels=8*2, kernel_size=(3, 3), padding=1)
        self.conv22dt = nn.ConvTranspose2d(in_channels=8*2, out_channels=1, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)

        # self.conv31 = nn.Conv2d(in_channels=8*2, out_channels=8*4, kernel_size=(3, 3), padding=1, bias=False,
        #                         padding_mode='zeros')
        # self.conv32 = nn.Conv2d(in_channels=8*4, out_channels=8*4, kernel_size=(3, 3), padding=1, bias=False,
        #                         padding_mode='zeros')
        # self.conv32dt = nn.ConvTranspose2d(in_channels=8*4, out_channels=1, kernel_size=3, stride=2, padding=1,
        #                                    output_padding=1)

    def forward(self, x):
        # latent_w = int(np.sqrt(self.latent_dim))
        x = self.dense_layer(x)
        x = torch.reshape(x, (x.shape[0], 8*2, 8, 8))
        x = F.relu(self.conv12(F.relu(self.conv11(x))))
        x = F.relu(self.conv12dt(x))
        x = F.relu(self.conv22(F.relu(self.conv21(x))))
        x = torch.sigmoid(self.conv22dt(x))
        # x = torch.sigmoid(self.conv22dt(x))
        #
        # # x = F.relu(self.conv12(F.relu(self.conv11(x))))
        # # upsample
        # x = F.relu(self.conv12dt(x))
        # x = F.relu(self.conv22(F.relu(self.conv21(x))))
        # x = F.relu(self.conv22dt(x))

        # x = F.relu(self.conv32(F.relu(self.conv31(x))))
        # x = F.relu(self.conv32dt(x))
        return x

class vae(nn.Module):
    def __init__(self, latent_dim=256, do_train=True):
        super(vae, self).__init__()
        self.latent_dim = latent_dim
        self.do_train = do_train

        self.encoder = vae_encoder(latent_dim=self.latent_dim, do_train=do_train)
        self.decoder = vae_decoder(latent_dim=self.latent_dim)

    def forward(self, x):
        mu, log_sigma, z = self.encoder(x)
        xhat = self.decoder(z)
        return mu, log_sigma, z, xhat


# define the losses
# def kl_loss(mu, log_sigma):
#     kl_loss = -0.5 * (1 + log_sigma - tf.square(mu) - tf.exp(log_sigma))
#     return kl_loss


def L_logpx_z(x, xhat):
    # xhat is a sigmoid prob output. x is binary.
    # log p (x | z) = \sum_i (log [xhat_i^x_i] + log [(1 - xhat_i)^(1 - x_i)]
    # x_i log xhat + (1 - x_i) log (1 - xhat)

    # (B, C, H, W)
    # logpx_z = x * torch.log(xhat) + (1 - x) * torch.log(1 - xhat)
    bce_loss = nn.BCELoss(reduction='none')

    logpx_z = -bce_loss(xhat, x)
    # sum over all the pixels
    sum_logpx_z = torch.sum(logpx_z, dim=[2, 3])

    # avg over batch
    avg_sum_logpx_z = torch.mean(sum_logpx_z, dim=0)
    return avg_sum_logpx_z

def L_logpz(z):
    # dimensionality
    D = z.shape[-1]
    z_sq = z*z

    # log p_z = \log [ (1/2\pi)^D/2 ]  - \log [exp(-1/2 z_i T z_i) ]
    # log p_z = -D/2 \log 2pi  -  1/2 z_i ' z_i
    # logp_z = (-0.5) * D * torch.log(torch.tensor(np.pi)) + (-0.5)*torch.sum(z_sq, dim=1)
    logp_z = -0.5 * (D*torch.log(torch.tensor(2*np.pi)) + torch.sum(z_sq, dim=1))
    avg_logp_z = torch.mean(logp_z)
    return avg_logp_z

def L_logqz_x(z, mu, log_sigma):
    D = z.shape[-1]
    epsilons = z - mu
    eps_sq = epsilons * epsilons
    #logqz_x = -torch.sum(log_sigma, dim=1) + - (0.5)*torch.sum(eps_sq, dim=1)
    logqz_x = -(torch.sum(log_sigma, dim=1)) - 0.5*(D*torch.log(torch.tensor(2*np.pi)) + torch.sum(eps_sq, dim=1))
    avg_logqz_x = torch.mean(logqz_x, dim=0)
    return avg_logqz_x


def kl_loss(mu, log_sigma):
    # = E_q[ \log ( q(z|x) / p(z)) ]
    # q(z | x) = N(mu, exp(log_sigma)^2)
    # p(z) = N(0, I)
    # log q(z | x) = log( 1/[(2pi)^D/2 det(Sigma)^1/2] - 0.5 (z - mu) T Sigma^-1 (z - mu)
    #              = log(1/((2pi)^D/2)) - 0.5 log(det(Sigma) - 0.5 \sigma* eps'Sigma^-1 \sigma*eps
    #              = c - 0.5 log(\prod \sigma_i ^2) - 0.5 \sum_i \eps_i ^2
    #              = c - \sum_i (log \sigma_i) - 0.5 \sum_i \eps_i^2
    #
    # log p (z) = log(1/(2pi)^D/2) - 0.5zTz
    # log p(z) = c - 0.5 \sum z_i ^ 2
    # E_eps[log q(z|x) - log p(z)]
    # E_eps[- \sum_i (log \sigma_i) - 0.5 \sum_i eps_i^2 + 0.5 \sum z_i^2]
    # \sum_i[ - log \sigma_i] - 0.5 E[\sum_i eps_i^2] + 0.5 E[(\mu + \sigma*\eps)^2]
    # \sum_i[ - log \sigma_i] - 0.5 * \sum_i 1 + 0.5 (\sum_i \mu_i^2) + E[\sum_i 2*mu_i*eps_i] + \sum_i \sigma_i^2
    # \sum_i[ - log \sigma_i] + 0.5 \sum_i [ - 1 +  mu_i^2 + \sigma_i^2]
    
    loss = -torch.sum(log_sigma, dim=1) + 0.5 * torch.sum(-1 + mu*mu + torch.exp(log_sigma)*torch.exp(log_sigma), dim=1)
    avg_batch_loss = torch.mean(loss)
    return avg_batch_loss

def vae_loss_closed_form(x, mu, log_sigma, z, xhat):
    kl = kl_loss(mu, log_sigma)
    recon_loss = - L_logpx_z(x=x, xhat=xhat)
    loss = recon_loss + kl
    return loss, kl

def vae_loss(x, mu, log_sigma, z, xhat):
    # argmax ELBO = E_q[ \log p(x | z) + \log p(z) - \log q(z | x)]
    #      = E_q[\log p( x | z)] + E_q[\log p(z) - \log q(z | x)]
    #      = L_logpx_z() + E_q[\log ( p(z) / q(z|x) )]
    #      = L_logpx_z() - E_q[\log (q(z|x) / p(z)) ]
    #      = L_logpx_z() - KL(q(z|x) || p(z))
    #      = argmin KL(q(z|x) || p(z)) - L_logpx_z()
    #      = argmin KL(q(z|x) || p(z)) + ReconLoss

    loss = - (L_logpx_z(x, xhat) + L_logpz(z) - L_logqz_x(z, mu, log_sigma))
    return loss

def train_model(config, model, dataloaders, dataset_sizes, num_epochs, optimizer, scheduler, device):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 99999.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-"*10)

        for phase in ['train', 'val']:
            print(phase)
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_recon_loss = 0.0

            # iterate over data
            for samples in tqdm(dataloaders[phase]):
                imgs = samples[0]
                imgs = imgs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    mu, log_sigma, z, imgs_hat = model(imgs)
                    vloss, kl = vae_loss_closed_form(imgs, mu, log_sigma, z, imgs_hat)
                    if phase == 'train':
                        vloss.backward()
                        optimizer.step()
                        # print(vloss.item())
                        # print(kl.item())
                        
                        # for n, p in model.named_parameters():
                            # if n == 'encoder.conv11.weight':
                            #     print(vloss.item())
                            #     print(torch.max(torch.abs(p.grad)))

                # stats
                running_loss = running_loss + vloss.item()*imgs.size()[0]
                running_recon_loss = running_recon_loss + kl.item()*imgs.size()[0]
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_recon_loss = running_recon_loss / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step()
            print("{} Total Loss {:.4f}, Recon Loss {:.4f}".format(phase, epoch_loss, epoch_recon_loss))


            if (phase == 'val') and (epoch_loss < best_loss):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save_file = os.path.join(config['model_save_dir'], "model_loss {:.2f}".format(best_loss) + ".pth")
                torch.save(model.state_dict(), save_file)
                print("VALIDATIOON LOSS REDUCED TO: {:.4f}".format(best_loss))
                print("SAVING MODEL TO {}".format(save_file))

    time_elapsed = time.time() - since
    print("Training completed in {:.0f} mins {:.0f} seconds".format(time_elapsed//60, time_elapsed % 60))
    print("Best validation loss {:.4f}".format(best_loss))

    return model

def visualize_batch(batch):
    # batch (batch_size, 1, img_h, img_w)
    fig, ax = plt.subplots(2,8)
    for iter in range(8):
        img = batch[iter, 0, :, :]
        ax[0,iter].imshow(batch[iter,0,:,:])
        ax[1,iter].hist(batch[iter,0,:,:].flatten(), 500)

    plt.show()



if __name__ == "__main__":
    # load mnist datasets
    train_data = datasets.MNIST(root='../data/',
                                train=True,
                                download=False,
                                transform=torchvision.transforms.Compose([Resize(size=(32, 32)), ToTensor()])
                                )
    test_data = datasets.MNIST(root='../data/',
                               train=False,
                               download=False,
                               transform=torchvision.transforms.Compose([Resize(size=(32, 32)), ToTensor()])
                               )
    # train_subset = Subset(training_data, indices=np.arange(128))
    # test_subset = Subset(test_data, indices=np.arange(128))

    training_loader = DataLoader(dataset=train_data,
                                 batch_size=128,
                                 shuffle=False,
                                 )
    test_loader = DataLoader(dataset=test_data,
                             batch_size=128,
                             shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    config = dict()
    config['model_save_dir'] = './saved_models'
    subprocess.call(['mkdir', '-p', 'saved_models'])

    data_loaders = {}
    data_loaders['train'] = training_loader
    data_loaders['val'] = test_loader

    dataset_sizes = {}
    dataset_sizes['train'] = len(train_data)
    dataset_sizes['val'] = len(test_data)
    mnist_vae = vae(latent_dim=10)
    optimizer = torch.optim.Adam(params=mnist_vae.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    mnist_vae = train_model(config=config, model=mnist_vae, dataloaders=data_loaders,
                            dataset_sizes=dataset_sizes, num_epochs=100, optimizer=optimizer,
                            scheduler=scheduler, device=device)

    # vae_e = vae_encoder()
    # vae_d = vae_decoder(latent_dim=64)
    batch_test = next(iter(test_loader))
    tmu, tlog_sigma, tz, txhat_sigmoid  = mnist_vae(batch_test[0])
    print(txhat_sigmoid.shape)
    loss = vae_loss(batch_test[0], tmu, tlog_sigma, tz, txhat_sigmoid)

    visualize_batch(batch_test[0])
    visualize_batch(txhat_sigmoid.detach())
    # out_e = (vae_e(batch_test[0])[0])
    # out_d = vae_d(out_e)
    # vae_0.eval()
    # x = batch_test[0]
    # mu, log_sigma, z, xhat = vae_0(x)
    # logp_z = L_logpz(z)
    # logpx_z = L_logpx_z(x, xhat)
    # logqz_x = L_logqz_x(z, mu, log_sigma)
    # total_loss = vae_loss(x, mu, log_sigma, z, xhat)



    # model = train_model(vae_0, data_loaders)


    # t = torch.randn(32, 1, 4, 4)
    # conv1 = nn.ConvTranspose2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1)
    # conv2 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1)
    # conv3 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
    #
    # tout = conv3(conv2(conv1(t)))
    # print(tout.shape)

    # for n, p in mnist_vae.named_parameters():
    #     if n == 'decoder.conv22dt.weight':
    #         print(torch.max(p.grad))
    #         print(torch.min(p.grad))