import os
import torch

import torchvision.datasets
from torch.utils.data import DataLoader, Dataset
from vae import vae, L_logpx_z
from torchvision.transforms import Resize, ToTensor
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
if __name__=="__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_file = 'saved_models/model_loss 150.05.pth'
    mnist_vae = vae(latent_dim=10, do_train=False)
    mnist_vae.load_state_dict(torch.load(model_file))
    mnist_vae.eval()

    test_dataset = torchvision.datasets.MNIST(root='../data/',
                                              train=False,
                                              download=False,
                                              transform=torchvision.transforms.Compose([Resize((32,32)), ToTensor()]))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)
    test_mu = []
    test_xhat = []
    test_labels = []
    test_x = []
    test_recon_loss = []

    for batch in tqdm(test_loader):
        tmu, tlog_sigma, tz, txhat_sigmoid = mnist_vae(batch[0])
        test_labels.append(batch[1])
        test_mu.append(tmu.detach().numpy())
        test_xhat.append(txhat_sigmoid.cpu().detach().numpy())
        test_x.append(batch[0].cpu().detach().numpy())
        recon_loss = - L_logpx_z(batch[0], txhat_sigmoid).cpu().detach().numpy()
        test_recon_loss.append(recon_loss)


    test_mu = np.concatenate(test_mu)
    test_xhat = np.concatenate(test_xhat).squeeze()
    test_labels = np.concatenate(test_labels)
    test_x = np.concatenate(test_x).squeeze()
    test_recon_loss = np.concatenate(test_recon_loss)

    # plot top 10 samples with max recon loss
    idxs = np.argsort(test_recon_loss)
    fig, ax = plt.subplots(2, 10)
    fig.set_figheight(4)
    fig.set_figwidth(20)
    for i in np.arange(0,10):
        ax[0, i].imshow(test_x[idxs[-(i+1)]])
        ax[1, i].imshow(test_xhat[idxs[-(i+1)]])
        ax[0, i].title.set_text("{:.1f}".format(test_recon_loss[idxs[-(i+1)]]))
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(5, 5)
    plt.show()


    # plot best fit 10 samples with min recon loss
    fig, ax = plt.subplots(2, 10)
    fig.set_figheight(4)
    fig.set_figwidth(20)
    for i in np.arange(0,10):
        ax[0, i].imshow(test_x[idxs[(i)]])
        ax[1, i].imshow(test_xhat[idxs[(i)]])
        ax[0, i].title.set_text("{:.1f}".format(test_recon_loss[idxs[(i)]]))
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.show()

    # sne of features with their labels

