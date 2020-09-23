"""
Author: Duy-Phuong Dao
Email: phuongdd.1997@gmail.com (or duyphuongcri@gmail.com)
"""

import nibabel as ni
import numpy as np
import os, glob
import torch 
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt 

import model
import loss
import dataloader
import visualize
##---------Settings--------------------------
batch_size = 32
##############
path_data = "/home/ubuntu/Desktop/DuyPhuong/VAE/data/test"
path_model = "./checkpoint/vae_t1/model_vae_epoch_43.pt"
####################
verbose = True
log = print if verbose else lambda *x, **i: None
np.random.seed(10)
torch.manual_seed(10)
###################
###################
criterion_rec = loss.L1Loss()
criterion_dis = loss.KLDivergence()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(" GPU is activated" if device else " CPU is activated")
no_images = len(glob.glob(path_data + "/*.nii"))
print("Number of MRI images: ", no_images)
if __name__=="__main__":
    vae_model = torch.load(path_model)
    vae_model.to(device)
    #log(vae_model)
    loss_rec_batch, loss_KL_batch, total_loss_batch = 0, 0, 0
    loss_rec, loss_KL, total_loss = 0, 0, 0
        
    # interfere phrase
    vae_model.eval()
    z = []
    with torch.no_grad():
        for batch_images in dataloader.load_mri_images(path_data, batch_size):
            batch_images = batch_images.to(device)
            y, z_mean, z_log_sigma = vae_model(batch_images)
            z.append((z_mean + z_log_sigma.exp()*vae_model.epsilon).cpu().detach().numpy())

            # Measure loss
            loss_rec_batch = criterion_rec(batch_images, y)
            loss_KL_batch = criterion_dis(z_mean.to(device), z_log_sigma.to(device))
            total_loss_batch = loss_rec_batch + loss_KL_batch

            loss_rec += loss_rec_batch.item() * batch_images.shape[0]
            loss_KL += loss_KL_batch.item() * batch_images.shape[0]
            total_loss += total_loss_batch.item() * batch_images.shape[0]

            # display
            #visualize.display_image(batch_images, y)
    print("Reconstruct Loss: {:.4f} | KL Loss: {:.4f}".format(loss_rec/ no_images, loss_KL/ no_images))
    z = np.concatenate(z, axis=0)
    dataloader.save_data_to_csv("./latent_space_z.csv", z)
    print(z.shape)






           

    