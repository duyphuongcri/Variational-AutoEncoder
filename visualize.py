import matplotlib.pyplot as plt 
import numpy as np
import nibabel as ni 
import torch

def display_image(x, y):
    assert x.shape == y.shape
    if x.dim() == 5:
        x = torch.reshape(x, (x.shape[0], 80, 96, 80)).cpu().detach().numpy()
        y = torch.reshape(y, (y.shape[0], 80, 96, 80)).cpu().detach().numpy()
        for i in range(len(x)):
            rows = 2
            columns = 1
            fig=plt.figure()
            for idx in range(rows*columns):
                fig.add_subplot(rows, columns, idx+1)
                if idx < columns:
                    plt.imshow(x[i, :, 57, :], cmap="gray", origin="lower")
                else:
                    plt.imshow(y[i, :, 57, :], cmap="gray", origin="lower")
            plt.show()


