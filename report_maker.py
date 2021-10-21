#%%
import torch
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import rescale

epoch, batch = 0, 0

report_folder = "/home/mriva/Recherche/PhD/SATANN/nnUNet/reports/report{}-{}".format(epoch, batch)

output = torch.load(join(report_folder, "output.pt"), map_location="cpu").detach().numpy()
output_thresholded = torch.load(join(report_folder, "output_thresholded.pt"), map_location="cpu").detach().numpy()
centroids_y, centroids_x, centroids_z = torch.load(join(report_folder, "centroids.pt"), map_location="cpu").detach().numpy()
deltas = torch.load(join(report_folder, "deltas.pt"), map_location="cpu").detach().numpy()

colors = [(1,0,0,1),(0,1,0,1),(0,0,1,1),(0,1,1,1)]

#%%

# Getting each element of the batch
for idx_element, batch_element in enumerate(output_thresholded):
    # Plotting the labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    batch_element = rescale(batch_element.astype(float), (1,0.5,0.3,0.3), mode='constant')

    plt.axis("off")
    for fg_class, fg_class_map in enumerate(batch_element[1:]):
        # Plotting labelmaps
        facecolors = np.resize(colors[fg_class], (*fg_class_map.shape,4)).astype(float)

        den = np.max(fg_class_map)
        if den > 0:
            # normalizing class map like a brute
            #fg_class_map = fg_class_map / den
            fg_class_map = np.clip(fg_class_map, 0, 0.5)

            # assigning alphas
            facecolors[...,3] = fg_class_map + 0.25

            #ax.voxels(fg_class_map, facecolors=facecolors)
        else: print("Class {} has no values".format(fg_class+1))
        # Plotting centroids
        # TODO
    
    plt.title("Epoch {} - Batch {} - Element {}".format(epoch, batch, idx_element))
    plt.show()


# %%
