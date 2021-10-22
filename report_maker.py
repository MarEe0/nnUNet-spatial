#%%
import torch
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import rescale
from batchgenerators.utilities.file_and_folder_operations import *

#%%
# These values are for CHAOS
relations_gt = [[1, 2, 0.3174332015268699, -0.1333684479096222, -0.0015823315859921916], [1, 3, 0.2831431796169534, -0.1478951149120119, -0.31131461534490806], [1, 4, 0.03368983195113984, -0.14924252339596134, -0.381475596739336],
                [2, 3, -0.03429002190991648, -0.014526667002389693, -0.30973228375891587], [2, 4, -0.28374336957573004, -0.015874075486339123, -0.3798932651533438], [3, 4, -0.24945334766581356, -0.0013474084839494305, -0.07016098139442795]]
patch_size_gt = [40, 192, 256]
print("Expected patch size is {}".format(patch_size_gt))

epochs = range(20)
batches = [0,1,249]
for epoch in epochs:
    for batch in batches :

        report_folder = "/home/mriva/Recherche/PhD/SATANN/nnUNet/reports/report{}-{}".format(epoch, batch)

        output = torch.load(join(report_folder, "output.pt"), map_location="cpu").detach().numpy()
        output_thresholded = torch.load(join(report_folder, "output_thresholded.pt"), map_location="cpu").detach().numpy()
        centroids_y, centroids_x, centroids_z = torch.load(join(report_folder, "centroids.pt"), map_location="cpu").detach().numpy()
        deltas = torch.load(join(report_folder, "deltas.pt"), map_location="cpu").detach().numpy()

        colors = [(1,0,0,1),(0,1,0,1),(0,0,1,1),(0,1,1,1)]

        print("Reporting Epoch {}, batch {}: patch is {} and RG loss is {}".format(epoch, batch, output.shape, np.sum(deltas)))

        # Getting each element of the batch
        for idx_element, batch_element in enumerate(output_thresholded):
            # Plotting the labels
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')

            original_dimensions = batch_element.shape

            #batch_element = rescale(batch_element.astype(float), (1,0.5,0.3,0.3), mode='constant')

            rescaled_dimensions = batch_element.shape

            plt.axis("off")
            for fg_class, fg_class_map in enumerate(batch_element[1:]):
                # Computing FRESH centroid
                _, h, w, d = rescaled_dimensions
                coordinates_map = torch.meshgrid([torch.arange(h), torch.arange(w), torch.arange(d)])
                coordinates_map = (coordinates_map[0].numpy()/h, coordinates_map[1].numpy()/w, coordinates_map[2].numpy()/d)  # normalizing
                coords_y, coords_x, coords_z = coordinates_map
                
                output_sum = np.sum(fg_class_map)
                centroid_y = np.sum(fg_class_map * coords_y) / output_sum
                centroid_x = np.sum(fg_class_map * coords_x) / output_sum
                centroid_z = np.sum(fg_class_map * coords_z) / output_sum

                # Plotting labelmaps
                facecolors = np.resize(colors[fg_class], (*fg_class_map.shape,4)).astype(float)

                den = np.max(fg_class_map)
                if den > 0:
                    # normalizing class map like a brute
                    fg_class_map = np.clip(fg_class_map, 0, 0.5)

                    # assigning alphas
                    facecolors[...,3] = fg_class_map + 0.25

                    #ax.voxels(fg_class_map, facecolors=facecolors)
                    
                    # Plotting centroids
                    y = centroids_y[idx_element, fg_class+1] * rescaled_dimensions[1]
                    x = centroids_x[idx_element, fg_class+1] * rescaled_dimensions[2]
                    z = centroids_z[idx_element, fg_class+1] * rescaled_dimensions[3]
                    #y = centroid_y * rescaled_dimensions[1]
                    #x = centroid_x * rescaled_dimensions[2]
                    #z = centroid_z * rescaled_dimensions[3]

                    #ax.scatter(y, x, z, marker="o", c=[colors[fg_class]], edgecolors="k")
                else: print("Class {} has no values".format(fg_class+1))
            
            #plt.title("Epoch {} - Batch {} - Element {}".format(epoch, batch, idx_element))
            #ax.view_init(0, 90)
            #plt.show()
            #plt.savefig(join(report_folder, "plot.png"))
            #plt.close("all")


# %%
