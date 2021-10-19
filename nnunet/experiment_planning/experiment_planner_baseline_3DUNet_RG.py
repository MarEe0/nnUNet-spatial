#    Copyright 2021 Information Processing and Communications Laboratory (LTCI), Télécom-Paris, Palaiseau, France
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# Authors:
#  * Mateus Riva (mateus.riva@telecom-paris.fr)

import glob

import numpy as np
import torch
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
#from nnunet.preprocessing.cropping import get_case_identifier_from_npz
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import *


class ExperimentPlanner3D_RG(ExperimentPlanner3D_v21):
    """
    Allows for performing Relational Graph computations
    """

    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_RG, self).__init__(
            folder_with_cropped_data, preprocessed_output_folder)

    def compute_relations(self):
        print("\nComputing relations...")
        # We normalize by the patch size value, since this is what will be used in training
        patch_size = self.plans["plans_per_stage"][0]["patch_size"]
        print("normalizing by patch size {}".format(patch_size))

        # We currently only compute relations for stage0
        processed_files = subfiles(
            join(self.preprocessed_output_folder, self.plans["data_identifier"] + "_stage0"), True, None, ".npz", True)

        # Computing all centroids
        foreground_classes = [_class for _class in self.plans["all_classes"] if _class > 0]
        centroids = np.empty((len(processed_files), len(foreground_classes), 3))  # Files X Classes X 3

        print("We have {} files and {} classes".format(centroids.shape[0], centroids.shape[1]))
        # Iterating over all files
        for i, file in enumerate(processed_files):
            #case_identifier = get_case_identifier_from_npz(file)
            seg = np.load(file)['data'][-1]

            h, w, d = seg.shape
            coordinates_map = [t.numpy() for t in torch.meshgrid([torch.arange(h), torch.arange(
                w), torch.arange(d)])]
            # normalizing
            patch_h, patch_w, patch_d = patch_size
            coordinates_map = (coordinates_map[0]/patch_h, coordinates_map[1]/patch_w, coordinates_map[2]/patch_d)
            coords_y, coords_x, coords_z = coordinates_map

            # Iterating over all classes
            for j, current_class in enumerate(foreground_classes):
                labelmap = np.array(seg == current_class).astype(int)
                y = np.sum(labelmap * coords_y)/np.sum(labelmap)
                x = np.sum(labelmap * coords_x)/np.sum(labelmap)
                z = np.sum(labelmap * coords_z)/np.sum(labelmap)
                centroids[i,j] = (y,x,z)

        # Computing the mean centroid per class (over all patients)
        mean_centroids = centroids.mean(axis=0)
        print("mean_centroids are:")
        print(mean_centroids)

        # Assembling relations
        relations = []
        for class1 in range(len(foreground_classes)):
            for class2 in range(class1+1, len(foreground_classes)):
                centroid_diff = mean_centroids[class1] - mean_centroids[class2]
                relations.append([class1+1, class2+1, *centroid_diff])
        
        # Saving relations
        save_pickle(relations, join(self.preprocessed_output_folder, "relations.pkl"))