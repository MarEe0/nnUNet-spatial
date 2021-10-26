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
#  * Mireia Alenyà (mireia.Alenyà@upf.edu)
#  * Maria Inmaculada (mariainmaculada.villanueva@upf.edu)

import torch
from torch import nn

from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, SoftDiceLossSquared
from nnunet.utilities.nd_softmax import softmax_helper

def get_coordinates_map(image_dimensions):
    _, _, h, w, d = image_dimensions
    coordinates_map = torch.meshgrid([torch.arange(h), torch.arange(
        w), torch.arange(d)])
    coordinates_map = (coordinates_map[0].to(torch.device("cuda"))/h, coordinates_map[1].to(torch.device("cuda"))/w, coordinates_map[2].to(torch.device("cuda"))/d)  # normalizing
    return coordinates_map


class GraphSpatialLoss3D(nn.Module):
    def __init__(self, relations, image_dimensions=None):
        """Spatial loss based on Relational Graph.

        Args:
            relations (list): List of spatial relationships in the format `(source, target, dy, dx, dz)`
            image_dimensions (tuple or None): shape of input images. If None, computed on-the-fly.
        """
        super(GraphSpatialLoss3D, self).__init__()

        self.relations = relations

        if image_dimensions is not None:
            self.image_dimensions = image_dimensions
            self.coordinates_map = get_coordinates_map(image_dimensions)
        else:
            self.image_dimensions = None
            self.coordinates_map = None

        # TODO: verify if we should switch to 1/n_class
        self.threshold = nn.Threshold(0.5, 0)

    def forward(self, output):
        # Computing centroids
        if self.image_dimensions is None:  # Initializing coordinates_map on-the-fly
            self.coordinates_map = get_coordinates_map(output.size())

        coords_y, coords_x, coords_z = self.coordinates_map  # Getting coordinates map


        coords_y = coords_y.expand(output.size())  # Fitting to the batch shape
        coords_x = coords_x.expand(output.size())
        coords_z = coords_z.expand(output.size())


        # Thresholding with the defined threshold
        output_thresholded = self.threshold(output)
        # The total sum will be used for norm
        output_sum = torch.sum(output_thresholded, dim=[2, 3, 4])

        centroids_y = torch.sum(output_thresholded * coords_y,
                                dim=[2, 3, 4]) / output_sum
        centroids_x = torch.sum(output_thresholded * coords_x,
                                dim=[2, 3, 4]) / output_sum
        centroids_z = torch.sum(output_thresholded * coords_z,
                                dim=[2, 3, 4]) / output_sum

        # Computing loss per relation
        dy_all = torch.empty(len(self.relations))
        dx_all = torch.empty(len(self.relations))
        dz_all = torch.empty(len(self.relations))
        for relation_index, relation in enumerate(self.relations):
            i, j, dy_gt, dx_gt, dz_gt = relation
            dy = centroids_y[:, i] - centroids_y[:, j]
            dx = centroids_x[:, i] - centroids_x[:, j]
            dz = centroids_z[:, i] - centroids_z[:, j]

            diff_y = dy - dy_gt
            diff_x = dx - dx_gt
            diff_z = dz - dz_gt

            dy_error = torch.mean(torch.square(
                torch.nan_to_num(diff_y, nan=0, posinf=0, neginf=0)))
            dx_error = torch.mean(torch.square(
                torch.nan_to_num(diff_x, nan=0, posinf=0, neginf=0)))
            dz_error = torch.mean(torch.square(
                torch.nan_to_num(diff_z, nan=0, posinf=0, neginf=0)))

            dy_all[relation_index] = dy_error
            dx_all[relation_index] = dx_error
            dz_all[relation_index] = dz_error

        # Aggregating the errors - TODO: other aggregations?
        error = dy_all.sum() + dx_all.sum() + dz_all.sum()
        return error


class DC_and_CE_and_RG_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, rg_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1, weight_rg=1,
                 log_dice=False, ignore_label=None):
        """Obtains an weighted sum of Dice, Cross-Entropy and Relational-Graph loss.

        Args:
            soft_dice_kwargs ([type]): [description]
            ce_kwargs ([type]): [description]
            rg_kwargs ([type]): [description]
            aggregate (str, optional): [description]. Defaults to "sum".
            square_dice (bool, optional): [description]. Defaults to False.
            weight_ce (int, optional): [description]. Defaults to 1.
            weight_dice (int, optional): [description]. Defaults to 1.
            weight_rg (int, optional): [description]. Defaults to 1.
            log_dice (bool, optional): [description]. Defaults to False.
            ignore_label ([type], optional): [description]. Defaults to None.
        """
        super(DC_and_CE_and_RG_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_rg = weight_rg
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(
                apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(
                apply_nonlin=softmax_helper, **soft_dice_kwargs)

        self.rg = GraphSpatialLoss3D(**rg_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target,
                          loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(
            net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        rg_loss = self.rg(net_output) if self.weight_rg != 0 else 0

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * \
                dc_loss + self.weight_rg * rg_loss
        else:
            # reserved for other stuff (later)
            raise NotImplementedError("nah son")
        return result
