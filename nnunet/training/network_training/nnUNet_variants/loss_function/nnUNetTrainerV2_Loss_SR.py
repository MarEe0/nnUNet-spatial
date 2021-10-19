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

from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.training.loss_functions.spatial_loss import DC_and_CE_and_RG_loss
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_Loss_SR_RG(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        # Loading relations file
        relations_file = join(self.dataset_directory, "relations.pkl")
        if not isfile(relations_file):
            raise FileNotFoundError("Couldn't find {}. Make sure you add the --compute_relations flag when running nnUNet_plan_and_preprocess.".format(relations_file))
        self.relations = load_pickle(relations_file)
        self.print_to_log_file("Loaded {} relations.".format(len(self.relations)))
        for relation in self.relations:
            self.print_to_log_file("\t ({}->{}): {}, {}, {}".format(*relation))

        # Loading weights file, if available
        loss_weights_file = join(self.dataset_directory, "loss_weights.pkl")
        if isfile(loss_weights_file):
            self.weight_ce, self.weight_dice, self.weight_rg = load_pickle(loss_weights_file)
        else:
            self.print_to_log_file("WARNING! No loss_weights.pkl available.")
            self.weight_ce, self.weight_dice, self.weight_rg = 1,1,1
        self.print_to_log_file("Loss weights: CE={}, Dice={}, RG={}".format(
            self.weight_ce, self.weight_dice, self.weight_rg))

        self.loss = DC_and_CE_and_RG_loss(
            {'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, {"relations": self.relations, "image_dimensions": None},
            weight_ce=self.weight_ce, weight_dice=self.weight_dice, weight_rg=self.weight_rg)
