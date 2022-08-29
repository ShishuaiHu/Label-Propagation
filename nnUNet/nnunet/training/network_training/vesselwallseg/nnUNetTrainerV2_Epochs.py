# -*- coding:utf-8 -*-
from nnunet.training.network_training.vesselwallseg.nnUNetTrainerV2_100Epoch_4Fold import nnUNetTrainerV2_100Epoch_4Fold
from nnunet.training.network_training.vesselwallseg.nnUNetTrainerV2_100Epoch_4Fold import nnUNetTrainerV2_ResencUNet_4Fold


class nnUNetTrainerV2_200Epoch(nnUNetTrainerV2_100Epoch_4Fold):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 200


class nnUNetTrainerV2_400Epoch(nnUNetTrainerV2_100Epoch_4Fold):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 400


class nnUNetTrainerV2_500Epoch(nnUNetTrainerV2_100Epoch_4Fold):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 500


class nnUNetTrainerV2_1000Epoch(nnUNetTrainerV2_100Epoch_4Fold):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000


class nnUNetTrainerV2_ResencUNet_500Epoch(nnUNetTrainerV2_ResencUNet_4Fold):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 500
