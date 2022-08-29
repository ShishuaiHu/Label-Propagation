# -*- coding:utf-8 -*-
import shutil

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data


if __name__ == "__main__":
    base = "/media/userdisk0/Sync/VesselWallSeg/data/img"
    labelsdir = "/media/userdisk0/Sync/VesselWallSeg/data/seg"

    task_id = 1001
    task_name = "VesselWallSeg"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []

    all_cases = subfiles(base, suffix='_anno.nii.gz', join=False)

    train_patients = all_cases

    for p in train_patients:
        image_file = join(base, p)
        label_file = join(labelsdir, p.replace('anno', 'anno_all'))
        shutil.copy(image_file, join(imagestr, p.replace('_anno.nii.gz', "_0000.nii.gz")))
        shutil.copy(label_file, join(labelstr, p.replace('_anno.nii.gz', ".nii.gz")))
        train_patient_names.append(p.replace('_anno.nii.gz', ''))

    json_dict = {}
    json_dict['name'] = "VesselWallSeg"
    json_dict['description'] = "Carotid Vessel Wall Segmentation and Atherosclerosis Diagnosis Challenge "
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "https://vessel-wall-segmentation-2022.grand-challenge.org/vessel-wall-segmentation-2022/"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MR",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "vessel wall",
        "2": "lumen",
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i
        in
        train_patient_names]
    json_dict['test'] = []

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
