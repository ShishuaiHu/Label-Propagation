# -*- coding:utf-8 -*-
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
import numpy as np


def calculate_min_max(out_seg_npy):
    tmp_npy = out_seg_npy.copy()

    min_value = None
    max_value = None
    for i in range(tmp_npy.shape[0]):
        if 1 in tmp_npy[i] and (2 in tmp_npy[i] or 3 in tmp_npy[i]):
            min_value = i
            break
    for i in range(tmp_npy.shape[0])[::-1]:
        if 1 in tmp_npy[i] and (2 in tmp_npy[i] or 3 in tmp_npy[i]):
            max_value = i + 1
            break

    return min_value, max_value


def process_img_seg(img_file, abnormal_file, lumen_file, wall_file, out_img_file, out_seg_file):
    img_sitk = sitk.ReadImage(img_file)
    abnormal_sitk = sitk.ReadImage(abnormal_file)
    lumen_sitk = sitk.ReadImage(lumen_file)
    wall_sitk = sitk.ReadImage(wall_file)

    img_npy = sitk.GetArrayFromImage(img_sitk)
    abnormal_npy = sitk.GetArrayFromImage(abnormal_sitk)
    lumen_npy = sitk.GetArrayFromImage(lumen_sitk)
    wall_npy = sitk.GetArrayFromImage(wall_sitk)

    out_seg_npy = np.zeros(img_npy.shape)

    if abnormal_npy.max() < 1 or lumen_npy.max() < 1 or wall_npy.max() < 1:
        return

    out_seg_npy[lumen_npy == 1] = 1
    out_seg_npy[(wall_npy == 1) * (abnormal_npy == 1)] = 2
    out_seg_npy[(wall_npy == 1) * (abnormal_npy == 2)] = 3

    this_min, this_max = calculate_min_max(out_seg_npy)

    out_seg_sitk = sitk.GetImageFromArray(out_seg_npy[this_min:this_max])
    out_seg_sitk.SetSpacing(img_sitk.GetSpacing())
    out_seg_sitk.SetDirection(img_sitk.GetDirection())
    sitk.WriteImage(out_seg_sitk, out_seg_file)

    out_img_sitk = sitk.GetImageFromArray(img_npy[this_min:this_max])
    out_img_sitk.SetSpacing(img_sitk.GetSpacing())
    out_img_sitk.SetDirection(img_sitk.GetDirection())
    sitk.WriteImage(out_img_sitk, out_img_file)


def main():
    base = r"/home/sshu/Downloads/2022.Vessel-Wall-Seg/split/"
    img_folder = join(base, 'img')
    seg_folder = join(base, 'seg')
    cases = subfiles(img_folder, join=False, suffix='.nii.gz')
    out_base = r"/home/sshu/Downloads/2022.Vessel-Wall-Seg/split_label"
    out_img_folder = join(out_base, 'img')
    out_seg_folder = join(out_base, 'seg')
    maybe_mkdir_p(out_img_folder)
    maybe_mkdir_p(out_seg_folder)

    for c in cases:
        img_file = join(img_folder, c)
        abnormal_file = join(seg_folder, c.replace('.nii.gz', '_abnormal.nii.gz'))
        lumen_file = join(seg_folder, c.replace('.nii.gz', '_lumen.nii.gz'))
        wall_file = join(seg_folder, c.replace('.nii.gz', '_wall.nii.gz'))

        out_img_file = join(out_img_folder, c)
        out_seg_file = join(out_seg_folder, c)

        process_img_seg(img_file, abnormal_file, lumen_file, wall_file, out_img_file, out_seg_file)


if __name__ == '__main__':
    main()
