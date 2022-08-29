# -*- coding:utf-8 -*-
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk


def split(img_file, out_img_l_file=None, out_img_r_file=None):
    img_sitk = sitk.ReadImage(img_file)
    img_sitk.GetSpacing()
    img_sitk.GetDirection()
    img_sitk.GetSize()

    img_npy = sitk.GetArrayFromImage(img_sitk)
    cut_value = img_npy.shape[-1] // 2

    if out_img_l_file is not None:
        img_l_npy = img_npy[:, :, cut_value:]
        img_l_sitk = sitk.GetImageFromArray(img_l_npy)
        # img_l_sitk.SetOrigin(img_sitk.GetOrigin())
        img_l_sitk.SetSpacing(img_sitk.GetSpacing())
        img_l_sitk.SetDirection(img_sitk.GetDirection())
        sitk.WriteImage(img_l_sitk, out_img_l_file)

    if out_img_r_file is not None:
        img_r_npy = img_npy[:, :, :cut_value]
        img_r_sitk = sitk.GetImageFromArray(img_r_npy)
        # img_r_sitk.SetOrigin(img_sitk.GetOrigin())
        img_r_sitk.SetSpacing(img_sitk.GetSpacing())
        img_r_sitk.SetDirection(img_sitk.GetDirection())
        sitk.WriteImage(img_r_sitk, out_img_r_file)


def main():
    base = r"/home/sshu/Downloads/2022.Vessel-Wall-Seg/nii/"
    img_folder = join(base, 'img')
    seg_folder = join(base, 'seg')
    cases = subfiles(img_folder, join=False, suffix='.nii.gz')
    out_base = r"/home/sshu/Downloads/2022.Vessel-Wall-Seg/split"
    out_img_folder = join(out_base, 'img')
    out_seg_folder = join(out_base, 'seg')
    maybe_mkdir_p(out_img_folder)
    maybe_mkdir_p(out_seg_folder)

    for c in cases:
        img_file = join(img_folder, c)
        abnormal_l_file = join(seg_folder, c.replace('.nii.gz', '_abnormal_l.nii.gz'))
        abnormal_r_file = join(seg_folder, c.replace('.nii.gz', '_abnormal_r.nii.gz'))
        lumen_l_file = join(seg_folder, c.replace('.nii.gz', '_lumen_l.nii.gz'))
        lumen_r_file = join(seg_folder, c.replace('.nii.gz', '_lumen_r.nii.gz'))
        wall_l_file = join(seg_folder, c.replace('.nii.gz', '_wall_l.nii.gz'))
        wall_r_file = join(seg_folder, c.replace('.nii.gz', '_wall_r.nii.gz'))

        out_img_l_file = join(out_img_folder, c.replace('.nii.gz', '_l.nii.gz'))
        out_img_r_file = join(out_img_folder, c.replace('.nii.gz', '_r.nii.gz'))
        out_abnormal_l_file = join(out_seg_folder, c.replace('.nii.gz', '_l_abnormal.nii.gz'))
        out_abnormal_r_file = join(out_seg_folder, c.replace('.nii.gz', '_r_abnormal.nii.gz'))
        out_lumen_l_file = join(out_seg_folder, c.replace('.nii.gz', '_l_lumen.nii.gz'))
        out_lumen_r_file = join(out_seg_folder, c.replace('.nii.gz', '_r_lumen.nii.gz'))
        out_wall_l_file = join(out_seg_folder, c.replace('.nii.gz', '_l_wall.nii.gz'))
        out_wall_r_file = join(out_seg_folder, c.replace('.nii.gz', '_r_wall.nii.gz'))

        split(img_file, out_img_l_file, out_img_r_file)

        split(lumen_l_file, out_lumen_l_file, None)
        split(lumen_r_file, None, out_lumen_r_file)

        split(wall_l_file, out_wall_l_file, None)
        split(wall_r_file, None, out_wall_r_file)

        split(abnormal_l_file, out_abnormal_l_file, None)
        split(abnormal_r_file, None, out_abnormal_r_file)


if __name__ == '__main__':
    main()
