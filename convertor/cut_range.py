# -*- coding:utf-8 -*-
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk


def cut_label_file(case, s_min, s_max, in_seg_folder, out_seg_folder):
    file_suffixes = ['_abnormal_l.nii.gz', '_abnormal_r.nii.gz', '_lumen_l.nii.gz', '_lumen_r.nii.gz', '_wall_l.nii.gz', '_wall_r.nii.gz']

    for file_suffix in file_suffixes:
        label_file = join(in_seg_folder, case.replace('.nii.gz', file_suffix))
        if isfile(label_file):
            label_sitk = sitk.ReadImage(label_file)
            label_npy = sitk.GetArrayFromImage((label_sitk))
            out_label_npy = label_npy[s_min-1:s_max+1]
            out_label_sitk = sitk.GetImageFromArray(out_label_npy)
            out_label_sitk.SetSpacing(label_sitk.GetSpacing())
            out_label_sitk.SetDirection(label_sitk.GetDirection())
            sitk.WriteImage(out_label_sitk, join(out_seg_folder, case.replace('.nii.gz', file_suffix)))


def main():
    in_base = r"/home/sshu/Downloads/2022.Vessel-Wall-Seg/nii_newest"
    in_img_folder = join(in_base, 'img')
    in_seg_folder = join(in_base, 'seg')
    out_base = r"/home/sshu/Downloads/2022.Vessel-Wall-Seg/nii_cut"
    out_img_folder = join(out_base, 'img')
    out_seg_folder = join(out_base, 'seg')
    maybe_mkdir_p(out_img_folder)
    maybe_mkdir_p(out_seg_folder)
    cases = subfiles(in_img_folder, suffix='.nii.gz', join=False)

    with open('/home/sshu/Downloads/2022.Vessel-Wall-Seg/range.csv', 'r') as f:
        cases_info_list = f.read().split('\n')[1:-1]
    cases_info = dict()
    for i in cases_info_list:
        this_info = i.split(',')
        cases_info[this_info[0]] = dict()
        cases_info[this_info[0]]['min'] = this_info[1]
        cases_info[this_info[0]]['max'] = this_info[2]

    for case in cases:
        s_min = int(cases_info[str(int(case[5:9]))]['min'])
        s_max = int(cases_info[str(int(case[5:9]))]['max'])
        in_img_file = join(in_img_folder, case)
        in_img_sitk = sitk.ReadImage(in_img_file)
        in_img_npy = sitk.GetArrayFromImage(in_img_sitk)
        out_img_npy = in_img_npy[s_min-1:s_max+1]
        out_img_sitk = sitk.GetImageFromArray(out_img_npy)
        out_img_sitk.SetSpacing(in_img_sitk.GetSpacing())
        out_img_sitk.SetDirection(in_img_sitk.GetDirection())
        sitk.WriteImage(out_img_sitk, join(out_img_folder, case))

        cut_label_file(case, s_min, s_max, in_seg_folder, out_seg_folder)


if __name__ == '__main__':
    main()
