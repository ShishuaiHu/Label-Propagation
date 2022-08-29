# -*- coding:utf-8 -*-
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes


def load_dicom(dcm_dir):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(dcm_dir))
    return reader.Execute()


def get_qvs_fname(qvj_path):
    qvs_element = ET.parse(qvj_path).getroot().find('QVAS_Loaded_Series_List').find('QVASSeriesFileName')
    return qvs_element.text


def list_contour_slices(qvs_root):
    """
    :param qvs_root: xml root
    :return: slices with annotations
    """
    avail_slices = []
    image_elements = qvs_root.findall('QVAS_Image')
    for slice_id, element in enumerate(image_elements):
        conts = element.findall('QVAS_Contour')
        if len(conts) > 0:
            avail_slices.append(slice_id)
    return avail_slices


def get_contour(qvsroot, slice_id, cont_type, height, width):
    qvas_img = qvsroot.findall('QVAS_Image')
    conts = qvas_img[slice_id].findall('QVAS_Contour')
    pts = None
    for cont_id, cont in enumerate(conts):
        if cont.find('ContourType').text == cont_type:
            pts = cont.find('Contour_Point').findall('Point')
            break
    if pts is not None:
        contours = []
        for p in pts:
            contx = float(p.get('x')) / 512 * width
            conty = float(p.get('y')) / 512 * height
            # if current pt is different from last pt, add to contours
            if len(contours) == 0 or contours[-1][0] != contx or contours[-1][1] != conty:
                contours.append([contx, conty])
        return np.array(contours)
    return None


def get_bir_slice(qvjroot):
    if qvjroot.find('QVAS_System_Info').find('BifurcationLocation'):
        bif_slice = int(qvjroot.find('QVAS_System_Info').find('BifurcationLocation').find('BifurcationImageIndex').get('ImageIndex'))
        return bif_slice
    else:
        return -1


def get_loc_prop(qvj_root, bif_slice):
    loc_prop = qvj_root.find('Location_Property')
    status_list = dict()
    for loc in loc_prop.iter('Location'):
        loc_ind = int(loc.get('Index')) + bif_slice
        image_quality = int(loc.find('IQ').text)
        # only slices with Image Quality (IQ) > 1 were labeled
        # AHAStatus: 1: Normal; > 1 : Atherosclerotic
        AHA_status = float(loc.find('AHAStatus').text)
        if image_quality>1 and AHA_status == 1:
            # print(f"{loc_ind}: Normal")
            status_list[f"{loc_ind}"] = 0
        elif image_quality>1 and AHA_status >1:
            # print(f"{loc_ind}: Atherosclerotic")
            status_list[f"{loc_ind}"] = 1
    return status_list


def main():
    base = r""  # Decompressed training dataset path
    target = r""  # Expected output path
    target_img = join(target, 'img')
    target_seg = join(target, 'seg')
    maybe_mkdir_p(target_img)
    maybe_mkdir_p(target_seg)
    cases = subfolders(base, join=False)
    for c in cases:
        c_sitk = load_dicom(join(base, c))
        c_npy = sitk.GetArrayFromImage(c_sitk)
        d, h, w = c_npy.shape
        sitk.WriteImage(c_sitk, join(target_img, 'case_{0:04}.nii.gz'.format(int(c))))

        qvj_l_file = join(base, c, c+'L.QVJ')
        qvj_r_file = join(base, c, c+'R.QVJ')
        anno_npy_lumen_l = np.zeros_like(c_npy)
        anno_npy_wall_l = np.zeros_like(c_npy)
        anno_npy_lumen_r = np.zeros_like(c_npy)
        anno_npy_wall_r = np.zeros_like(c_npy)
        anno_npy_abnormal_l = np.zeros_like(c_npy)
        anno_npy_abnormal_r = np.zeros_like(c_npy)
        if isfile(qvj_l_file):
            qvs_file = join(base, c, get_qvs_fname(qvj_l_file))
            qvs_root = ET.parse(qvs_file).getroot()
            annotated_slices = list_contour_slices(qvs_root)
            # print(f"annotated_slices: {annotated_slices}")
            qvj_root = ET.parse(qvj_l_file).getroot()
            bif_slice = get_bir_slice(qvj_root)
            status_list = get_loc_prop(qvj_root, bif_slice)
            # print(f"status_slices: {status_list}")

            for anno_id in status_list.keys():
                anno_npy_abnormal_l[int(anno_id)] = status_list[anno_id] + 1

            for anno_id in annotated_slices:
                lumen_cont = get_contour(qvs_root, anno_id, 'Lumen', height=h, width=w)
                wall_cont = get_contour(qvs_root, anno_id, 'Outer Wall', height=h, width=w)

                for cord in lumen_cont:
                    anno_npy_lumen_l[anno_id, round(cord[1]), round(cord[0])] = 1
                anno_npy_lumen_l[anno_id] = binary_fill_holes(anno_npy_lumen_l[anno_id])
                for cord in wall_cont:
                    anno_npy_wall_l[anno_id, round(cord[1]), round(cord[0])] = 1
                anno_npy_wall_l[anno_id] = binary_fill_holes(anno_npy_wall_l[anno_id]) - anno_npy_lumen_l[anno_id]
            unanno_slices = list()
            annotated_abnormal = [int(i) for i in status_list.keys()]
            l_min, l_max = np.array(annotated_abnormal).min(), np.array(annotated_abnormal).max()+1
            for s in range(l_min, l_max):
                if s not in annotated_abnormal:
                    unanno_slices.append(s)
            for s in unanno_slices:
                anno_npy_abnormal_l[s] = anno_npy_abnormal_l[min(annotated_abnormal, key=lambda x: abs(x - s))]

            unanno_slices = list()
            l_min, l_max = np.array(annotated_slices).min(), np.array(annotated_slices).max() + 1
            for s in range(l_min, l_max):
                if s not in annotated_slices:
                    unanno_slices.append(s)
            for s in unanno_slices:
                anno_npy_lumen_l[s] = anno_npy_lumen_l[min(annotated_slices, key=lambda x: abs(x - s))]
                anno_npy_wall_l[s] = anno_npy_wall_l[min(annotated_slices, key=lambda x: abs(x - s))]

        if isfile(qvj_r_file):
            qvs_file = join(base, c, get_qvs_fname(qvj_r_file))
            qvs_root = ET.parse(qvs_file).getroot()
            annotated_slices = list_contour_slices(qvs_root)
            # print(f"annotated_slices: {annotated_slices}")
            qvj_root = ET.parse(qvj_r_file).getroot()
            bif_slice = get_bir_slice(qvj_root)
            status_list = get_loc_prop(qvj_root, bif_slice)
            # print(f"status_slices: {status_list}")

            for anno_id in status_list.keys():
                anno_npy_abnormal_r[int(anno_id)] = status_list[anno_id] + 1

            for anno_id in annotated_slices:
                lumen_cont = get_contour(qvs_root, anno_id, 'Lumen', height=h, width=w)
                wall_cont = get_contour(qvs_root, anno_id, 'Outer Wall', height=h, width=w)

                for cord in lumen_cont:
                    anno_npy_lumen_r[anno_id, round(cord[1]), round(cord[0])] = 1
                anno_npy_lumen_r[anno_id] = binary_fill_holes(anno_npy_lumen_r[anno_id])
                for cord in wall_cont:
                    anno_npy_wall_r[anno_id, round(cord[1]), round(cord[0])] = 1
                anno_npy_wall_r[anno_id] = binary_fill_holes(anno_npy_wall_r[anno_id]) - anno_npy_lumen_r[anno_id]
            unanno_slices = list()
            r_min, r_max = np.array(annotated_slices).min(), np.array(annotated_slices).max()+1
            for s in range(r_min, r_max):
                if s not in annotated_slices:
                    unanno_slices.append(s)
            for s in unanno_slices:
                anno_npy_lumen_r[s] = anno_npy_lumen_r[min(annotated_slices, key=lambda x: abs(x - s))]
                anno_npy_wall_r[s] = anno_npy_wall_r[min(annotated_slices, key=lambda x: abs(x - s))]

            unanno_slices = list()
            annotated_abnormal = [int(i) for i in status_list.keys()]
            r_min, r_max = np.array(annotated_abnormal).min(), np.array(annotated_abnormal).max() + 1
            for s in range(r_min, r_max):
                if s not in annotated_abnormal:
                    unanno_slices.append(s)
            for s in unanno_slices:
                anno_npy_abnormal_r[s] = anno_npy_abnormal_r[min(annotated_abnormal, key=lambda x: abs(x - s))]

        anno_lumen_l_sitk = sitk.GetImageFromArray(anno_npy_lumen_l)
        anno_lumen_l_sitk.CopyInformation(c_sitk)
        anno_wall_l_sitk = sitk.GetImageFromArray(anno_npy_wall_l)
        anno_wall_l_sitk.CopyInformation(c_sitk)
        anno_abnormal_l_sitk = sitk.GetImageFromArray(anno_npy_abnormal_l)
        anno_abnormal_l_sitk.CopyInformation(c_sitk)
        sitk.WriteImage(anno_lumen_l_sitk, join(target_seg, 'case_{0:04}_lumen_l.nii.gz'.format(int(c))))
        sitk.WriteImage(anno_wall_l_sitk, join(target_seg, 'case_{0:04}_wall_l.nii.gz'.format(int(c))))
        sitk.WriteImage(anno_abnormal_l_sitk, join(target_seg, 'case_{0:04}_abnormal_l.nii.gz'.format(int(c))))

        anno_lumen_r_sitk = sitk.GetImageFromArray(anno_npy_lumen_r)
        anno_lumen_r_sitk.CopyInformation(c_sitk)
        anno_wall_r_sitk = sitk.GetImageFromArray(anno_npy_wall_r)
        anno_wall_r_sitk.CopyInformation(c_sitk)
        anno_abnormal_r_sitk = sitk.GetImageFromArray(anno_npy_abnormal_r)
        anno_abnormal_r_sitk.CopyInformation(c_sitk)
        sitk.WriteImage(anno_lumen_r_sitk, join(target_seg, 'case_{0:04}_lumen_r.nii.gz'.format(int(c))))
        sitk.WriteImage(anno_wall_r_sitk, join(target_seg, 'case_{0:04}_wall_r.nii.gz'.format(int(c))))
        sitk.WriteImage(anno_abnormal_r_sitk, join(target_seg, 'case_{0:04}_abnormal_r.nii.gz'.format(int(c))))


if __name__ == '__main__':
    main()
