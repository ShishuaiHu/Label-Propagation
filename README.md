# Label Propagation for 3D Carotid Vessel Wall Segmentation and Atherosclerosis Diagnosis

## Overview
<p align="center"><img src="https://github.com/ShishuaiHu/Label-Propagation/blob/master/figures/overview.png" width="90%"></p>

## Preparing

Clone this repo:
```bash
git clone https://github.com/ShishuaiHu/Label-Propagation.git
cd Label-Propagation
```

Create experimental environment using virtual env:
```bash
virtualenv .env --python=3.8 # create
source .env/bin/activate # activate

cd nnUNet
pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .
pip install hiddenlayer graphviz IPython
```

Due to the limitation of the license attached to the official dataset, we can not provide the preprocessed dataset.
But we provide the data preprocessing scripts in `convertor`.
You can follow the instructions bellow to preprocess the dataset.

1. Download the dataset from [Grand-Challenge](https://vessel-wall-segmentation-2022.grand-challenge.org/data/) and decompress it.
2. Set the `base` and `target` path in `convert.py`.
3. Run `pip install requirements.txt` in `convertor` folder to install the dependencies, and run `python convert.py` to preprocess the dataset.

Also, you need to cut the image and labels to single-side and preserve only the slices with interpolated labels. The scripts on this are not provided.
You also need to place the data into nnUNet raw data path.

Configure the paths in `.envrc` to the proper path:
```bash
echo -e '
export nnUNet_raw_data_base="nnUNet raw data path you want to store in"
export nnUNet_preprocessed="nnUNet preprocessed data path you want to store in, SSD is prefered"
export RESULTS_FOLDER="nnUNet trained models path you want to store in"' > .envrc

source .envrc # make the variables take effect
```

## Training

### Train Seg-Model-A
```bash
nnUNet_plan_and_preprocess -t 1001 --verify_dataset_integrity

nnUNet_train 3d_fullres nnUNetTrainerV2_100Epoch_4Fold 1001 0
nnUNet_train 3d_fullres nnUNetTrainerV2_100Epoch_4Fold 1001 1
nnUNet_train 3d_fullres nnUNetTrainerV2_100Epoch_4Fold 1001 2
nnUNet_train 3d_fullres nnUNetTrainerV2_100Epoch_4Fold 1001 3
```

### Generate pseudo labels for Seg-Model-B
```bash
nnUNet_predict -i $TRAINING_IMAGE_FOLDER -o $OUTPUT_FOLDER -t 1001 -m 3d_fullres -tr nnUNetTrainerV2_100Epoch_4Fold --save_npz
```

Move the predicted nii files to Task1002 in `$nnUNet_raw_data_base/nnUNet_raw_data`, and generate Task1002.

### Train Seg-Model-B
```bash
nnUNet_plan_and_preprocess -t 1002 --verify_dataset_integrity

nnUNet_train 3d_fullres nnUNetTrainerV2_500Epoch 1002 0
nnUNet_train 3d_fullres nnUNetTrainerV2_500Epoch 1002 1
nnUNet_train 3d_fullres nnUNetTrainerV2_500Epoch 1002 2
nnUNet_train 3d_fullres nnUNetTrainerV2_500Epoch 1002 3
```

### Inference
```bash
nnUNet_predict -i $TESTING_IMAGE_FOLDER -o $OUTPUT_FOLDER -t 1002 -m 3d_fullres -tr nnUNetTrainerV2_500Epoch --save_npz
```

## Pretrained Weights

Can be downloaded from [Releases](https://github.com/ShishuaiHu/Label-Propagation/releases/download/public/Seg-Model.zip).

### Citation ✏️ 📄

If you find this repo useful for your research, please consider citing the paper as follows:

```
@misc{https://doi.org/10.48550/arxiv.2208.13337,
  doi = {10.48550/ARXIV.2208.13337},
  url = {https://arxiv.org/abs/2208.13337},
  author = {Hu, Shishuai and Liao, Zehui and Xia, Yong},
  title = {Label Propagation for 3D Carotid Vessel Wall Segmentation and Atherosclerosis Diagnosis},
  publisher = {arXiv},
  year = {2022},
}
```

### Acknowledgements

- The whole framework is based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
