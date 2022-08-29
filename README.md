# Label Propagation for 3D Carotid Vessel Wall Segmentation and Atherosclerosis Diagnosis

## Overview
<p align="center"><img src="https://github.com/ShishuaiHu/Label-Propagation/blob/master/page_files/overview.png" width="90%"></p>

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
bash ./install.sh # install torch and nnUNet (equipped with BA-Net)
```

Due to the limitation of the license attached to the official dataset, we can not provide the preprocessed dataset.
But we provide the data preprocessing scripts in `convertor`.
You can follow the instructions bellow to preprocess the dataset.

1. Download the dataset from [Grand-Challenge](https://vessel-wall-segmentation-2022.grand-challenge.org/data/) and decompressed it.
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

### Generate pseudo labels for Seg-Model-B

### Train Seg-Model-B

### Inference

### Pretrained Weights

Can be downloaded from [here]().

### Citation ✏️ 📄

If you find this repo useful for your research, please consider citing the paper as follows:

```

```

### Acknowledgements

- The whole framework is based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
