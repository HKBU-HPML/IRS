# IRS
IRS: A Large Naturalistic Indoor Robotics Stereo Dataset to Train Deep Models for Disparity and Surface Normal Estimation

# Introduction

**IRS** is an open dataset for indoor robotics vision tasks, especially disparity and surface normal estimation. It contains totally 103,316 samples covering a wide range of indoor scenes, such as home, office, store and restaurant. 

|<img src="/imgs/left.png" width="100%" > | <img src="/imgs/right.png" width="100%" > |
|:--:|:--:|
|Left image|Right image|
|<img src="/imgs/disparity.png" width="100%" > | <img src="/imgs/normal.png" width="100%" > | 
|Disparity map|Surface normal map|

*****

# Overview of IRS

Rendering Characteristic|Options
:--|:--:
indoor scene class|home(31145), office(43417), restaurant(22058), store(6696)
object class|desk, chair, sofa, glass, mirror, bed, bedside table, lamp, wardrobe, etc.
brightness|over-exposure(>1300), darkness(>1700)
light behavior|bloom(>1700), lens flare(>1700), glass transmission(>3600), mirror reflection(>3600)

We give some sample of different indoor scene characteristics as follows.

|<img src="/imgs/home.png" width="100%" > | <img src="/imgs/office.png" width="100%">  | <img src="/imgs/restaurant.png" width="100%" >|
|:--:|:--:|:--:|
|Home|Office|Restaurant|
|<img src="/imgs/normal_light.png" width="100%" > | <img src="/imgs/over_exposure.png" width="100%" > | <img src="/imgs/dark.png" width="100%" >|
|Normal light|Over exposure|Darkness|
|<img src="/imgs/glass.png" width="100%" > | <img src="/imgs/mirror.png" width="100%" > | <img src="/imgs/metal.png" width="100%" >|
|Glass|Mirror|Metal|

# Network Structure of DTN-Net

We designed a novel deep model, DTN-Net, to predict the surface normal map by refining the initial one transformed from the predicted disparity. DTN-Net (**D**isparity **T**o **N**ormal Network) is comprised of two modules, **[RD-Net](https://arxiv.org/pdf/1512.02134.pdf)** and NormNetS. First, RD-Net predicts the disparity map for the input stereo images. Then we apply the transformation from disparity to normal in **[GeoNet](https://arxiv.org/abs/2012.06980)**, denoted by D2N Transform, to produces the initial coarse normal map. Finally, NormNetS takes the stereo images, the predicted disparity map by RD-Net and the initial normal map as input and predicts the final normal map. The structure of NormNetS is similar to  **[DispNetS](https://arxiv.org/pdf/1512.02134.pdf)** except that the final convolution layer outputs three channels instead of one, as each pixel normal has three dimension (x,y,z).

<div align="center">
<img src="/imgs/DTN-Net.png" width="95%" >
</div>

# Paper
Q. Wang<sup>\*,1</sup>, S. Zheng<sup>\*,1</sup>, Q. Yan<sup>\*,2</sup>, F. Deng<sup>2</sup>, K. Zhao<sup>&#8224;,1</sup>, X. Chu<sup>&#8224;,1</sup>.

IRS: A Large Naturalistic Indoor Robotics Stereo Dataset to Train Deep Models for Disparity and Surface Normal Estimation. [\[preprint\]](https://arxiv.org/abs/1912.09678)

<font size=2>
* indicates equal contribution. &#8224; indicates corresponding authors.<br>
<sup>1</sup>Department of Computer Science, Hong Kong Baptist University. <sup>2</sup>School of Geodesy and Geomatics, Wuhan University.
</font>

<!--
Q. Wang<sup>*,1</sup>, S. Zheng<sup>*,1</sup>, Q. Yan<sup>*,2</sup>, F. Deng<sup>2</sup>, K. Zhao<sup>&#8224;,1</sup>, X. Chu<sup>&#8224;,1</sup>.[preprint](/pdfs/IRS_indoor_robotics_stereo_dataset.pdf)

[IRS : A Large Synthetic Indoor Robotics Stereo Dataset for Disparity and Surface Normal Estimation](https://www.github.com)

<font size=2>
* indicates equal contribution. &#8224; indicates corresponding authors.<br>
<sup>1</sup>Department of Computer Science, Hong Kong Baptist University. <sup>2</sup>School of Geodesy and Geomatics, Wuhan University.
</font>

-->

# Download 
You can use the following BaiduYun link to download our dataset. More download links, including Google Drive and OneDrive, will be provided soon.

BaiduYun: <a href="https://pan.baidu.com/s/1iWZt3JklcX5iXdQqotY4uA" target="_blank">https://pan.baidu.com/s/1iWZt3JklcX5iXdQqotY4uA</a> (code: gxlw)

Google Drive: <a href="https://drive.google.com/drive/folders/1s6zUHkyQdCfxIq4OVzCp1CI6-_e4kGtu" target="_blank">https://drive.google.com/drive/folders/1s6zUHkyQdCfxIq4OVzCp1CI6-_e4kGtu</a>

Tips for Google Drive:
- Go to OAuth 2.0 Playground https://developers.google.com/oauthplayground/
- In the Select the Scope box, paste https://www.googleapis.com/auth/drive.readonly
- Click Authorize APIs and then Exchange authorization code for tokens
- Copy the Access token
- Run in terminal
```
curl -H "Authorization: Bearer ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/FILE_ID?alt=media -o FILE_NAME
```

# Video Demonstration

<!--
<iframe width="560" height="315" src="https://www.youtube.com/embed/jThNQFHNU_s" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
-->

[![IRS Dataset and DTN-Net](http://img.youtube.com/vi/jThNQFHNU_s/0.jpg)](http://www.youtube.com/watch?v=jThNQFHNU_s)

# Usage

### Dependencies

- [Python 3.7](https://www.python.org/downloads/)
- [PyTorch 1.6.0+](http://pytorch.org)
- torchvision 0.5.0+
- CUDA 10.1 (https://developer.nvidia.com/cuda-downloads)

### Install

We recommend using [conda](https://www.anaconda.com/distribution/) for installation: 

```shell
conda env create -f environment.yml
```

Install dependencies:

```
cd layers_package
./install.sh

# install OpenEXR (https://www.openexr.com/)
sudo apt-get update
sudo apt-get install openexr
```

### Dataset

Download IRS dataset from https://pan.baidu.com/s/1iWZt3JklcX5iXdQqotY4uA (BaiduYun). \
Extract zip files and put them in correct folder:
```
data
└── IRSDataset
    ├── Home
    ├── Office
    ├── Restaurant
    └── Store
```

### Pretrained Model
"FT3D" denotes FlyingThings3D.

| | IRS | FT3D | IRS+FT3D |
|---|---|---|---|
|FADNet|[fadnet-ft3d.pth](https://drive.google.com/file/d/1XH0l9wS-CDfVCQJDE1CZOx-g0qgb4VmE/view?usp=sharing)|[fadnet-irs.pth](https://drive.google.com/file/d/13mKnv4Z19jpxu_t3gWstrPAx3Cqaxq04/view?usp=sharing)|[fadnet-ft3d-irs.pth](https://drive.google.com/file/d/1EPkCeedWFo0xqL4W1Zoe9VTde4zv8bKT/view?usp=sharing)|
|GwcNet|[gwcnet-ft3d.pth](https://drive.google.com/file/d/1Ptk92srE_WicztONIm1m0vm6edsgllac/view?usp=sharing)|[gwcnet-irs.pth](https://drive.google.com/file/d/1Y7RVxWOGOHqAY7J2y52RzvukIySETgzs/view?usp=sharing)|[gwcnet-ft3d-irs.pth](https://drive.google.com/file/d/1YTsDlyFr8FjqCjBE8AJkt2UaTWgBbUG1/view?usp=sharing)|

| | DTN-Net | DNF-Net | NormNetS |
|---|---|---|---|
|IRS|[dtonnet-irs.pth](https://drive.google.com/file/d/1jdg35tPK7Ii2bYL-q1WIDpFhJY8l2RJK/view?usp=sharing)|[dnfusionnet-irs.pth](https://drive.google.com/file/d/1xnT-0aaVLKjb4engBiZXPRFui3GEH_GN/view?usp=sharing)|[normnets-irs.pth](https://drive.google.com/file/d/1PZub6sQKeHH9HP9JjzQ_fmJXZ6y_u2Lu/view?usp=sharing)|


### Train

There are configurations for train in "exp_configs" folder. You can create your own configuration file as samples. \
As an example, following configuration can be used to train a DispNormNet on IRS dataset: \
\
/exp_configs/dtonnet.conf
```
net=dispnormnet
loss=loss_configs/dispnetcres_irs.json
outf_model=models/${net}-irs
logf=logs/${net}-irs.log

lr=1e-4
devices=0,1,2,3

dataset=irs #sceneflow, irs, sintel
trainlist=lists/IRSDataset_TRAIN.list
vallist=lists/IRSDataset_TEST.list

startR=0
startE=0
endE=10
batchSize=16
maxdisp=-1
model=none
```

Then, the configuration should be specified in the "train.sh"\
\
/train.sh
```
dnn="${dnn:-dispnormnet}"
source exp_configs/$dnn.conf

python main.py --cuda --net $net --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE --endEpoch $endE \
               --model $model \
               --maxdisp $maxdisp \
               --manualSeed 1024 \
```

Lastly, use the following command to start a train
```
./train.sh
```

### Evaluation

There is a script for evaluation with a model from a train \
\
/detech.sh
```
dataset=irs
net=dispnormnet

model=models/dispnormnet-irs/model_best.pth
outf=detect_results/${net}-${dataset}/

filelist=lists/IRSDataset_TEST.list
filepath=data

CUDA_VISIBLE_DEVICES=0 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0 --net ${net} --disp-on --norm-on
```

Use the script in your configuration, and then get result in detect_result folder.\
\
Disparity results are saved in png format as default. \
Normal results are saved in exr format as default. \
\
If you want to change the output format, you need to modify "detecter.py" and use save function as follow
```
# png
skimage.io.imsave(filepath, image)

# pfm
save_pfm(filepath, data)

# exr
save_exr(data, filepath)
```


### EXR Viewer

For viewing files in exr format, we recommand a free software
- [RenderDoc](https://renderdoc.org/)


# Contact

Please contact us at [qiangwang@comp.hkbu.edu.hk](mailto:qiangwang@comp.hkbu.edu.hk) if you have any question. 
