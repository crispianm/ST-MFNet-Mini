# ST-MFNet Mini: Knowledge Distillation-Driven Frame Interpolation

[Original Project](https://danielism97.github.io/ST-MFNet) | [Original Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Danier_ST-MFNet_A_Spatio-Temporal_Multi-Flow_Network_for_Frame_Interpolation_CVPR_2022_paper.html) | [arXiv](https://arxiv.org/abs/2302.08455)


## Dependencies and Installation
The following packages were used to evaluate the model.

- python==3.8.8
- pytorch==1.7.1
- torchvision==0.8.2
- cudatoolkit==11.3
- opencv-python==4.5.1.48
- numpy==1.19.2
- pillow==8.1.2
- cupy==9.0.0
- sk-video==1.1.10

Installation with anaconda:

```
conda create -n stmfnet python=3.8.8
conda activate stmfnet
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge cupy
pip install opencv-python==4.5.1.48
pip install sk-video==1.1.10
```

## Preparing datasets
### Training sets:
[[Vimeo-90K]](http://toflow.csail.mit.edu/) | [[BVI-DVC quintuplets]](https://uob-my.sharepoint.com/:f:/g/personal/mt20523_bristol_ac_uk/EnHgdYU1cwNEhl-3BXFL8ncBXXGpg7u3N_oiXQ4OJuLXtw?e=fxdITc)

### Test sets: 
[[UCF101]](https://sites.google.com/view/xiangyuxu/qvi_nips19) | [[DAVIS]](https://sites.google.com/view/xiangyuxu/qvi_nips19) | [[SNU-FILM]](https://myungsub.github.io/CAIN/) | [[VFITex]](https://uob-my.sharepoint.com/:f:/g/personal/mt20523_bristol_ac_uk/EsRvHziGSA9BpQ7J02GO9PoBpVzoXFlrHHjwHCYYAsDIOQ?e=gYxehR)


The dataset folder names should be lower-case and structured as follows.
```
└──── <data directory>/
    ├──── bvidvc/quintuplets
    |   ├──── 00000/
    |   ├──── 00001/
    |   ├──── ...
    |   └──── 17599/
    ├──── davis90/
    |   ├──── bear/
    |   ├──── bike-packing/
    |   ├──── ...
    |   └──── walking/
    ├──── snufilm/
    |   ├──── test-easy.txt
    |   ├──── test-medium.txt
    |   ├──── test-hard.txt
    |   ├──── test-extreme.txt
    |   └──── data/SNU-FILM/test/...
    ├──── ucf101/
    |   ├──── 0/
    |   ├──── 1/
    |   ├──── ...
    |   └──── 99/
    ├──── vfitex/
    |   ├──── beach02_4K_mitch/
    |   ├──── bluewater_4K_pexels/
    |   ├──── ...
    |   └──── waterfall_4K_pexels/
    └──── vimeo_septuplet/
        ├──── sequences/
        ├──── readme/
        ├──── sep_testlist.txt
        └──── sep_trainlist.txt

```

## Downloading the pre-trained models
Download the pre-trained ST-MFNet from [here](https://drive.google.com/file/d/1s5JJdt5X69AO2E2uuaes17aPwlWIQagG/view?usp=sharing).
The pre-trained ST-MFNet Mini is included in the repository, inside the models folder, or as a download [here](https://github.com/crispianm/ST-MFNet-Mini/raw/main/models/stmfnet_mini.pth).

## Training
### Phase 1: Architecture Compression
OBProxSG options can be modified in `utility.py` inside of the make_optimizer function. Feel free to experiment with other options, but here is an example:
```
python train.py \
--load <path to pre-trained stmfnet model.pth> \
--data_dir <path to data directory> \
--out_dir "./train_results" \
--epochs 20 \
--optimizer "OBProxSG"
```


### Phase 2: Knowledge Distillation
Feel free to experiment with other options, but here is an example:
```
python distill.py \
--student "student_STMFNet" \
--teacher "STMFNet" \
--distill_loss_fn "1*Lap" \
--student_loss_fn  "1*Lap" \
--alpha 0.1 \
--data_dir <path to data directory> \
--out_dir "./distill_results" \
--optimizer "ADAMax"
```


## Evaluation (on test sets)
```
python evaluate.py \
--net student_STMFNet \
--data_dir <data directory> \
--checkpoint <path to pre-trained model (.pth file)> \
--out_dir eval_results \
--dataset <dataset name>
```
where `<dataset name>` should be the same as the class names defined in `data/testsets.py`, e.g. `Snufilm_extreme_quintuplet`.


## Acknowledgement
Lots of code in this repository are adapted/taken from the following repositories:

- [AdaCoF-pytorch](https://github.com/HyeongminLEE/AdaCoF-pytorch)
- [softmax-splatting](https://github.com/sniklaus/softmax-splatting) 
- [ST-MFNet](https://github.com/danielism97/ST-MFNet)
- [obproxsg](https://github.com/tianyic/obproxsg)
- [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc)

We would like to thank the authors for sharing their code.
