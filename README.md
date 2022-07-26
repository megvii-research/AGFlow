<h2 align="center"> Learning Optical Flow with Adaptive Graph Reasoning </h2>

<h4 align="center">Ao Luo$^1$, Fan Fang$^2$, Kunming Luo$^1$, Xin Li$^2$, Haoqiang Fan$^1$, Shuaicheng Liu$^3$</h4>
<h4 align="center">1. Megvii Research,             2. Group 42</h4>
<h4 align="center">3. University of Electronic Science and Technology of China</h4>



This project provides the source code for '[**Learning Optical Flow with Adaptive Graph Reasoning**](https://arxiv.org/pdf/2202.03857.pdf)'. (AAAI-2022)

## Presentation Video
[[Youtube](https://www.youtube.com/watch?v=7ywAgSTaj1A)], [[Bilibili](https://www.bilibili.com/video/BV1Fm4y1f7QC/)]


## Overview

We propose a novel graph-based approach, called adaptive graph reasoning for optical flow (AGFlow), to emphasize the value of scene context in optical flow. Our key idea is to decouple the context reasoning from the matching procedure, and exploit scene information to effectively assist motion estimation by learning to reason over the adaptive graph. 

<img width="1000" alt="overview" src="https://user-images.githubusercontent.com/47421121/147655606-bd8a1640-5c57-4c23-a50d-57661ec49f54.png">

## Requirements

Python 3.6 with following packages
```Shell
pytorch==1.6.0
torchvision==0.7.0
matplotlib
scipy
opencv-python
tensorboard
```
(The code has been tested on Cuda 10.0.)

## Usage

1. The trained weights are available on [GoogleDrive](https://drive.google.com/drive/folders/1Bnijg9VPJwc9RPk0wOJNx8ngxXBnrGsV?usp=sharing). Put `*.pth` files into folder `./weights`.

2. Download [Sintel](http://sintel.is.tue.mpg.de/) and [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) dataset, and set the root path of each class in `./core/datasets.py`.


3. Evaluation on Sintel
```Shell
./eval_sintel.sh
```

4. Evaluation on KITTI
```Shell
./eval_kitti.sh
```

![results](https://user-images.githubusercontent.com/1344482/180935818-1f77400a-6a60-48e5-aed2-7cd274269785.JPG)


## Citation

If you think this work is helpful, please cite
```
@InProceedings{luo2022learning,
  title={Learning Optical Flow with Adaptive Graph Reasoning},
  author={Luo, Ao and Yang, Fan and Luo, Kunming and Li, Xin and Fan, Haoqiang and Liu, Shuaicheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2022},
}
```

If you have any questions, please contact me at (aoluo_uestc@hotmail.com).

## Acknowledgement

The main framework is adapted from [RAFT](https://github.com/princeton-vl/RAFT). We thank the authors for the contribution.
