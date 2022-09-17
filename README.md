<h2 align="center"> [AAAI2022]Learning Optical Flow with Adaptive Graph Reasoning </h2> 


<h4 align="center">Ao Luo<sup>1</sup>, Fan Fang<sup>2</sup>, Kunming Luo<sup>1</sup>, Xin Li<sup>2</sup>, Haoqiang Fan<sup>1</sup>, Shuaicheng Liu<sup>3</sup></h4>
<h4 align="center">1. Megvii Research,             2. Group 42</h4>
<h4 align="center">3. University of Electronic Science and Technology of China</h4>





This is the official implementation of 'Learning Optical Flow with Adaptive Graph Reasoning', named as 'AGFlow' for short, AAAI 2022, [[paper](https://www.aaai.org/AAAI22Papers/AAAI-1843.LuoA.pdf)]

## Presentation Video
[[Youtube](https://www.youtube.com/watch?v=7ywAgSTaj1A)], [[Bilibili](https://www.bilibili.com/video/BV1Fm4y1f7QC/)]

## Abstract
Estimating per-pixel motion between video frames, known as optical flow, is a long-standing problem in video understanding and analysis. Most contemporary optical flow techniques largely focus on addressing the cross-image matching with feature similarity, with few methods considering how to explicitly reason over the given scene for achieving a holistic motion understanding. In this work, taking a fresh perspective, we introduce a novel graph-based approach, called adaptive graph reasoning for optical flow (AGFlow), to emphasize the value of scene/context information in optical flow. Our key idea is to decouple the context reasoning from the matching procedure, and exploit scene information to effectively assist motion estimation by learning to reason over the adaptive graph. The proposed AGFlow can effectively exploit the context information and incorporate it within the matching procedure, producing more robust and accurate results. On both Sintel clean and final passes, our AGFlow achieves the best accuracy with EPE of 1.43 and 2.47 pixels, outperforming state-of-the-art approaches by 11.2% and 13.6%, respectively.


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
