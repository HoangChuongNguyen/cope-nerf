# [CVPR2025] Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video

**[Project Page]() | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_Joint_Optimization_of_Neural_Radiance_Fields_and_Continuous_Camera_Motion_CVPR_2025_paper.pdf) | [Data]() | [Pretrained Model]()**


Hoang Chuong Nguyen<sup>1</sup>, Wei Mao, Jose M. Alvarez<sup>2</sup>, Miaomiao Liu<sup>1</sup>.

<sup>1</sup>Australian National University, <sup>2</sup>NVIDIA


## Installation

```
git clone https://github.com/HoangChuongNguyen/cope-nerf
cd cope-nerf
conda env create -f environment.yaml
conda activate cope-nerf
```

## Data

We used the preprocessed Tanks and Temples dataset from <a href="https://github.com/ActiveVisionLab/nope-nerf">nope-nerf</a>. It can be downloaded <a href="https://www.robots.ox.ac.uk/~wenjing/Tanks.zip">here</a>.

Our preprocessed Co3D and Scannet dataset can be downloaded <a href="https://drive.google.com/drive/folders/1TQ5R73OuYvogKXZnbyCTcyE_MMH57_WJ?usp=sharing">here</a>.

## Training

Train a new model from scratch:

```
python train.py configs/Co3D/skateboard.yaml
```
where you can replace `configs/Co3D/skateboard.yaml` with other config files. 

To resume training from the latest checkpoint, simply use the same command as above. 

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
tensorboard --logdir ./out --port 6006
```

## Evaluation

To evaluate the method in all tasks (novel-view synthesis, depth estimation and pose estimation), simply run:
```
python eval.py configs/Co3D/skateboard.yaml
```

The latest check point will be automatically loaded. If ground-depth is not available in the dataset, depth evaluation will be skipped. 


## Citation
```
@InProceedings{Nguyen_2025_CVPR,
    author    = {Nguyen, Hoang Chuong and Mao, Wei and Alvarez, Jose M. and Liu, Miaomiao},
    title     = {Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {11472-11481}
}
```

## Acknowledgement

This research was supported in part by the Australia Research Council ARC Discovery Grant (DP200102274).

We develop our method upon the code base provided by <a href="https://github.com/ActiveVisionLab/nope-nerf">nope-nerf</a>. We appreciate and thank the authors for providing their excellent code. 
