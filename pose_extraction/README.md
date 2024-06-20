## Data preprocessing
High-quality training data is essential for an AI model. In order to extract 3D joints (point cloud) data from videos provided by Ilya, we have tested two open-source tools for pose estimation, [OpenPose](https://github.com/Devashi-Choudhary/AI-Dance-based-on-Human-Pose-Estimation) and [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose). 

We had no luck in managing configuration of OpenPose, so we switched to AlphaPose, which provided many different kinds of models for both 2D and 3D inference. 

### AlphaPose Installation
I would like to thank Mariel for providing the computing resources on Perlmutter. The installation document of AlphaPose on Perlmutter can be found [here](INSTALL.md). I have to admit that it focuses on a specific server, but I believe that it can be applied to any Linux server with CUDA driver version >= 12.0.

### Model Inference

#### 2D

I have tested with [this model](https://github.com/MVIG-SJTU/AlphaPose/blob/master/configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-dcn-regression.yaml) for 2D keypoints inference.

![image](imgs/halpe136_2d.png)

#### 3D

AlphaPose utilizes [Hybrik](https://github.com/MVIG-SJTU/AlphaPose/blob/master/configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml) for 3D pose estimation. This image shows the body mesh using [SMPL](https://github.com/Jeff-sjtu/HybrIK).

![body_mesh](imgs/body_mesh.png) 

Then, AlphaPose will use the body mesh to estimate 3D joints. 

## Raw data link
This [link](https://drive.google.com/drive/folders/1QkkAjVaKEuPBDzz7mN1BVYxUYZaYdtJF?usp=sharing) shares the raw json data extracted from AlphaPose.