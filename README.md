# 基于PaddlePaddle实现的声纹识别系统

![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/VoiceprintRecognition-PaddlePaddle)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/VoiceprintRecognition-PaddlePaddle)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/VoiceprintRecognition-PaddlePaddle)
![支持系统](https://img.shields.io/badge/支持系统-Win/Linux/MAC-9cf)

本分支为1.0版本，如果要使用之前的0.3版本请在[0.x分支](https://github.com/yeyupiaoling/VoiceprintRecognition-PaddlePaddle/tree/release/0.x)使用。本项目使用了EcapaTdnn、ResNetSE、ERes2Net、CAM++等多种先进的声纹识别模型，不排除以后会支持更多模型，同时本项目也支持了MelSpectrogram、Spectrogram、MFCC、Fbank等多种数据预处理方法，使用了ArcFace Loss，ArcFace loss：Additive Angular Margin Loss（加性角度间隔损失函数），对应项目中的AAMLoss，对特征向量和权重归一化，对θ加上角度间隔m，角度间隔比余弦间隔在对角度的影响更加直接，除此之外，还支持AMLoss、ARMLoss、CELoss等多种损失函数。

**欢迎大家扫码入知识星球或者QQ群讨论，知识星球里面提供项目的模型文件和博主其他相关项目的模型文件，也包括其他一些资源。**

<div align="center">
  <img src="https://yeyupiaoling.cn/zsxq.png" alt="知识星球" width="400">
  <img src="https://yeyupiaoling.cn/qq.png" alt="QQ群" width="400">
</div>


使用环境：

 - Anaconda 3
 - Python 3.8
 - PaddlePaddle 2.4.1
 - Windows 10 or Ubuntu 18.04

# 项目特性

1. 支持模型：EcapaTdnn、TDNN、Res2Net、ResNetSE、ERes2Net、CAM++
2. 支持池化层：AttentiveStatsPool(ASP)、SelfAttentivePooling(SAP)、TemporalStatisticsPooling(TSP)、TemporalAveragePooling(TAP)、TemporalStatsPool(TSTP)
3. 支持损失函数：AAMLoss、SphereFace2、AMLoss、ARMLoss、CELoss
4. 支持预处理方法：MelSpectrogram、Spectrogram、MFCC、Fbank

**模型论文：**

- EcapaTdnn：[ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/abs/2005.07143v3)
- TDNN：[Prediction of speech intelligibility with DNN-based performance measures](https://arxiv.org/abs/2203.09148)
- Res2Net：[Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)
- ResNetSE：[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- CAMPPlus：[CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking](https://arxiv.org/abs/2303.00332v3)
- ERes2Net：[An Enhanced Res2Net with Local and Global Feature Fusion for Speaker Verification](https://arxiv.org/abs/2305.12838v1)

# 模型下载

|    模型     | Params(M) | 预处理方法 |                数据集                 | train speakers | threshold |   EER   | MinDCF  |                                      模型下载                                       |
|:---------:|:---------:|:-----:|:----------------------------------:|:--------------:|:---------:|:-------:|:-------:|:-------------------------------------------------------------------------------:|
|   CAM++   |    7.5    | Fbank | [CN-Celeb](http://openslr.org/82/) |      2796      |   0.25    | 0.09485 | 0.56214 | 加入知识星球获取/[CSDN下载](https://download.csdn.net/download/qq_33200967/88265940)(不建议) |
| ERes2Net  |    8.2    | Fbank | [CN-Celeb](http://openslr.org/82/) |      2796      |   0.22    | 0.09637 | 0.52627 |                                    加入知识星球获取                                     |
| ResNetSE  |   10.7    | Fbank | [CN-Celeb](http://openslr.org/82/) |      2796      |   0.19    | 0.10222 | 0.57981 |                                    加入知识星球获取                                     |
| EcapaTdnn |    6.7    | Fbank | [CN-Celeb](http://openslr.org/82/) |      2796      |   0.25    | 0.10465 | 0.58521 |                                    加入知识星球获取                                     |
|   TDNN    |    3.2    | Fbank | [CN-Celeb](http://openslr.org/82/) |      2796      |   0.23    | 0.11804 | 0.61070 |                                    加入知识星球获取                                     |
|  Res2Net  |    7.2    | Fbank | [CN-Celeb](http://openslr.org/82/) |      2796      |   0.18    | 0.14126 | 0.68511 |                                    加入知识星球获取                                     |
|   CAM++   |    7.5    | Fbank |               更大数据集                |      2W+       |   0.34    | 0.07884 | 0.52738 |                                    加入知识星球获取                                     |
| ERes2Net  |    8.2    | Fbank |               其他数据集                |      20W       |   0.36    | 0.02939 | 0.18355 |                                    加入知识星球获取                                     |
|   CAM++   |    7.5    | Flank |               其他数据集                |      20W       |   0.29    | 0.04768 | 0.31429 |                                    加入知识星球获取                                     |

说明：
1. 评估的测试集为[CN-Celeb的测试集](https://aistudio.baidu.com/aistudio/datasetdetail/233361)，包含196个说话人。

## 安装环境

 - 首先安装的是PaddlePaddle的GPU版本，如果已经安装过了，请跳过。
```shell
conda install paddlepaddle-gpu==2.4.1 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

 - 安装ppvector库。
 
使用pip安装，命令如下：
```shell
python -m pip install ppvector -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**建议源码安装**，源码安装能保证使用最新代码。
```shell
git clone https://github.com/yeyupiaoling/VoiceprintRecognition-PaddlePaddle.git
cd VoiceprintRecognition-PaddlePaddle/
pip install .
```

# 修改预处理方法

配置文件中默认使用的是Fbank预处理方法，如果要使用其他预处理方法，可以修改配置文件中的安装下面方式修改，具体的值可以根据自己情况修改。如果不清楚如何设置参数，可以直接删除该部分，直接使用默认值。

```yaml
# 数据预处理参数
preprocess_conf:
  # 音频预处理方法，支持：LogMelSpectrogram、MelSpectrogram、Spectrogram、MFCC、Fbank
  feature_method: 'Fbank'
  # 设置API参数，更参数查看对应API，不清楚的可以直接删除该部分，直接使用默认值
  method_args:
    sr: 16000
    n_mels: 80
```

# 训练模型
使用`train.py`训练模型，本项目支持多个音频预处理方式，通过`configs/ecapa_tdnn.yml`配置文件的参数`preprocess_conf.feature_method`可以指定，`MelSpectrogram`为梅尔频谱，`Spectrogram`为语谱图，`MFCC`梅尔频谱倒谱系数。通过参数`augment_conf_path`可以指定数据增强方式。训练过程中，会使用VisualDL保存训练日志，通过启动VisualDL可以随时查看训练结果，启动命令`visualdl --logdir=log --host 0.0.0.0`
```shell
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python train.py
# 多卡训练
python -m paddle.distributed.launch --gpus '0,1' train.py
```

训练输出日志：
```
[2023-08-05 09:52:06.497988 INFO   ] utils:print_arguments:13 - ----------- 额外配置参数 -----------
[2023-08-05 09:52:06.498094 INFO   ] utils:print_arguments:15 - configs: configs/ecapa_tdnn.yml
[2023-08-05 09:52:06.498149 INFO   ] utils:print_arguments:15 - do_eval: True
[2023-08-05 09:52:06.498191 INFO   ] utils:print_arguments:15 - local_rank: 0
[2023-08-05 09:52:06.498230 INFO   ] utils:print_arguments:15 - pretrained_model: None
[2023-08-05 09:52:06.498269 INFO   ] utils:print_arguments:15 - resume_model: None
[2023-08-05 09:52:06.498306 INFO   ] utils:print_arguments:15 - save_model_path: models/
[2023-08-05 09:52:06.498342 INFO   ] utils:print_arguments:15 - use_gpu: True
[2023-08-05 09:52:06.498378 INFO   ] utils:print_arguments:16 - ------------------------------------------------
[2023-08-05 09:52:06.513761 INFO   ] utils:print_arguments:18 - ----------- 配置文件参数 -----------
[2023-08-05 09:52:06.513906 INFO   ] utils:print_arguments:21 - dataset_conf:
[2023-08-05 09:52:06.513957 INFO   ] utils:print_arguments:24 -         dataLoader:
[2023-08-05 09:52:06.513995 INFO   ] utils:print_arguments:26 -                 batch_size: 64
[2023-08-05 09:52:06.514031 INFO   ] utils:print_arguments:26 -                 num_workers: 4
[2023-08-05 09:52:06.514066 INFO   ] utils:print_arguments:28 -         do_vad: False
[2023-08-05 09:52:06.514101 INFO   ] utils:print_arguments:28 -         enroll_list: dataset/enroll_list.txt
[2023-08-05 09:52:06.514135 INFO   ] utils:print_arguments:24 -         eval_conf:
[2023-08-05 09:52:06.514169 INFO   ] utils:print_arguments:26 -                 batch_size: 1
[2023-08-05 09:52:06.514203 INFO   ] utils:print_arguments:26 -                 max_duration: 20
[2023-08-05 09:52:06.514237 INFO   ] utils:print_arguments:28 -         max_duration: 3
[2023-08-05 09:52:06.514274 INFO   ] utils:print_arguments:28 -         min_duration: 0.5
[2023-08-05 09:52:06.514308 INFO   ] utils:print_arguments:28 -         noise_aug_prob: 0.2
[2023-08-05 09:52:06.514342 INFO   ] utils:print_arguments:28 -         noise_dir: dataset/noise
[2023-08-05 09:52:06.514374 INFO   ] utils:print_arguments:28 -         num_speakers: 3242
[2023-08-05 09:52:06.514408 INFO   ] utils:print_arguments:28 -         sample_rate: 16000
[2023-08-05 09:52:06.514441 INFO   ] utils:print_arguments:28 -         speed_perturb: True
[2023-08-05 09:52:06.514475 INFO   ] utils:print_arguments:28 -         target_dB: -20
[2023-08-05 09:52:06.514508 INFO   ] utils:print_arguments:28 -         train_list: dataset/train_list.txt
[2023-08-05 09:52:06.514542 INFO   ] utils:print_arguments:28 -         trials_list: dataset/trials_list.txt
[2023-08-05 09:52:06.514575 INFO   ] utils:print_arguments:28 -         use_dB_normalization: True
[2023-08-05 09:52:06.514609 INFO   ] utils:print_arguments:21 - loss_conf:
[2023-08-05 09:52:06.514643 INFO   ] utils:print_arguments:24 -         args:
[2023-08-05 09:52:06.514678 INFO   ] utils:print_arguments:26 -                 easy_margin: False
[2023-08-05 09:52:06.514713 INFO   ] utils:print_arguments:26 -                 margin: 0.2
[2023-08-05 09:52:06.514746 INFO   ] utils:print_arguments:26 -                 scale: 32
[2023-08-05 09:52:06.514779 INFO   ] utils:print_arguments:24 -         margin_scheduler_args:
[2023-08-05 09:52:06.514814 INFO   ] utils:print_arguments:26 -                 final_margin: 0.3
[2023-08-05 09:52:06.514848 INFO   ] utils:print_arguments:28 -         use_loss: AAMLoss
[2023-08-05 09:52:06.514882 INFO   ] utils:print_arguments:28 -         use_margin_scheduler: True
[2023-08-05 09:52:06.514915 INFO   ] utils:print_arguments:21 - model_conf:
[2023-08-05 09:52:06.514950 INFO   ] utils:print_arguments:24 -         backbone:
[2023-08-05 09:52:06.514984 INFO   ] utils:print_arguments:26 -                 embd_dim: 192
[2023-08-05 09:52:06.515017 INFO   ] utils:print_arguments:26 -                 pooling_type: ASP
[2023-08-05 09:52:06.515050 INFO   ] utils:print_arguments:24 -         classifier:
[2023-08-05 09:52:06.515084 INFO   ] utils:print_arguments:26 -                 num_blocks: 0
[2023-08-05 09:52:06.515118 INFO   ] utils:print_arguments:21 - optimizer_conf:
[2023-08-05 09:52:06.515154 INFO   ] utils:print_arguments:28 -         learning_rate: 0.001
[2023-08-05 09:52:06.515188 INFO   ] utils:print_arguments:28 -         optimizer: Adam
[2023-08-05 09:52:06.515221 INFO   ] utils:print_arguments:28 -         scheduler: CosineAnnealingLR
[2023-08-05 09:52:06.515254 INFO   ] utils:print_arguments:28 -         scheduler_args: None
[2023-08-05 09:52:06.515289 INFO   ] utils:print_arguments:28 -         weight_decay: 1e-06
[2023-08-05 09:52:06.515323 INFO   ] utils:print_arguments:21 - preprocess_conf:
[2023-08-05 09:52:06.515357 INFO   ] utils:print_arguments:28 -         feature_method: MelSpectrogram
[2023-08-05 09:52:06.515390 INFO   ] utils:print_arguments:24 -         method_args:
[2023-08-05 09:52:06.515426 INFO   ] utils:print_arguments:26 -                 f_max: 14000.0
[2023-08-05 09:52:06.515460 INFO   ] utils:print_arguments:26 -                 f_min: 50.0
[2023-08-05 09:52:06.515493 INFO   ] utils:print_arguments:26 -                 hop_length: 320
[2023-08-05 09:52:06.515527 INFO   ] utils:print_arguments:26 -                 n_fft: 1024
[2023-08-05 09:52:06.515560 INFO   ] utils:print_arguments:26 -                 n_mels: 64
[2023-08-05 09:52:06.515593 INFO   ] utils:print_arguments:26 -                 sample_rate: 16000
[2023-08-05 09:52:06.515626 INFO   ] utils:print_arguments:26 -                 win_length: 1024
[2023-08-05 09:52:06.515660 INFO   ] utils:print_arguments:21 - train_conf:
[2023-08-05 09:52:06.515694 INFO   ] utils:print_arguments:28 -         log_interval: 100
[2023-08-05 09:52:06.515728 INFO   ] utils:print_arguments:28 -         max_epoch: 30
[2023-08-05 09:52:06.515761 INFO   ] utils:print_arguments:30 - use_model: EcapaTdnn
[2023-08-05 09:52:06.515794 INFO   ] utils:print_arguments:31 - ------------------------------------------------
----------------------------------------------------------------------------------------
        Layer (type)             Input Shape          Output Shape         Param #    
========================================================================================
          Conv1D-2              [[1, 64, 102]]        [1, 512, 98]         164,352    
          Conv1d-1              [[1, 64, 98]]         [1, 512, 98]            0       
           ReLU-1               [[1, 512, 98]]        [1, 512, 98]            0       
       BatchNorm1D-2            [[1, 512, 98]]        [1, 512, 98]          2,048     
       BatchNorm1d-1            [[1, 512, 98]]        [1, 512, 98]            0       
        TDNNBlock-1             [[1, 64, 98]]         [1, 512, 98]            0       
          Conv1D-4              [[1, 512, 98]]        [1, 512, 98]         262,656    
          Conv1d-3              [[1, 512, 98]]        [1, 512, 98]            0       
           ReLU-2               [[1, 512, 98]]        [1, 512, 98]            0       
       BatchNorm1D-4            [[1, 512, 98]]        [1, 512, 98]          2,048     
       BatchNorm1d-3            [[1, 512, 98]]        [1, 512, 98]            0       
        TDNNBlock-2             [[1, 512, 98]]        [1, 512, 98]            0       
··········································
         SEBlock-3           [[1, 512, 98], None]     [1, 512, 98]            0       
      SERes2NetBlock-3          [[1, 512, 98]]        [1, 512, 98]            0       
         Conv1D-70             [[1, 1536, 98]]       [1, 1536, 98]        2,360,832   
         Conv1d-69             [[1, 1536, 98]]       [1, 1536, 98]            0       
          ReLU-32              [[1, 1536, 98]]       [1, 1536, 98]            0       
       BatchNorm1D-58          [[1, 1536, 98]]       [1, 1536, 98]          6,144     
       BatchNorm1d-57          [[1, 1536, 98]]       [1, 1536, 98]            0       
        TDNNBlock-29           [[1, 1536, 98]]       [1, 1536, 98]            0       
         Conv1D-72             [[1, 4608, 98]]        [1, 128, 98]         589,952    
         Conv1d-71             [[1, 4608, 98]]        [1, 128, 98]            0       
          ReLU-33               [[1, 128, 98]]        [1, 128, 98]            0       
       BatchNorm1D-60           [[1, 128, 98]]        [1, 128, 98]           512      
       BatchNorm1d-59           [[1, 128, 98]]        [1, 128, 98]            0       
        TDNNBlock-30           [[1, 4608, 98]]        [1, 128, 98]            0       
           Tanh-1               [[1, 128, 98]]        [1, 128, 98]            0       
         Conv1D-74              [[1, 128, 98]]       [1, 1536, 98]         198,144    
         Conv1d-73              [[1, 128, 98]]       [1, 1536, 98]            0       
AttentiveStatisticsPooling-1   [[1, 1536, 98]]        [1, 3072, 1]            0       
       BatchNorm1D-62           [[1, 3072, 1]]        [1, 3072, 1]         12,288     
       BatchNorm1d-61           [[1, 3072, 1]]        [1, 3072, 1]            0       
         Conv1D-76              [[1, 3072, 1]]        [1, 192, 1]          590,016    
         Conv1d-75              [[1, 3072, 1]]        [1, 192, 1]             0       
        EcapaTdnn-1             [[1, 98, 64]]           [1, 192]              0       
  SpeakerIdentification-1         [[1, 192]]           [1, 9726]          1,867,392   
========================================================================================
Total params: 8,039,808
Trainable params: 8,020,480
Non-trainable params: 19,328
----------------------------------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 35.60
Params size (MB): 30.67
Estimated Total Size (MB): 66.30
----------------------------------------------------------------------------------------
[2023-08-05 09:52:08.084231 INFO   ] trainer:train:388 - 训练数据：874175
[2023-08-05 09:52:09.186542 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [0/13659], loss: 11.95824, accuracy: 0.00000, learning rate: 0.00100000, speed: 58.09 data/sec, eta: 5 days, 5:24:08
[2023-08-05 09:52:22.477905 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [100/13659], loss: 10.35675, accuracy: 0.00278, learning rate: 0.00100000, speed: 481.65 data/sec, eta: 15:07:15
[2023-08-05 09:52:35.948581 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [200/13659], loss: 10.22089, accuracy: 0.00505, learning rate: 0.00100000, speed: 475.27 data/sec, eta: 15:19:12
[2023-08-05 09:52:49.249098 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [300/13659], loss: 10.00268, accuracy: 0.00706, learning rate: 0.00100000, speed: 481.45 data/sec, eta: 15:07:11
[2023-08-05 09:53:03.716015 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [400/13659], loss: 9.76052, accuracy: 0.00830, learning rate: 0.00100000, speed: 442.74 data/sec, eta: 16:26:16
[2023-08-05 09:53:18.258807 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [500/13659], loss: 9.50189, accuracy: 0.01060, learning rate: 0.00100000, speed: 440.46 data/sec, eta: 16:31:08
[2023-08-05 09:53:31.618354 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [600/13659], loss: 9.26083, accuracy: 0.01256, learning rate: 0.00100000, speed: 479.50 data/sec, eta: 15:10:12
[2023-08-05 09:53:45.439642 INFO   ] trainer:__train_epoch:334 - Train epoch: [1/30], batch: [700/13659], loss: 9.03548, accuracy: 0.01449, learning rate: 0.00099999, speed: 463.63 data/sec, eta: 15:41:08
```

VisualDL页面：
![VisualDL页面](./docs/images/log.jpg)


# 评估模型
训练结束之后会保存预测模型，我们用预测模型来预测测试集中的音频特征，然后使用音频特征进行两两对比，计算EER和MinDCF。
```shell
python eval.py
```

输出类似如下：
```
······
------------------------------------------------
W0425 08:27:32.057426 17654 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0425 08:27:32.065165 17654 device_context.cc:465] device: 0, cuDNN Version: 7.6.
[2023-03-16 20:20:47.195908 INFO   ] trainer:evaluate:341 - 成功加载模型：models/EcapaTdnn_Fbank/best_model/model.pth
100%|███████████████████████████| 84/84 [00:28<00:00,  2.95it/s]
开始两两对比音频特征...
100%|███████████████████████████| 5332/5332 [00:05<00:00, 1027.83it/s]
评估消耗时间：65s，threshold：0.26，EER: 0.14739, MinDCF: 0.41999
```

# 声纹对比
下面开始实现声纹对比，创建`infer_contrast.py`程序，首先介绍几个重要的函数，`predict()`函数是可以获取声纹特征，`predict_batch()`函数是可以获取一批的声纹特征，`contrast()`函数可以对比两条音频的相似度，`register()`函数注册一条音频到声纹库里面，`recognition()`函输入一条音频并且从声纹库里面对比识别，`remove_user()`函数移除你好。声纹库里面的注册人。我们输入两个语音，通过预测函数获取他们的特征数据，使用这个特征数据可以求他们的对角余弦值，得到的结果可以作为他们相识度。对于这个相识度的阈值`threshold`，读者可以根据自己项目的准确度要求进行修改。
```shell
python infer_contrast.py --audio_path1=audio/a_1.wav --audio_path2=audio/b_2.wav
```

输出类似如下：
```
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:13 - ----------- 额外配置参数 -----------
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:15 - audio_path1: dataset/a_1.wav
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:15 - audio_path2: dataset/b_2.wav
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:15 - configs: configs/ecapa_tdnn.yml
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:15 - model_path: models/EcapaTdnn_Fbank/best_model/
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:15 - threshold: 0.6
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:15 - use_gpu: True
[2023-04-02 18:30:48.009149 INFO   ] utils:print_arguments:16 - ------------------------------------------------
······································································
W0425 08:29:10.006249 21121 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0425 08:29:10.008555 21121 device_context.cc:465] device: 0, cuDNN Version: 7.6.
成功加载模型参数和优化方法参数：models/ecapa_tdnn/model.pdparams
audio/a_1.wav 和 audio/b_2.wav 不是同一个人，相似度为：-0.09565544128417969
```

# 声纹识别

在新闻识别里面主要使用到`register()`函数和`recognition()`函数，首先使用`register()`函数函数来注册音频到声纹库里面，也可以直接把文件添加到`audio_db`文件夹里面，使用的时候通过`recognition()`函数来发起识别，输入一条音频，就可以从声纹库里面识别到所需要的说话人。

有了上面的声纹识别的函数，读者可以根据自己项目的需求完成声纹识别的方式，例如笔者下面提供的是通过录音来完成声纹识别。首先必须要加载语音库中的语音，语音库文件夹为`audio_db`，然后用户回车后录音3秒钟，然后程序会自动录音，并使用录音到的音频进行声纹识别，去匹配语音库中的语音，获取用户的信息。通过这样方式，读者也可以修改成通过服务请求的方式完成声纹识别，例如提供一个API供APP调用，用户在APP上通过声纹登录时，把录音到的语音发送到后端完成声纹识别，再把结果返回给APP，前提是用户已经使用语音注册，并成功把语音数据存放在`audio_db`文件夹中。
```shell
python infer_recognition.py
```

输出类似如下：
```
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:13 - ----------- 额外配置参数 -----------
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:15 - audio_db_path: audio_db/
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:15 - configs: configs/ecapa_tdnn.yml
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:15 - model_path: models/EcapaTdnn_Fbank/best_model/
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:15 - record_seconds: 3
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:15 - threshold: 0.6
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:15 - use_gpu: True
[2023-04-02 18:31:20.521040 INFO   ] utils:print_arguments:16 - ------------------------------------------------
······································································
W0425 08:30:13.257884 23889 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.6, Runtime API Version: 10.2
W0425 08:30:13.260191 23889 device_context.cc:465] device: 0, cuDNN Version: 7.6.
成功加载模型参数和优化方法参数：models/ecapa_tdnn/model.pdparams
Loaded 沙瑞金 audio.
Loaded 李达康 audio.
请选择功能，0为注册音频到声纹库，1为执行声纹识别：0
按下回车键开机录音，录音3秒中：
开始录音......
录音已结束!
请输入该音频用户的名称：夜雨飘零
请选择功能，0为注册音频到声纹库，1为执行声纹识别：1
按下回车键开机录音，录音3秒中：
开始录音......
录音已结束!
识别说话的为：夜雨飘零，相似度为：0.920434
```

# 其他版本
 - Tensorflow：[VoiceprintRecognition-Tensorflow](https://github.com/yeyupiaoling/VoiceprintRecognition-Tensorflow)
 - Pytorch：[VoiceprintRecognition-Pytorch](https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch)
 - Keras：[VoiceprintRecognition-Keras](https://github.com/yeyupiaoling/VoiceprintRecognition-Keras)

## 打赏作者

<br/>
<div align="center">
<p>打赏一块钱支持一下作者</p>
<img src="https://yeyupiaoling.cn/reward.png" alt="打赏作者" width="400">
</div>

# 参考资料
1. https://github.com/PaddlePaddle/PaddleSpeech
2. https://github.com/yeyupiaoling/PaddlePaddle-MobileFaceNets
3. https://github.com/yeyupiaoling/PPASR
4. https://github.com/alibaba-damo-academy/3D-Speaker
5. https://github.com/wenet-e2e/wespeaker
