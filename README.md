# DTFED

## Introduction
DTFED: 面向机器学习的纹理滤波与边缘检测训练数据集
> 本数据集可于纹理滤波和边缘检测任务的训练

## Requirements
- Python 3.7
- pytorch 1.8.1
- matplotlib 3.4.1 

## Usage
### Installation
- Download the repository.
```
git clone https://gitee.com/Chinazjk/DTFED.git
```
- Install python dependencies.
```
pip3 install torch torchvision torchaudio
pip3 install matplotlib
```
- Download the dataset.
   > [百度云盘](https://pan.baidu.com/s/1-AriDUY8-m-LaFCrCYTrqg) 提取码: ccik

### Test
#### BDCN
> run `test_DTFED.py` in `.\BDCN` 
>```
># 如果需要测试BSDS500、NYUD数据集训练的模型
>run test_DTFED.py -t [BSDS500\NYUD]
>```
#### BEPS
> run  `BEPS_train_by_TED.py` in `.\BEPS`
>
> 由于版本原因，本文复现使用自己的代码模型结构和官方相同,使用pytorch复现。运行`BEPS_train_by_TED.py`即可运行，结果存储在`BEPS_train_byTED`中 我们提供了重新训练的模型`model45_2`
而运行官方实现的代码通过运行`VDCNN_tf/test.py`  结果会保存在`VDCNN_tf/result`中。
> 
>*note*: 官方是使用pytorch实现的

### Result
> 结果分别保存在对应方法的 `result` 目录下


