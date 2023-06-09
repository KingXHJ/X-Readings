# X-Readings
My books reading notes and codes

- [flashlight minst note](#flashlight-minst-note)
- [ssd tf2 colab](#ssd-tf2-colab)
    1. [性能情况 Performance](#性能情况)
    1. [所需环境 Environment](#所需环境)
    1. [文件下载 Download](#文件下载)
    1. [预测步骤 How2predict](#预测步骤)
    1. [训练步骤 How2train](#训练步骤)
    1. [评估步骤 How2eval](#评估步骤)
    1. [参考资料 Reference](#Reference)
- [nn-nlp-BookSourceCode](#nn-nlp-booksourcecode)

## flashlight minst note

## ssd tf2 colab
### SSD: Single-Shot MultiBox Detector目标检测模型在TF2当中的实现
---

**2021年2月8日更新：**   
**加入letterbox_image的选项，关闭letterbox_image后网络的map一般可以得到提升。**

### 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [评估步骤 How2eval](#评估步骤)
7. [参考资料 Reference](#Reference)

### 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07++12+COCO | [ssd_weights.h5](https://github.com/bubbliiiing/ssd-tf2/releases/download/v1.0/ssd_weights.h5) | VOC-Test12 | 300x300| - | 79.4

### 所需环境
tensorflow-gpu==2.2.0  

### 文件下载
训练所需的ssd_weights.h5可以在百度云下载。  
链接: https://pan.baidu.com/s/10l02SiObuzbrv-oYRwYNAg 提取码: 7jck

VOC数据集下载地址如下：  
VOC2007+2012训练集    
链接: https://pan.baidu.com/s/16pemiBGd-P9q2j7dZKGDFA 提取码: eiw9    

VOC2007测试集   
链接: https://pan.baidu.com/s/1BnMiFwlNwIWG9gsd4jHLig 提取码: dsda   

### 预测步骤
#### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载ssd_weights.h5，放入model_data，运行predict.py，输入  
```python
img/street.jpg
``` 
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
#### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在ssd.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    "model_path": 'model_data/ssd_weights.h5',
    "classes_path": 'model_data/voc_classes.txt',
    "model_image_size" : (300, 300, 3),
    "confidence": 0.5,
}
```
3. 运行predict.py，输入  
```python
img/street.jpg
``` 
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。  

### 训练步骤
1. 本文使用VOC格式进行训练。  
2. 训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3. 训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4. 在训练前利用voc2ssd.py文件生成对应的txt。  
5. 再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**   
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6. 此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。  
7. **在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类**，示例如下：   
model_data/new_classes.txt文件内容为：   
```python
cat
dog
...
```
8. 将train.py的NUM_CLASSSES修改成所需要分的类的个数+1，运行train.py即可开始训练。

### 评估步骤
评估过程可参考视频https://www.bilibili.com/video/BV1zE411u7Vw  
步骤是一样的，不需要自己再建立get_dr_txt.py、get_gt_txt.py等文件。  
1. 本文使用VOC格式进行评估。  
2. 评估前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3. 评估前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4. 在评估前利用voc2ssd.py文件生成对应的txt，评估用的txt为VOCdevkit/VOC2007/ImageSets/Main/test.txt，需要注意的是，如果整个VOC2007里面的数据集都是用于评估，那么直接将trainval_percent设置成0即可。  
5. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
6. 运行get_dr_txt.py和get_gt_txt.py，在./input/detection-results和./input/ground-truth文件夹下生成对应的txt。  
7. 运行get_map.py即可开始计算模型的mAP。

### mAP目标检测精度计算更新
更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  
具体mAP计算过程可参考：https://www.bilibili.com/video/BV1zE411u7Vw

### Reference
https://github.com/Cartucho/mAP  
https://github.com/pierluigiferrari/ssd_keras  
https://github.com/kuhung/SSD_keras  



# nn-nlp-BookSourceCode
包含两本书的源码：
1. 深度学习入门 基于Python的理论与实现(图灵出品)
    - 附带自己跑过的代码(Neural Network With Python)
1. 深度学习进阶 自然语言处理（图灵出品）
    - 附带曾经学习的一份代码(NLP With Python)