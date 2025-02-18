# 环境配置

```
torch==2.5.1+cu118
torchvision==0.20.1+cu118
segmentation-models-pytorch
torchmetrics
albumentations
loguru
tqdm
optuna==4.0.0
optuna-integration==4.0.0
kornia==0.7.4
opencv-python==4.7.0.68
opencv-contrib-python==4.9.0.80
opencv-python-headless==4.10.0.84
opencv-contrib-python-headless==4.10.0.84
PyQt5-tools
PyQt5
PyQtChart
pandas==2.2.3
numpy==1.24.4
```

有可能有遗漏，缺啥安啥就行

# 硬件连接

只需要把相机的USB线与电脑相连即可

# 代码使用

软件代码都在ui文件夹中，其余文件与文件夹均为测试文件

Mainwindows.ui与mainwindows.py文件中都是UI设置代码，如需修改UI界面，**<u>请在mainwindows.py文件中进行修改！！！</u>**

算法实现与接口都在run.py文件中，如需修改算法与功能添加**<u>请在run.py文件中进行修改！！！</u>**

直接运行run.py文件即可打开软件，如下，需要首先点击打开相机

![](E:\熔渣流速\Slag\code\src\ui1.png)

点击打开相机即可在右上角信息框内显示相机信息

![](E:\熔渣流速\Slag\code\src\ui2.png)

点击单帧采集或者连续采集即可显示相机采集的图像

开始测速前先点击框选测速范围，此时会弹窗需要进行框选区域：**点击选择矩形框的左上角与右下角后按下空格即可完成框选（<u>框选过程中矩形框不会显示</u>）**，选择后效果应如下，其中绿色框即为框选区域。

![](E:\熔渣流速\Slag\code\src\ui3.png)

然后点击开始传统测速或者开始深度学习测速即可开始测速，右边中间框为分割结果、右下方图表可以查看测速结果。

**<u>在关闭软件前请先点击关闭相机</u>**，否则可能产生资源无法释放的问题。

