```
# parameters
nc: 2  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]  from接收哪一层结果-1就是上一层，number该模块的个数除了1以外要乘上depth_multiple
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

![image-20240307094505311](https://github.com/jiangsu415/YOLOV5/assets/130949548/6ce6cf5d-e8a6-42b9-b060-c474a2114b84)

![yolov5s onnx](https://github.com/jiangsu415/YOLOV5/assets/130949548/5b2b7797-8d5d-4cbf-9ff9-78b76c801180)

![image-20240307101033596](https://github.com/jiangsu415/YOLOV5/assets/130949548/a04bc21f-f7e1-418f-9234-bcd38af68db3)

![image-20240307094807143](https://github.com/jiangsu415/YOLOV5/assets/130949548/90331fa1-4cc0-46b0-8125-f6705151e791)

![image-20240307101254736](https://github.com/jiangsu415/YOLOV5/assets/130949548/00d5d815-96f3-4884-8ed6-9e91ba3ee3a7)

![image-20240307101442535](https://github.com/jiangsu415/YOLOV5/assets/130949548/13cd1d99-833c-4214-abbb-043e86b0a771)

![image-20240307101513548](https://github.com/jiangsu415/YOLOV5/assets/130949548/130404c1-52b7-471b-a975-86233c8c7f5b)


[3,32,3]输入，输出，卷积核大小

[32,64,3,2]输入，输出，卷积核大小，stride为2特征图大小减小一半

![image-20240307104222400](https://github.com/jiangsu415/YOLOV5/assets/130949548/340f549e-fe39-4bc7-8882-645235d0684d)

SPP层，先经过一个1*1的卷积做一个降维的操作

![image-20240307205116413](https://github.com/jiangsu415/YOLOV5/assets/130949548/af0dcf97-b8de-43e5-ba2f-837657ed22ad)

![image-20240307205116413](https://github.com/jiangsu415/YOLOV5/assets/130949548/861a948a-8b64-4b6d-b235-f77b6b8a1ff0)


分别对1 * 1分块，2 * 2分块和4 * 4子图里分别取每一个框内的max值（即取蓝框框内的最大值），这一步就是作最大池化，这样最后提取出来的特征值（即取出来的最大值）一共有1 * 1 + 2 * 2 + 4 * 4 = 21个。得出的特征再concat在一起。

![img](https://img-blog.csdnimg.cn/d87b6e3992c84da1ba8962088e622d05.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAdHTkuKs=,size_20,color_FFFFFF,t_70,g_se,x_16)

通过不同的kenersize得到最终的特征结果是不同的，指定padding让特征图大小保持一致，之后在连接一个1*1的卷积降维
![image-20240307205736701](https://github.com/jiangsu415/YOLOV5/assets/130949548/31695582-8436-4ad8-b582-79f6a8d775b8)

![image-20240307211248380](https://github.com/jiangsu415/YOLOV5/assets/130949548/3938bfa4-7c42-4bf5-9a68-11ef89de65bc)

三个检测头，大目标，中目标和小目标，最终做1*1的卷积

na是候选框的个数，nc是类别的个数，5是候选框的置信度加上x,y,w,h
![CB3F3E256910970DB46CEA9A4FD96F09](https://github.com/jiangsu415/YOLOV5/assets/130949548/2f0a2423-0eb9-4cf2-9a4b-d48421c699d6)


首先输入x经过focus层(common.py，93行)，focus的模型结构就是这样的，他是间隔着分块取值，最终得到的结果再按通道维度进行拼接，拼接后的结果是（320，320,12,1）在经过一卷积和一个HardWish激活函数，然后进入到BottleneckCSP层（common.py,65,46），中间有一个bottleneck结构和Resnet结构一样有一个shortcut结构，外面再嵌套一层卷积之后做一个拼接，之后到达SPP层，首先经过一个SPP层(common.py，83)降维，将512转化为256进入分别进入到卷积核大小不同的maxpooling，目的是实现局部特征和全局特征的featherMap级别的融合。到达检测头部分，一共要做四次拼接操作前两次需要进行一个上采样，因为第六层特征图大小为8*8,第11层特征图大小为4 * 4 所以要进行一个上采样操作，同样15层要做上采样后第16层和第四层进行拼接，要进行一个上采样操作  因为要乘width_multiple: 0.50，所以第三个维度通道大小是一半，得到最终的输出大小，128,256,512三个维度分别连接上一个1*1的卷积
![image-20240313155735499](https://github.com/jiangsu415/YOLOV5/assets/130949548/e71fcc12-fc8d-4ea5-abd4-993d5d27ef80)
![image-20240313155556433](https://github.com/jiangsu415/YOLOV5/assets/130949548/3b93fbb7-c6d5-4672-acbf-c7c7ddaec04d)
