# 每日从arXiv中获取最新YOLO相关论文


## DriveIndia: An Object Detection Dataset for Diverse Indian Traffic Scenes / DriveIndia：印度不同交通场景的目标检测数据集

发布日期：2025-07-26

作者：Rishav Kumar

摘要：We introduce DriveIndia, a large\-scale object detection dataset purpose\-built to capture the complexity and unpredictability of Indian traffic environments. The dataset contains 66,986 high\-resolution images annotated in YOLO format across 24 traffic\-relevant object categories, encompassing diverse conditions such as varied weather \(fog, rain\), illumination changes, heterogeneous road infrastructure, and dense, mixed traffic patterns and collected over 120\+ hours and covering 3,400\+ kilometers across urban, rural, and highway routes. DriveIndia offers a comprehensive benchmark for real\-world autonomous driving challenges. We provide baseline results using state\-of\-the\-art YOLO family models, with the top\-performing variant achieving a mAP50 of 78.7%. Designed to support research in robust, generalizable object detection under uncertain road conditions, DriveIndia will be publicly available via the TiHAN\-IIT Hyderabad dataset repository \(https://tihan.iith.ac.in/tiand\-datasets/\).

中文摘要：我们介绍DriveIndia，这是一个大规模的对象检测数据集，旨在捕捉印度交通环境的复杂性和不可预测性。该数据集包含66986幅以YOLO格式注释的高分辨率图像，涵盖24个交通相关对象类别，包括不同的天气（雾、雨）、光照变化、异质道路基础设施和密集、混合的交通模式等不同条件，收集时间超过120个小时，覆盖城市、农村和公路路线3400多公里。DriveIndia为现实世界的自动驾驶挑战提供了一个全面的基准。我们使用最先进的YOLO系列模型提供基线结果，其中性能最佳的变体实现了78.7%的mAP50。DriveIndia旨在支持在不确定道路条件下进行稳健、通用的目标检测研究，将通过TiHAN IIT Hyderabad数据集库公开提供(https://tihan.iith.ac.in/tiand-datasets/).


代码链接：https://tihan.iith.ac.in/tiand-datasets/).

论文链接：[阅读更多](http://arxiv.org/abs/2507.19912v2)

---


## Underwater Waste Detection Using Deep Learning A Performance Comparison of YOLOv7 to 10 and Faster RCNN / 基于深度学习的水下废物检测——YOLOv7与10及更快RCNN的性能比较

发布日期：2025-07-25

作者：UMMPK Nawarathne

摘要：Underwater pollution is one of today's most significant environmental concerns, with vast volumes of garbage found in seas, rivers, and landscapes around the world. Accurate detection of these waste materials is crucial for successful waste management, environmental monitoring, and mitigation strategies. In this study, we investigated the performance of five cutting\-edge object recognition algorithms, namely YOLO \(You Only Look Once\) models, including YOLOv7, YOLOv8, YOLOv9, YOLOv10, and Faster Region\-Convolutional Neural Network \(R\-CNN\), to identify which model was most effective at recognizing materials in underwater situations. The models were thoroughly trained and tested on a large dataset containing fifteen different classes under diverse conditions, such as low visibility and variable depths. From the above\-mentioned models, YOLOv8 outperformed the others, with a mean Average Precision \(mAP\) of 80.9%, indicating a significant performance. This increased performance is attributed to YOLOv8's architecture, which incorporates advanced features such as improved anchor\-free mechanisms and self\-supervised learning, allowing for more precise and efficient recognition of items in a variety of settings. These findings highlight the YOLOv8 model's potential as an effective tool in the global fight against pollution, improving both the detection capabilities and scalability of underwater cleanup operations.

中文摘要：水下污染是当今最重要的环境问题之一，在世界各地的海洋、河流和景观中发现了大量垃圾。准确检测这些废料对于成功的废物管理、环境监测和缓解策略至关重要。在这项研究中，我们研究了五种尖端物体识别算法的性能，即YOLO（You Only Look Once）模型，包括YOLOv7、YOLOv8、YOLOv9、YOLOv10和更快区域卷积神经网络（R-CNN），以确定哪种模型在水下情况下识别材料最有效。这些模型在包含15个不同类别的大型数据集上进行了全面的训练和测试，这些数据集在不同的条件下，如低能见度和可变深度。从上述模型来看，YOLOv8的表现优于其他模型，平均精度（mAP）为80.9%，表明其表现显著。这种性能的提高归功于YOLOv8的架构，该架构融合了改进的无锚机制和自我监督学习等高级功能，允许在各种设置中更精确、更高效地识别物品。这些发现突显了YOLOv8模型作为全球对抗污染的有效工具的潜力，提高了水下清理作业的检测能力和可扩展性。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.18967v1)

---


## YOLO for Knowledge Extraction from Vehicle Images: A Baseline Study / YOLO用于车辆图像知识提取：一项基线研究

发布日期：2025-07-25

作者：Saraa Al\-Saddik

摘要：Accurate identification of vehicle attributes such as make, colour, and shape is critical for law enforcement and intelligence applications. This study evaluates the effectiveness of three state\-of\-the\-art deep learning approaches YOLO\-v11, YOLO\-World, and YOLO\-Classification on a real\-world vehicle image dataset. This dataset was collected under challenging and unconstrained conditions by NSW Police Highway Patrol Vehicles. A multi\-view inference \(MVI\) approach was deployed to enhance the performance of the models' predictions. To conduct the analyses, datasets with 100,000 plus images were created for each of the three metadata prediction tasks, specifically make, shape and colour. The models were tested on a separate dataset with 29,937 images belonging to 1809 number plates. Different sets of experiments have been investigated by varying the models sizes. A classification accuracy of 93.70%, 82.86%, 85.19%, and 94.86% was achieved with the best performing make, shape, colour, and colour\-binary models respectively. It was concluded that there is a need to use MVI to get usable models within such complex real\-world datasets. Our findings indicated that the object detection models YOLO\-v11 and YOLO\-World outperformed classification\-only models in make and shape extraction. Moreover, smaller YOLO variants perform comparably to larger counterparts, offering substantial efficiency benefits for real\-time predictions. This work provides a robust baseline for extracting vehicle metadata in real\-world scenarios. Such models can be used in filtering and sorting user queries, minimising the time required to search large vehicle images datasets.

中文摘要：准确识别车辆的品牌、颜色和形状等属性对于执法和情报应用至关重要。本研究评估了三种最先进的深度学习方法YOLO-v11、YOLO World和YOLO Classification在真实车辆图像数据集上的有效性。该数据集是由新南威尔士州警察公路巡逻车在具有挑战性和不受约束的条件下收集的。部署了多视图推理（MVI）方法来提高模型预测的性能。为了进行分析，为三个元数据预测任务中的每一个创建了包含100000多张图像的数据集，特别是品牌、形状和颜色。这些模型在一个单独的数据集上进行了测试，该数据集包含属于1809个车牌的29937张图像。通过改变模型大小，对不同的实验集进行了研究。使用性能最佳的品牌、形状、颜色和颜色二元模型分别实现了93.70%、82.86%、85.19%和94.86%的分类准确率。结论是，有必要使用MVI在如此复杂的现实世界数据集中获得可用的模型。我们的研究结果表明，对象检测模型YOLO-v11和YOLO World在品牌和形状提取方面优于仅分类模型。此外，较小的YOLO变体与较大的变体表现相当，为实时预测提供了巨大的效率优势。这项工作为在现实场景中提取车辆元数据提供了一个强大的基线。此类模型可用于过滤和排序用户查询，最大限度地减少搜索大型车辆图像数据集所需的时间。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.18966v1)

---


## Real\-Time Object Detection and Classification using YOLO for Edge FPGAs / 基于YOLO的边缘FPGA实时目标检测与分类

发布日期：2025-07-24

作者：Rashed Al Amin

摘要：Object detection and classification are crucial tasks across various application domains, particularly in the development of safe and reliable Advanced Driver Assistance Systems \(ADAS\). Existing deep learning\-based methods such as Convolutional Neural Networks \(CNNs\), Single Shot Detectors \(SSDs\), and You Only Look Once \(YOLO\) have demonstrated high performance in terms of accuracy and computational speed when deployed on Field\-Programmable Gate Arrays \(FPGAs\). However, despite these advances, state\-of\-the\-art YOLO\-based object detection and classification systems continue to face challenges in achieving resource efficiency suitable for edge FPGA platforms. To address this limitation, this paper presents a resource\-efficient real\-time object detection and classification system based on YOLOv5 optimized for FPGA deployment. The proposed system is trained on the COCO and GTSRD datasets and implemented on the Xilinx Kria KV260 FPGA board. Experimental results demonstrate a classification accuracy of 99%, with a power consumption of 3.5W and a processing speed of 9 frames per second \(FPS\). These findings highlight the effectiveness of the proposed approach in enabling real\-time, resource\-efficient object detection and classification for edge computing applications.

中文摘要：物体检测和分类是各个应用领域的关键任务，特别是在开发安全可靠的高级驾驶辅助系统（ADAS）方面。现有的基于深度学习的方法，如卷积神经网络（CNN）、单镜头检测器（SSD）和你只看一次（YOLO），在部署在现场可编程门阵列（FPGA）上时，在准确性和计算速度方面表现出了很高的性能。然而，尽管取得了这些进步，但最先进的基于YOLO的对象检测和分类系统在实现适用于边缘FPGA平台的资源效率方面仍面临挑战。为了解决这一局限性，本文提出了一种基于YOLOv5的资源高效的实时目标检测和分类系统，该系统针对FPGA部署进行了优化。所提出的系统在COCO和GTSRD数据集上进行训练，并在Xilinx Kria KV260 FPGA板上实现。实验结果表明，分类准确率为99%，功耗为3.5W，处理速度为每秒9帧（FPS）。这些发现突出了所提出的方法在为边缘计算应用程序实现实时、资源高效的对象检测和分类方面的有效性。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.18174v1)

---


## Bearded Dragon Activity Recognition Pipeline: An AI\-Based Approach to Behavioural Monitoring / 胡须龙活动识别管道：一种基于人工智能的行为监测方法

发布日期：2025-07-23

作者：Arsen Yermukan

摘要：Traditional monitoring of bearded dragon \(Pogona Viticeps\) behaviour is time\-consuming and prone to errors. This project introduces an automated system for real\-time video analysis, using You Only Look Once \(YOLO\) object detection models to identify two key behaviours: basking and hunting. We trained five YOLO variants \(v5, v7, v8, v11, v12\) on a custom, publicly available dataset of 1200 images, encompassing bearded dragons \(600\), heating lamps \(500\), and crickets \(100\). YOLOv8s was selected as the optimal model due to its superior balance of accuracy \(mAP@0.5:0.95 = 0.855\) and speed. The system processes video footage by extracting per\-frame object coordinates, applying temporal interpolation for continuity, and using rule\-based logic to classify specific behaviours. Basking detection proved reliable. However, hunting detection was less accurate, primarily due to weak cricket detection \(mAP@0.5 = 0.392\). Future improvements will focus on enhancing cricket detection through expanded datasets or specialised small\-object detectors. This automated system offers a scalable solution for monitoring reptile behaviour in controlled environments, significantly improving research efficiency and data quality.

中文摘要：传统上对须龙（Pogona Viticeps）行为的监测既费时又容易出错。该项目引入了一个用于实时视频分析的自动化系统，使用You Only Look Once（YOLO）对象检测模型来识别两个关键行为：晒太阳和狩猎。我们在一个定制的、公开可用的1200张图像数据集上训练了五种YOLO变体（v5、v7、v8、v11、v12），包括须龙（600）、加热灯（500）和蟋蟀（100）。YOLOv8s因其卓越的精度平衡而被选为最佳模型(mAP@0.50.95=0.855）和速度。该系统通过提取每帧对象坐标、应用时间插值以实现连续性以及使用基于规则的逻辑对特定行为进行分类来处理视频片段。Basking检测被证明是可靠的。然而，狩猎检测的准确性较低，主要是由于蟋蟀检测较弱(mAP@0.5 = 0.392).未来的改进将侧重于通过扩展数据集或专门的小物体探测器来增强蟋蟀检测。该自动化系统为在受控环境中监测爬行动物行为提供了一种可扩展的解决方案，显著提高了研究效率和数据质量。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.17987v1)

---

