# 每日从arXiv中获取最新YOLO相关论文


## You Only Look Once at Anytime \(AnytimeYOLO\): Analysis and Optimization of Early\-Exits for Object\-Detection / 

发布日期：2025-03-21

作者：Daniel Kuhse

摘要：We introduce AnytimeYOLO, a family of variants of the YOLO architecture that enables anytime object detection. Our AnytimeYOLO networks allow for interruptible inference, i.e., they provide a prediction at any point in time, a property desirable for safety\-critical real\-time applications.   We present structured explorations to modify the YOLO architecture, enabling early termination to obtain intermediate results. We focus on providing fine\-grained control through high granularity of available termination points. First, we formalize Anytime Models as a special class of prediction models that offer anytime predictions. Then, we discuss a novel transposed variant of the YOLO architecture, that changes the architecture to enable better early predictions and greater freedom for the order of processing stages. Finally, we propose two optimization algorithms that, given an anytime model, can be used to determine the optimal exit execution order and the optimal subset of early\-exits to select for deployment in low\-resource environments. We evaluate the anytime performance and trade\-offs of design choices, proposing a new anytime quality metric for this purpose. In particular, we also discuss key challenges for anytime inference that currently make its deployment costly.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.17497v1)

---


## UltraFlwr \-\- An Efficient Federated Medical and Surgical Object Detection Framework / 

发布日期：2025-03-19

作者：Yang Li

摘要：Object detection shows promise for medical and surgical applications such as cell counting and tool tracking. However, its faces multiple real\-world edge deployment challenges including limited high\-quality annotated data, data sharing restrictions, and computational constraints. In this work, we introduce UltraFlwr, a framework for federated medical and surgical object detection. By leveraging Federated Learning \(FL\), UltraFlwr enables decentralized model training across multiple sites without sharing raw data. To further enhance UltraFlwr's efficiency, we propose YOLO\-PA, a set of novel Partial Aggregation \(PA\) strategies specifically designed for YOLO models in FL. YOLO\-PA significantly reduces communication overhead by up to 83% per round while maintaining performance comparable to Full Aggregation \(FA\) strategies. Our extensive experiments on BCCD and m2cai16\-tool\-locations datasets demonstrate that YOLO\-PA not only provides better client models compared to client\-wise centralized training and FA strategies, but also facilitates efficient training and deployment across resource\-constrained edge devices. Further, we also establish one of the first benchmarks in federated medical and surgical object detection. This paper advances the feasibility of training and deploying detection models on the edge, making federated object detection more practical for time\-critical and resource\-constrained medical and surgical applications. UltraFlwr is publicly available at https://github.com/KCL\-BMEIS/UltraFlwr.

中文摘要：


代码链接：https://github.com/KCL-BMEIS/UltraFlwr.

论文链接：[阅读更多](http://arxiv.org/abs/2503.15161v1)

---


## YOLO\-LLTS: Real\-Time Low\-Light Traffic Sign Detection via Prior\-Guided Enhancement and Multi\-Branch Feature Interaction / 

发布日期：2025-03-18

作者：Ziyu Lin

摘要：Detecting traffic signs effectively under low\-light conditions remains a significant challenge. To address this issue, we propose YOLO\-LLTS, an end\-to\-end real\-time traffic sign detection algorithm specifically designed for low\-light environments. Firstly, we introduce the High\-Resolution Feature Map for Small Object Detection \(HRFM\-TOD\) module to address indistinct small\-object features in low\-light scenarios. By leveraging high\-resolution feature maps, HRFM\-TOD effectively mitigates the feature dilution problem encountered in conventional PANet frameworks, thereby enhancing both detection accuracy and inference speed. Secondly, we develop the Multi\-branch Feature Interaction Attention \(MFIA\) module, which facilitates deep feature interaction across multiple receptive fields in both channel and spatial dimensions, significantly improving the model's information extraction capabilities. Finally, we propose the Prior\-Guided Enhancement Module \(PGFE\) to tackle common image quality challenges in low\-light environments, such as noise, low contrast, and blurriness. This module employs prior knowledge to enrich image details and enhance visibility, substantially boosting detection performance. To support this research, we construct a novel dataset, the Chinese Nighttime Traffic Sign Sample Set \(CNTSSS\), covering diverse nighttime scenarios, including urban, highway, and rural environments under varying weather conditions. Experimental evaluations demonstrate that YOLO\-LLTS achieves state\-of\-the\-art performance, outperforming the previous best methods by 2.7% mAP50 and 1.6% mAP50:95 on TT100K\-night, 1.3% mAP50 and 1.9% mAP50:95 on CNTSSS, and achieving superior results on the CCTSDB2021 dataset. Moreover, deployment experiments on edge devices confirm the real\-time applicability and effectiveness of our proposed approach.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.13883v1)

---


## 8\-Calves Image dataset / 

发布日期：2025-03-17

作者：Xuyang Fang

摘要：We introduce the 8\-Calves dataset, a benchmark for evaluating object detection and identity classification in occlusion\-rich, temporally consistent environments. The dataset comprises a 1\-hour video \(67,760 frames\) of eight Holstein Friesian calves in a barn, with ground truth bounding boxes and identities, alongside 900 static frames for detection tasks. Each calf exhibits a unique coat pattern, enabling precise identity distinction.   For cow detection, we fine\-tuned 28 models \(25 YOLO variants, 3 transformers\) on 600 frames, testing on the full video. Results reveal smaller YOLO models \(e.g., YOLOV9c\) outperform larger counterparts despite potential bias from a YOLOv8m\-based labeling pipeline. For identity classification, embeddings from 23 pretrained vision models \(ResNet, ConvNextV2, ViTs\) were evaluated via linear classifiers and KNN. Modern architectures like ConvNextV2 excelled, while larger models frequently overfit, highlighting inefficiencies in scaling.   Key findings include: \(1\) Minimal, targeted augmentations \(e.g., rotation\) outperform complex strategies on simpler datasets; \(2\) Pretraining strategies \(e.g., BEiT, DinoV2\) significantly boost identity recognition; \(3\) Temporal continuity and natural motion patterns offer unique challenges absent in synthetic or domain\-specific benchmarks. The dataset's controlled design and extended sequences \(1 hour vs. prior 10\-minute benchmarks\) make it a pragmatic tool for stress\-testing occlusion handling, temporal consistency, and efficiency.   The link to the dataset is https://github.com/tonyFang04/8\-calves.

中文摘要：


代码链接：https://github.com/tonyFang04/8-calves.

论文链接：[阅读更多](http://arxiv.org/abs/2503.13777v1)

---


## Comparative Analysis of Advanced AI\-based Object Detection Models for Pavement Marking Quality Assessment during Daytime / 

发布日期：2025-03-14

作者：Gian Antariksa

摘要：Visual object detection utilizing deep learning plays a vital role in computer vision and has extensive applications in transportation engineering. This paper focuses on detecting pavement marking quality during daytime using the You Only Look Once \(YOLO\) model, leveraging its advanced architectural features to enhance road safety through precise and real\-time assessments. Utilizing image data from New Jersey, this study employed three YOLOv8 variants: YOLOv8m, YOLOv8n, and YOLOv8x. The models were evaluated based on their prediction accuracy for classifying pavement markings into good, moderate, and poor visibility categories. The results demonstrated that YOLOv8n provides the best balance between accuracy and computational efficiency, achieving the highest mean Average Precision \(mAP\) for objects with good visibility and demonstrating robust performance across various Intersections over Union \(IoU\) thresholds. This research enhances transportation safety by offering an automated and accurate method for evaluating the quality of pavement markings.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.11008v2)

---

