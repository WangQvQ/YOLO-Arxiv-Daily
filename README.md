# 每日从arXiv中获取最新YOLO相关论文


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


## Object detection characteristics in a learning factory environment using YOLOv8 / 

发布日期：2025-03-13

作者：Toni Schneidereit

摘要：AI\-based object detection, and efforts to explain and investigate their characteristics, is a topic of high interest. The impact of, e.g., complex background structures with similar appearances as the objects of interest, on the detection accuracy and, beforehand, the necessary dataset composition are topics of ongoing research. In this paper, we present a systematic investigation of background influences and different features of the object to be detected. The latter includes various materials and surfaces, partially transparent and with shiny reflections in the context of an Industry 4.0 learning factory. Different YOLOv8 models have been trained for each of the materials on different sized datasets, where the appearance was the only changing parameter. In the end, similar characteristics tend to show different behaviours and sometimes unexpected results. While some background components tend to be detected, others with the same features are not part of the detection. Additionally, some more precise conclusions can be drawn from the results. Therefore, we contribute a challenging dataset with detailed investigations on 92 trained YOLO models, addressing some issues on the detection accuracy and possible overfitting.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.10356v1)

---


## YOLOE: Real\-Time Seeing Anything / 

发布日期：2025-03-10

作者：Ao Wang

摘要：Object detection and segmentation are widely employed in computer vision applications, yet conventional models like YOLO series, while efficient and accurate, are limited by predefined categories, hindering adaptability in open scenarios. Recent open\-set methods leverage text prompts, visual cues, or prompt\-free paradigm to overcome this, but often compromise between performance and efficiency due to high computational demands or deployment complexity. In this work, we introduce YOLOE, which integrates detection and segmentation across diverse open prompt mechanisms within a single highly efficient model, achieving real\-time seeing anything. For text prompts, we propose Re\-parameterizable Region\-Text Alignment \(RepRTA\) strategy. It refines pretrained textual embeddings via a re\-parameterizable lightweight auxiliary network and enhances visual\-textual alignment with zero inference and transferring overhead. For visual prompts, we present Semantic\-Activated Visual Prompt Encoder \(SAVPE\). It employs decoupled semantic and activation branches to bring improved visual embedding and accuracy with minimal complexity. For prompt\-free scenario, we introduce Lazy Region\-Prompt Contrast \(LRPC\) strategy. It utilizes a built\-in large vocabulary and specialized embedding to identify all objects, avoiding costly language model dependency. Extensive experiments show YOLOE's exceptional zero\-shot performance and transferability with high inference efficiency and low training cost. Notably, on LVIS, with 3$times$ less training cost and 1.4$times$ inference speedup, YOLOE\-v8\-S surpasses YOLO\-Worldv2\-S by 3.5 AP. When transferring to COCO, YOLOE\-v8\-L achieves 0.6 AP$^b$ and 0.4 AP$^m$ gains over closed\-set YOLOv8\-L with nearly 4$times$ less training time. Code and models are available at https://github.com/THU\-MIG/yoloe.

中文摘要：


代码链接：https://github.com/THU-MIG/yoloe.

论文链接：[阅读更多](http://arxiv.org/abs/2503.07465v1)

---

