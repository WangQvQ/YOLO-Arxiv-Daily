# 每日从arXiv中获取最新YOLO相关论文


## Detection Fire in Camera RGB\-NIR / 

发布日期：2025-12-29

作者：Nguyen Truong Khai

摘要：Improving the accuracy of fire detection using infrared night vision cameras remains a challenging task. Previous studies have reported strong performance with popular detection models. For example, YOLOv7 achieved an mAP50\-95 of 0.51 using an input image size of 640 x 1280, RT\-DETR reached an mAP50\-95 of 0.65 with an image size of 640 x 640, and YOLOv9 obtained an mAP50\-95 of 0.598 at the same resolution. Despite these results, limitations in dataset construction continue to cause issues, particularly the frequent misclassification of bright artificial lights as fire.   This report presents three main contributions: an additional NIR dataset, a two\-stage detection model, and Patched\-YOLO. First, to address data scarcity, we explore and apply various data augmentation strategies for both the NIR dataset and the classification dataset. Second, to improve night\-time fire detection accuracy while reducing false positives caused by artificial lights, we propose a two\-stage pipeline combining YOLOv11 and EfficientNetV2\-B0. The proposed approach achieves higher detection accuracy compared to previous methods, particularly for night\-time fire detection. Third, to improve fire detection in RGB images, especially for small and distant objects, we introduce Patched\-YOLO, which enhances the model's detection capability through patch\-based processing. Further details of these contributions are discussed in the following sections.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.23594v1)

---


## YOLO\-Master: MOE\-Accelerated with Specialized Transformers for Enhanced Real\-time Detection / 

发布日期：2025-12-29

作者：Xu Lin

摘要：Existing Real\-Time Object Detection \(RTOD\) methods commonly adopt YOLO\-like architectures for their favorable trade\-off between accuracy and speed. However, these models rely on static dense computation that applies uniform processing to all inputs, misallocating representational capacity and computational resources such as over\-allocating on trivial scenes while under\-serving complex ones. This mismatch results in both computational redundancy and suboptimal detection performance. To overcome this limitation, we propose YOLO\-Master, a novel YOLO\-like framework that introduces instance\-conditional adaptive computation for RTOD. This is achieved through a Efficient Sparse Mixture\-of\-Experts \(ES\-MoE\) block that dynamically allocates computational resources to each input according to its scene complexity. At its core, a lightweight dynamic routing network guides expert specialization during training through a diversity enhancing objective, encouraging complementary expertise among experts. Additionally, the routing network adaptively learns to activate only the most relevant experts, thereby improving detection performance while minimizing computational overhead during inference. Comprehensive experiments on five large\-scale benchmarks demonstrate the superiority of YOLO\-Master. On MS COCO, our model achieves 42.4% AP with 1.62ms latency, outperforming YOLOv13\-N by \+0.8% mAP and 17.8% faster inference. Notably, the gains are most pronounced on challenging dense scenes, while the model preserves efficiency on typical inputs and maintains real\-time inference speed. Code will be available.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.23273v1)

---


## YOLO\-IOD: Towards Real Time Incremental Object Detection / 

发布日期：2025-12-28

作者：Shizhou Zhang

摘要：Current methods for incremental object detection \(IOD\) primarily rely on Faster R\-CNN or DETR series detectors; however, these approaches do not accommodate the real\-time YOLO detection frameworks. In this paper, we first identify three primary types of knowledge conflicts that contribute to catastrophic forgetting in YOLO\-based incremental detectors: foreground\-background confusion, parameter interference, and misaligned knowledge distillation. Subsequently, we introduce YOLO\-IOD, a real\-time Incremental Object Detection \(IOD\) framework that is constructed upon the pretrained YOLO\-World model, facilitating incremental learning via a stage\-wise parameter\-efficient fine\-tuning process. Specifically, YOLO\-IOD encompasses three principal components: 1\) Conflict\-Aware Pseudo\-Label Refinement \(CPR\), which mitigates the foreground\-background confusion by leveraging the confidence levels of pseudo labels and identifying potential objects relevant to future tasks. 2\) Importancebased Kernel Selection \(IKS\), which identifies and updates the pivotal convolution kernels pertinent to the current task during the current learning stage. 3\) Cross\-Stage Asymmetric Knowledge Distillation \(CAKD\), which addresses the misaligned knowledge distillation conflict by transmitting the features of the student target detector through the detection heads of both the previous and current teacher detectors, thereby facilitating asymmetric distillation between existing and newly introduced categories. We further introduce LoCo COCO, a more realistic benchmark that eliminates data leakage across stages. Experiments on both conventional and LoCo COCO benchmarks show that YOLO\-IOD achieves superior performance with minimal forgetting.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.22973v1)

---


## CLIP\-Joint\-Detect: End\-to\-End Joint Training of Object Detectors with Contrastive Vision\-Language Supervision / 

发布日期：2025-12-28

作者：Behnam Raoufi

摘要：Conventional object detectors rely on cross\-entropy classification, which can be vulnerable to class imbalance and label noise. We propose CLIP\-Joint\-Detect, a simple and detector\-agnostic framework that integrates CLIP\-style contrastive vision\-language supervision through end\-to\-end joint training. A lightweight parallel head projects region or grid features into the CLIP embedding space and aligns them with learnable class\-specific text embeddings via InfoNCE contrastive loss and an auxiliary cross\-entropy term, while all standard detection losses are optimized simultaneously. The approach applies seamlessly to both two\-stage and one\-stage architectures. We validate it on Pascal VOC 2007\+2012 using Faster R\-CNN and on the large\-scale MS COCO 2017 benchmark using modern YOLO detectors \(YOLOv11\), achieving consistent and substantial improvements while preserving real\-time inference speed. Extensive experiments and ablations demonstrate that joint optimization with learnable text embeddings markedly enhances closed\-set detection performance across diverse architectures and datasets.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.22969v1)

---


## Comparative Analysis of Deep Learning Models for Perception in Autonomous Vehicles / 

发布日期：2025-12-25

作者：Jalal Khan

摘要：Recently, a plethora of machine learning \(ML\) and deep learning \(DL\) algorithms have been proposed to achieve the efficiency, safety, and reliability of autonomous vehicles \(AVs\). The AVs use a perception system to detect, localize, and identify other vehicles, pedestrians, and road signs to perform safe navigation and decision\-making. In this paper, we compare the performance of DL models, including YOLO\-NAS and YOLOv8, for a detection\-based perception task. We capture a custom dataset and experiment with both DL models using our custom dataset. Our analysis reveals that the YOLOv8s model saves 75% of training time compared to the YOLO\-NAS model. In addition, the YOLOv8s model \(83%\) outperforms the YOLO\-NAS model \(81%\) when the target is to achieve the highest object detection accuracy. These comparative analyses of these new emerging DL models will allow the relevant research community to understand the models' performance under real\-world use case scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.21673v1)

---

