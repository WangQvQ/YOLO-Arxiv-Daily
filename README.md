# 每日从arXiv中获取最新YOLO相关论文


## SARES\-DEIM: Sparse Mixture\-of\-Experts Meets DETR for Robust SAR Ship Detection / 

发布日期：2026-04-05

作者：Fenghao Song

摘要：Ship detection in Synthetic Aperture Radar \(SAR\) imagery is fundamentally challenged by inherent coherent speckle noise, complex coastal clutter, and the prevalence of small\-scale targets. Conventional detectors, primarily designed for optical imagery, often exhibit limited robustness against SAR\-specific degradation and suffer from the loss of fine\-grained ship signatures during spatial downsampling. To address these limitations, we propose SARES\-DEIM, a domain\-aware detection framework grounded in the DEtection TRansformer \(DETR\) paradigm. Central to our approach is SARESMoE \(SAR\-aware Expert Selection Mixture\-of\-Experts\), a module leveraging a sparse gating mechanism to selectively route features toward specialized frequency and wavelet experts. This sparsely\-activated architecture effectively filters speckle noise and semantic clutter while maintaining high computational efficiency. Furthermore, we introduce the Space\-to\-Depth Enhancement Pyramid \(SDEP\) neck to preserve high\-resolution spatial cues from shallow stages, significantly improving the localization of small targets. Extensive experiments on two benchmark datasets demonstrate the superiority of SARES\-DEIM. Notably, on the challenging HRSID dataset, our model achieves a mAP50:95 of 76.4% and a mAP50 of 93.8%, outperforming state\-of\-the\-art YOLO\-series and specialized SAR detectors.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.04127v1)

---


## Can VLMs Truly Forget? Benchmarking Training\-Free Visual Concept Unlearning / 

发布日期：2026-04-03

作者：Zhangyun Tan

摘要：VLMs trained on web\-scale data retain sensitive and copyrighted visual concepts that deployment may require removing. Training\-based unlearning methods share a structural flaw: fine\-tuning on a narrow forget set degrades general capabilities before unlearning begins, making it impossible to attribute subsequent performance drops to the unlearning procedure itself. Training\-free approaches sidestep this by suppressing concepts through prompts or system instructions, but no rigorous benchmark exists for evaluating them on visual tasks.   We introduce VLM\-UnBench, the first benchmark for training\-free visual concept unlearning in VLMs. It covers four forgetting levels, 7 source datasets, and 11 concept axes, and pairs a three\-level probe taxonomy with five evaluation conditions to separate genuine forgetting from instruction compliance. Across 8 evaluation settings and 13 VLM configurations, realistic unlearning prompts leave forget accuracy near the no\-instruction baseline; meaningful reductions appear only under oracle conditions that disclose the target concept to the model. Object and scene concepts are the most resistant to suppression, and stronger instruction\-tuned models remain capable despite explicit forget instructions. These results expose a clear gap between prompt\-level suppression and true visual concept erasure.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.03114v1)

---


## YOLOv11 Demystified: A Practical Guide to High\-Performance Object Detection / 

发布日期：2026-04-03

作者：Nikhileswara Rao Sulake

摘要：YOLOv11 is the latest iteration in the You Only Look Once \(YOLO\) series of real\-time object detectors, introducing novel architectural modules to improve feature extraction and small\-object detection. In this paper, we present a detailed analysis of YOLOv11, including its backbone, neck, and head components. The model key innovations, the C3K2 blocks, Spatial Pyramid Pooling \- Fast \(SPPF\), and C2PSA \(Cross Stage Partial with Spatial Attention\) modules enhance spatial feature processing while preserving speed. We compare YOLOv11 performance to prior YOLO versions on standard benchmarks, highlighting improvements in mean Average Precision \(mAP\) and inference speed. Our results demonstrate that YOLOv11 achieves superior accuracy without sacrificing real\-time capabilities, making it well\-suited for applications in autonomous driving, surveillance, and video analytics.This work formalizes YOLOv11 in a research context, providing a clear reference for future studies.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.03349v1)

---


## Deep Neural Network Based Roadwork Detection for Autonomous Driving / 

发布日期：2026-04-02

作者：Sebastian Wullrich

摘要：Road construction sites create major challenges for both autonomous vehicles and human drivers due to their highly dynamic and heterogeneous nature. This paper presents a real\-time system that detects and localizes roadworks by combining a YOLO neural network with LiDAR data. The system identifies individual roadwork objects while driving, merges them into coherent construction sites and records their outlines in world coordinates. The model training was based on an adapted US dataset and a new dataset collected from test drives with a prototype vehicle in Berlin, Germany. Evaluations on real\-world road construction sites showed a localization accuracy below 0.5 m. The system can support traffic authorities with up\-to\-date roadwork data and could enable autonomous vehicles to navigate construction sites more safely in the future.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.02282v1)

---


## Gaze to Insight: A Scalable AI Approach for Detecting Gaze Behaviours in Face\-to\-Face Collaborative Learning / 

发布日期：2026-04-01

作者：Junyuan Liang

摘要：Previous studies have illustrated the potential of analysing gaze behaviours in collaborative learning to provide educationally meaningful information for students to reflect on their learning. Over the past decades, machine learning approaches have been developed to automatically detect gaze behaviours from video data. Yet, since these approaches often require large amounts of labelled data for training, human annotation remains necessary. Additionally, researchers have questioned the cross\-configuration robustness of machine learning models developed, as training datasets often fail to encompass the full range of situations encountered in educational contexts. To address these challenges, this study proposes a scalable artificial intelligence approach that leverages pretrained and foundation models to automatically detect gaze behaviours in face\-to\-face collaborative learning contexts without requiring human\-annotated data. The approach utilises pretrained YOLO11 for person tracking, YOLOE\-26 with text\-prompt capability for education\-related object detection, and the Gaze\-LLE model for gaze target prediction. The results indicate that the proposed approach achieves an F1\-score of 0.829 in detecting students' gaze behaviours from video data, with strong performance for laptop\-directed gaze and peer\-directed gaze, yet weaker performance for other gaze targets. Furthermore, when compared to other supervised machine learning approaches, the proposed method demonstrates superior and more stable performance in complex contexts, highlighting its better cross\-configuration robustness. The implications of this approach for supporting students' collaborative learning in real\-world environments are also discussed.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.03317v1)

---

