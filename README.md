# 每日从arXiv中获取最新YOLO相关论文


## BLO\-Inst: Bi\-Level Optimization Based Alignment of YOLO and SAM for Robust Instance Segmentation / 

发布日期：2026-01-29

作者：Li Zhang

摘要：The Segment Anything Model has revolutionized image segmentation with its zero\-shot capabilities, yet its reliance on manual prompts hinders fully automated deployment. While integrating object detectors as prompt generators offers a pathway to automation, existing pipelines suffer from two fundamental limitations: objective mismatch, where detectors optimized for geometric localization do not correspond to the optimal prompting context required by SAM, and alignment overfitting in standard joint training, where the detector simply memorizes specific prompt adjustments for training samples rather than learning a generalizable policy. To bridge this gap, we introduce BLO\-Inst, a unified framework that aligns detection and segmentation objectives by bi\-level optimization. We formulate the alignment as a nested optimization problem over disjoint data splits. In the lower level, the SAM is fine\-tuned to maximize segmentation fidelity given the current detection proposals on a subset \($D\_1$\). In the upper level, the detector is updated to generate bounding boxes that explicitly minimize the validation loss of the fine\-tuned SAM on a separate subset \($D\_2$\). This effectively transforms the detector into a segmentation\-aware prompt generator, optimizing the bounding boxes not just for localization accuracy, but for downstream mask quality. Extensive experiments demonstrate that BLO\-Inst achieves superior performance, outperforming standard baselines on tasks in general and biomedical domains.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.22061v1)

---


## An AI Framework for Microanastomosis Motion Assessment / 

发布日期：2026-01-28

作者：Yan Meng

摘要：Proficiency in microanastomosis is a fundamental competency across multiple microsurgical disciplines. These procedures demand exceptional precision and refined technical skills, making effective, standardized assessment methods essential. Traditionally, the evaluation of microsurgical techniques has relied heavily on the subjective judgment of expert raters. They are inherently constrained by limitations such as inter\-rater variability, lack of standardized evaluation criteria, susceptibility to cognitive bias, and the time\-intensive nature of manual review. These shortcomings underscore the urgent need for an objective, reliable, and automated system capable of assessing microsurgical performance with consistency and scalability. To bridge this gap, we propose a novel AI framework for the automated assessment of microanastomosis instrument handling skills. The system integrates four core components: \(1\) an instrument detection module based on the You Only Look Once \(YOLO\) architecture; \(2\) an instrument tracking module developed from Deep Simple Online and Realtime Tracking \(DeepSORT\); \(3\) an instrument tip localization module employing shape descriptors; and \(4\) a supervised classification module trained on expert\-labeled data to evaluate instrument handling proficiency. Experimental results demonstrate the effectiveness of the framework, achieving an instrument detection precision of 97%, with a mean Average Precision \(mAP\) of 96%, measured by Intersection over Union \(IoU\) thresholds ranging from 50% to 95% \(mAP50\-95\).

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.21120v1)

---


## VisGuardian: A Lightweight Group\-based Privacy Control Technique For Front Camera Data From AR Glasses in Home Environments / 

发布日期：2026-01-27

作者：Shuning Zhang

摘要：Always\-on sensing of AI applications on AR glasses makes traditional permission techniques ill\-suited for context\-dependent visual data, especially within home environments. The home presents a highly challenging privacy context due to the high density of sensitive objects, and the frequent presence of non\-consenting family members, and the intimate nature of daily routines, making it a critical focus area for scalable privacy control mechanisms. Existing fine\-grained controls, while offering nuanced choices, are inefficient for managing multiple private objects. We propose VisGuardian, a fine\-grained content\-based visual permission technique for AR glasses. VisGuardian features a group\-based control mechanism that enables users to efficiently manage permissions for multiple private objects. VisGuardian detects objects using YOLO and adopts a pre\-classified schema to group them. By selecting a single object, users can efficiently obscure groups of related objects based on criteria including privacy sensitivity, object category, or spatial proximity. A technical evaluation shows VisGuardian achieves mAP50 of 0.6704 with only 14.0 ms latency and a 1.7% increase in battery consumption per hour. Furthermore, a user study \(N=24\) comparing VisGuardian to slider\-based and object\-based baselines found it to be significantly faster for setting permissions and was preferred by users for its efficiency, effectiveness, and ease of use.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.19502v1)

---


## YOLO\-DS: Fine\-Grained Feature Decoupling via Dual\-Statistic Synergy Operator for Object Detection / 

发布日期：2026-01-26

作者：Lin Huang

摘要：One\-stage object detection, particularly the YOLO series, strikes a favorable balance between accuracy and efficiency. However, existing YOLO detectors lack explicit modeling of heterogeneous object responses within shared feature channels, which limits further performance gains. To address this, we propose YOLO\-DS, a framework built around a novel Dual\-Statistic Synergy Operator \(DSO\). The DSO decouples object features by jointly modeling the channel\-wise mean and the peak\-to\-mean difference. Building upon the DSO, we design two lightweight gating modules: the Dual\-Statistic Synergy Gating \(DSG\) module for adaptive channel\-wise feature selection, and the Multi\-Path Segmented Gating \(MSG\) module for depth\-wise feature weighting. On the MS\-COCO benchmark, YOLO\-DS consistently outperforms YOLOv8 across five model scales \(N, S, M, L, X\), achieving AP gains of 1.1% to 1.7% with only a minimal increase in inference latency. Extensive visualization, ablation, and comparative studies validate the effectiveness of our approach, demonstrating its superior capability in discriminating heterogeneous objects with high efficiency.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.18172v1)

---


## The Latency Wall: Benchmarking Off\-the\-Shelf Emotion Recognition for Real\-Time Virtual Avatars / 

发布日期：2026-01-22

作者：Yarin Benyamin

摘要：In the realm of Virtual Reality \(VR\) and Human\-Computer Interaction \(HCI\), real\-time emotion recognition shows promise for supporting individuals with Autism Spectrum Disorder \(ASD\) in improving social skills. This task requires a strict latency\-accuracy trade\-off, with motion\-to\-photon \(MTP\) latency kept below 140 ms to maintain contingency. However, most off\-the\-shelf Deep Learning models prioritize accuracy over the strict timing constraints of commodity hardware. As a first step toward accessible VR therapy, we benchmark State\-of\-the\-Art \(SOTA\) models for Zero\-Shot Facial Expression Recognition \(FER\) on virtual characters using the UIBVFED dataset. We evaluate Medium and Nano variants of YOLO \(v8, v11, and v12\) for face detection, alongside general\-purpose Vision Transformers including CLIP, SigLIP, and ViT\-FER.Our results on CPU\-only inference demonstrate that while face detection on stylized avatars is robust \(100% accuracy\), a "Latency Wall" exists in the classification stage. The YOLOv11n architecture offers the optimal balance for detection \(~54 ms\). However, general\-purpose Transformers like CLIP and SigLIP fail to achieve viable accuracy \(<23%\) or speed \(>150 ms\) for real\-time loops. This study highlights the necessity for lightweight, domain\-specific architectures to enable accessible, real\-time AI in therapeutic settings.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.15914v1)

---

