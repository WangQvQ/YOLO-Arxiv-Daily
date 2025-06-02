# 每日从arXiv中获取最新YOLO相关论文


## Leadership Assessment in Pediatric Intensive Care Unit Team Training / 

发布日期：2025-05-30

作者：Liangyang Ouyang

摘要：This paper addresses the task of assessing PICU team's leadership skills by developing an automated analysis framework based on egocentric vision. We identify key behavioral cues, including fixation object, eye contact, and conversation patterns, as essential indicators of leadership assessment. In order to capture these multimodal signals, we employ Aria Glasses to record egocentric video, audio, gaze, and head movement data. We collect one\-hour videos of four simulated sessions involving doctors with different roles and levels. To automate data processing, we propose a method leveraging REMoDNaV, SAM, YOLO, and ChatGPT for fixation object detection, eye contact detection, and conversation classification. In the experiments, significant correlations are observed between leadership skills and behavioral metrics, i.e., the output of our proposed methods, such as fixation time, transition patterns, and direct orders in speech. These results indicate that our proposed data collection and analysis framework can effectively solve skill assessment for training PICU teams.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.24389v1)

---


## Robust and Annotation\-Free Wound Segmentation on Noisy Real\-World Pressure Ulcer Images: Towards Automated DESIGN\-R\textsuperscript\{\textregistered\} Assessment / 

发布日期：2025-05-29

作者：Yun\-Cheng Tsai

摘要：Purpose: Accurate wound segmentation is essential for automated DESIGN\-R scoring. However, existing models such as FUSegNet, which are trained primarily on foot ulcer datasets, often fail to generalize to wounds on other body sites.   Methods: We propose an annotation\-efficient pipeline that combines a lightweight YOLOv11n\-based detector with the pre\-trained FUSegNet segmentation model. Instead of relying on pixel\-level annotations or retraining for new anatomical regions, our method achieves robust performance using only 500 manually labeled bounding boxes. This zero fine\-tuning approach effectively bridges the domain gap and enables direct deployment across diverse wound types. This is an advance not previously demonstrated in the wound segmentation literature.   Results: Evaluated on three real\-world test sets spanning foot, sacral, and trochanter wounds, our YOLO plus FUSegNet pipeline improved mean IoU by 23 percentage points over vanilla FUSegNet and increased end\-to\-end DESIGN\-R size estimation accuracy from 71 percent to 94 percent \(see Table 3 for details\).   Conclusion: Our pipeline generalizes effectively across body sites without task\-specific fine\-tuning, demonstrating that minimal supervision, with 500 annotated ROIs, is sufficient for scalable, annotation\-light wound segmentation. This capability paves the way for real\-world DESIGN\-R automation, reducing reliance on pixel\-wise labeling, streamlining documentation workflows, and supporting objective and consistent wound scoring in clinical practice. We will publicly release the trained detector weights and configuration to promote reproducibility and facilitate downstream deployment.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.23392v1)

---


## YOLO\-SPCI: Enhancing Remote Sensing Object Detection via Selective\-Perspective\-Class Integration / 

发布日期：2025-05-27

作者：Xinyuan Wang

摘要：Object detection in remote sensing imagery remains a challenging task due to extreme scale variation, dense object distributions, and cluttered backgrounds. While recent detectors such as YOLOv8 have shown promising results, their backbone architectures lack explicit mechanisms to guide multi\-scale feature refinement, limiting performance on high\-resolution aerial data. In this work, we propose YOLO\-SPCI, an attention\-enhanced detection framework that introduces a lightweight Selective\-Perspective\-Class Integration \(SPCI\) module to improve feature representation. The SPCI module integrates three components: a Selective Stream Gate \(SSG\) for adaptive regulation of global feature flow, a Perspective Fusion Module \(PFM\) for context\-aware multi\-scale integration, and a Class Discrimination Module \(CDM\) to enhance inter\-class separability. We embed two SPCI blocks into the P3 and P5 stages of the YOLOv8 backbone, enabling effective refinement while preserving compatibility with the original neck and head. Experiments on the NWPU VHR\-10 dataset demonstrate that YOLO\-SPCI achieves superior performance compared to state\-of\-the\-art detectors.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.21370v1)

---


## YOLO\-FireAD: Efficient Fire Detection via Attention\-Guided Inverted Residual Learning and Dual\-Pooling Feature Preservation / 

发布日期：2025-05-27

作者：Weichao Pan

摘要：Fire detection in dynamic environments faces continuous challenges, including the interference of illumination changes, many false detections or missed detections, and it is difficult to achieve both efficiency and accuracy. To address the problem of feature extraction limitation and information loss in the existing YOLO\-based models, this study propose You Only Look Once for Fire Detection with Attention\-guided Inverted Residual and Dual\-pooling Downscale Fusion \(YOLO\-FireAD\) with two core innovations: \(1\) Attention\-guided Inverted Residual Block \(AIR\) integrates hybrid channel\-spatial attention with inverted residuals to adaptively enhance fire features and suppress environmental noise; \(2\) Dual Pool Downscale Fusion Block \(DPDF\) preserves multi\-scale fire patterns through learnable fusion of max\-average pooling outputs, mitigating small\-fire detection failures. Extensive evaluation on two public datasets shows the efficient performance of our model. Our proposed model keeps the sum amount of parameters \(1.45M, 51.8% lower than YOLOv8n\) \(4.6G, 43.2% lower than YOLOv8n\), and mAP75 is higher than the mainstream real\-time object detection models YOLOv8n, YOL\-Ov9t, YOLOv10n, YOLO11n, YOLOv12n and other YOLOv8 variants 1.3\-5.5%.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.20884v1)

---


## Knowledge Distillation Approach for SOS Fusion Staging: Towards Fully Automated Skeletal Maturity Assessment / 

发布日期：2025-05-27

作者：Omid Halimi Milani

摘要：We introduce a novel deep learning framework for the automated staging of spheno\-occipital synchondrosis \(SOS\) fusion, a critical diagnostic marker in both orthodontics and forensic anthropology. Our approach leverages a dual\-model architecture wherein a teacher model, trained on manually cropped images, transfers its precise spatial understanding to a student model that operates on full, uncropped images. This knowledge distillation is facilitated by a newly formulated loss function that aligns spatial logits as well as incorporates gradient\-based attention spatial mapping, ensuring that the student model internalizes the anatomically relevant features without relying on external cropping or YOLO\-based segmentation. By leveraging expert\-curated data and feedback at each step, our framework attains robust diagnostic accuracy, culminating in a clinically viable end\-to\-end pipeline. This streamlined approach obviates the need for additional pre\-processing tools and accelerates deployment, thereby enhancing both the efficiency and consistency of skeletal maturation assessment in diverse clinical settings.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.21561v1)

---

