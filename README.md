# 每日从arXiv中获取最新YOLO相关论文


## The Latency Wall: Benchmarking Off\-the\-Shelf Emotion Recognition for Real\-Time Virtual Avatars / 

发布日期：2026-01-22

作者：Yarin Benyamin

摘要：In the realm of Virtual Reality \(VR\) and Human\-Computer Interaction \(HCI\), real\-time emotion recognition shows promise for supporting individuals with Autism Spectrum Disorder \(ASD\) in improving social skills. This task requires a strict latency\-accuracy trade\-off, with motion\-to\-photon \(MTP\) latency kept below 140 ms to maintain contingency. However, most off\-the\-shelf Deep Learning models prioritize accuracy over the strict timing constraints of commodity hardware. As a first step toward accessible VR therapy, we benchmark State\-of\-the\-Art \(SOTA\) models for Zero\-Shot Facial Expression Recognition \(FER\) on virtual characters using the UIBVFED dataset. We evaluate Medium and Nano variants of YOLO \(v8, v11, and v12\) for face detection, alongside general\-purpose Vision Transformers including CLIP, SigLIP, and ViT\-FER.Our results on CPU\-only inference demonstrate that while face detection on stylized avatars is robust \(100% accuracy\), a "Latency Wall" exists in the classification stage. The YOLOv11n architecture offers the optimal balance for detection \(~54 ms\). However, general\-purpose Transformers like CLIP and SigLIP fail to achieve viable accuracy \(<23%\) or speed \(>150 ms\) for real\-time loops. This study highlights the necessity for lightweight, domain\-specific architectures to enable accessible, real\-time AI in therapeutic settings.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.15914v1)

---


## HumanDiffusion: A Vision\-Based Diffusion Trajectory Planner with Human\-Conditioned Goals for Search and Rescue UAV / 

发布日期：2026-01-21

作者：Faryal Batool

摘要：Reliable human\-\-robot collaboration in emergency scenarios requires autonomous systems that can detect humans, infer navigation goals, and operate safely in dynamic environments. This paper presents HumanDiffusion, a lightweight image\-conditioned diffusion planner that generates human\-aware navigation trajectories directly from RGB imagery. The system combines YOLO\-11 based human detection with diffusion\-driven trajectory generation, enabling a quadrotor to approach a target person and deliver medical assistance without relying on prior maps or computationally intensive planning pipelines. Trajectories are predicted in pixel space, ensuring smooth motion and a consistent safety margin around humans. We evaluate HumanDiffusion in simulation and real\-world indoor mock\-disaster scenarios. On a 300\-sample test set, the model achieves a mean squared error of 0.02 in pixel\-space trajectory reconstruction. Real\-world experiments demonstrate an overall mission success rate of 80% across accident\-response and search\-and\-locate tasks with partial occlusions. These results indicate that human\-conditioned diffusion planning offers a practical and robust solution for human\-aware UAV navigation in time\-critical assistance settings.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.14973v2)

---


## YOLO26: An Analysis of NMS\-Free End to End Framework for Real\-Time Object Detection / 

发布日期：2026-01-19

作者：Sudip Chakrabarty

摘要：The "You Only Look Once" \(YOLO\) framework has long served as the benchmark for real\-time object detection, yet traditional iterations \(YOLOv1 through YOLO11\) remain constrained by the latency and hyperparameter sensitivity of Non\-Maximum Suppression \(NMS\) post\-processing. This paper analyzes a comprehensive analysis of YOLO26, an architecture that fundamentally redefines this paradigm by eliminating NMS in favor of a native end\-to\-end learning strategy. This study examines the critical innovations that enable this transition, specifically the introduction of the MuSGD optimizer for stabilizing lightweight backbones, STAL for small\-target\-aware assignment, and ProgLoss for dynamic supervision. Through a systematic review of official performance benchmarks, the results demonstrate that YOLO26 establishes a new Pareto front, outperforming a comprehensive suite of predecessors and state\-of\-the\-art competitors \(including RTMDet and DAMO\-YOLO\) in both inference speed and detection accuracy. The analysis confirms that by decoupling representation learning from heuristic post\-processing, YOLOv26 successfully resolves the historical trade\-off between latency and precision, signaling the next evolutionary step in edge\-based computer vision.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.12882v1)

---


## A Multimodal Assistive System for Product Localization and Retrieval for People who are Blind or have Low Vision / 

发布日期：2026-01-18

作者：Ligao Ruan

摘要：Shopping is a routine activity for sighted individuals, yet for people who are blind or have low vision \(pBLV\), locating and retrieving products in physical environments remains a challenge. This paper presents a multimodal wearable assistive system that integrates object detection with vision\-language models to support independent product or item retrieval, with the goal of enhancing users'autonomy and sense of agency. The system operates through three phases: product search, which identifies target products using YOLO\-World detection combined with embedding similarity and color histogram matching; product navigation, which provides spatialized sonification and VLM\-generated verbal descriptions to guide users toward the target; and product correction, which verifies whether the user has reached the correct product and provides corrective feedback when necessary. Technical evaluation demonstrated promising performance across all modules, with product detection achieving near\-perfect accuracy at close range and high accuracy when facing shelves within 1.5 m. VLM\-based navigation achieved up to 94.4% accuracy, and correction accuracy exceeded 86% under optimal model configurations. These results demonstrate the system's potential to address the last\-meter problem in assistive shopping. Future work will focus on user studies with pBLV participants and integration with multi\-scale navigation ecosystems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.12486v1)

---


## SME\-YOLO: A Real\-Time Detector for Tiny Defect Detection on PCB Surfaces / 

发布日期：2026-01-16

作者：Meng Han

摘要：Surface defects on Printed Circuit Boards \(PCBs\) directly compromise product reliability and safety. However, achieving high\-precision detection is challenging because PCB defects are typically characterized by tiny sizes, high texture similarity, and uneven scale distributions. To address these challenges, this paper proposes a novel framework based on YOLOv11n, named SME\-YOLO \(Small\-target Multi\-scale Enhanced YOLO\). First, we employ the Normalized Wasserstein Distance Loss \(NWDLoss\). This metric effectively mitigates the sensitivity of Intersection over Union \(IoU\) to positional deviations in tiny objects. Second, the original upsampling module is replaced by the Efficient Upsampling Convolution Block \(EUCB\). By utilizing multi\-scale convolutions, the EUCB gradually recovers spatial resolution and enhances the preservation of edge and texture details for tiny defects. Finally, this paper proposes the Multi\-Scale Focused Attention \(MSFA\) module. Tailored to the specific spatial distribution of PCB defects, this module adaptively strengthens perception within key scale intervals, achieving efficient fusion of local fine\-grained features and global context information. Experimental results on the PKU\-PCB dataset demonstrate that SME\-YOLO achieves state\-of\-the\-art performance. Specifically, compared to the baseline YOLOv11n, SME\-YOLO improves mAP by 2.2% and Precision by 4%, validating the effectiveness of the proposed method.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.11402v1)

---

