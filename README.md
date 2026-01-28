# 每日从arXiv中获取最新YOLO相关论文


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

