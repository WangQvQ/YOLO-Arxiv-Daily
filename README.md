# 每日从arXiv中获取最新YOLO相关论文


## AnyDepth\-DETR/\-YOLO: Any\-depth object detection with a single network / 

发布日期：2026-05-10

作者：Woochul Kang

摘要：Modern object detectors are static, fixed\-depth networks optimized for a single operating point, requiring separate models for different deployment scenarios. We present an any\-depth detection framework that enables a single network to span a continuous range of accuracy\-\-efficiency trade\-offs by controlling depth at inference time without retraining. Each backbone and neck stage is divided into an essential path, which always executes, and a skippable refinement path; this decomposition preserves the full multi\-scale feature hierarchy at every depth configuration, unlike conventional early exiting that discards entire stages. To train such a network, jointly optimizing many sub\-networks of varying depth introduces conflicting gradient signals. We address this via self\-distillation between only the two extremes, with prediction\-level and feature\-level alignment losses that enforce stage\-wise modularity, ensuring the outputs of each stage remain compatible regardless of the paths taken. Instantiated on RT\-DETR and YOLOv12, our full\-depth configurations match or surpass their respective SOTA baselines with negligible parameter overhead, while the most efficient configurations achieve up to $1.82times$ speedup at a cost of only 2.0 AP, all from a single set of weights.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2605.09407v1)

---


## A Marine Debris Detection Framework for Ocean Robots via Self\-Attention Enhancement and Feature Interaction Optimization / 

发布日期：2026-05-08

作者：Yuyang Li

摘要：Marine debris detection for ocean robot is crucial for ecological protection, yet performance is often degraded by low\-quality images with blur, complex backgrounds, and small targets. To address these challenges, we propose YOLO\-MD, an enhanced YOLO\-based detection framework. A Dual\-Branch Convolutional Enhanced Self\-Attention \(DB\-CASA\) module is designed to strengthen spatial\-channel interactions, improving feature representation in degraded images. Additionally, a lightweight shift\-based operation is introduced to enhance fine\-grained feature extraction for objects of varying scales while maintaining parameter efficiency. We further propose SFG\-Loss to mitigate class imbalance and optimization instability via dynamic sample reweighting. Experiments on the UODM dataset demonstrate that YOLO\-MD achieves 0.875 precision, 0.822 F1\-score, and 0.849 mAP50, outperforming the latest state\-of\-the\-art methods. The effectiveness of this method has also been verified through real\-world robotic edge deployment experiments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2605.07388v1)

---


## XiYOLO: Energy\-Aware Object Detection via Iterative Architecture Search and Scaling / 

发布日期：2026-05-07

作者：Tony Tran

摘要：Object detection on heterogeneous edge devices must satisfy strict energy, latency, and memory constraints while still providing reliable perception for downstream autonomy. Existing energy\-aware NAS methods often target limited deployment settings, while real energy remains difficult to optimize because it is highly device\-dependent and costly to measure. We address these challenges with an energy\-adaptive framework that combines an energy\-aware XiResOFA search space, a two\-stage energy estimator, and iterative search to identify a single energy\-efficient base architecture. We then apply compound scaling to transform this base design into the XiYOLO family across deployment budgets, enabling interpretable accuracy\-energy tradeoffs under sparse hardware measurements. Experiments on PascalVOC, COCO, and real\-device deployment show that XiYOLO achieves a stronger energy\-accuracy tradeoff than YOLO baselines. On PascalVOC, the medium XiYOLO model reaches 86.15 mAP50 while reducing energy relative to YOLOv12m by 20.6% on GPU and 35.9% on NPU. On COCO, XiYOLO reduces energy relative to YOLOv12 by up to 53.7% on GPU and 51.6% on NPU at the small scale. The proposed two\-stage estimator also improves sample efficiency over a joint predictor under few\-shot adaptation with only 2\-20 target\-device samples.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2605.06927v1)

---


## Low\-Cost Stereo Vision for Robust 3D Positioning of Thin Radiata Pine Branches in Autonomous Drone Pruning / 

发布日期：2026-05-06

作者：Yida Lin

摘要：Manual pruning of radiata pine, a species of major economic importance to New Zealand forestry, is hazardous, labour\-intensive, and increasingly constrained by workforce shortages. Existing autonomous pruning platforms typically rely on expensive sensors such as LiDAR and are limited to thick branches, which restricts their wider adoption. This paper investigates whether a single low\-cost stereo camera mounted on a drone can provide sufficiently accurate branch detection and three\-dimensional positioning to support autonomous pruning of branches as thin as 10 mm, thereby removing the need for auxiliary depth sensors. The proposed pipeline comprises two stages: branch segmentation and depth estimation. For segmentation, Mask R\-CNN variants and the YOLOv8 and YOLOv9 families are compared on a custom dataset of 71 stereo image pairs captured with a ZED Mini camera; YOLOv8 and YOLOv9 are selected as representative state\-of\-the\-art real\-time segmentors at the time of data collection, and the framework is designed to remain compatible with newer YOLO releases. For depth estimation, a traditional method \(SGBM with WLS filtering\) and deep\-learning\-based methods \(PSMNet, ACVNet, GWCNet, MobileStereoNet, RAFT\-Stereo, and NeRF\-Supervised Deep Stereo\) are evaluated, including cross\-dataset fine\-tuning experiments that expose the domain gap between urban driving benchmarks and natural forestry scenes. The main novelty of this work lies in coupling stereo segmentation with a centroid\-based triangulation algorithm and Median\-Absolute\-Deviation outlier rejection that converts a segmentation mask and disparity map into a single robust branch\-to\-camera distance, addressing the challenges of sparse texture, thin structures, and noisy disparity values typical of forest scenes. Qualitative evaluations at distances of 1\-2 m show that the learning\-based stereo methods produce more coherent depth es...

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2605.08213v1)

---


## StreakMind: AI detection and analysis of satellite streaks in astronomical images with automated database integration / 

发布日期：2026-05-05

作者：Rafael Carrillo Navarro

摘要：Artificial satellites and space debris increasingly contaminate astronomical images, affecting scientific surveys and producing large volumes of streaked exposures. Manual inspection is no longer feasible at scale, and reliable detection and characterisation of streaks has become essential for both data\-quality control and the monitoring of objects in Earth orbit. We present StreakMind, an automated pipeline designed to detect Near\-Earth Objects and satellite streaks in astronomical images, characterise their geometry, and cross\-identify them with known orbital objects. The system integrates all inference results into a structured database suitable for large surveys. A YOLO OBB model was trained on a hybrid dataset of 2335 images and applied to processed FITS frames. Geometric refinement, inter\-frame association, satellite cross\-identification, and Gaussian\-based confidence scoring were then used to produce final identifications stored in a relational database. Observations from La Sagra Observatory were used to develop and test the method. On the test set, the model achieved a precision of 94 percent and a recall of 97 percent. It reliably detected faint streaks, delivered consistent geometric reconstructions, and performed robust satellite cross\-identification. StreakMind demonstrates strong potential for large\-scale automated analysis of linear streaks produced by both Near\-Earth Objects and artificial satellites, contributing to space situational awareness.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2605.03429v1)

---

