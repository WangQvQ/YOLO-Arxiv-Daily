# 每日从arXiv中获取最新YOLO相关论文


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


## StreakMind: AI detection and analysis of satellite streaks in astronomical images with automated database integration / 

发布日期：2026-05-05

作者：Rafael Carrillo Navarro

摘要：Artificial satellites and space debris increasingly contaminate astronomical images, affecting scientific surveys and producing large volumes of streaked exposures. Manual inspection is no longer feasible at scale, and reliable detection and characterisation of streaks has become essential for both data\-quality control and the monitoring of objects in Earth orbit. We present StreakMind, an automated pipeline designed to detect Near\-Earth Objects and satellite streaks in astronomical images, characterise their geometry, and cross\-identify them with known orbital objects. The system integrates all inference results into a structured database suitable for large surveys. A YOLO OBB model was trained on a hybrid dataset of 2335 images and applied to processed FITS frames. Geometric refinement, inter\-frame association, satellite cross\-identification, and Gaussian\-based confidence scoring were then used to produce final identifications stored in a relational database. Observations from La Sagra Observatory were used to develop and test the method. On the test set, the model achieved a precision of 94 percent and a recall of 97 percent. It reliably detected faint streaks, delivered consistent geometric reconstructions, and performed robust satellite cross\-identification. StreakMind demonstrates strong potential for large\-scale automated analysis of linear streaks produced by both Near\-Earth Objects and artificial satellites, contributing to space situational awareness.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2605.03429v1)

---


## An Extended Evaluation Split for DeepSpaceYoloDataset / 

发布日期：2026-04-30

作者：Olivier Parisot

摘要：Recent technological advances in astronomy, particularly the growing popularity of smart telescopes for the general public, make it possible to develop highly effective detection solutions that are accessible to a wide audience, rather than being reserved for major scientific observatories. Published in 2023, DeepSpaceYoloDataset is a collection of annotated images created to train YOLO\-based models for detecting Deep Sky Objects, particularly suited for Electronically Assisted Astronomy. In this paper, we present an update to DeepSpaceYoloDataset with the addition of a new split, test2026, designed to evaluate detection models with a greater diversity of images.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.27593v1)

---


## Comparative Evaluation of Convolutional and Transformer\-Based Detectors for Automated Weed Detection in Precision Agriculture / 

发布日期：2026-04-29

作者：Alcides Toledo Espinosa

摘要：This paper presents a comparative evaluation of convolutional and transformer\-based object detection architectures for early weed detection in realistic scenarios. Representative models from each paradigm are considered, including YOLOv26\-nano, a recent variant of the YOLO family, and transformer\-based approaches such as RTDETR and RF\-DETR. Experiments were conducted on the GROUNDBASED\_ WEED dataset, allowing performance to be evaluated in terms of detection accuracy and computational efficiency using metrics such as precision, recall, average precision, and inference speed. The results highlight a clear trade\-off between efficiency and contextual modeling: CNN\-based detectors achieve high performance at a lower computational cost, while transformer\-based approaches offer better global context capture at the expense of higher resource demands. These results provide practical criteria for model selection in precision agriculture applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2605.00908v1)

---

