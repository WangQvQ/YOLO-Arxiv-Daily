# 每日从arXiv中获取最新YOLO相关论文


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


## Compilation and Execution of an Embeddable YOLO\-NAS on the VTA / 

发布日期：2026-04-27

作者：Anthony Faure\-Gignoux

摘要：Deploying complex Convolutional Neural Networks \(CNNs\) on FPGA\-based accelerators is a promising way forward for safety\-critical domains such as aeronautics. In a previous work, we have explored the Versatile Tensor Accelerator \(VTA\) and showed its suitability for avionic applications. For that, we developed an initial stand\-alone compiler designed with certification in mind. However, this compiler still suffers from some limitations that are overcome in this paper. The contributions consist in extending and fully automating the VTA compilation chain to allow complete CNN compilation and support larger CNNs \(which parameters do not fit in the on\-chip memory\). The effectiveness is demonstrated by the successful compilation and simulated execution of a YOLO\-NAS object detection model.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.24455v1)

---


## Resource\-Constrained UAV\-Based Weed Detection for Site\-Specific Management on Edge Devices / 

发布日期：2026-04-25

作者：Linyuan Wang

摘要：Weeds compete with crops for light, water, and nutrients, reducing yield and crop quality. Efficient weed detection is essential for site\-specific weed management \(SSWM\). Although deep learning models have been deployed on UAV\-based edge systems, a systematic understanding of how different model architectures perform under real\-world resource constraints is still lacking. To address this gap, this study proposes a deployment\-oriented framework for real\-time UAV\-based weed detection on resource\-constrained edge platforms. The framework integrates UAV data acquisition, model development, and on\-device inference, with a focus on balancing detection accuracy and computational efficiency. A diverse set of state\-of\-the\-art object detection models is evaluated, including convolution\-based YOLO models \(v8\-v12\) and transformer\-based RT\-DETR models \(v1\-v2\). Experiments on three edge devices \(Jetson Orin Nano, Jetson AGX Xavier, and Jetson AGX Orin\) demonstrate clear trade\-offs between accuracy and inference latency across models and hardware configurations. Results show that high\-capacity models achieve up to 86.9% mAP50 but suffer from high latency, limiting real\-time deployment. In contrast, lightweight models achieve 66%\-71% mAP50 with significantly lower latency, enabling real\-time performance. Among all models, RT\-DETRv2\-R50\-M achieves competitive accuracy \(79% mAP50\) with improved efficiency, while YOLOv10n provides the fastest inference speed. YOLOv11s and RT\-DETRv2\-R50\-M offer the best balance between accuracy and speed, making them strong candidates for real\-time UAV deployment.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.23442v1)

---

