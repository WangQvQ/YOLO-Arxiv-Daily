# 每日从arXiv中获取最新YOLO相关论文


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


## EgoMAGIC\- An Egocentric Video Field Medicine Dataset for Training Perception Algorithms / 

发布日期：2026-04-23

作者：Brian VanVoorst

摘要：This paper introduces EgoMAGIC \(Medical Assistance, Guidance, Instruction, and Correction\), an egocentric medical activity dataset collected as part of DARPA's Perceptually\-enabled Task Guidance \(PTG\) program. This dataset comprises 3,355 videos of 50 medical tasks, with at least 50 labeled videos per task. The primary objective of the PTG program was to develop virtual assistants integrated into augmented reality headsets to assist users in performing complex tasks.   To encourage exploration and research using this dataset, the medical training data has been released along with an action detection challenge focused on eight medical tasks. The majority of the videos were recorded using a head\-mounted stereo camera with integrated audio. From this dataset, 40 YOLO models were trained using 1.95 million labels to detect 124 medical objects, providing a robust starting point for developers working on medical AI applications.   In addition to introducing the dataset, this paper presents baseline results on action detection for the eight selected medical tasks across three models, with the best\-performing method achieving average mAP 0.526. Although this paper primarily addresses action detection as the benchmark, the EgoMAGIC dataset is equally suitable for action recognition, object identification and detection, error detection, and other challenging computer vision tasks.   The dataset is accessible via zenodo.org \(DOI: 10.5281/zenodo.19239154\).

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.22036v1)

---


## Proactive Detection of GUI Defects in Multi\-Window Scenarios via Multimodal Reasoning / 

发布日期：2026-04-21

作者：Xinyao Zhang

摘要：Multi\-window mobile scenarios, such as split\-screen and foldable modes, make GUI display defects more likely by forcing applications to adapt to changing window sizes and dynamic layout reflow. Existing detection techniques are limited in two ways: they are largely passive, analyzing screenshots only after problematic states have been reached, and they are mainly designed for conventional full\-screen interfaces, making them less effective in multi\-window settings.We propose an end\-to\-end framework for GUI display defect detection in multi\-window mobile scenarios. The framework proactively triggers split\-screen, foldable, and window\-transition states during app exploration, uses Set\-of\-Mark \(SoM\) to align screenshots with widget\-level interface elements, and leverages multimodal large language models with chain\-of\-thought prompting to detect, localize, and explain display defects. We also construct a benchmark of GUI display defects using 50 real\-world Android applications.Experimental results show that multi\-window settings substantially increase the exposure of layout\-related defects, with text truncation increasing by 184% compared with conventional full\-screen settings. At the application level, our method detects 40 defect\-prone apps with a false positive rate of 10.00% and a false negative rate of 11.11%, outperforming OwlEye and YOLO\-based baselines. At the fine\-grained level, it achieves the best F1 score of 87.2% for widget occlusion detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.19081v1)

---


## Domain\-Specialized Object Detection via Model\-Level Mixtures of Experts / 

发布日期：2026-04-20

作者：Svetlana Pavlitska

摘要：Mixture\-of\-Experts \(MoE\) models provide a structured approach to combining specialized neural networks and offer greater interpretability than conventional ensembles. While MoEs have been successfully applied to image classification and semantic segmentation, their use in object detection remains limited due to challenges in merging dense and structured predictions. In this work, we investigate model\-level mixtures of object detectors and analyze their suitability for improving performance and interpretability in object detection. We propose an MoE architecture that combines YOLO\-based detectors trained on semantically disjoint data subsets, with a learned gating network that dynamically weights expert contributions. We study different strategies for fusing detection outputs and for training the gating mechanism, including balancing losses to prevent expert collapse. Experiments on the BDD100K dataset demonstrate that the proposed MoE consistently outperforms standard ensemble approaches and provides insights into expert specialization across domains, highlighting model\-level MoEs as a viable alternative to traditional ensembling for object detection. Our code is available at https://github.com/KASTEL\-MobilityLab/mixtures\-of\-experts/.

中文摘要：


代码链接：https://github.com/KASTEL-MobilityLab/mixtures-of-experts/.

论文链接：[阅读更多](http://arxiv.org/abs/2604.18256v1)

---

