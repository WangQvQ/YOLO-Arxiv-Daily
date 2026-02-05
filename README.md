# 每日从arXiv中获取最新YOLO相关论文


## From Vision to Assistance: Gaze and Vision\-Enabled Adaptive Control for a Back\-Support Exoskeleton / 

发布日期：2026-02-04

作者：Alessandro Leanza

摘要：Back\-support exoskeletons have been proposed to mitigate spinal loading in industrial handling, yet their effectiveness critically depends on timely and context\-aware assistance. Most existing approaches rely either on load\-estimation techniques \(e.g., EMG, IMU\) or on vision systems that do not directly inform control. In this work, we present a vision\-gated control framework for an active lumbar occupational exoskeleton that leverages egocentric vision with wearable gaze tracking. The proposed system integrates real\-time grasp detection from a first\-person YOLO\-based perception system, a finite\-state machine \(FSM\) for task progression, and a variable admittance controller to adapt torque delivery to both posture and object state. A user study with 15 participants performing stooping load lifting trials under three conditions \(no exoskeleton, exoskeleton without vision, exoskeleton with vision\) shows that vision\-gated assistance significantly reduces perceived physical demand and improves fluency, trust, and comfort. Quantitative analysis reveals earlier and stronger assistance when vision is enabled, while questionnaire results confirm user preference for the vision\-gated mode. These findings highlight the potential of egocentric vision to enhance the responsiveness, ergonomics, safety, and acceptance of back\-support exoskeletons.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.04648v1)

---


## Real\-Time 2D LiDAR Object Detection Using Three\-Frame RGB Scan Encoding / 

发布日期：2026-02-02

作者：Soheil Behnam Roudsari

摘要：Indoor service robots need perception that is robust, more privacy\-friendly than RGB video, and feasible on embedded hardware. We present a camera\-free 2D LiDAR object detection pipeline that encodes short\-term temporal context by stacking three consecutive scans as RGB channels, yielding a compact YOLOv8n input without occupancy\-grid construction while preserving angular structure and motion cues. Evaluated in Webots across 160 randomized indoor scenarios with strict scenario\-level holdout, the method achieves 98.4% mAP@0.5 \(0.778 mAP@0.5:0.95\) with 94.9% precision and 94.7% recall on four object classes. On a Raspberry Pi 5, it runs in real time with a mean post\-warm\-up end\-to\-end latency of 47.8ms per frame, including scan encoding and postprocessing. Relative to a closely related occupancy\-grid LiDAR\-YOLO pipeline reported on the same platform, the proposed representation is associated with substantially lower reported end\-to\-end latency. Although results are simulation\-based, they suggest that lightweight temporal encoding can enable accurate and real\-time LiDAR\-only detection for embedded indoor robotics without capturing RGB appearance.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.02167v1)

---


## Cross\-Modal Alignment and Fusion for RGB\-D Transmission\-Line Defect Detection / 

发布日期：2026-02-02

作者：Jiaming Cui

摘要：Transmission line defect detection remains challenging for automated UAV inspection due to the dominance of small\-scale defects, complex backgrounds, and illumination variations. Existing RGB\-based detectors, despite recent progress, struggle to distinguish geometrically subtle defects from visually similar background structures under limited chromatic contrast. This paper proposes CMAFNet, a Cross\-Modal Alignment and Fusion Network that integrates RGB appearance and depth geometry through a principled purify\-then\-fuse paradigm. CMAFNet consists of a Semantic Recomposition Module that performs dictionary\-based feature purification via a learned codebook to suppress modality\-specific noise while preserving defect\-discriminative information, and a Contextual Semantic Integration Framework that captures global spatial dependencies using partial\-channel attention to enhance structural semantic reasoning. Position\-wise normalization within the purification stage enforces explicit reconstruction\-driven cross\-modal alignment, ensuring statistical compatibility between heterogeneous features prior to fusion. Extensive experiments on the TLRGBD benchmark, where 94.5% of instances are small objects, demonstrate that CMAFNet achieves 32.2% mAP@50 and 12.5% APs, outperforming the strongest baseline by 9.8 and 4.0 percentage points, respectively. A lightweight variant reaches 24.8% mAP50 at 228 FPS with only 4.9M parameters, surpassing all YOLO\-based detectors while matching transformer\-based methods at substantially lower computational cost.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.01696v2)

---


## Omni\-Judge: Can Omni\-LLMs Serve as Human\-Aligned Judges for Text\-Conditioned Audio\-Video Generation? / 

发布日期：2026-02-02

作者：Susan Liang

摘要：State\-of\-the\-art text\-to\-video generation models such as Sora 2 and Veo 3 can now produce high\-fidelity videos with synchronized audio directly from a textual prompt, marking a new milestone in multi\-modal generation. However, evaluating such tri\-modal outputs remains an unsolved challenge. Human evaluation is reliable but costly and difficult to scale, while traditional automatic metrics, such as FVD, CLAP, and ViCLIP, focus on isolated modality pairs, struggle with complex prompts, and provide limited interpretability. Omni\-modal large language models \(omni\-LLMs\) present a promising alternative: they naturally process audio, video, and text, support rich reasoning, and offer interpretable chain\-of\-thought feedback. Driven by this, we introduce Omni\-Judge, a study assessing whether omni\-LLMs can serve as human\-aligned judges for text\-conditioned audio\-video generation. Across nine perceptual and alignment metrics, Omni\-Judge achieves correlation comparable to traditional metrics and excels on semantically demanding tasks such as audio\-text alignment, video\-text alignment, and audio\-video\-text coherence. It underperforms on high\-FPS perceptual metrics, including video quality and audio\-video synchronization, due to limited temporal resolution. Omni\-Judge provides interpretable explanations that expose semantic or physical inconsistencies, enabling practical downstream uses such as feedback\-based refinement. Our findings highlight both the potential and current limitations of omni\-LLMs as unified evaluators for multi\-modal generation.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.01623v1)

---


## Deep Learning\-Based Object Detection for Autonomous Vehicles: A Comparative Study of One\-Stage and Two\-Stage Detectors on Basic Traffic Objects / 

发布日期：2026-01-30

作者：Bsher Karbouj

摘要：Object detection is a crucial component in autonomous vehicle systems. It enables the vehicle to perceive and understand its environment by identifying and locating various objects around it. By utilizing advanced imaging and deep learning techniques, autonomous vehicle systems can rapidly and accurately identify objects based on their features. Different deep learning methods vary in their ability to accurately detect and classify objects in autonomous vehicle systems. Selecting the appropriate method significantly impacts system performance, robustness, and efficiency in real\-world driving scenarios. While several generic deep learning architectures like YOLO, SSD, and Faster R\-CNN have been proposed, guidance on their suitability for specific autonomous driving applications is often limited. The choice of method affects detection accuracy, processing speed, environmental robustness, sensor integration, scalability, and edge case handling. This study provides a comprehensive experimental analysis comparing two prominent object detection models: YOLOv5 \(a one\-stage detector\) and Faster R\-CNN \(a two\-stage detector\). Their performance is evaluated on a diverse dataset combining real and synthetic images, considering various metrics including mean Average Precision \(mAP\), recall, and inference speed. The findings reveal that YOLOv5 demonstrates superior performance in terms of mAP, recall, and training efficiency, particularly as dataset size and image resolution increase. However, Faster R\-CNN shows advantages in detecting small, distant objects and performs well in challenging lighting conditions. The models' behavior is also analyzed under different confidence thresholds and in various real\-world scenarios, providing insights into their applicability for autonomous driving systems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.00385v1)

---

