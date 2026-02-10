# 每日从arXiv中获取最新YOLO相关论文


## Weak to Strong: VLM\-Based Pseudo\-Labeling as a Weakly Supervised Training Strategy in Multimodal Video\-based Hidden Emotion Understanding Tasks / 

发布日期：2026-02-08

作者：Yufei Wang

摘要：To tackle the automatic recognition of "concealed emotions" in videos, this paper proposes a multimodal weak\-supervision framework and achieves state\-of\-the\-art results on the iMiGUE tennis\-interview dataset. First, YOLO 11x detects and crops human portraits frame\-by\-frame, and DINOv2\-Base extracts visual features from the cropped regions. Next, by integrating Chain\-of\-Thought and Reflection prompting \(CoT \+ Reflection\), Gemini 2.5 Pro automatically generates pseudo\-labels and reasoning texts that serve as weak supervision for downstream models. Subsequently, OpenPose produces 137\-dimensional key\-point sequences, augmented with inter\-frame offset features; the usual graph neural network backbone is simplified to an MLP to efficiently model the spatiotemporal relationships of the three key\-point streams. An ultra\-long\-sequence Transformer independently encodes both the image and key\-point sequences, and their representations are concatenated with BERT\-encoded interview transcripts. Each modality is first pre\-trained in isolation, then fine\-tuned jointly, with pseudo\-labeled samples merged into the training set for further gains. Experiments demonstrate that, despite severe class imbalance, the proposed approach lifts accuracy from under 0.6 in prior work to over 0.69, establishing a new public benchmark. The study also validates that an "MLP\-ified" key\-point backbone can match \- or even surpass \- GCN\-based counterparts in this task.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.08057v1)

---


## CA\-YOLO: Cross Attention Empowered YOLO for Biomimetic Localization / 

发布日期：2026-02-07

作者：Zhen Zhang

摘要：In modern complex environments, achieving accurate and efficient target localization is essential in numerous fields. However, existing systems often face limitations in both accuracy and the ability to recognize small targets. In this study, we propose a bionic stabilized localization system based on CA\-YOLO, designed to enhance both target localization accuracy and small target recognition capabilities. Acting as the "brain" of the system, the target detection algorithm emulates the visual focusing mechanism of animals by integrating bionic modules into the YOLO backbone network. These modules include the introduction of a small target detection head and the development of a Characteristic Fusion Attention Mechanism \(CFAM\). Furthermore, drawing inspiration from the human Vestibulo\-Ocular Reflex \(VOR\), a bionic pan\-tilt tracking control strategy is developed, which incorporates central positioning, stability optimization, adaptive control coefficient adjustment, and an intelligent recapture function. The experimental results show that CA\-YOLO outperforms the original model on standard datasets \(COCO and VisDrone\), with average accuracy metrics improved by 3.94%and 4.90%, respectively.Further time\-sensitive target localization experiments validate the effectiveness and practicality of this bionic stabilized localization system.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.07523v1)

---


## PTB\-XL\-Image\-17K: A Large\-Scale Synthetic ECG Image Dataset with Comprehensive Ground Truth for Deep Learning\-Based Digitization / 

发布日期：2026-02-07

作者：Naqcho Ali Mehdi

摘要：Electrocardiogram \(ECG\) digitization\-converting paper\-based or scanned ECG images back into time\-series signals\-is critical for leveraging decades of legacy clinical data in modern deep learning applications. However, progress has been hindered by the lack of large\-scale datasets providing both ECG images and their corresponding ground truth signals with comprehensive annotations. We introduce PTB\-XL\-Image\-17K, a complete synthetic ECG image dataset comprising 17,271 high\-quality 12\-lead ECG images generated from the PTB\-XL signal database. Our dataset uniquely provides five complementary data types per sample: \(1\) realistic ECG images with authentic grid patterns and annotations \(50% with visible grid, 50% without\), \(2\) pixel\-level segmentation masks, \(3\) ground truth time\-series signals, \(4\) bounding box annotations in YOLO format for both lead regions and lead name labels, and \(5\) comprehensive metadata including visual parameters and patient information. We present an open\-source Python framework enabling customizable dataset generation with controllable parameters including paper speed \(25/50 mm/s\), voltage scale \(5/10 mm/mV\), sampling rate \(500 Hz\), grid appearance \(4 colors\), and waveform characteristics. The dataset achieves 100% generation success rate with an average processing time of 1.35 seconds per sample. PTB\-XL\-Image\-17K addresses critical gaps in ECG digitization research by providing the first large\-scale resource supporting the complete pipeline: lead detection, waveform segmentation, and signal extraction with full ground truth for rigorous evaluation. The dataset, generation framework, and documentation are publicly available at https://github.com/naqchoalimehdi/PTB\-XL\-Image\-17K and https://doi.org/10.5281/zenodo.18197519.

中文摘要：


代码链接：https://github.com/naqchoalimehdi/PTB-XL-Image-17K，https://doi.org/10.5281/zenodo.18197519.

论文链接：[阅读更多](http://arxiv.org/abs/2602.07446v1)

---


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

