# 每日从arXiv中获取最新YOLO相关论文


## A Comprehensive Evaluation of YOLO\-based Deer Detection Performance on Edge Devices / 

发布日期：2025-09-24

作者：Bishal Adhikari

摘要：The escalating economic losses in agriculture due to deer intrusion, estimated to be in the hundreds of millions of dollars annually in the U.S., highlight the inadequacy of traditional mitigation strategies since these methods are often labor\-intensive, costly, and ineffective for modern farming systems. To overcome this, there is a critical need for intelligent, autonomous solutions which require accurate and efficient deer detection. But the progress in this field is impeded by a significant gap in the literature, mainly the lack of a domain\-specific, practical dataset and limited study on the on\-field deployability of deer detection systems. Addressing this gap, this study presents a comprehensive evaluation of state\-of\-the\-art deep learning models for deer detection in challenging real\-world scenarios. The contributions of this work are threefold. First, we introduce a curated, publicly available dataset of 3,095 annotated images with bounding\-box annotations of deer, derived from the Idaho Cameratraps project. Second, we provide an extensive comparative analysis of 12 model variants across four recent YOLO architectures\(v8, v9, v10, and v11\). Finally, we benchmarked performance on a high\-end NVIDIA RTX 5090 GPU and evaluated on two representative edge computing platforms: Raspberry Pi 5 and NVIDIA Jetson AGX Xavier. Results show that the real\-time detection is not feasible in Raspberry Pi without hardware\-specific model optimization, while NVIDIA Jetson provides greater than 30 FPS with GPU\-accelerated inference on 's' and 'n' series models. This study also reveals that smaller, architecturally advanced models such as YOLOv11n, YOLOv8s, and YOLOv9s offer the optimal balance of high accuracy \(AP@.5 > 0.85\) and computational efficiency \(FPS > 30\). To support further research, both the source code and datasets are publicly available at https://github.com/WinnerBishal/track\-the\-deer.

中文摘要：


代码链接：https://github.com/WinnerBishal/track-the-deer.

论文链接：[阅读更多](http://arxiv.org/abs/2509.20318v1)

---


## SDE\-DET: A Precision Network for Shatian Pomelo Detection in Complex Orchard Environments / 

发布日期：2025-09-24

作者：Yihao Hu

摘要：Pomelo detection is an essential process for their localization, automated robotic harvesting, and maturity analysis. However, detecting Shatian pomelo in complex orchard environments poses significant challenges, including multi\-scale issues, obstructions from trunks and leaves, small object detection, etc. To address these issues, this study constructs a custom dataset STP\-AgriData and proposes the SDE\-DET model for Shatian pomelo detection. SDE\-DET first utilizes the Star Block to effectively acquire high\-dimensional information without increasing the computational overhead. Furthermore, the presented model adopts Deformable Attention in its backbone, to enhance its ability to detect pomelos under occluded conditions. Finally, multiple Efficient Multi\-Scale Attention mechanisms are integrated into our model to reduce the computational overhead and extract deep visual representations, thereby improving the capacity for small object detection. In the experiment, we compared SDE\-DET with the Yolo series and other mainstream detection models in Shatian pomelo detection. The presented SDE\-DET model achieved scores of 0.883, 0.771, 0.838, 0.497, and 0.823 in Precision, Recall, mAP@0.5, mAP@0.5:0.95 and F1\-score, respectively. SDE\-DET has achieved state\-of\-the\-art performance on the STP\-AgriData dataset. Experiments indicate that the SDE\-DET provides a reliable method for Shatian pomelo detection, laying the foundation for the further development of automatic harvest robots.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.19990v1)

---


## High Clockrate Free\-space Optical In\-Memory Computing / 

发布日期：2025-09-23

作者：Yuanhao Liang

摘要：The ability to process and act on data in real time is increasingly critical for applications ranging from autonomous vehicles, three\-dimensional environmental sensing and remote robotics. However, the deployment of deep neural networks \(DNNs\) in edge devices is hindered by the lack of energy\-efficient scalable computing hardware. Here, we introduce a fanout spatial time\-of\-flight optical neural network \(FAST\-ONN\) that calculates billions of convolutions per second with ultralow latency and power consumption. This is enabled by the combination of high\-speed dense arrays of vertical\-cavity surface\-emitting lasers \(VCSELs\) for input modulation with spatial light modulators of high pixel counts for in\-memory weighting. In a three\-dimensional optical system, parallel differential readout allows signed weight values accurate inference in a single shot. The performance is benchmarked with feature extraction in You\-Only\-Look\-Once \(YOLO\) for convolution at 100 million frames per second \(MFPS\), and in\-system backward propagation training with photonic reprogrammability. The VCSEL transmitters are implementable in any free\-space optical computing systems to improve the clockrate to over gigahertz. The high scalability in device counts and channel parallelism enables a new avenue to scale up free space computing hardware.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.19642v1)

---


## YOLO\-LAN: Precise Polyp Detection via Optimized Loss, Augmentations and Negatives / 

发布日期：2025-09-23

作者：Siddharth Gupta

摘要：Colorectal cancer \(CRC\), a lethal disease, begins with the growth of abnormal mucosal cell proliferation called polyps in the inner wall of the colon. When left undetected, polyps can become malignant tumors. Colonoscopy is the standard procedure for detecting polyps, as it enables direct visualization and removal of suspicious lesions. Manual detection by colonoscopy can be inconsistent and is subject to oversight. Therefore, object detection based on deep learning offers a better solution for a more accurate and real\-time diagnosis during colonoscopy. In this work, we propose YOLO\-LAN, a YOLO\-based polyp detection pipeline, trained using M2IoU loss, versatile data augmentations and negative data to replicate real clinical situations. Our pipeline outperformed existing methods for the Kvasir\-seg and BKAI\-IGH NeoPolyp datasets, achieving mAP$\_\{50\}$ of 0.9619, mAP$\_\{50:95\}$ of 0.8599 with YOLOv12 and mAP$\_\{50\}$ of 0.9540, mAP$\_\{50:95\}$ of 0.8487 with YOLOv8 on the Kvasir\-seg dataset. The significant increase is achieved in mAP$\_\{50:95\}$ score, showing the precision of polyp detection. We show robustness based on polyp size and precise location detection, making it clinically relevant in AI\-assisted colorectal screening.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.19166v1)

---


## Investigating Traffic Accident Detection Using Multimodal Large Language Models / 

发布日期：2025-09-23

作者：Ilhan Skender

摘要：Traffic safety remains a critical global concern, with timely and accurate accident detection essential for hazard reduction and rapid emergency response. Infrastructure\-based vision sensors offer scalable and efficient solutions for continuous real\-time monitoring, facilitating automated detection of accidents directly from captured images. This research investigates the zero\-shot capabilities of multimodal large language models \(MLLMs\) for detecting and describing traffic accidents using images from infrastructure cameras, thus minimizing reliance on extensive labeled datasets. Main contributions include: \(1\) Evaluation of MLLMs using the simulated DeepAccident dataset from CARLA, explicitly addressing the scarcity of diverse, realistic, infrastructure\-based accident data through controlled simulations; \(2\) Comparative performance analysis between Gemini 1.5 and 2.0, Gemma 3 and Pixtral models in accident identification and descriptive capabilities without prior fine\-tuning; and \(3\) Integration of advanced visual analytics, specifically YOLO for object detection, Deep SORT for multi\-object tracking, and Segment Anything \(SAM\) for instance segmentation, into enhanced prompts to improve model accuracy and explainability. Key numerical results show Pixtral as the top performer with an F1\-score of 71% and 83% recall, while Gemini models gained precision with enhanced prompts \(e.g., Gemini 1.5 rose to 90%\) but suffered notable F1 and recall losses. Gemma 3 offered the most balanced performance with minimal metric fluctuation. These findings demonstrate the substantial potential of integrating MLLMs with advanced visual analytics techniques, enhancing their applicability in real\-world automated traffic monitoring systems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.19096v2)

---

