# 每日从arXiv中获取最新YOLO相关论文


## Performance Optimization of YOLO\-FEDER FusionNet for Robust Drone Detection in Visually Complex Environments / 

发布日期：2025-09-17

作者：Tamara R. Lenhard

摘要：Drone detection in visually complex environments remains challenging due to background clutter, small object scale, and camouflage effects. While generic object detectors like YOLO exhibit strong performance in low\-texture scenes, their effectiveness degrades in cluttered environments with low object\-background separability. To address these limitations, this work presents an enhanced iteration of YOLO\-FEDER FusionNet \-\- a detection framework that integrates generic object detection with camouflage object detection techniques. Building upon the original architecture, the proposed iteration introduces systematic advancements in training data composition, feature fusion strategies, and backbone design. Specifically, the training process leverages large\-scale, photo\-realistic synthetic data, complemented by a small set of real\-world samples, to enhance robustness under visually complex conditions. The contribution of intermediate multi\-scale FEDER features is systematically evaluated, and detection performance is comprehensively benchmarked across multiple YOLO\-based backbone configurations. Empirical results indicate that integrating intermediate FEDER features, in combination with backbone upgrades, contributes to notable performance improvements. In the most promising configuration \-\- YOLO\-FEDER FusionNet with a YOLOv8l backbone and FEDER features derived from the DWD module \-\- these enhancements lead to a FNR reduction of up to 39.1 percentage points and a mAP increase of up to 62.8 percentage points at an IoU threshold of 0.5, compared to the initial baseline.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.14012v1)

---


## MOCHA: Multi\-modal Objects\-aware Cross\-arcHitecture Alignment / 

发布日期：2025-09-17

作者：Elena Camuffo

摘要：We introduce MOCHA \(Multi\-modal Objects\-aware Cross\-arcHitecture Alignment\), a knowledge distillation approach that transfers region\-level multimodal semantics from a large vision\-language teacher \(e.g., LLaVa\) into a lightweight vision\-only object detector student \(e.g., YOLO\). A translation module maps student features into a joint space, where the training of the student and translator is guided by a dual\-objective loss that enforces both local alignment and global relational consistency. Unlike prior approaches focused on dense or global alignment, MOCHA operates at the object level, enabling efficient transfer of semantics without modifying the teacher or requiring textual input at inference. We validate our method across four personalized detection benchmarks under few\-shot regimes. Results show consistent gains over baselines, with a \+10.1 average score improvement. Despite its compact architecture, MOCHA reaches performance on par with larger multimodal models, proving its suitability for real\-world deployment.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.14001v1)

---


## Federated Learning for Deforestation Detection: A Distributed Approach with Satellite Imagery / 

发布日期：2025-09-17

作者：Yuvraj Dutta

摘要：Accurate identification of deforestation from satellite images is essential in order to understand the geographical situation of an area. This paper introduces a new distributed approach to identify as well as locate deforestation across different clients using Federated Learning \(FL\). Federated Learning enables distributed network clients to collaboratively train a model while maintaining data privacy and security of the active users. In our framework, a client corresponds to an edge satellite center responsible for local data processing. Moreover, FL provides an advantage over centralized training method which requires combining data, thereby compromising with data security of the clients. Our framework leverages the FLOWER framework with RAY framework to execute the distributed learning workload. Furthermore, efficient client spawning is ensured by RAY as it can select definite amount of users to create an emulation environment. Our FL framework uses YOLOS\-small \(a Vision Transformer variant\), Faster R\-CNN with a ResNet50 backbone, and Faster R\-CNN with a MobileNetV3 backbone models trained and tested on publicly available datasets. Our approach provides us a different view for image segmentation\-based tasks on satellite imagery.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.13631v1)

---


## Layout\-Aware OCR for Black Digital Archives with Unsupervised Evaluation / 

发布日期：2025-09-16

作者：Fitsum Sileshi Beyene

摘要：Despite their cultural and historical significance, Black digital archives continue to be a structurally underrepresented area in AI research and infrastructure. This is especially evident in efforts to digitize historical Black newspapers, where inconsistent typography, visual degradation, and limited annotated layout data hinder accurate transcription, despite the availability of various systems that claim to handle optical character recognition \(OCR\) well. In this short paper, we present a layout\-aware OCR pipeline tailored for Black newspaper archives and introduce an unsupervised evaluation framework suited to low\-resource archival contexts. Our approach integrates synthetic layout generation, model pretraining on augmented data, and a fusion of state\-of\-the\-art You Only Look Once \(YOLO\) detectors. We used three annotation\-free evaluation metrics, the Semantic Coherence Score \(SCS\), Region Entropy \(RE\), and Textual Redundancy Score \(TRS\), which quantify linguistic fluency, informational diversity, and redundancy across OCR regions. Our evaluation on a 400\-page dataset from ten Black newspaper titles demonstrates that layout\-aware OCR improves structural diversity and reduces redundancy compared to full\-page baselines, with modest trade\-offs in coherence. Our results highlight the importance of respecting cultural layout logic in AI\-driven document understanding and lay the foundation for future community\-driven and ethically grounded archival AI systems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.13236v1)

---


## A Comparative Study of YOLOv8 to YOLOv11 Performance in Underwater Vision Tasks / 

发布日期：2025-09-16

作者：Gordon Hung

摘要：Autonomous underwater vehicles \(AUVs\) increasingly rely on on\-board computer\-vision systems for tasks such as habitat mapping, ecological monitoring, and infrastructure inspection. However, underwater imagery is hindered by light attenuation, turbidity, and severe class imbalance, while the computational resources available on AUVs are limited. One\-stage detectors from the YOLO family are attractive because they fuse localization and classification in a single, low\-latency network; however, their terrestrial benchmarks \(COCO, PASCAL\-VOC, Open Images\) leave open the question of how successive YOLO releases perform in the marine domain. We curate two openly available datasets that span contrasting operating conditions: a Coral Disease set \(4,480 images, 18 classes\) and a Fish Species set \(7,500 images, 20 classes\). For each dataset, we create four training regimes \(25 %, 50 %, 75 %, 100 % of the images\) while keeping balanced validation and test partitions fixed. We train YOLOv8\-s, YOLOv9\-s, YOLOv10\-s, and YOLOv11\-s with identical hyperparameters \(100 epochs, 640 px input, batch = 16, T4 GPU\) and evaluate precision, recall, mAP50, mAP50\-95, per\-image inference time, and frames\-per\-second \(FPS\). Post\-hoc Grad\-CAM visualizations probe feature utilization and localization faithfulness. Across both datasets, accuracy saturates after YOLOv9, suggesting architectural innovations primarily target efficiency rather than accuracy. Inference speed, however, improves markedly. Our results \(i\) provide the first controlled comparison of recent YOLO variants on underwater imagery, \(ii\) show that lightweight YOLOv10 offers the best speed\-accuracy trade\-off for embedded AUV deployment, and \(iii\) deliver an open, reproducible benchmark and codebase to accelerate future marine\-vision research.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.12682v1)

---

