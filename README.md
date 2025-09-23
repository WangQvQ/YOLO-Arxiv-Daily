# 每日从arXiv中获取最新YOLO相关论文


## An Empirical Study on the Robustness of YOLO Models for Underwater Object Detection / 

发布日期：2025-09-22

作者：Edwine Nabahirwa

摘要：Underwater object detection \(UOD\) remains a critical challenge in computer vision due to underwater distortions which degrade low\-level features and compromise the reliability of even state\-of\-the\-art detectors. While YOLO models have become the backbone of real\-time object detection, little work has systematically examined their robustness under these uniquely challenging conditions. This raises a critical question: Are YOLO models genuinely robust when operating under the chaotic and unpredictable conditions of underwater environments? In this study, we present one of the first comprehensive evaluations of recent YOLO variants \(YOLOv8\-YOLOv12\) across six simulated underwater environments. Using a unified dataset of 10,000 annotated images from DUO and Roboflow100, we not only benchmark model robustness but also analyze how distortions affect key low\-level features such as texture, edges, and color. Our findings show that \(1\) YOLOv12 delivers the strongest overall performance but is highly vulnerable to noise, and \(2\) noise disrupts edge and texture features, explaining the poor detection performance in noisy images. Class imbalance is a persistent challenge in UOD. Experiments revealed that \(3\) image counts and instance frequency primarily drive detection performance, while object appearance exerts only a secondary influence. Finally, we evaluated lightweight training\-aware strategies: noise\-aware sample injection, which improves robustness in both noisy and real\-world conditions, and fine\-tuning with advanced enhancement, which boosts accuracy in enhanced domains but slightly lowers performance in original data, demonstrating strong potential for domain adaptation, respectively. Together, these insights provide practical guidance for building resilient and cost\-efficient UOD systems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.17561v1)

---


## Vision\-Based Driver Drowsiness Monitoring: Comparative Analysis of YOLOv5\-v11 Models / 

发布日期：2025-09-22

作者：Dilshara Herath

摘要：Driver drowsiness remains a critical factor in road accidents, accounting for thousands of fatalities and injuries each year. This paper presents a comprehensive evaluation of real\-time, non\-intrusive drowsiness detection methods, focusing on computer vision based YOLO \(You Look Only Once\) algorithms. A publicly available dataset namely, UTA\-RLDD was used, containing both awake and drowsy conditions, ensuring variability in gender, eyewear, illumination, and skin tone. Seven YOLO variants \(v5s, v9c, v9t, v10n, v10l, v11n, v11l\) are fine\-tuned, with performance measured in terms of Precision, Recall, mAP0.5, and mAP 0.5\-0.95. Among these, YOLOv9c achieved the highest accuracy \(0.986 mAP 0.5, 0.978 Recall\) while YOLOv11n strikes the optimal balance between precision \(0.954\) and inference efficiency, making it highly suitable for embedded deployment. Additionally, we implement an Eye Aspect Ratio \(EAR\) approach using Dlib's facial landmarks, which despite its low computational footprint exhibits reduced robustness under pose variation and occlusions. Our findings illustrate clear trade offs between accuracy, latency, and resource requirements, and offer practical guidelines for selecting or combining detection methods in autonomous driving and industrial safety applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.17498v1)

---


## SFN\-YOLO: Towards Free\-Range Poultry Detection via Scale\-aware Fusion Networks / 

发布日期：2025-09-21

作者：Jie Chen

摘要：Detecting and localizing poultry is essential for advancing smart poultry farming. Despite the progress of detection\-centric methods, challenges persist in free\-range settings due to multiscale targets, obstructions, and complex or dynamic backgrounds. To tackle these challenges, we introduce an innovative poultry detection approach named SFN\-YOLO that utilizes scale\-aware fusion. This approach combines detailed local features with broader global context to improve detection in intricate environments. Furthermore, we have developed a new expansive dataset \(M\-SCOPE\) tailored for varied free\-range conditions. Comprehensive experiments demonstrate our model achieves an mAP of 80.7% with just 7.2M parameters, which is 35.1% fewer than the benchmark, while retaining strong generalization capability across different domains. The efficient and real\-time detection capabilities of SFN\-YOLO support automated smart poultry farming. The code and dataset can be accessed at https://github.com/chenjessiee/SFN\-YOLO.

中文摘要：


代码链接：https://github.com/chenjessiee/SFN-YOLO.

论文链接：[阅读更多](http://arxiv.org/abs/2509.17086v1)

---


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

