# 每日从arXiv中获取最新YOLO相关论文


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

摘要：Traffic safety remains a critical global concern, with timely and accurate accident detection essential for hazard reduction and rapid emergency response. Infrastructure\-based vision sensors offer scalable and efficient solutions for continuous real\-time monitoring, facilitating automated detection of acci\- dents directly from captured images. This research investigates the zero\-shot capabilities of multimodal large language models \(MLLMs\) for detecting and describing traffic accidents using images from infrastructure cameras, thus minimizing reliance on extensive labeled datasets. Main contributions include: \(1\) Evaluation of MLLMs using the simulated DeepAccident dataset from CARLA, explicitly addressing the scarcity of diverse, realistic, infrastructure\-based accident data through controlled simulations; \(2\) Comparative performance analysis between Gemini 1.5 and 2.0, Gemma 3 and Pixtral models in acci\- dent identification and descriptive capabilities without prior fine\-tuning; and \(3\) Integration of advanced visual analytics, specifically YOLO for object detection, Deep SORT for multi\- object tracking, and Segment Anything \(SAM\) for instance segmentation, into enhanced prompts to improve model accuracy and explainability. Key numerical results show Pixtral as the top performer with an F1\-score of 0.71 and 83% recall, while Gemini models gained precision with enhanced prompts \(e.g., Gemini 1.5 rose to 90%\) but suffered notable F1 and recall losses. Gemma 3 offered the most balanced performance with minimal metric fluctuation. These findings demonstrate the substantial potential of integrating MLLMs with advanced visual analytics techniques, enhancing their applicability in real\-world automated traffic monitoring systems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.19096v1)

---


## Generative data augmentation for biliary tract detection on intraoperative images / 

发布日期：2025-09-23

作者：Cristina Iacono

摘要：Cholecystectomy is one of the most frequently performed procedures in gastrointestinal surgery, and the laparoscopic approach is the gold standard for symptomatic cholecystolithiasis and acute cholecystitis. In addition to the advantages of a significantly faster recovery and better cosmetic results, the laparoscopic approach bears a higher risk of bile duct injury, which has a significant impact on quality of life and survival. To avoid bile duct injury, it is essential to improve the intraoperative visualization of the bile duct. This work aims to address this problem by leveraging a deep\-learning approach for the localization of the biliary tract from white\-light images acquired during the surgical procedures. To this end, the construction and annotation of an image database to train the Yolo detection algorithm has been employed. Besides classical data augmentation techniques, the paper proposes Generative Adversarial Network \(GAN\) for the generation of a synthetic portion of the training dataset. Experimental results have been discussed along with ethical considerations.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.18958v1)

---


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

