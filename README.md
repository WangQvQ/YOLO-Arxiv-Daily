# 每日从arXiv中获取最新YOLO相关论文


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


## YOLO\-CIANNA: Galaxy detection with deep learning in radio data: II. Winning the SKA SDC2 using a generalized 3D\-YOLO network / 

发布日期：2025-09-15

作者：D. Cornu

摘要：As the scientific exploitation of the Square Kilometre Array \(SKA\) approaches, there is a need for new advanced data analysis and visualization tools capable of processing large high\-dimensional datasets. In this study, we aim to generalize the YOLO\-CIANNA deep learning source detection and characterization method for 3D hyperspectral HI emission cubes. We present the adaptations we made to the regression\-based detection formalism and the construction of an end\-to\-end 3D convolutional neural network \(CNN\) backbone. We then describe a processing pipeline for applying the method to simulated 3D HI cubes from the SKA Observatory Science Data Challenge 2 \(SDC2\) dataset. The YOLO\-CIANNA method was originally developed and used by the MINERVA team that won the official SDC2 competition. Despite the public release of the full SDC2 dataset, no published result has yet surpassed MINERVA's top score. In this paper, we present an updated version of our method that improves our challenge score by 9.5%. The resulting catalog exhibits a high detection purity of 92.3%, best\-in\-class characterization accuracy, and contains 45% more confirmed sources than concurrent classical detection tools. The method is also computationally efficient, processing the full ~1TB SDC2 data cube in 30 min on a single GPU. These state\-of\-the\-art results highlight the effectiveness of 3D CNN\-based detectors for processing large hyperspectral data cubes and represent a promising step toward applying YOLO\-CIANNA to observational data from SKA and its precursors.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.12082v1)

---


## Drone\-Based Multispectral Imaging and Deep Learning for Timely Detection of Branched Broomrape in Tomato Farms / 

发布日期：2025-09-12

作者：Mohammadreza Narimani

摘要：This study addresses the escalating threat of branched broomrape \(Phelipanche ramosa\) to California's tomato industry, which supplies over 90 percent of U.S. processing tomatoes. The parasite's largely underground life cycle makes early detection difficult, while conventional chemical controls are costly, environmentally harmful, and often ineffective. To address this, we combined drone\-based multispectral imagery with Long Short\-Term Memory \(LSTM\) deep learning networks, using the Synthetic Minority Over\-sampling Technique \(SMOTE\) to handle class imbalance. Research was conducted on a known broomrape\-infested tomato farm in Woodland, Yolo County, CA, across five key growth stages determined by growing degree days \(GDD\). Multispectral images were processed to isolate tomato canopy reflectance. At 897 GDD, broomrape could be detected with 79.09 percent overall accuracy and 70.36 percent recall without integrating later stages. Incorporating sequential growth stages with LSTM improved detection substantially. The best\-performing scenario, which integrated all growth stages with SMOTE augmentation, achieved 88.37 percent overall accuracy and 95.37 percent recall. These results demonstrate the strong potential of temporal multispectral analysis and LSTM networks for early broomrape detection. While further real\-world data collection is needed for practical deployment, this study shows that UAV\-based multispectral sensing coupled with deep learning could provide a powerful precision agriculture tool to reduce losses and improve sustainability in tomato production.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.09972v1)

---


## Zero\-Shot Referring Expression Comprehension via Visual\-Language True/False Verification / 

发布日期：2025-09-12

作者：Jeffrey Liu

摘要：Referring Expression Comprehension \(REC\) is usually addressed with task\-trained grounding models. We show that a zero\-shot workflow, without any REC\-specific training, can achieve competitive or superior performance. Our approach reformulates REC as box\-wise visual\-language verification: given proposals from a COCO\-clean generic detector \(YOLO\-World\), a general\-purpose VLM independently answers True/False queries for each region. This simple procedure reduces cross\-box interference, supports abstention and multiple matches, and requires no fine\-tuning. On RefCOCO, RefCOCO\+, and RefCOCOg, our method not only surpasses a zero\-shot GroundingDINO baseline but also exceeds reported results for GroundingDINO trained on REC and GroundingDINO\+CRG. Controlled studies with identical proposals confirm that verification significantly outperforms selection\-based prompting, and results hold with open VLMs. Overall, we show that workflow design, rather than task\-specific pretraining, drives strong zero\-shot REC performance.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.09958v1)

---

