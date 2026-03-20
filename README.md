# 每日从arXiv中获取最新YOLO相关论文


## EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task\-Specialized Distillation / 

发布日期：2026-03-19

作者：Longfei Liu

摘要：Deploying high\-performance dense prediction models on resource\-constrained edge devices remains challenging due to strict limits on computation and memory. In practice, lightweight systems for object detection, instance segmentation, and pose estimation are still dominated by CNN\-based architectures such as YOLO, while compact Vision Transformers \(ViTs\) often struggle to achieve similarly strong accuracy efficiency tradeoff, even with large scale pretraining. We argue that this gap is largely due to insufficient task specific representation learning in small scale ViTs, rather than an inherent mismatch between ViTs and edge dense prediction. To address this issue, we introduce EdgeCrafter, a unified compact ViT framework for edge dense prediction centered on ECDet, a detection model built from a distilled compact backbone and an edge\-friendly encoder decoder design. On the COCO dataset, ECDet\-S achieves 51.7 AP with fewer than 10M parameters using only COCO annotations. For instance segmentation, ECInsSeg achieves performance comparable to RF\-DETR while using substantially fewer parameters. For pose estimation, ECPose\-X reaches 74.8 AP, significantly outperforming YOLO26Pose\-X \(71.6 AP\) despite the latter's reliance on extensive Objects365 pretraining. These results show that compact ViTs, when paired with task\-specialized distillation and edge\-aware design, can be a practical and competitive option for edge dense prediction. Code is available at: https://intellindust\-ai\-lab.github.io/projects/EdgeCrafter/

中文摘要：


代码链接：https://intellindust-ai-lab.github.io/projects/EdgeCrafter/

论文链接：[阅读更多](http://arxiv.org/abs/2603.18739v1)

---


## HOMEY: Heuristic Object Masking with Enhanced YOLO for Property Insurance Risk Detection / 

发布日期：2026-03-19

作者：Teerapong Panboonyuen

摘要：Automated property risk detection is a high\-impact yet underexplored frontier in computer vision with direct implications for real estate, underwriting, and insurance operations. We introduce HOMEY \(Heuristic Object Masking with Enhanced YOLO\), a novel detection framework that combines YOLO with a domain\-specific masking mechanism and a custom\-designed loss function. HOMEY is trained to detect 17 risk\-related property classes, including structural damages \(e.g., cracked foundations, roof issues\), maintenance neglect \(e.g., dead yards, overgrown bushes\), and liability hazards \(e.g., falling gutters, garbage, hazard signs\). Our approach introduces heuristic object masking to amplify weak signals in cluttered backgrounds and risk\-aware loss calibration to balance class skew and severity weighting. Experiments on real\-world property imagery demonstrate that HOMEY achieves superior detection accuracy and reliability compared to baseline YOLO models, while retaining fast inference. Beyond detection, HOMEY enables interpretable and cost\-efficient risk analysis, laying the foundation for scalable AI\-driven property insurance workflows.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.18502v1)

---


## Does YOLO Really Need to See Every Training Image in Every Epoch? / 

发布日期：2026-03-18

作者：Xingxing Xie

摘要：YOLO detectors are known for their fast inference speed, yet training them remains unexpectedly time\-consuming due to their exhaustive pipeline that processes every training image in every epoch, even when many images have already been sufficiently learned. This stands in clear contrast to the efficiency suggested by the \`\`You Only Look Once'' philosophy. This naturally raises an important question: textit\{Does YOLO really need to see every training image in every epoch?\} To explore this, we propose an Anti\-Forgetting Sampling Strategy \(AFSS\) that dynamically determines which images should be used and which can be skipped during each epoch, allowing the detector to learn more effectively and efficiently. Specifically, AFSS measures the learning sufficiency of each training image as the minimum of its detection recall and precision, and dynamically categorizes training images into easy, medium, or hard levels accordingly. Easy training images are sparsely resampled during training in a continuous review manner, with priority given to those that have not been used for a long time to reduce redundancy and prevent forgetting. Moderate training images are partially selected, prioritizing recently unused ones and randomly choosing the rest from unselected images to ensure coverage and prevent forgetting. Hard training images are fully sampled in every epoch to ensure sufficient learning. The learning sufficiency of each training image is periodically updated, enabling detectors to adaptively shift its focus toward the informative training images over time while progressively discarding redundant ones. On widely used natural image detection benchmarks \(MS COCO 2017 and PASCAL VOC 2007\) and remote sensing detection datasets \(DOTA\-v1.0 and DIOR\-R\), AFSS achieves more than $1.43times$ training speedup for YOLO\-series detectors while also improving accuracy.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.17684v1)

---


## Automated identification of Ichneumonoidea wasps via YOLO\-based deep learning: Integrating HiresCam for Explainable AI / 

发布日期：2026-03-17

作者：Joao Manoel Herrera Pinheiro

摘要：Accurate taxonomic identification of parasitoid wasps within the superfamily Ichneumonoidea is essential for biodiversity assessment, ecological monitoring, and biological control programs. However, morphological similarity, small body size, and fine\-grained interspecific variation make manual identification labor\-intensive and expertise\-dependent. This study proposes a deep learning\-based framework for the automated identification of Ichneumonoidea wasps using a YOLO\-based architecture integrated with High\-Resolution Class Activation Mapping \(HiResCAM\) to enhance interpretability. The proposed system simultaneously identifies wasp families from high\-resolution images. The dataset comprises 3556 high\-resolution images of Hymenoptera specimens. The taxonomic distribution is primarily concentrated among the families Ichneumonidae \(n = 786\), Braconidae \(n = 648\), Apidae \(n = 466\), and Vespidae \(n = 460\). Extensive experiments were conducted using a curated dataset, with model performance evaluated through precision, recall, F1 score, and accuracy. The results demonstrate high accuracy of over 96 % and robust generalization across morphological variations. HiResCAM visualizations confirm that the model focuses on taxonomically relevant anatomical regions, such as wing venation, antennae segmentation, and metasomal structures, thereby validating the biological plausibility of the learned features. The integration of explainable AI techniques improves transparency and trustworthiness, making the system suitable for entomological research to accelerate biodiversity characterization in an under\-described parasitoid superfamily.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.16351v1)

---


## Automatic Characterization of Mid\-latitude Multiple Ionospheric Plasma Structures from All\-sky Airglow Images using Deep Learning Technique / 

发布日期：2026-03-16

作者：Jeevan Upadhyaya

摘要：The F\-region ionospheric plasma structures are propagating high and or low electron density regions in the Earth ionosphere. These plasma structures can be observed using ground based all\-sky airglow imagers which can capture faint airglow emissions originating from the F\-region of ionosphere. This study introduces a novel automatic method for determining the propagation parameters \(horizontal velocity and orientation\) of these multiple ionospheric plasma structures observed in O\(1D\) 630.0 nm all\-sky airglow images from Hanle, India located in the mid\-latitude region. We have used a deep learning\-based segmentation model called YOLOv8 \(You Only Look Once\) to localize and BoT\-SORT tracker to track individual mid\-latitude ionospheric plasma structures. Three different automatic algorithms are used to characterize the observed plasma structures utilizing the segmented outputs from the YOLO model. Finally, an additional quality control step is introduced that filters the results from the three automatic algorithms and generates a flag to retain the most reliable estimate. The results of the proposed fully automated pipeline are systematically compared with a previously developed semi\-automatic approach to assess the estimation efficacy. The automatic technique developed in this study is particularly valuable for all\-sky airglow imaging systems having large datasets, where manual intervention or semi\-automatic analysis is impractical.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.15333v1)

---

