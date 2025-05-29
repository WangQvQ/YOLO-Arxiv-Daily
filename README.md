# 每日从arXiv中获取最新YOLO相关论文


## YOLO\-SPCI: Enhancing Remote Sensing Object Detection via Selective\-Perspective\-Class Integration / 

发布日期：2025-05-27

作者：Xinyuan Wang

摘要：Object detection in remote sensing imagery remains a challenging task due to extreme scale variation, dense object distributions, and cluttered backgrounds. While recent detectors such as YOLOv8 have shown promising results, their backbone architectures lack explicit mechanisms to guide multi\-scale feature refinement, limiting performance on high\-resolution aerial data. In this work, we propose YOLO\-SPCI, an attention\-enhanced detection framework that introduces a lightweight Selective\-Perspective\-Class Integration \(SPCI\) module to improve feature representation. The SPCI module integrates three components: a Selective Stream Gate \(SSG\) for adaptive regulation of global feature flow, a Perspective Fusion Module \(PFM\) for context\-aware multi\-scale integration, and a Class Discrimination Module \(CDM\) to enhance inter\-class separability. We embed two SPCI blocks into the P3 and P5 stages of the YOLOv8 backbone, enabling effective refinement while preserving compatibility with the original neck and head. Experiments on the NWPU VHR\-10 dataset demonstrate that YOLO\-SPCI achieves superior performance compared to state\-of\-the\-art detectors.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.21370v1)

---


## YOLO\-FireAD: Efficient Fire Detection via Attention\-Guided Inverted Residual Learning and Dual\-Pooling Feature Preservation / 

发布日期：2025-05-27

作者：Weichao Pan

摘要：Fire detection in dynamic environments faces continuous challenges, including the interference of illumination changes, many false detections or missed detections, and it is difficult to achieve both efficiency and accuracy. To address the problem of feature extraction limitation and information loss in the existing YOLO\-based models, this study propose You Only Look Once for Fire Detection with Attention\-guided Inverted Residual and Dual\-pooling Downscale Fusion \(YOLO\-FireAD\) with two core innovations: \(1\) Attention\-guided Inverted Residual Block \(AIR\) integrates hybrid channel\-spatial attention with inverted residuals to adaptively enhance fire features and suppress environmental noise; \(2\) Dual Pool Downscale Fusion Block \(DPDF\) preserves multi\-scale fire patterns through learnable fusion of max\-average pooling outputs, mitigating small\-fire detection failures. Extensive evaluation on two public datasets shows the efficient performance of our model. Our proposed model keeps the sum amount of parameters \(1.45M, 51.8% lower than YOLOv8n\) \(4.6G, 43.2% lower than YOLOv8n\), and mAP75 is higher than the mainstream real\-time object detection models YOLOv8n, YOL\-Ov9t, YOLOv10n, YOLO11n, YOLOv12n and other YOLOv8 variants 1.3\-5.5%.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.20884v1)

---


## Knowledge Distillation Approach for SOS Fusion Staging: Towards Fully Automated Skeletal Maturity Assessment / 

发布日期：2025-05-27

作者：Omid Halimi Milani

摘要：We introduce a novel deep learning framework for the automated staging of spheno\-occipital synchondrosis \(SOS\) fusion, a critical diagnostic marker in both orthodontics and forensic anthropology. Our approach leverages a dual\-model architecture wherein a teacher model, trained on manually cropped images, transfers its precise spatial understanding to a student model that operates on full, uncropped images. This knowledge distillation is facilitated by a newly formulated loss function that aligns spatial logits as well as incorporates gradient\-based attention spatial mapping, ensuring that the student model internalizes the anatomically relevant features without relying on external cropping or YOLO\-based segmentation. By leveraging expert\-curated data and feedback at each step, our framework attains robust diagnostic accuracy, culminating in a clinically viable end\-to\-end pipeline. This streamlined approach obviates the need for additional pre\-processing tools and accelerates deployment, thereby enhancing both the efficiency and consistency of skeletal maturation assessment in diverse clinical settings.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.21561v1)

---


## Distill CLIP \(DCLIP\): Enhancing Image\-Text Retrieval via Cross\-Modal Transformer Distillation / 

发布日期：2025-05-25

作者：Daniel Csizmadia

摘要：We present Distill CLIP \(DCLIP\), a fine\-tuned variant of the CLIP model that enhances multimodal image\-text retrieval while preserving the original model's strong zero\-shot classification capabilities. CLIP models are typically constrained by fixed image resolutions and limited context, which can hinder their effectiveness in retrieval tasks that require fine\-grained cross\-modal understanding. DCLIP addresses these challenges through a meta teacher\-student distillation framework, where a cross\-modal transformer teacher is fine\-tuned to produce enriched embeddings via bidirectional cross\-attention between YOLO\-extracted image regions and corresponding textual spans. These semantically and spatially aligned global representations guide the training of a lightweight student model using a hybrid loss that combines contrastive learning and cosine similarity objectives. Despite being trained on only ~67,500 samples curated from MSCOCO, Flickr30k, and Conceptual Captions\-just a fraction of CLIP's original dataset\-DCLIP significantly improves image\-text retrieval metrics \(Recall@K, MAP\), while retaining approximately 94% of CLIP's zero\-shot classification performance. These results demonstrate that DCLIP effectively mitigates the trade\-off between task specialization and generalization, offering a resource\-efficient, domain\-adaptive, and detail\-sensitive solution for advanced vision\-language tasks. Code available at https://anonymous.4open.science/r/DCLIP\-B772/README.md.

中文摘要：


代码链接：https://anonymous.4open.science/r/DCLIP-B772/README.md.

论文链接：[阅读更多](http://arxiv.org/abs/2505.21549v1)

---


## Detailed Evaluation of Modern Machine Learning Approaches for Optic Plastics Sorting / 

发布日期：2025-05-22

作者：Vaishali Maheshkar

摘要：According to the EPA, only 25% of waste is recycled, and just 60% of U.S. municipalities offer curbside recycling. Plastics fare worse, with a recycling rate of only 8%; an additional 16% is incinerated, while the remaining 76% ends up in landfills. The low plastic recycling rate stems from contamination, poor economic incentives, and technical difficulties, making efficient recycling a challenge. To improve recovery, automated sorting plays a critical role. Companies like AMP Robotics and Greyparrot utilize optical systems for sorting, while Materials Recovery Facilities \(MRFs\) employ Near\-Infrared \(NIR\) sensors to detect plastic types.   Modern optical sorting uses advances in computer vision such as object recognition and instance segmentation, powered by machine learning. Two\-stage detectors like Mask R\-CNN use region proposals and classification with deep backbones like ResNet. Single\-stage detectors like YOLO handle detection in one pass, trading some accuracy for speed. While such methods excel under ideal conditions with a large volume of labeled training data, challenges arise in realistic scenarios, emphasizing the need to further examine the efficacy of optic detection for automated sorting.   In this study, we compiled novel datasets totaling 20,000\+ images from varied sources. Using both public and custom machine learning pipelines, we assessed the capabilities and limitations of optical recognition for sorting. Grad\-CAM, saliency maps, and confusion matrices were employed to interpret model behavior. We perform this analysis on our custom trained models from the compiled datasets. To conclude, our findings are that optic recognition methods have limited success in accurate sorting of real\-world plastics at MRFs, primarily because they rely on physical properties such as color and shape.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.16513v1)

---

