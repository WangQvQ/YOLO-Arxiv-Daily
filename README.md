# 每日从arXiv中获取最新YOLO相关论文


## VisioFirm: Cross\-Platform AI\-assisted Annotation Tool for Computer Vision / 

发布日期：2025-09-04

作者：Safouane El Ghazouali

摘要：AI models rely on annotated data to learn pattern and perform prediction. Annotation is usually a labor\-intensive step that require associating labels ranging from a simple classification label to more complex tasks such as object detection, oriented bounding box estimation, and instance segmentation. Traditional tools often require extensive manual input, limiting scalability for large datasets. To address this, we introduce VisioFirm, an open\-source web application designed to streamline image labeling through AI\-assisted automation. VisioFirm integrates state\-of\-the\-art foundation models into an interface with a filtering pipeline to reduce human\-in\-the\-loop efforts. This hybrid approach employs CLIP combined with pre\-trained detectors like Ultralytics models for common classes and zero\-shot models such as Grounding DINO for custom labels, generating initial annotations with low\-confidence thresholding to maximize recall. Through this framework, when tested on COCO\-type of classes, initial prediction have been proven to be mostly correct though the users can refine these via interactive tools supporting bounding boxes, oriented bounding boxes, and polygons. Additionally, VisioFirm has on\-the\-fly segmentation powered by Segment Anything accelerated through WebGPU for browser\-side efficiency. The tool supports multiple export formats \(YOLO, COCO, Pascal VOC, CSV\) and operates offline after model caching, enhancing accessibility. VisioFirm demonstrates up to 90% reduction in manual effort through benchmarks on diverse datasets, while maintaining high annotation accuracy via clustering of connected CLIP\-based disambiguate components and IoU\-graph for redundant detection suppression. VisioFirm can be accessed from href\{https://github.com/OschAI/VisioFirm\}\{https://github.com/OschAI/VisioFirm\}.

中文摘要：


代码链接：https://github.com/OschAI/VisioFirm}{https://github.com/OschAI/VisioFirm}.

论文链接：[阅读更多](http://arxiv.org/abs/2509.04180v1)

---


## YOLO Ensemble for UAV\-based Multispectral Defect Detection in Wind Turbine Components / 

发布日期：2025-09-04

作者：Serhii Svystun

摘要：Unmanned aerial vehicles \(UAVs\) equipped with advanced sensors have opened up new opportunities for monitoring wind power plants, including blades, towers, and other critical components. However, reliable defect detection requires high\-resolution data and efficient methods to process multispectral imagery. In this research, we aim to enhance defect detection accuracy through the development of an ensemble of YOLO\-based deep learning models that integrate both visible and thermal channels. We propose an ensemble approach that integrates a general\-purpose YOLOv8 model with a specialized thermal model, using a sophisticated bounding box fusion algorithm to combine their predictions. Our experiments show this approach achieves a mean Average Precision \(mAP@.5\) of 0.93 and an F1\-score of 0.90, outperforming a standalone YOLOv8 model, which scored an mAP@.5 of 0.91. These findings demonstrate that combining multiple YOLO architectures with fused multispectral data provides a more reliable solution, improving the detection of both visual and thermal defects.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.04156v1)

---


## YOLO\-based Bearing Fault Diagnosis With Continuous Wavelet Transform / 

发布日期：2025-09-03

作者：Po\-Heng Chou

摘要：This letter proposes a YOLO\-based framework for spatial bearing fault diagnosis using time\-frequency spectrograms derived from continuous wavelet transform \(CWT\). One\-dimensional vibration signals are first transformed into time\-frequency spectrograms using Morlet wavelets to capture transient fault signatures. These spectrograms are then processed by YOLOv9, v10, and v11 models to classify fault types. Evaluated on three benchmark datasets, including Case Western Reserve University \(CWRU\), Paderborn University \(PU\), and Intelligent Maintenance System \(IMS\), the proposed CWT\-\-YOLO pipeline achieves significantly higher accuracy and generalizability than the baseline MCNN\-\-LSTM model. Notably, YOLOv11 reaches mAP scores of 99.4% \(CWRU\), 97.8% \(PU\), and 99.5% \(IMS\). In addition, its region\-aware detection mechanism enables direct visualization of fault locations in spectrograms, offering a practical solution for condition monitoring in rotating machinery.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.03070v1)

---


## Ensemble YOLO Framework for Multi\-Domain Mitotic Figure Detection in Histopathology Images / 

发布日期：2025-09-03

作者：Navya Sri Kelam

摘要：Accurate detection of mitotic figures in whole slide histopathological images remains a challenging task due to their scarcity, morphological heterogeneity, and the variability introduced by tissue preparation and staining protocols. The MIDOG competition series provides standardized benchmarks for evaluating detection approaches across diverse domains, thus motivating the development of generalizable deep learning models. In this work, we investigate the performance of two modern one\-stage detectors, YOLOv5 and YOLOv8, trained on MIDOG\+\+, CMC, and CCMCT datasets. To enhance robustness, training incorporated stain\-invariant color perturbations and texture preserving augmentations. In internal validation, YOLOv5 achieved superior precision, while YOLOv8 provided improved recall, reflecting architectural trade\-offs between anchor\-based and anchor\-free detection. To capitalize on these complementary strengths, we employed an ensemble of the two models, which improved sensitivity without a major reduction in precision. These findings highlight the effectiveness of ensemble strategies built upon contemporary object detectors to advance automated mitosis detection in digital pathology.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.02957v1)

---


## Explaining What Machines See: XAI Strategies in Deep Object Detection Models / 

发布日期：2025-09-02

作者：FatemehSadat Seyedmomeni

摘要：In recent years, deep learning has achieved unprecedented success in various computer vision tasks, particularly in object detection. However, the black\-box nature and high complexity of deep neural networks pose significant challenges for interpretability, especially in critical domains such as autonomous driving, medical imaging, and security systems. Explainable Artificial Intelligence \(XAI\) aims to address this challenge by providing tools and methods to make model decisions more transparent, interpretable, and trust\-worthy for humans. This review provides a comprehensive analysis of state\-of\-the\-art explain\-ability methods specifically applied to object detection models. The paper be\-gins by categorizing existing XAI techniques based on their underlying mechanisms\-perturbation\-based, gradient\-based, backpropagation\-based, and graph\-based methods. Notable methods such as D\-RISE, BODEM, D\-CLOSE, and FSOD are discussed in detail. Furthermore, the paper investigates their applicability to various object detection architectures, including YOLO, SSD, Faster R\-CNN, and EfficientDet. Statistical analysis of publication trends from 2022 to mid\-2025 shows an accelerating interest in explainable object detection, indicating its increasing importance. The study also explores common datasets and evaluation metrics, and highlights the major challenges associated with model interpretability. By providing a structured taxonomy and a critical assessment of existing methods, this review aims to guide researchers and practitioners in selecting suitable explainability techniques for object detection applications and to foster the development of more interpretable AI systems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.01991v1)

---

