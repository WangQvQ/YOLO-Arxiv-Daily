# 每日从arXiv中获取最新YOLO相关论文


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


## A Single Detect Focused YOLO Framework for Robust Mitotic Figure Detection / 

发布日期：2025-09-01

作者：Yasemin Topuz

摘要：Mitotic figure detection is a crucial task in computational pathology, as mitotic activity serves as a strong prognostic marker for tumor aggressiveness. However, domain variability that arises from differences in scanners, tissue types, and staining protocols poses a major challenge to the robustness of automated detection methods. In this study, we introduce SDF\-YOLO \(Single Detect Focused YOLO\), a lightweight yet domain\-robust detection framework designed specifically for small, rare targets such as mitotic figures. The model builds on YOLOv11 with task\-specific modifications, including a single detection head aligned with mitotic figure scale, coordinate attention to enhance positional sensitivity, and improved cross\-channel feature mixing. Experiments were conducted on three datasets that span human and canine tumors: MIDOG \+\+, canine cutaneous mast cell tumor \(CCMCT\), and canine mammary carcinoma \(CMC\). When submitted to the preliminary test set for the MIDOG2025 challenge, SDF\-YOLO achieved an average precision \(AP\) of 0.799, with a precision of 0.758, a recall of 0.775, an F1 score of 0.766, and an FROC\-AUC of 5.793, demonstrating both competitive accuracy and computational efficiency. These results indicate that SDF\-YOLO provides a reliable and efficient framework for robust mitotic figure detection across diverse domains.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.02637v1)

---


## Quantization Robustness to Input Degradations for Object Detection / 

发布日期：2025-08-27

作者：Toghrul Karimov

摘要：Post\-training quantization \(PTQ\) is crucial for deploying efficient object detection models, like YOLO, on resource\-constrained devices. However, the impact of reduced precision on model robustness to real\-world input degradations such as noise, blur, and compression artifacts is a significant concern. This paper presents a comprehensive empirical study evaluating the robustness of YOLO models \(nano to extra\-large scales\) across multiple precision formats: FP32, FP16 \(TensorRT\), Dynamic UINT8 \(ONNX\), and Static INT8 \(TensorRT\). We introduce and evaluate a degradation\-aware calibration strategy for Static INT8 PTQ, where the TensorRT calibration process is exposed to a mix of clean and synthetically degraded images. Models were benchmarked on the COCO dataset under seven distinct degradation conditions \(including various types and levels of noise, blur, low contrast, and JPEG compression\) and a mixed\-degradation scenario. Results indicate that while Static INT8 TensorRT engines offer substantial speedups \(~1.5\-3.3x\) with a moderate accuracy drop \(~3\-7% mAP50\-95\) on clean data, the proposed degradation\-aware calibration did not yield consistent, broad improvements in robustness over standard clean\-data calibration across most models and degradations. A notable exception was observed for larger model scales under specific noise conditions, suggesting model capacity may influence the efficacy of this calibration approach. These findings highlight the challenges in enhancing PTQ robustness and provide insights for deploying quantized detectors in uncontrolled environments. All code and evaluation tables are available at https://github.com/AllanK24/QRID.

中文摘要：


代码链接：https://github.com/AllanK24/QRID.

论文链接：[阅读更多](http://arxiv.org/abs/2508.19600v1)

---

