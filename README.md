# 每日从arXiv中获取最新YOLO相关论文


## COTONET: A custom cotton detection algorithm based on YOLO11 for stage of growth cotton boll detection / 

发布日期：2026-03-12

作者：Guillem González

摘要：Cotton harvesting is a critical phase where cotton capsules are physically manipulated and can lead to fibre degradation. To maintain the highest quality, harvesting methods must emulate delicate manual grasping, to preserve cotton's intrinsic properties. Automating this process requires systems capable of recognising cotton capsules across various phenological stages. To address this challenge, we propose COTONET, an enhanced custom YOLO11 model tailored with attention mechanisms to improve the detection of difficult instances. The architecture incorporates gradients in non\-learnable operations to enhance shape and feature extraction. Key architectural modifications include: the replacement of convolutional blocks with Squeeze\-and\-Exitation blocks, a redesigned backbone integrating attention mechanisms, and the substitution of standard upsampling operations for Content Aware Reassembly of Features \(CARAFE\). Additionally, we integrate Simple Attention Modules \(SimAM\) for primary feature aggregation and Parallel Hybrid Attention Mechanisms \(PHAM\) for channel\-wise, spatial\-wise and coordinate\-wise attention in the downward neck path. This configuration offers increased flexibility and robustness for interpreting the complexity of cotton crop growth. COTONET aligns with small\-to\-medium YOLO models utilizing 7.6M parameters and 27.8 GFLOPS, making it suitable for low\-resource edge computing and mobile robotics. COTONET outperforms the standard YOLO baselines, achieving a mAP50 of 81.1% and a mAP50\-95 of 60.6%.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.11717v1)

---


## TornadoNet: Real\-Time Building Damage Detection with Ordinal Supervision / 

发布日期：2026-03-12

作者：Robinson Umeike

摘要：We present TornadoNet, a comprehensive benchmark for automated street\-level building damage assessment evaluating how modern real\-time object detection architectures and ordinal\-aware supervision strategies perform under realistic post\-disaster conditions. TornadoNet provides the first controlled benchmark demonstrating how architectural design and loss formulation jointly influence multi\-level damage detection from street\-view imagery, delivering methodological insights and deployable tools for disaster response. Using 3,333 high\-resolution geotagged images and 8,890 annotated building instances from the 2021 Midwest tornado outbreak, we systematically compare CNN\-based detectors from the YOLO family against transformer\-based models \(RT\-DETR\) for multi\-level damage detection. Models are trained under standardized protocols using a five\-level damage classification framework based on IN\-CORE damage states, validated through expert cross\-annotation. Baseline experiments reveal complementary architectural strengths. CNN\-based YOLO models achieve highest detection accuracy and throughput, with larger variants reaching 46.05% mAP@0.5 at 66\-276 FPS on A100 GPUs. Transformer\-based RT\-DETR models exhibit stronger ordinal consistency, achieving 88.13% Ordinal Top\-1 Accuracy and MAOE of 0.65, indicating more reliable severity grading despite lower baseline mAP. To align supervision with the ordered nature of damage severity, we introduce soft ordinal classification targets and evaluate explicit ordinal\-distance penalties. RT\-DETR trained with calibrated ordinal supervision achieves 44.70% mAP@0.5, a 4.8 percentage\-point improvement, with gains in ordinal metrics \(91.15% Ordinal Top\-1 Accuracy, MAOE = 0.56\). These findings establish that ordinal\-aware supervision improves damage severity estimation when aligned with detector architecture. Model & Data: https://github.com/crumeike/TornadoNet

中文摘要：


代码链接：https://github.com/crumeike/TornadoNet

论文链接：[阅读更多](http://arxiv.org/abs/2603.11557v1)

---


## GroundCount: Grounding Vision\-Language Models with Object Detection for Mitigating Counting Hallucinations / 

发布日期：2026-03-11

作者：Boyuan Chen

摘要：Vision Language Models \(VLMs\) exhibit persistent hallucinations in counting tasks, with accuracy substantially lower than other visual reasoning tasks \(excluding sentiment\). This phenomenon persists even in state\-of\-the\-art reasoning\-capable VLMs. Conversely, CNN\-based object detection models \(ODMs\) such as YOLO excel at spatial localization and instance counting with minimal computational overhead. We propose GroundCount, a framework that augments VLMs with explicit spatial grounding from ODMs to mitigate counting hallucinations. In the best case, our prompt\-based augmentation strategy achieves 81.3% counting accuracy on the best\-performing model \(Ovis2.5\-2B\) \- a 6.6pp improvement \- while reducing inference time by 22% through elimination of hallucination\-driven reasoning loops for stronger models. We conduct comprehensive ablation studies demonstrating that positional encoding is a critical component, being beneficial for stronger models but detrimental for weaker ones. Confidence scores, by contrast, introduce noise for most architectures and their removal improves performance in four of five evaluated models. We further evaluate feature\-level fusion architectures, finding that explicit symbolic grounding via structured prompts outperforms implicit feature fusion despite sophisticated cross\-attention mechanisms. Our approach yields consistent improvements across four of five evaluated VLM architectures \(6.2\-\-7.5pp\), with one architecture exhibiting degraded performance due to incompatibility between its iterative reflection mechanisms and structured prompts. These results suggest that counting failures stem from fundamental spatial\-semantic integration limitations rather than architecture\-specific deficiencies, while highlighting the importance of architectural compatibility in augmentation strategies.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.10978v1)

---


## Phase\-Interface Instance Segmentation as a Visual Sensor for Laboratory Process Monitoring / 

发布日期：2026-03-11

作者：Mingyue Li

摘要：Reliable visual monitoring of chemical experiments remains challenging in transparent glassware, where weak phase boundaries and optical artifacts degrade conventional segmentation. We formulate laboratory phenomena as the time evolution of phase interfaces and introduce the Chemical Transparent Glasses dataset 2.0 \(CTG 2.0\), a vessel\-aware benchmark with 3,668 images, 23 glassware categories, and five multiphase interface types for phase\-interface instance segmentation. Building on YOLO11m\-seg, we propose LGA\-RCM\-YOLO, which combines Local\-Global Attention \(LGA\) for robust semantic representation and a Rectangular Self\-Calibration Module \(RCM\) for boundary refinement of thin, elongated interfaces. On CTG 2.0, the proposed model achieves 84.4% AP@0.5 and 58.43% AP@0.5\-0.95, improving over the YOLO11m baseline by 6.42 and 8.75 AP points, respectively, while maintaining near real\-time inference \(13.67 FPS, RTX 3060\). An auxiliary color\-attribute head further labels liquid instances as colored or colorless with 98.71% precision and 98.32% recall. Finally, we demonstrate continuous process monitoring in separatory\-funnel phase separation and crystallization, showing that phase\-interface instance segmentation can serve as a practical visual sensor for laboratory automation.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.10782v1)

---


## A Robust Deep Learning Framework for Bangla License Plate Recognition Using YOLO and Vision\-Language OCR / 

发布日期：2026-03-10

作者：Nayeb Hasin

摘要：An Automatic License Plate Recognition \(ALPR\) system constitutes a crucial element in an intelligent traffic management system. However, the detection of Bangla license plates remains challenging because of the complicated character scheme and uneven layouts. This paper presents a robust Bangla License Plate Recognition system that integrates a deep learning\-based object detection model for license plate localization with Optical Character Recognition for text extraction. Multiple object detection architectures, including U\-Net and several YOLO \(You Only Look Once\) variants, are compared for license plate localization. This study proposes a novel two\-stage adaptive training strategy built upon the YOLOv8 architecture to improve localization performance. The proposed approach outperforms the established models, achieving an accuracy of 97.83% and an Intersection over Union \(IoU\) of 91.3%. The text recognition problem is phrased as a sequence generation problem with a VisionEncoderDecoder architecture, with a combination of encoder\-decoders evaluated. It was demonstrated that the ViT \+ BanglaBERT model gives better results at the character level, with a Character Error Rate of 0.1323 and Word Error Rate of 0.1068. The proposed system also shows a consistent performance when tested on an external dataset that has been curated for this study purpose. The dataset offers completely different environment and lighting conditions compared to the training sample, indicating the robustness of the proposed framework. Overall, our proposed system provides a robust and reliable solution for Bangla license plate recognition and performs effectively across diverse real\-world scenarios, including variations in lighting, noise, and plate styles. These strengths make it well suited for deployment in intelligent transportation applications such as automated law enforcement and access control.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.10267v1)

---

