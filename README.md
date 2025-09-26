# 每日从arXiv中获取最新YOLO相关论文


## Real\-Time Object Detection Meets DINOv3 / 

发布日期：2025-09-25

作者：Shihua Huang

摘要：Benefiting from the simplicity and effectiveness of Dense O2O and MAL, DEIM has become the mainstream training framework for real\-time DETRs, significantly outperforming the YOLO series. In this work, we extend it with DINOv3 features, resulting in DEIMv2. DEIMv2 spans eight model sizes from X to Atto, covering GPU, edge, and mobile deployment. For the X, L, M, and S variants, we adopt DINOv3\-pretrained or distilled backbones and introduce a Spatial Tuning Adapter \(STA\), which efficiently converts DINOv3's single\-scale output into multi\-scale features and complements strong semantics with fine\-grained details to enhance detection. For ultra\-lightweight models \(Nano, Pico, Femto, and Atto\), we employ HGNetv2 with depth and width pruning to meet strict resource budgets. Together with a simplified decoder and an upgraded Dense O2O, this unified design enables DEIMv2 to achieve a superior performance\-cost trade\-off across diverse scenarios, establishing new state\-of\-the\-art results. Notably, our largest model, DEIMv2\-X, achieves 57.8 AP with only 50.3 million parameters, surpassing prior X\-scale models that require over 60 million parameters for just 56.5 AP. On the compact side, DEIMv2\-S is the first sub\-10 million model \(9.71 million\) to exceed the 50 AP milestone on COCO, reaching 50.9 AP. Even the ultra\-lightweight DEIMv2\-Pico, with just 1.5 million parameters, delivers 38.5 AP, matching YOLOv10\-Nano \(2.3 million\) with around 50 percent fewer parameters.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.20787v1)

---


## Building Information Models to Robot\-Ready Site Digital Twins \(BIM2RDT\): An Agentic AI Safety\-First Framework / 

发布日期：2025-09-25

作者：Reza Akhavian

摘要：The adoption of cyber\-physical systems and jobsite intelligence that connects design models, real\-time site sensing, and autonomous field operations can dramatically enhance digital management in the construction industry. This paper introduces BIM2RDT \(Building Information Models to Robot\-Ready Site Digital Twins\), an agentic artificial intelligence \(AI\) framework designed to transform static Building Information Modeling \(BIM\) into dynamic, robot\-ready digital twins \(DTs\) that prioritize safety during execution. The framework bridges the gap between pre\-existing BIM data and real\-time site conditions by integrating three key data streams: geometric and semantic information from BIM models, activity data from IoT sensor networks, and visual\-spatial data collected by robots during site traversal. The methodology introduces Semantic\-Gravity ICP \(SG\-ICP\), a point cloud registration algorithm that leverages large language model \(LLM\) reasoning. Unlike traditional methods, SG\-ICP utilizes an LLM to infer object\-specific, plausible orientation priors based on BIM semantics, improving alignment accuracy by avoiding convergence on local minima. This creates a feedback loop where robot\-collected data updates the DT, which in turn optimizes paths for missions. The framework employs YOLOE object detection and Shi\-Tomasi corner detection to identify and track construction elements while using BIM geometry as a priori maps. The framework also integrates real\-time Hand\-Arm Vibration \(HAV\) monitoring, mapping sensor\-detected safety events to the digital twin using IFC standards for intervention. Experiments demonstrate SG\-ICP's superiority over standard ICP, achieving RMSE reductions of 64.3%\-\-88.3% in alignment across scenarios with occluded features, ensuring plausible orientations. HAV integration triggers warnings upon exceeding exposure limits, enhancing compliance with ISO 5349\-1.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.20705v1)

---


## A Comparative Benchmark of Real\-time Detectors for Blueberry Detection towards Precision Orchard Management / 

发布日期：2025-09-24

作者：Xinyang Mu

摘要：Blueberry detection in natural environments remains challenging due to variable lighting, occlusions, and motion blur due to environmental factors and imaging devices. Deep learning\-based object detectors promise to address these challenges, but they demand a large\-scale, diverse dataset that captures the real\-world complexities. Moreover, deploying these models in practical scenarios often requires the right accuracy/speed/memory trade\-off in model selection. This study presents a novel comparative benchmark analysis of advanced real\-time object detectors, including YOLO \(You Only Look Once\) \(v8\-v12\) and RT\-DETR \(Real\-Time Detection Transformers\) \(v1\-v2\) families, consisting of 36 model variants, evaluated on a newly curated dataset for blueberry detection. This dataset comprises 661 canopy images collected with smartphones during the 2022\-2023 seasons, consisting of 85,879 labelled instances \(including 36,256 ripe and 49,623 unripe blueberries\) across a wide range of lighting conditions, occlusions, and fruit maturity stages. Among the YOLO models, YOLOv12m achieved the best accuracy with a mAP@50 of 93.3%, while RT\-DETRv2\-X obtained a mAP@50 of 93.6%, the highest among all the RT\-DETR variants. The inference time varied with the model scale and complexity, and the mid\-sized models appeared to offer a good accuracy\-speed balance. To further enhance detection performance, all the models were fine\-tuned using Unbiased Mean Teacher\-based semi\-supervised learning \(SSL\) on a separate set of 1,035 unlabeled images acquired by a ground\-based machine vision platform in 2024. This resulted in accuracy gains ranging from \-1.4% to 2.9%, with RT\-DETR\-v2\-X achieving the best mAP@50 of 94.8%. More in\-depth research into SSL is needed to better leverage cross\-domain unlabeled data. Both the dataset and software programs of this study are made publicly available to support further research.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.20580v1)

---


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

