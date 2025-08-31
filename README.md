# 每日从arXiv中获取最新YOLO相关论文


## Quantization Robustness to Input Degradations for Object Detection / 目标检测中输入退化的量化鲁棒性

发布日期：2025-08-27

作者：Toghrul Karimov

摘要：Post\-training quantization \(PTQ\) is crucial for deploying efficient object detection models, like YOLO, on resource\-constrained devices. However, the impact of reduced precision on model robustness to real\-world input degradations such as noise, blur, and compression artifacts is a significant concern. This paper presents a comprehensive empirical study evaluating the robustness of YOLO models \(nano to extra\-large scales\) across multiple precision formats: FP32, FP16 \(TensorRT\), Dynamic UINT8 \(ONNX\), and Static INT8 \(TensorRT\). We introduce and evaluate a degradation\-aware calibration strategy for Static INT8 PTQ, where the TensorRT calibration process is exposed to a mix of clean and synthetically degraded images. Models were benchmarked on the COCO dataset under seven distinct degradation conditions \(including various types and levels of noise, blur, low contrast, and JPEG compression\) and a mixed\-degradation scenario. Results indicate that while Static INT8 TensorRT engines offer substantial speedups \(~1.5\-3.3x\) with a moderate accuracy drop \(~3\-7% mAP50\-95\) on clean data, the proposed degradation\-aware calibration did not yield consistent, broad improvements in robustness over standard clean\-data calibration across most models and degradations. A notable exception was observed for larger model scales under specific noise conditions, suggesting model capacity may influence the efficacy of this calibration approach. These findings highlight the challenges in enhancing PTQ robustness and provide insights for deploying quantized detectors in uncontrolled environments. All code and evaluation tables are available at https://github.com/AllanK24/QRID.

中文摘要：训练后量化（PTQ）对于在资源受限的设备上部署高效的目标检测模型（如YOLO）至关重要。然而，精度降低对模型对真实世界输入退化（如噪声、模糊和压缩伪影）的鲁棒性的影响是一个值得关注的问题。本文介绍了一项全面的实证研究，评估了YOLO模型（纳米到超大尺度）在多种精度格式下的鲁棒性：FP32、FP16（TensorRT）、动态UINT8（ONNX）和静态INT8（张量RT）。我们介绍并评估了静态INT8 PTQ的退化感知校准策略，其中TensorRT校准过程暴露于干净和合成退化图像的混合中。在七种不同的退化条件下（包括各种类型和级别的噪声、模糊、低对比度和JPEG压缩）和混合退化场景下，模型在COCO数据集上进行了基准测试。结果表明，虽然静态INT8 TensorRT引擎在干净数据上提供了可观的加速（约1.5-3.3倍），但精度下降适中（约3-7%mAP50-95），但所提出的退化感知校准在大多数模型和退化中并没有在鲁棒性方面比标准干净数据校准产生一致、广泛的改进。在特定噪声条件下，观察到较大模型尺度的一个显著例外，表明模型容量可能会影响这种校准方法的有效性。这些发现突显了增强PTQ鲁棒性的挑战，并为在不受控制的环境中部署量化探测器提供了见解。所有代码和评估表均可在https://github.com/AllanK24/QRID.


代码链接：https://github.com/AllanK24/QRID.

论文链接：[阅读更多](http://arxiv.org/abs/2508.19600v1)

---


## Spatial\-temporal risk field\-based coupled dynamic\-static driving risk assessment and trajectory planning in weaving segments / 基于时空风险场的交织段动静耦合驾驶风险评估与轨迹规划

发布日期：2025-08-27

作者：Guodong Ma

摘要：In this paper, we first propose a spatial\-temporal coupled risk assessment paradigm by constructing a three\-dimensional spatial\-temporal risk field \(STRF\). Specifically, we introduce spatial\-temporal distances to quantify the impact of future trajectories of dynamic obstacles. We also incorporate a geometrically configured specialized field for the weaving segment to constrain vehicle movement directionally. To enhance the STRF's accuracy, we further developed a parameter calibration method using real\-world aerial video data, leveraging YOLO\-based machine vision and dynamic risk balance theory. A comparative analysis with the traditional risk field demonstrates the STRF's superior situational awareness of anticipatory risk. Building on these results, we final design a STRF\-based CAV trajectory planning method in weaving segments. We integrate spatial\-temporal risk occupancy maps, dynamic iterative sampling, and quadratic programming to enhance safety, comfort, and efficiency. By incorporating both dynamic and static risk factors during the sampling phase, our method ensures robust safety performance. Additionally, the proposed method simultaneously optimizes path and speed using a parallel computing approach, reducing computation time. Real\-world cases show that, compared to the dynamic planning \+ quadratic programming schemes, and real human driving trajectories, our method significantly improves safety, reduces lane\-change completion time, and minimizes speed fluctuations.

中文摘要：本文首先通过构建三维时空风险场（STRF），提出了一种时空耦合风险评估范式。具体来说，我们引入时空距离来量化动态障碍物未来轨迹的影响。我们还为交织段引入了一个几何配置的专用字段，以定向约束车辆运动。为了提高STRF的准确性，我们利用基于YOLO的机器视觉和动态风险平衡理论，进一步开发了一种使用真实世界航空视频数据的参数校准方法。与传统风险领域的比较分析表明，STRF对预期风险具有卓越的情境意识。基于这些结果，我们最终设计了一种基于STRF的CAV交织段轨迹规划方法。我们整合了时空风险占用图、动态迭代采样和二次规划，以提高安全性、舒适性和效率。通过在采样阶段结合动态和静态风险因素，我们的方法确保了稳健的安全性能。此外，所提出的方法使用并行计算方法同时优化路径和速度，减少了计算时间。实际案例表明，与动态规划+二次规划方案和真实的人类驾驶轨迹相比，我们的方法显著提高了安全性，缩短了变道完成时间，并最大限度地减少了速度波动。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.19513v1)

---


## Weed Detection in Challenging Field Conditions: A Semi\-Supervised Framework for Overcoming Shadow Bias and Data Scarcity / 挑战性田间条件下的杂草检测：克服阴影偏差和数据稀缺的半监督框架

发布日期：2025-08-27

作者：Alzayat Saleh

摘要：The automated management of invasive weeds is critical for sustainable agriculture, yet the performance of deep learning models in real\-world fields is often compromised by two factors: challenging environmental conditions and the high cost of data annotation. This study tackles both issues through a diagnostic\-driven, semi\-supervised framework. Using a unique dataset of approximately 975 labeled and 10,000 unlabeled images of Guinea Grass in sugarcane, we first establish strong supervised baselines for classification \(ResNet\) and detection \(YOLO, RF\-DETR\), achieving F1 scores up to 0.90 and mAP50 scores exceeding 0.82. Crucially, this foundational analysis, aided by interpretability tools, uncovered a pervasive "shadow bias," where models learned to misidentify shadows as vegetation. This diagnostic insight motivated our primary contribution: a semi\-supervised pipeline that leverages unlabeled data to enhance model robustness. By training models on a more diverse set of visual information through pseudo\-labeling, this framework not only helps mitigate the shadow bias but also provides a tangible boost in recall, a critical metric for minimizing weed escapes in automated spraying systems. To validate our methodology, we demonstrate its effectiveness in a low\-data regime on a public crop\-weed benchmark. Our work provides a clear and field\-tested framework for developing, diagnosing, and improving robust computer vision systems for the complex realities of precision agriculture.

中文摘要：入侵杂草的自动化管理对于可持续农业至关重要，但深度学习模型在现实世界中的性能往往受到两个因素的影响：具有挑战性的环境条件和数据注释的高昂成本。本研究通过一个诊断驱动的半监督框架解决了这两个问题。使用一个包含约975张标记和10000张未标记甘蔗几内亚草图像的独特数据集，我们首先建立了用于分类（ResNet）和检测（YOLO、RF-DETR）的强监督基线，F1得分高达0.90，mAP50得分超过0.82。至关重要的是，在可解释性工具的帮助下，这一基础分析揭示了一种普遍的“阴影偏见”，即模型学会了将阴影误认为植被。这种诊断见解激发了我们的主要贡献：一种利用未标记数据来增强模型鲁棒性的半监督管道。通过伪标签在一组更多样化的视觉信息上训练模型，该框架不仅有助于减轻阴影偏差，而且还能显著提高召回率，这是减少自动喷洒系统中杂草逃逸的关键指标。为了验证我们的方法，我们在公共作物杂草基准的低数据制度下证明了其有效性。我们的工作为开发、诊断和改进用于精准农业复杂现实的强大计算机视觉系统提供了一个清晰且经过现场测试的框架。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.19511v1)

---


## HOTSPOT\-YOLO: A Lightweight Deep Learning Attention\-Driven Model for Detecting Thermal Anomalies in Drone\-Based Solar Photovoltaic Inspections / HOTSPOT-YOLO：一种轻量级的深度学习注意力驱动模型，用于检测基于无人机的太阳能光伏检测中的热异常

发布日期：2025-08-26

作者：Mahmoud Dhimish

摘要：Thermal anomaly detection in solar photovoltaic \(PV\) systems is essential for ensuring operational efficiency and reducing maintenance costs. In this study, we developed and named HOTSPOT\-YOLO, a lightweight artificial intelligence \(AI\) model that integrates an efficient convolutional neural network backbone and attention mechanisms to improve object detection. This model is specifically designed for drone\-based thermal inspections of PV systems, addressing the unique challenges of detecting small and subtle thermal anomalies, such as hotspots and defective modules, while maintaining real\-time performance. Experimental results demonstrate a mean average precision of 90.8%, reflecting a significant improvement over baseline object detection models. With a reduced computational load and robustness under diverse environmental conditions, HOTSPOT\-YOLO offers a scalable and reliable solution for large\-scale PV inspections. This work highlights the integration of advanced AI techniques with practical engineering applications, revolutionizing automated fault detection in renewable energy systems.

中文摘要：太阳能光伏（PV）系统中的热异常检测对于确保运行效率和降低维护成本至关重要。在这项研究中，我们开发并命名了HOTSPOT-YOLO，这是一种轻量级的人工智能（AI）模型，它集成了高效的卷积神经网络骨干和注意力机制来提高目标检测。该模型专为基于无人机的光伏系统热检测而设计，解决了检测小而微妙的热异常（如热点和有缺陷的模块）的独特挑战，同时保持实时性能。实验结果表明，平均精度为90.8%，比基线目标检测模型有了显著提高。HOTSPOT-YOLO在各种环境条件下具有降低的计算负载和鲁棒性，为大规模光伏检测提供了一种可扩展且可靠的解决方案。这项工作强调了先进的人工智能技术与实际工程应用的融合，彻底改变了可再生能源系统的自动故障检测。


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.18912v1)

---


## LPLC: A Dataset for License Plate Legibility Classification / 车牌易读性分类数据集

发布日期：2025-08-25

作者：Lucas Wojcik

摘要：Automatic License Plate Recognition \(ALPR\) faces a major challenge when dealing with illegible license plates \(LPs\). While reconstruction methods such as super\-resolution \(SR\) have emerged, the core issue of recognizing these low\-quality LPs remains unresolved. To optimize model performance and computational efficiency, image pre\-processing should be applied selectively to cases that require enhanced legibility. To support research in this area, we introduce a novel dataset comprising 10,210 images of vehicles with 12,687 annotated LPs for legibility classification \(the LPLC dataset\). The images span a wide range of vehicle types, lighting conditions, and camera/image quality levels. We adopt a fine\-grained annotation strategy that includes vehicle\- and LP\-level occlusions, four legibility categories \(perfect, good, poor, and illegible\), and character labels for three categories \(excluding illegible LPs\). As a benchmark, we propose a classification task using three image recognition networks to determine whether an LP image is good enough, requires super\-resolution, or is completely unrecoverable. The overall F1 score, which remained below 80% for all three baseline models \(ViT, ResNet, and YOLO\), together with the analyses of SR and LP recognition methods, highlights the difficulty of the task and reinforces the need for further research. The proposed dataset is publicly available at https://github.com/lmlwojcik/lplc\-dataset.

中文摘要：自动车牌识别（ALPR）在处理难以辨认的车牌（LP）时面临着重大挑战。虽然超分辨率（SR）等重建方法已经出现，但识别这些低质量LP的核心问题仍未得到解决。为了优化模型性能和计算效率，图像预处理应选择性地应用于需要增强易读性的情况。为了支持这一领域的研究，我们引入了一个新的数据集，其中包括10210张车辆图像和12687个带注释的车牌，用于易读性分类（车牌识别数据集）。这些图像涵盖了各种车辆类型、照明条件和摄像头/图像质量水平。我们采用了一种细粒度的注释策略，包括车辆和LP级别的遮挡、四个易读性类别（完美、良好、较差和难以辨认）以及三个类别的字符标签（不包括难以辨认的LP）。作为基准，我们提出了一种使用三个图像识别网络的分类任务，以确定LP图像是否足够好、是否需要超分辨率或是否完全不可恢复。所有三个基线模型（ViT、ResNet和YOLO）的总体F1得分均低于80%，再加上对SR和LP识别方法的分析，突显了这项任务的难度，并强调了进一步研究的必要性。拟议的数据集可在以下网址公开获取https://github.com/lmlwojcik/lplc-dataset.


代码链接：https://github.com/lmlwojcik/lplc-dataset.

论文链接：[阅读更多](http://arxiv.org/abs/2508.18425v1)

---

