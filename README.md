<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. SPARK: Low Latency Single\-Camera 3D Pose Estimation for Autonomous Racing using Keypoints
> **🔹 中文标题：** SPARK：基于关键点技术的自动驾驶赛车低延迟单摄像头3D姿态估计
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-16 |
> | 👤 作者 | Dominic Ebner |
>
> **📄 英文摘要：**
> In autonomous racing, fast detection of other participants' movements is required to plan safe, collision\-free trajectories with non\-cooperative opponents. LiDAR detection is inherently slower and harder to deploy on edge devices than vision methods, causing delayed detections that limit object tracking performance during high\-dynamic maneuvering. Utilizing monocular 3D detection enables an easy\-to\-deploy, low\-latency detection of other participants on the racetrack. We present SPARK, a single\-camera pose\-estimation algorithm for autonomous racing using keypoint detection. It achieves long\-range detection with high accuracy, exceeding the performance of state\-of\-the\-art monocular camera detection algorithms while maintaining lower latency. By employing well\-optimized YOLO models and leveraging the fixed geometry in the autonomous racing domain, the algorithm also exhibits low latency and resource usage. We evaluate the performance of our approach on real\-world autonomous racing data and compare it to state\-of\-the\-art LiDAR and camera detection algorithms. The source code is available at: https://github.com/TUMFTM/SPARK\-camera\-det
>
> **📝 中文摘要：**
> 在自动驾驶赛车运动中，为规划安全、无碰撞的轨迹以应对非合作对手，需快速检测其他参赛者的移动情况。激光雷达检测固有的速度较慢且比视觉方法更难部署在边缘设备上，导致检测延迟，限制了高动态机动时的目标跟踪性能。利用单目3D检测可实现赛场上其他参赛者的易部署、低延迟检测。我们提出了SPARK——一种基于关键点检测的自动驾驶赛车单相机位姿估计算法。该算法实现了远距离高精度检测，在保持更低延迟的同时，性能超越了当前最先进的单目相机检测算法。通过采用优化良好的YOLO模型并利用自动驾驶赛车领域的固定几何特性，该算法展现出低延迟和低资源占用的特点。我们在真实世界的自动驾驶赛车数据中评估了该方法的性能，并与当前先进的激光雷达和相机检测算法进行了比较。源代码可通过以下地址获取：https://github.com/TUMFTM/SPARK-camera-det
>
> **💻 代码链接：** https://github.com/TUMFTM/SPARK-camera-det
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.17936v1)

---

> ### 2. MOSAIC: Mobile Object Segmentation under Adverse Imaging Conditions for Rapid L\-PBF Keyhole Behavior Characterization
> **🔹 中文标题：** MOSAIC：恶劣成像条件下的移动目标分割算法及其在L-PBF匙孔行为快速表征中的应用
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-15 |
> | 👤 作者 | Garrett Mathesen |
>
> **📄 英文摘要：**
> In laser powder bed fusion \(L\-PBF\) processes, the rapid evolution of gas and fluid interactions complicates our ability to properly monitor or control the process, with unstable keyholes leading to porosity and spatter formation. High\-speed operando x\-ray imaging of the keyhole has been used to better understand the impact of these interactions on the monitoring and control of the L\-PBF process. MOSAIC, a Mobile Object Segmentation algorithm for experiments under Adverse Imaging Conditions, is designed to perform rapid analysis of keyhole dynamics during active beamline experimentation without needing time consuming manual labeling or model training. Validation studies performed on 12 unique samples proved the robustness of MOSAIC with an average F1 score of 0.894 and a precision of 0.953 when compared to manually segmented images, performing equally or better than the SAM and YOLO machine learning methods tested. MOSAIC is efficient, processing frames cropped to a moving window approximately 150x250 pixels at 19.9 milliseconds per image on CPU, compared to 54 and 5284 milliseconds per image for inference on CPU for YOLO and SAM models.
>
> **📝 中文摘要：**
> 在激光粉末床熔融（L-PBF）工艺中，气体与流体的快速相互作用演变增加了过程监控与控制的难度，不稳定的匙孔形态易导致孔隙形成与飞溅物产生。通过高速操作式X射线成像技术观察匙孔行为，可更深入理解这些相互作用对L-PBF过程监控与控制的影响。MOSAIC（适用于恶劣成像条件的移动物体分割算法）专为实验束线运行时的匙孔动力学快速分析而设计，无需耗时的人工标注或模型训练。通过对12种独特样品的验证研究证明，MOSAIC算法具有强鲁棒性：与人工分割图像相比，其平均F1分数达0.894，精确率达0.953，性能达到或优于测试中的SAM与YOLO机器学习方法。MOSAIC处理效率显著，在CPU上对约150×250像素的移动窗口裁剪帧进行处理时，每张图像仅需19.9毫秒；而YOLO与SAM模型在CPU上的单张图像推理时间分别为54毫秒和5284毫秒。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.16186v1)

---

> ### 3. TimeLens: On\-Device Artifact Recognition with Retrieval\-Augmented Question Answering for the Grand Egyptian Museum
> **🔹 中文标题：** TimeLens：基于检索增强问答的设备端文物识别技术在大埃及博物馆的应用
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-11 |
> | 👤 作者 | Rawan Hesham |
>
> **📄 英文摘要：**
> TimeLens is an AI\-powered bilingual mobile guide for the Grand Egyptian Museum \(GEM\). Pointing a phone at an exhibit, a visitor sees the artifact recognized in real time and can ask follow\-up questions answered in English or Arabic. The work addresses three problems specific to in\-gallery deployment: fine\-grained visual similarity among 51 catalogued artifacts \(many near\-identical Ramesside statues\), the gap between curated training data and handheld camera conditions, and the risk of an AI guide stating unsupported historical facts. Two engineering contributions are reported. First, an on\-device artifact detector was developed through a data\-quality\-driven iteration study \-\- from foundation\-model auto\-annotation \(YOLO\-World\), through spatial label\-cleaning rules, to a fully hand\-annotated dataset \-\- isolating label quality as the decisive factor: the final YOLOv8n model resolves every previously failing class while remaining a 5.97 MB TensorFlow Lite asset that runs in real time on a mid\-range phone \(mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.924\). Second, a bilingual Retrieval\-Augmented Generation \(RAG\) guide, grounded in a 108\-record ChromaDB knowledge base, was benchmarked across seven candidate language models, with Gemma 4 E2B \(Q4 K M\) selected; ten targeted optimizations reduce end\-to\-end latency from over 30 s to approximately 10 s. Both subsystems are integrated in a production Flutter application with bilingual interface, museum location gating, and text\-to\-speech support.
>
> **📝 中文摘要：**
> TimeLens是大埃及博物馆（GEM）的双语AI移动导览系统。参观者将手机对准展品时，可实时识别文物并选择用英语或阿拉伯语提问互动。该系统重点解决了展厅部署面临的三大挑战：51件展品间高度视觉相似性（包括多尊近乎相同的拉美西斯雕像）、精心标注的训练数据与手持拍摄条件的差异，以及AI导览可能生成无依据历史陈述的风险。

本研究实现两项关键技术突破：
首先，通过数据质量驱动的迭代研究开发了端侧文物检测器——从基础模型自动标注（YOLO-World）、基于空间规则的标签清洗，到最终全人工标注数据集构建，验证了标签质量的决定性作用。最终的YOLOv8n模型成功识别所有先前难以分类的文物，以5.97 MB的TensorFlow Lite模型实现中端手机实时运行（mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.924）。

其次，构建了基于108条ChromaDB知识库的双语检索增强生成（RAG）导览系统，经七种候选语言模型横向评测后选定Gemma 4 E2B（Q4 K M）；通过十项定向优化将端到端延迟从30余秒压缩至约10秒。

两个子系统已集成于生产级Flutter应用，具备双语界面、博物馆地理围栏和文本转语音功能。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.13267v1)

---

> ### 4. YOLO\-AMC: An Improved YOLO Architecture with Attention Mechanisms for Building Crack Detection
> **🔹 中文标题：** YOLO-AMC：一种基于注意力机制的改进YOLO架构用于建筑裂缝检测

摘要：本文提出了一种改进的YOLO架构，称为YOLO-AMC，专门用于建筑裂缝检测。该架构通过引入注意力机制来增强特征提取能力，显著提升了裂缝检测的准确性和鲁棒性。我们设计了一种新颖的注意力模块，能够自适应地增强裂缝区域的特征表达，同时抑制背景干扰。此外，我们还优化了特征融合策略，以实现多尺度特征的有效整合。在多个公开数据集上的实验结果表明，YOLO-AMC在检测精度和效率方面均优于现有方法，为建筑结构健康监测提供了有效的解决方案。
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-11 |
> | 👤 作者 | Ching\-Yu Tsai |
>
> **📄 英文摘要：**
> Crack detection plays an important role in infrastructure inspection and Structural Health Monitoring \(SHM\). However, cracks typically appear as thin, low\-contrast structures and are easily affected by background noise, posing challenges for existing object detection models. This study proposes an improved YOLO\-based architecture with integrated attention mechanisms, termed YOLO\-AMC \(YOLO with Attention Mechanisms for Crack Detection\), to enhance automated crack detection performance. Based on YOLOv11, the original C2PSA module is removed, and multiple attention mechanisms, including Global Attention Mechanism \(GAM\), Residual Convolutional Block Attention Module \(Res\-CBAM\), and Shuffle Attention \(SA\), are introduced into the multi\-scale feature fusion layers of the Neck to strengthen cross\-scale feature integration. Experimental results demonstrate that YOLO\-AMC consistently outperforms baseline models YOLOv11n and YOLOv8n across multiple evaluation metrics. Among the evaluated attention modules, GAM achieves the best detection performance, obtaining mAP@0.5 = 0.9917 and mAP@0.5:0.95 = 0.9506 on the test dataset, which are higher than those of YOLOv11 \(0.9833 / 0.9112\) and YOLOv8 \(0.9707 / 0.8921\). Furthermore, while maintaining a computational complexity of 7.6 GFLOPs, the proposed model achieves 110.95 FPS on an NVIDIA RTX 4090 platform and approximately 5 FPS on a Raspberry Pi 5 edge device, demonstrating a favorable trade\-off between accuracy and deployment efficiency. The implementation code for this study is available on GitHub at https://github.com/CY\-Tsai24/YOLO\-AMC.
>
> **📝 中文摘要：**
> 裂缝检测在基础设施检查与结构健康监测（SHM）中扮演着重要角色。然而，裂缝通常呈现为细薄、低对比度的结构，且易受背景噪声干扰，这给现有目标检测模型带来了挑战。本研究提出了一种集成注意力机制的改进型YOLO架构——YOLO-AMC（面向裂缝检测的注意力机制YOLO），以提升自动化裂缝检测性能。该模型基于YOLOv11架构，移除了原始C2PSA模块，并在颈部网络的多尺度特征融合层中引入了全局注意力机制（GAM）、残差卷积注意力模块（Res-CBAM）和通道重组注意力（SA）等多种注意力机制，以增强跨尺度特征融合能力。

实验结果表明，YOLO-AMC在多项评估指标上均优于基线模型YOLOv11n和YOLOv8n。在所评估的注意力模块中，GAM取得了最佳检测性能：在测试集上实现了mAP@0.5 = 0.9917和mAP@0.5:0.95 = 0.9506，显著高于YOLOv11（0.9833 / 0.9112）和YOLOv8（0.9707 / 0.8921）的性能。此外，该模型在保持7.6 GFLOPs计算复杂度的前提下，在NVIDIA RTX 4090平台上达到110.95 FPS的推理速度，在树莓派5边缘设备上约为5 FPS，展现出精度与部署效率之间的良好平衡。本研究实现代码已开源于GitHub仓库：https://github.com/CY-Tsai24/YOLO-AMC。
>
> **💻 代码链接：** https://github.com/CY-Tsai24/YOLO-AMC.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12958v1)

---

> ### 5. Performance Analysis of YOLOv11 and YOLOv8 for Mixed Traffic Object Detection under Adverse Weather Conditions in Developing Countries
> **🔹 中文标题：** YOLOv11与YOLOv8在发展中国家恶劣天气条件下混合交通目标检测的性能分析
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-10 |
> | 👤 作者 | Quoc Thuan Nguyen |
>
> **📄 英文摘要：**
> In modern vehicular systems, robust performance under harsh conditions has become a critical problem of autonomous driving. Our study delivers a comprehensive evaluation of the newest iteration of the YOLO series, which is YOLOv11 Nano architecture benchmarked against the widely adopted YOLOv8 Nano as a baseline on a custom fused dataset that combines the Indian Driving Dataset \(IDD\) \[1\] and Berkeley Deep Drive Dataset \(BDD100K\) \[2\]. We have analyzed the trade\-offs among detection accuracy, inference speed, and computational efficiency in high\-entropy scenarios involving dense mixed traffic, rain, and low\-light conditions. Specifically, YOLOv11n achieves a mean Average Precision \(mAP@50\) of 46.6%, with a notable 3.2% improvement in Precision over the baseline, effectively reducing false positives in cluttered scenes. Furthermore, the proposed model exhibits enhanced energy efficiency, requiring 22% fewer FLOPs \(6.3G vs. 8.1G\) while maintaining real\-time inference speed of 70.9 FPS on a Tesla T4 GPU, offering an optimal trade\-off for safety\-critical edge deployment.
>
> **📝 中文摘要：**
> 在现代车载系统中，恶劣环境下的鲁棒性能已成为自动驾驶的关键问题。本研究对YOLO系列最新架构YOLOv11 Nano进行了全面评估，以广泛使用的YOLOv8 Nano作为基线模型，在融合了印度驾驶数据集(Indian Driving Dataset, IDD)[1]与伯克利深度驾驶数据集(Berkeley Deep Drive, BDD100K)[2]的定制融合数据集上进行基准测试。我们分析了在包含密集混合交通流、降雨及低光照条件等高熵场景下，检测精度、推理速度与计算效率之间的权衡关系。实验表明：YOLOv11n在mAP@50指标上达到46.6%，较基线模型精度提升3.2%，有效降低了复杂场景中的误检率；该模型计算能效显著优化，在保持70.9 FPS实时推理速度（Tesla T4 GPU）的同时，浮点运算量减少22%（6.3G vs. 8.1G），为安全关键型边缘部署提供了更优的效能平衡方案。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12066v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>