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
> **🔹 中文标题：** SPARK：基于关键点的低延迟单摄像头自动驾驶赛车三维姿态估计
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
> 在自动驾驶赛车领域，需要快速检测其他参与者的运动状态，以便为与非合作对手规划安全无碰撞的轨迹。相比视觉方法，激光雷达检测本身速度较慢且更难部署于边缘设备，这导致在高动态机动过程中检测延迟，从而限制了目标跟踪性能。采用单目三维检测技术，可实现对赛道上其他参与者的易部署、低延迟检测。本文提出SPARK——一种基于关键点检测的单摄像头姿态估计算法。该算法实现了高精度的远距离检测，在保持更低延迟的同时，性能超越了当前最先进的单目摄像头检测算法。通过采用高度优化的YOLO模型并利用自动驾驶赛车领域的固定几何特征，该算法还表现出低延迟和低资源占用的特点。我们在真实自动驾驶赛车数据上评估了该方法的性能，并与最先进的激光雷达和摄像头检测算法进行了对比。源代码已开源：https://github.com/TUMFTM/SPARK-camera-det
>
> **💻 代码链接：** https://github.com/TUMFTM/SPARK-camera-det
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.17936v1)

---

> ### 2. Budget\-Aware Adaptive Adversarial Patches for Black\-Box Object Detection
> **🔹 中文标题：** 预算感知的自适应对抗补丁用于黑盒目标检测
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-16 |
> | 👤 作者 | Pedram MohajerAnsari |
>
> **📄 英文摘要：**
> Adversarial patches pose a practical threat to modern object detectors. Prior work shows vulnerability, but three gaps limit actionable insight: \(i\) few emph\{score\-based black\-box\} attacks emph\{jointly\} optimize patch emph\{location, texture, and size\} under tight query budgets; \(ii\) success is rarely tied to the patch's emph\{visual footprint\}; and \(iii\) evaluations often conflate EOT robustness with plain\-view suppression. We present method\{\}, a query\-efficient, budget\-adaptive black\-box attack that couples a lightweight emph\{Contextual Thompson\-Sampling\} placer with NES\-style pixel updates, growing the patch only when progress stalls. Reporting is anchored by a emph\{strict plain\-image\} suppression test; EOT is audited but never used as a substitute for success, and optional appearance/printability weights expose strength\-\-visibility trade\-offs. Across YOLOv5, Faster R\-CNN, and YOLOS, method\{\} achieves strong suppression on CNN\-based detectors and substantial suppression on the transformer\-based detector, using compact patches and exposing clear query\-\-footprint trade\-offs relative to fixed\-size and heuristic baselines. A print\-\-capture pilot further shows transfer across unseen physical objects and viewpoints.
>
> **📝 中文摘要：**
> 对抗性补丁对现代目标检测器构成实际威胁。现有研究虽已揭示其脆弱性，但三个关键局限制约了应用洞察：（i）极少有基于评分的黑盒攻击在严格查询预算下同步优化补丁的位置、纹理与尺寸；（ii）攻击效果很少与补丁的视觉特征相关联；（iii）评估常将期望过度变换鲁棒性与常规视角抑制效果混为一谈。本文提出method{}——一种查询高效且预算自适应的黑盒攻击方法，通过情境汤普森采样放置器与自然进化策略风格的像素更新相耦合，仅在攻击进展停滞时扩展补丁。评估体系严格锚定常规图像抑制测试，对期望过度变换进行审计但绝不替代攻击成功率指标，并通过可选外观/可打印性权重揭示强度-可见度的权衡关系。在YOLOv5、Faster R-CNN与YOLOS上的实验表明，method{}能对卷积神经网络检测器实现强力抑制，对基于Transformer的检测器产生显著抑制效果，采用紧凑补丁尺寸并展现出相对于固定尺寸基线与启发式基线的清晰查询-特征权衡关系。实物打印-捕获测试进一步验证了该方法在未接触物理对象与视角间的可迁移性。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.18318v1)

---

> ### 3. MOSAIC: Mobile Object Segmentation under Adverse Imaging Conditions for Rapid L\-PBF Keyhole Behavior Characterization
> **🔹 中文标题：** MOSAIC：恶劣成像条件下的移动物体分割与L-PBF匙孔行为快速表征
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
> 在激光粉末床熔融（L-PBF）工艺中，气体与流体相互作用的快速演变使得我们难以有效监控或调控该过程，不稳定的匙孔会导致气孔形成和飞溅现象。通过高速原位X射线成像技术对匙孔进行观测，能够更深入地理解这些相互作用对L-PBF工艺监控与控制的影响。MOSAIC（恶劣成像条件下移动物体分割算法）旨在无需耗时的人工标注或模型训练，即可在光束线实验过程中对匙孔动态进行快速分析。在12个不同样品上进行的验证研究证明了MOSAIC的鲁棒性，与人工分割图像相比，其平均F1分数达0.894，精确率达0.953，性能与SAM和YOLO机器学习方法相当甚至更优。MOSAIC算法具有高效性：在CPU上处理约150×250像素移动窗口裁剪帧的速度为每图像19.9毫秒，而YOLO和SAM模型在CPU上的推理速度分别为每图像54毫秒和5284毫秒。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.16186v1)

---

> ### 4. TimeLens: On\-Device Artifact Recognition with Retrieval\-Augmented Question Answering for the Grand Egyptian Museum
> **🔹 中文标题：** TimeLens：基于设备端检索增强问答的大埃及博物馆文物识别
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
> TimeLens是一款为大埃及博物馆（GEM）打造的人工智能双语移动导览系统。当游客将手机对准展品时，系统可实时识别文物并支持追问，随后以英语或阿拉伯语提供解答。该系统针对展厅部署的三大特殊挑战而设计：51件目录文物间的细粒度视觉相似性（包括多尊近乎一致的拉美西斯雕像）、策划训练数据与手持设备拍摄条件的差异、以及AI导览可能陈述未经证实历史事实的风险。本研究报道了两项工程贡献：首先，通过数据质量驱动的迭代研究——从基础模型自动标注（YOLO-World），到基于空间规则的标签清洗，最终建立全手工标注数据集——开发了文物检测器。研究最终表明标签质量是决定性因素：最终的YOLOv8n模型成功识别所有此前失败类别，同时保持仅5.97MB的TensorFlow Lite体积，可在中端手机实时运行（mAP@0.5=0.995，mAP@0.5:0.95=0.924）。其次，基于108条记录的ChromaDB知识库构建了双语检索增强生成（RAG）导览系统，在七种候选语言模型中通过基准测试选定Gemma 4 E2B（Q4 K M），并实施十项针对性优化将端到端延迟从超30秒降至约10秒。两大子系统已集成至生产级Flutter应用，具备双语界面、博物馆位置限定及文本转语音功能。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.13267v1)

---

> ### 5. YOLO\-AMC: An Improved YOLO Architecture with Attention Mechanisms for Building Crack Detection
> **🔹 中文标题：** YOLO-AMC：基于注意力机制的改进YOLO架构及其在建筑裂缝检测中的应用
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
> 裂缝检测在基础设施巡检与结构健康监测中具有重要作用。然而裂缝通常呈现为细长、低对比度的结构特征，且易受背景噪声干扰，这对现有目标检测模型提出了挑战。本研究提出一种融合注意力机制的改进型YOLO架构，命名为YOLO-AMC（基于注意力机制的裂缝检测YOLO模型），以提升自动化裂缝检测性能。该模型基于YOLOv11架构，移除了原有的C2PSA模块，并在Neck部分的多尺度特征融合层中引入全局注意力机制、残差卷积注意力模块和洗牌注意力等多种注意力机制，以增强跨尺度特征整合能力。实验结果表明，YOLO-AMC在多项评估指标上均优于基线模型YOLOv11n和YOLOv8n。在所评估的注意力模块中，全局注意力机制取得最佳检测性能，在测试集上获得mAP@0.5 = 0.9917和mAP@0.5:0.95 = 0.9506，分别高于YOLOv11（0.9833/0.9112）和YOLOv8（0.9707/0.8921）。此外，该模型在保持7.6 GFLOPs计算复杂度的同时，在NVIDIA RTX 4090平台上达到110.95 FPS，在Raspberry Pi 5边缘设备上实现约5 FPS，展现出精度与部署效率的良好平衡。本研究的实现代码已在GitHub平台开源：https://github.com/CY-Tsai24/YOLO-AMC
>
> **💻 代码链接：** https://github.com/CY-Tsai24/YOLO-AMC.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12958v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>