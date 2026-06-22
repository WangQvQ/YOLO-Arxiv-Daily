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
> **🔹 中文标题：** SPARK: 基于关键点的低延迟单摄像头自动驾驶竞速3D姿态估计
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
> 在自主赛车中，为规划安全无碰撞轨迹并应对非协作对手，需快速检测其他参与者的运动状态。激光雷达检测在部署于边缘设备时，其固有速度慢于视觉方法且更难实现，导致检测延迟，在高动态机动场景下限制了目标跟踪性能。采用单目3D检测技术，可在赛道上轻松部署低延迟的其他参与者检测系统。本文提出SPARK——一种基于关键点检测的单相机自主赛车姿态估计算法，该算法实现了高精度远距离检测，性能超越当前最先进的单目相机检测算法，同时保持更低延迟。通过优化的YOLO模型与赛车领域固定几何特征的结合，该算法兼具低延迟与低资源占用特性。我们在真实自主赛车数据上评估了该方法的性能，并与最先进的激光雷达及相机检测算法进行对比。源代码已开源：https://github.com/TUMFTM/SPARK-camera-det
>
> **💻 代码链接：** https://github.com/TUMFTM/SPARK-camera-det
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.17936v1)

---

> ### 2. Budget\-Aware Adaptive Adversarial Patches for Black\-Box Object Detection
> **🔹 中文标题：** 预算感知的自适应对抗性补丁用于黑盒目标检测
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
> 对抗性补丁对现代目标检测器构成了实际威胁。先前研究已揭示其脆弱性，但三个缺口限制了可操作的洞察：（i）鲜有基于**分数的黑盒攻击**能**在严格查询预算内联合优化**补丁的**位置、纹理与尺寸**；（ii）攻击成功率极少与补丁的**视觉足迹**相关联；（iii）评估常将EOT鲁棒性与普通视角下的抑制效果混为一谈。本文提出**方法{}**——一种查询高效、预算自适应的黑盒攻击方法，它将轻量化**上下文汤普森采样**定位器与NES式像素更新相结合，仅在进展停滞时扩大补丁尺寸。评估报告以**严格普通图像**抑制测试为核心；EOT鲁棒性被审计但从未作为成功指标的替代品，可选的外观/可打印性权重则揭示了攻击强度与可见性之间的权衡。在YOLOv5、Faster R-CNN和YOLOS上的实验表明，**方法{}**使用紧凑补丁即可在CNN检测器上实现强抑制效果，在基于Transformer的检测器上也能达到显著抑制，并相对固定尺寸和启发式基准方法展现出清晰的查询量-足迹权衡关系。打印-采集先导实验进一步验证了该方法在未见过的物理对象和视角间的迁移能力。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.18318v1)

---

> ### 3. MOSAIC: Mobile Object Segmentation under Adverse Imaging Conditions for Rapid L\-PBF Keyhole Behavior Characterization
> **🔹 中文标题：** MOSAIC：恶劣成像条件下移动目标分割技术及其在L-PBF匙孔行为快速表征中的应用
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
> 在激光粉末床融合（L-PBF）工艺中，气体与流体相互作用的快速演化过程，使得我们难以有效监控或调控工艺过程，其中不稳定的匙孔形态会导致孔隙形成与飞溅产生。通过高速原位X射线成像技术观测匙孔行为，有助于深入理解这些相互作用对L-PBF工艺监控与调控的影响。MOSAIC（适用于恶劣成像条件的移动目标分割算法）专为光束线实验中匙孔动力学的快速分析而设计，无需耗时的标注工作或模型训练。针对12组不同样品的验证研究表明，MOSAIC具有强鲁棒性：相比人工分割图像，其平均F1值达到0.894，精确率达0.953，性能与SAM和YOLO机器学习方法相当或更优。该算法处理效率显著：在CPU上处理约150×250像素的移动窗口裁剪帧时，处理速度达19.9毫秒/帧，而YOLO和SAM模型的CPU推理速度分别为54毫秒/帧和5284毫秒/帧。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.16186v1)

---

> ### 4. TimeLens: On\-Device Artifact Recognition with Retrieval\-Augmented Question Answering for the Grand Egyptian Museum
> **🔹 中文标题：** 时光透镜：基于检索增强问答的端侧文物识别技术在大埃及博物馆的应用
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
> TimeLens是一款基于人工智能的埃及大博物馆双语移动导览应用。访客将手机对准展品，即可实时识别文物并针对英语或阿拉伯语提问获得解答。本项目聚焦画廊部署的三大特定问题：51件目录文物间的细粒度视觉相似性（许多几乎相同的拉美西斯雕像）、策划训练数据与手持拍摄条件的差异，以及AI导览可能陈述无依据历史事实的风险。报告两项工程贡献：其一，通过数据质量驱动的迭代研究开发了端侧文物检测器——从基础模型自动标注（YOLO-World）经空间标签清洗规则，到全手人工标注数据集——证实标签质量为决定性因素：最终YOLOv8n模型解决了所有先前失效的类别，同时作为仅5.97MB的TensorFlow Lite模型资产，可在中端手机实时运行（mAP@0.5=0.995，mAP@0.5:0.95=0.924）。其二，构建了基于108条记录的ChromaDB知识库的双语检索增强生成导览系统，经七种候选语言模型基准测试选用Gemma 4 E2B（Q4 K M）；十项定向优化将端到端延迟从超30秒降至约10秒。两个子系统集成于具备双语界面、博物馆位置门控及文本转语音功能的生产级Flutter应用中。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.13267v1)

---

> ### 5. YOLO\-AMC: An Improved YOLO Architecture with Attention Mechanisms for Building Crack Detection
> **🔹 中文标题：** YOLO-AMC：一种采用注意力机制的改进型YOLO建筑裂缝检测模型
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
> 裂缝检测在基础设施检查与结构健康监测中扮演重要角色。然而裂缝通常呈现为细长的低对比度结构，且易受背景噪声干扰，这对现有目标检测模型构成了挑战。本研究提出一种改进的YOLO架构，通过集成注意力机制实现裂缝检测增强，命名为YOLO-AMC。该模型基于YOLOv11架构，移除原始C2PSA模块，并在Neck的多尺度特征融合层中引入全局注意力机制、残差卷积块注意力模块和通道混洗注意力机制，以强化跨尺度特征融合能力。实验结果表明，YOLO-AMC在多项评估指标上均优于基线模型YOLOv11n与YOLOv8n。在对比的注意力模块中，全局注意力机制表现最优，在测试集上达到mAP@0.5=0.9917和mAP@0.5:0.95=0.9506，显著超越YOLOv11（0.9833/0.9112）与YOLOv8（0.9707/0.8921）的性能。此外，该模型在保持7.6 GFLOPs计算复杂度的前提下，在NVIDIA RTX 4090平台实现110.95 FPS的推理速度，在Raspberry Pi 5边缘设备上可达约5 FPS，实现了精度与部署效率的良好平衡。研究代码已发布于GitHub：https://github.com/CY-Tsai24/YOLO-AMC。
>
> **💻 代码链接：** https://github.com/CY-Tsai24/YOLO-AMC.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12958v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>