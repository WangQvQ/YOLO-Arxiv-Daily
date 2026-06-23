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
> 在自主赛车场景中，快速检测其他参赛者的运动轨迹对于规划安全、无碰撞路径至关重要，尤其是应对非合作对手时。相较于视觉方案，激光雷达检测本身存在延迟更高、边缘设备部署难度大等问题，导致高动态机动过程中目标跟踪性能受限。采用单目3D检测技术能够实现赛道其他参赛者的低延迟、易部署感知方案。本文提出SPARK——一种基于关键点检测的单目位姿估计算法，专为自主赛车设计。该算法在保持更低延迟的同时，实现了高精度远距离检测，性能超越当前最先进的单目视觉检测算法。通过优化YOLO模型并结合自主赛车场景的固定几何约束，该算法展现出低延迟与低资源消耗的优势。我们在真实自主赛车数据集上评估了该算法性能，并与最先进的激光雷达及视觉检测算法进行对比分析。项目源代码已开源：https://github.com/TUMFTM/SPARK-camera-det
>
> **💻 代码链接：** https://github.com/TUMFTM/SPARK-camera-det
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.17936v1)

---

> ### 2. Budget\-Aware Adaptive Adversarial Patches for Black\-Box Object Detection
> **🔹 中文标题：** 预算自适应对抗性补丁用于黑盒目标检测
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
> 对抗性补丁对现代目标检测器构成切实威胁。现有研究虽已揭示其脆弱性，但三个关键缺口限制了实际洞察：（i）鲜有基于评分的黑箱攻击能在严苛查询预算下，同步优化补丁的位置、纹理与尺寸；（ii）攻击成功率很少与补丁的视觉影响范围关联；（iii）评估常将期望变换鲁棒性与正面视角抑制性混为一谈。本研究提出method{}——一种查询高效、预算自适应的黑箱攻击方法，该方法将轻量化上下文汤普森采样定位器与神经进化策略风格的像素更新机制相结合，仅在进展停滞时扩展补丁尺寸。评估体系以严格正面图像抑制测试为核心基准，对期望变换进行审计但绝不作为成功标准替代，同时提供可选的外观/可打印性权重以揭示强度-可见性权衡关系。在YOLOv5、Faster R-CNN与YOLOS上的实验表明，method{}能利用紧凑补丁对基于卷积神经网络的检测器实现强效抑制，对基于Transformer的检测器也能达到显著抑制效果，相比固定尺寸与启发式基线方法呈现出清晰的查询-影响范围权衡曲线。打印-采集实验进一步验证了该方法在未见物理对象与视角间的迁移能力。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.18318v1)

---

> ### 3. MOSAIC: Mobile Object Segmentation under Adverse Imaging Conditions for Rapid L\-PBF Keyhole Behavior Characterization
> **🔹 中文标题：** MOSAIC：用于快速L-PBF匙孔行为表征的恶劣成像条件下移动物体分割技术
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
> 在激光粉末床熔融（L-PBF）工艺中，气体与流体相互作用的快速演变过程增加了工艺监测与控制的难度，而不稳定的匙孔结构会导致孔隙和飞溅现象。通过对匙孔进行高速原位X射线成像，能够深入理解这些相互作用对L-PBF工艺监测与控制的影响。MOSAIC（Mobile Object Segmentation algorithm for experiments under Adverse Imaging Conditions）是一种面向恶劣成像条件实验的移动目标分割算法，旨在无需耗时的人工标注或模型训练，即可在线束实验过程中对匙孔动力学进行快速分析。在12个独立样品上的验证研究表明，MOSAIC具有优异的鲁棒性：与人工分割图像相比，其平均F1值达到0.894，精度达到0.953，性能与测试的SAM和YOLO机器学习方法持平或更优。该算法效率显著：在CPU上处理约150×250像素的移动窗口裁剪帧时，处理速度达19.9毫秒/帧；相比之下，YOLO与SAM模型在CPU上进行推理分别需要54毫秒和5284毫秒。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.16186v1)

---

> ### 4. TimeLens: On\-Device Artifact Recognition with Retrieval\-Augmented Question Answering for the Grand Egyptian Museum
> **🔹 中文标题：** TimeLens：基于设备端文物识别与检索增强问答系统在大埃及博物馆的应用
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
> TimeLens是为大埃及博物馆开发的AI驱动双语移动导览系统。游客将手机对准展品时，可实时识别文物并获取支持英语或阿拉伯语交互的后续问答服务。该系统专门解决展厅场景中的三大挑战：51件馆藏文物间高度相似的视觉特征（许多拉美西斯王朝雕像几乎完全一致）、策展级训练数据与手持设备拍摄条件之间的差距，以及AI导览提供未经证实的历史信息的风险。研究实现了两项工程突破：第一，通过数据质量驱动的迭代研究开发了端侧文物检测器——从基础模型自动标注（YOLO-World）、空间标签清洗规则到全人工标注数据集——最终确定标签质量是决定性因素：最终部署的YOLOv8n模型可识别所有曾识别失败的类别，同时作为仅5.97MB的TensorFlow Lite模型能在中端手机实时运行（mAP@0.5 = 0.995，mAP@0.5:0.95 = 0.924）。第二，基于包含108条记录的ChromaDB知识库构建双语检索增强生成导览系统，经七种候选语言模型基准测试后选用Gemma 4 E2B（Q4 K M）；通过十项定向优化将端到端延迟从30余秒缩短至约10秒。两个子系统已集成至具备双语界面、博物馆位置围栏及文本转语音功能的Flutter生产应用中。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.13267v1)

---

> ### 5. YOLO\-AMC: An Improved YOLO Architecture with Attention Mechanisms for Building Crack Detection
> **🔹 中文标题：** YOLO-AMC：一种融合注意力机制的改进YOLO架构用于建筑裂缝检测
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
> 裂缝检测在基础设施检测和结构健康监测中发挥着重要作用。然而，裂缝通常呈现为细薄的低对比度结构，且易受背景噪声干扰，这对现有目标检测模型构成了挑战。本研究提出一种基于改进YOLO架构并融合注意力机制的模型，命名为YOLO-AMC（用于裂缝检测的注意力机制增强YOLO模型），以提升裂缝自动化检测性能。基于YOLOv11架构，本研究移除了原有的C2PSA模块，并在颈部网络的多尺度特征融合层中引入了全局注意力机制、残差卷积块注意力模块及通道重排注意力等多种注意力机制，以增强跨尺度特征融合能力。实验结果表明，YOLO-AMC模型在多项评估指标上均优于基准模型YOLOv11n与YOLOv8n。在所评估的注意力模块中，全局注意力机制取得了最佳检测性能，在测试集上获得mAP@0.5=0.9917和mAP@0.5:0.95=0.9506，显著高于YOLOv11（0.9833/0.9112）和YOLOv8（0.9707/0.8921）。此外，该模型在保持7.6 GFLOPs计算复杂度的同时，在NVIDIA RTX 4090平台上达到110.95 FPS的推理速度，在树莓派5边缘设备上亦可实现约5 FPS的运行效率，展现出精度与部署效率的良好平衡。本研究实现代码已开源于GitHub平台：https://github.com/CY-Tsai24/YOLO-AMC。
>
> **💻 代码链接：** https://github.com/CY-Tsai24/YOLO-AMC.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12958v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>