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
> **🔹 中文标题：** SPARK：基于关键点的自动驾驶赛车低延迟单摄像头3D姿态估计
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
> 在自主赛车领域，需要快速检测其他参赛者的运动轨迹，以便与非合作对手规划安全无碰撞的行驶路径。相比视觉方法，激光雷达检测本质上更慢且在边缘设备上更难部署，其检测延迟会限制高动态机动过程中的目标跟踪性能。利用单目3D检测能够实现赛道上其他参赛者易于部署、低延迟的检测。我们提出SPARK——一种基于关键点检测的自主赛车单相机位姿估计算法。该算法实现了高精度的远距离检测，不仅超出现有单目相机检测算法的性能，同时保持更低的延迟。通过采用高度优化的YOLO模型，并利用自主赛车领域的固定几何特性，该算法还展现出低延迟和低资源占用的特点。我们在真实自主赛车数据上评估了本方法的性能，并与现有的激光雷达和相机检测算法进行对比。源代码获取地址：https://github.com/TUMFTM/SPARK-camera-det
>
> **💻 代码链接：** https://github.com/TUMFTM/SPARK-camera-det
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.17936v1)

---

> ### 2. Budget\-Aware Adaptive Adversarial Patches for Black\-Box Object Detection
> **🔹 中文标题：** 预算感知型自适应对抗补丁：面向黑盒目标检测
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
> 对抗性补丁对现代目标检测器构成了实际威胁。现有研究虽已揭示其脆弱性，但三个关键局限制约了实际应用洞察：其一，极少有基于评分的黑盒攻击能在严苛查询预算下协同优化补丁的**位置、纹理与尺寸**；其二，攻击成功率很少与补丁的**视觉覆盖范围**建立关联；其三，评估常将EOT鲁棒性与直接视角抑制效果混为一谈。我们提出**方法{}**——一种查询高效、预算自适应的黑盒攻击框架，通过轻量级**上下文汤普森采样**定位器与NES风格像素更新机制协同工作，仅在进展停滞时扩大补丁规模。评估体系以**严格直接图像抑制测试**为核心；EOT验证仅作为补充手段而非替代指标，可选的外观/可印刷性权重可直观展现强度-可见性权衡关系。在YOLOv5、Faster R-CNN及YOLOS测试中，该方法对基于CNN的检测器实现强力抑制，对基于Transformer的检测器达成显著抑制，采用紧凑补丁设计，并相较固定尺寸与启发式基线展现出清晰的查询量-覆盖范围权衡。打印-捕获先导实验进一步验证了该方法在未见过的物理对象与多视角间的迁移能力。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.18318v1)

---

> ### 3. MOSAIC: Mobile Object Segmentation under Adverse Imaging Conditions for Rapid L\-PBF Keyhole Behavior Characterization
> **🔹 中文标题：** MOSAIC: 恶劣成像条件下的移动目标分割技术及其在L-PBF匙孔行为快速表征中的应用
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
> 在激光粉末床熔融（L-PBF）工艺中，气体与流体相互作用的快速演变增加了工艺监测与控制的难度，不稳定的匙孔会导致孔隙和飞溅形成。通过对匙孔进行高速原位X射线成像，可深入理解这些相互作用对L-PBF工艺监测与控制的影响。MOSAIC（恶劣成像条件下移动物体分割算法）专为光束线实验中匙孔动态的快速分析设计，无需耗时的手动标注或模型训练。在12个独特样本上的验证研究表明MOSAIC具有鲁棒性，与人工分割图像相比，平均F1分数达0.894，精确度为0.953，性能达到或超过测试的SAM和YOLO机器学习方法。该算法处理效率高，在CPU上分析约150×250像素的移动窗口裁剪图像时，处理速度达19.9毫秒/帧，而YOLO和SAM模型在CPU上的推理速度分别为54毫秒/帧和5284毫秒/帧。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.16186v1)

---

> ### 4. TimeLens: On\-Device Artifact Recognition with Retrieval\-Augmented Question Answering for the Grand Egyptian Museum
> **🔹 中文标题：** 《时间之镜：面向大埃及博物馆的端侧文物识别与检索增强问答系统》
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
> TimeLens是一款面向大埃及博物馆的AI双语移动导览系统。游客将手机对准展品时，可实时识别文物并用英语或阿拉伯语提出后续问题。该系统针对展厅部署的三大挑战展开研究：51件展品（包括多尊高度相似的拉美西斯时期雕像）间的视觉细粒度相似性、策划训练数据与手持设备拍摄条件的差异，以及AI导览可能生成无史料依据的历史叙述的风险。

系统取得两项工程突破：其一，通过数据质量驱动的迭代实验开发端侧文物检测器——从基础模型自动标注（YOLO-World）到空间标签清洗规则，最终构建全手工标注数据集，证实标签质量是决定性因素。最终的YOLOv8n模型成功解决所有此前识别失败的类别，仅需5.97MB TensorFlow Lite资源即可在中端手机实时运行（mAP@0.5=0.995，mAP@0.5:0.95=0.924）。其二，基于108条ChromaDB知识库构建双语检索增强生成导览系统，经七种候选语言模型评估后选用Gemma 4 E2B（Q4 K M）；十项针对性优化将端到端延迟从30余秒降至约10秒。

两个子系统已集成至生产级Flutter应用，具备双语界面、博物馆地理围栏及文本转语音功能。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.13267v1)

---

> ### 5. YOLO\-AMC: An Improved YOLO Architecture with Attention Mechanisms for Building Crack Detection
> **🔹 中文标题：** YOLO-AMC：一种改进的YOLO架构与注意力机制在建筑裂缝检测中的应用
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
> 裂缝检测在基础设施检查与结构健康监测中具有重要作用。然而裂缝通常呈现为纤细的低对比度结构，且易受背景噪声干扰，这对现有目标检测模型构成了挑战。本研究提出一种融合注意力机制的改进型YOLO架构，命名为YOLO-AMC（基于注意力机制的裂缝检测YOLO模型），以提升裂缝自动化检测性能。该模型基于YOLOv11架构，移除原有C2PSA模块，并在颈部多尺度特征融合层中引入全局注意力机制（GAM）、残差卷积块注意力模块（Res-CBAM）和通道混洗注意力（SA）等多种注意力机制，以增强跨尺度特征融合能力。

实验结果表明，YOLO-AMC在多项评估指标上均优于基线模型YOLOv11n与YOLOv8n。其中采用全局注意力机制的模型检测性能最优，在测试集上达到mAP@0.5 = 0.9917和mAP@0.5:0.95 = 0.9506，优于YOLOv11（0.9833 / 0.9112）和YOLOv8（0.9707 / 0.8921）的表现。该模型在保持7.6 GFLOPs计算复杂度的同时，在NVIDIA RTX 4090平台上达到110.95 FPS，在树莓派5边缘设备上实现约5 FPS的推理速度，展现出精度与部署效率的良好平衡。本研究实现代码已开源至GitHub仓库：https://github.com/CY-Tsai24/YOLO-AMC。
>
> **💻 代码链接：** https://github.com/CY-Tsai24/YOLO-AMC.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12958v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>