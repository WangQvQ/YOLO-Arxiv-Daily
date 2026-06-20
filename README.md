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
> **🔹 中文标题：** SPARK: 基于关键点的自主赛车低延迟单摄像头3D姿态估计
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
> 在自动驾驶赛车中，需要快速检测其他参与者的运动，以便在面对非协作对手时规划安全无碰撞的轨迹。与视觉方法相比，激光雷达检测本质上速度更慢且更难部署在边缘设备上，这会导致检测延迟，从而在高动态机动过程中限制目标跟踪性能。利用单目3D检测可以实现对赛道上其他参与者的易部署、低延迟检测。我们提出了SPARK，一种基于关键点检测的自动驾驶赛车单相机位姿估计算法。它实现了高精度的远距离检测，性能超越了最先进的单目相机检测算法，同时保持较低的延迟。通过采用经过精心优化的YOLO模型并利用自动驾驶赛车领域的固定几何特性，该算法还表现出低延迟和低资源占用。我们在真实的自动驾驶赛车数据上评估了所提方法的性能，并将其与最先进的激光雷达和相机检测算法进行了比较。源代码可在以下地址获取：https://github.com/TUMFTM/SPARK-camera-det
>
> **💻 代码链接：** https://github.com/TUMFTM/SPARK-camera-det
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.17936v1)

---

> ### 2. Budget\-Aware Adaptive Adversarial Patches for Black\-Box Object Detection
> **🔹 中文标题：** 预算感知型自适应对抗性补丁用于黑盒目标检测
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
> 对抗性补丁对现代目标检测器构成实际威胁。现有研究虽揭示了相关漏洞，但存在三方面局限：其一，少有基于评分的黑盒攻击能在严格查询预算内同步优化补丁的位置、纹理与尺寸；其二，攻击效果很少与补丁的视觉覆盖范围明确关联；其三，评估常将期望变换优化鲁棒性与普通视角抑制效果混为一谈。我们提出**方法{}**——一种查询高效、预算自适应的黑盒攻击框架，通过轻量化**上下文汤普森采样**定位器与自然进化策略式像素更新相结合，仅在进展停滞时扩大补丁尺寸。评估体系以**严格普通图像抑制测试**为核心，对期望变换优化进行审计但不以其替代攻击成功指标，同时提供可选的外观/可打印性权重以揭示强度-可见性权衡。在YOLOv5、Faster R-CNN及YOLOS上的实验表明，**方法{}**能对基于卷积神经网络的检测器实现强抑制效果，对基于Transformer的检测器实现显著抑制，其紧凑补丁在查询次数-覆盖范围权衡上明显优于固定尺寸与启发式基线方法。打印捕获先导实验进一步验证了该方法在未见物理对象与视角间的可迁移性。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.18318v1)

---

> ### 3. MOSAIC: Mobile Object Segmentation under Adverse Imaging Conditions for Rapid L\-PBF Keyhole Behavior Characterization
> **🔹 中文标题：** MOSAIC：面向恶劣成像条件的移动目标分割方法及其在激光粉末床熔融匙孔行为快速表征中的应用
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
> 在激光粉末床熔融（L-PBF）过程中，气体与流体相互作用的快速演变增加了工艺监控与控制的复杂性，不稳定的匙孔形态易导致气孔和飞溅产生。通过匙孔的高速原位X射线成像，能够深入理解这些相互作用对L-PBF工艺监控与控制的影响。MOSAIC（恶劣成像条件下移动目标分割算法）专为动态实验设计，可在无需耗时人工标注或模型训练的情况下，实现光束线实验中匙孔动力学的快速分析。经12组独特样品验证，MOSAIC展现出优异鲁棒性：与人工分割图像相比，平均F1分数达0.894，精确率达0.953，性能等同或优于测试的SAM与YOLO机器学习方法。该算法效率突出，在CPU平台处理约150×250像素移动窗口裁剪帧时，单帧处理仅需19.9毫秒，而YOLO与SAM模型的CPU推理单帧耗时分别为54毫秒与5284毫秒。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.16186v1)

---

> ### 4. TimeLens: On\-Device Artifact Recognition with Retrieval\-Augmented Question Answering for the Grand Egyptian Museum
> **🔹 中文标题：** TimeLens：基于检索增强问答技术的移动端大埃及博物馆文物识别系统
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
> TimeLens是一款基于人工智能的双语移动导览应用，专为大埃及博物馆（GEM）设计。访客只需将手机对准展品，即可实时识别文物并获取英语或阿拉伯语的提问解答。本项目针对展厅部署的三大核心挑战展开研究：51件馆藏文物的细粒度视觉相似性问题（其中包含大量近乎一致的拉美西斯雕像）、策展训练数据与手持相机拍摄条件的差异，以及AI导览可能传播未经证实的历史信息风险。

研究实现了两项工程技术突破：其一，通过数据质量驱动的迭代实验开发了设备端文物检测器——从基础模型自动标注（YOLO-World）开始，经过空间标签清洗规则处理，最终构建全人工标注数据集，证实标签质量是决定性因素。最终优化的YOLOv8n模型（仅5.97MB TensorFlow Lite格式）在中端手机上实现实时运行，完全解决了所有原有检测难点（mAP@0.5=0.995，mAP@0.5:0.95=0.924）。其二，构建了基于108条记录的ChromaDB知识库的双语检索增强生成（RAG）导览系统，通过七种候选语言模型基准测试选定Gemma 4 E2B（Q4 K M），经十项定向优化将端到端延迟从30余秒压缩至10秒左右。

两个子系统已集成至生产环境Flutter应用，支持双语界面、博物馆区域定位与文本转语音功能。
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.13267v1)

---

> ### 5. YOLO\-AMC: An Improved YOLO Architecture with Attention Mechanisms for Building Crack Detection
> **🔹 中文标题：** YOLO-AMC：一种用于建筑裂缝检测的、基于注意力机制的改进YOLO架构
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
> 裂缝检测在基础设施检测和结构健康监测中发挥着重要作用。然而裂缝通常呈现为细薄、低对比度结构，且易受背景噪声干扰，这对现有目标检测模型构成了挑战。本研究提出一种融合注意力机制的改进YOLO架构——YOLO-AMC（基于注意力机制的裂缝检测YOLO），以提升裂缝自动化检测性能。该模型基于YOLOv11架构，移除原有C2PSA模块，并在Neck部分的多尺度特征融合层中引入全局注意力机制、残差卷积注意力模块及洗牌注意力机制，从而增强跨尺度特征整合能力。实验结果表明，YOLO-AMC在多项评估指标上均优于基线模型YOLOv11n与YOLOv8n。在所测试的注意力模块中，全局注意力机制取得最优检测性能，在测试集上实现mAP@0.5=0.9917和mAP@0.5:0.95=0.9506，较YOLOv11（0.9833/0.9112）和YOLOv8（0.9707/0.8921）显著提升。此外，该模型在保持7.6 GFLOPs计算复杂度的前提下，在NVIDIA RTX 4090平台上达到110.95 FPS，在树莓派5边缘设备上实现约5 FPS的推理速度，展现出精度与部署效率的良好平衡。本研究实现代码已发布于GitHub平台：https://github.com/CY-Tsai24/YOLO-AMC。
>
> **💻 代码链接：** https://github.com/CY-Tsai24/YOLO-AMC.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2606.12958v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>