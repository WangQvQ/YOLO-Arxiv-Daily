<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Evaluating Vision\-Language Models as a Zero\-Shot Learning Alternative to You Only Look Once and Optical Character Recognition for Nigerian License Plate Recognition
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-02 |
> | 👤 作者 | Ismail Ismail Tijjani |
>
> **📄 英文摘要：**
> License Plate Recognition \(LPR\) systems are critical tools in traffic monitoring, security enforcement, and urban mobility management. Traditional LPR systems often rely on a multi\-stage pipeline involving object detection using You Only Look Once \(YOLO\) and Optical Character Recognition \(OCR\), which suffer from limitations such as high resource demands, poor performance in unstructured environments, and the need for large annotated datasets. This study explores the potential of Vision\-Language Models \(VLMs\) as a unified, zeroshot learning solution for Nigerian license plate recognition. Using a curated dataset of 88 challenging real\-world images collected in Nigeria, we evaluate five selected VLMs: Gemini 2.0 Flash Exp \(Google DeepMind\), Qwen2.5\-VL\-7B\-Instruct \(Alibaba\), GPT\-4o \(OpenAI\), Claude 4 Sonnet \(Anthropic\), and Llama 3.2 Vision 90b \(Meta\). Results based on Character Error Rate \(CER\) reveal that Gemini and Qwen significantly outperform other models in both accuracy and robustness, on the challenging image scenarios. This work highlights the practical advantages of VLMs over YOLO\+OCR, questions the claims by model providers, and compares the performances of the VLMs.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.02025v1)

---

> ### 2. Computer Vision for Wildlife Monitoring: Detecting Brown Howler Monkeys using YOLO
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-01 |
> | 👤 作者 | Gabriel Ferri Schneider |
>
> **📄 英文摘要：**
> Urban expansion threatens global biodiversity, especially affecting arboreal species due to the fragmentation of forest habitats. The movement of arboreal species across disjointed forest patches increases mortality risk and, thus, compromises their conservation. In this context, the installation of canopy bridges can be a viable strategy; yet continuous monitoring of their use by arboreal species is essential for ensuring their effectiveness, typically carried out with the aid of camera traps. However, this method often produces false\-positive images that demand time from conservationists for review. In this context, computer vision algorithms can optimize the task of detecting target species using the canopy bridges. In this study, we explored the automatic detection of brown howler monkeys \(Alouatta guariba\) in videos obtained by camera traps. Given the need for a large number of annotated images of the target animals to train the algorithms, we tested the incorporation of auxiliary data to improve detection models, fine\-tuning the YOLOv10 framework using varying proportions of them. The improvement of these automatic detection techniques contributes to conservation efforts, by providing automatic tools to monitor solutions that minimize the impact of human interference in animals habitats.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.01396v1)

---

> ### 3. Image\-Domain Tilt Constrained Distributed Fusion for Maneuvering UAV Tracking with Multi\-Camera Electro\-Optical Observations
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-01 |
> | 👤 作者 | Minxing Sun |
>
> **📄 英文摘要：**
> Short\-horizon prediction is essential for electro\-optical UAV tracking, especially when the target is small, maneuvering, or intermittently observed. Image center, line\-of\-sight, and range measurements provide direct constraints on target position, but their constraints on acceleration are weak. As a result, prediction can lag during aggressive maneuvers.   This paper proposes an image\-domain tilt constrained distributed fusion method for maneuvering UAV tracking. The method uses the apparent roll and pitch of a rotorcraft target in the image as low\-level maneuver cues. A weak\-prior auto\-labeling pipeline first generates oriented bounding box and image\-domain tilt labels from synchronized video, gimbal IMU, and UAV IMU data. A YOLO\-OBB detector is then trained to provide online target position and tilt measurements. The front\-end Python implementation is publicly available at github.com/ShineMinxing/PythonYOLO.   In the fusion stage, the UAV state is modeled by position, velocity, and acceleration. Image\-domain roll and pitch are introduced as acceleration\-related pseudo\-observations. For distributed tracking, one mobile gimbal camera and two fixed ground cameras are fused asynchronously. Camera attitude error states are augmented into the filter to absorb extrinsic drift and cross\-camera systematic inconsistency. A Mahalanobis gate with time\-since\-last\-valid covariance widening is used to reject false detections and handle dropouts.   In simulation, adding roll/pitch observations reduces the prediction RMSE from 1.991 m to 0.821 m and decreases the cumulative prediction error by 60.75%. In real distributed experiments, a self\-consistency evaluation shows an 18.10% reduction in cumulative prediction error. The results show that image\-domain tilt can provide useful acceleration constraints for robust short\-horizon UAV prediction.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.01008v1)

---

> ### 4. Semantic\-Guided Reading Order Reconstruction in Historical Armenian Newspapers with LLMs
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-01 |
> | 👤 作者 | Chahan Vidal\-Gorène |
>
> **📄 英文摘要：**
> This paper addresses reading order reconstruction in historical Armenian newspapers, which combine complex layouts with limited language resources. We introduce a new annotated dataset of 66 pages and compare geometric heuristics, YOLO\-based layout parsing, an end\-to\-end document model ECLAIR, and a hybrid method combining semantic zone detection with a generative LLM. Our hybrid method achieves the lowest error rates of all evaluated approaches, reducing ordering errors by up to 76% over the strongest geometric baseline, and remains robust in multi\-page settings and under noisy OCR. Rather than targeting production the method is designed as a data bootstrapping strategy enabling rapid annotation in highly under\-resourced scenarios. Alongside the dataset, we release a specialized Tesseract OCR model for historical Armenian print.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.00596v1)

---

> ### 5. Classroom Behavior Monitoring with YOLO An Empirical Study in Higher Education Settings
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-06-30 |
> | 👤 作者 | Sinh Vu Trong |
>
> **📄 英文摘要：**
> Classroom behavior monitoring plays a vital role in evaluating student engagement and improving teaching effectiveness. Traditional observation methods remain subjective and lack scalability. This study introduces a real\-world dataset of classroom videos collected at the Banking Academy of Vietnam \(BAV\-Classroom dataset\), annotated with nine distinctive behavioral categories. State\-of\-the\-art Computer Vision models were evaluated and compared, with YOLOv11 achieving the best performance. Experimental results indicate that students' concentration often decreases notably during the final part of lectures, highlighting challenges in sustaining engagement. Our findings demonstrate the feasibility of applying computer vision for automated classroom monitoring, providing valuable insights for academic quality management.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.02580v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>