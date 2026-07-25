<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Synthetic data generation framework for quality control automation in gravure printing
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-23 |
> | 👤 作者 | Korota Arsène Coulibaly |
>
> **📄 英文摘要：**
> Quality control in printing, particularly in rotogravure printing, still depends on slow, costly, and subjective manual inspection. Automated surface defect detection is critical for maintaining high\-quality standards in rotogravure printing. Deep learning models give prospects for automation. However, training robust deep learning models, such as YOLO or Vision Transformers, is heavily hindered by the extreme scarcity of real\-world industrial defects images. To overcome this limitation, this paper introduces a novel synthetic data generation framework tailored for rotogravure printing quality control. The proposed pipeline automatically generates high\-fidelity images of specific printing defects \(creases, streaks, misregistration, etc.\) and outputs corresponding bounding boxes and annotations. To validate the framework, a synthetic dataset of 7533 images was generated and used to train the state\-of\-the\-art object\-detection model RFDETR. Experimental results demonstrate that the model trained on our synthetic data achieves a Mean Average Precision \(mAP\) of 80.9% on real industrial testing samples. This framework provides a zero\-cost, rapid\-deployment solution for automating defect inspection in printing lines without requiring massive manual data collection.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.21577v1)

---

> ### 2. Real\-Time EEG Cap Electrode Detection for Guided Point\-of\-Care Placement
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-22 |
> | 👤 作者 | William Lehn\-Schiøler |
>
> **📄 英文摘要：**
> We present a two\-stage vision system that detects EEG cap electrodes in a live webcam stream and validates their anatomical placement in real time. A single\-class YOLO detector localises electrodes; a geometric stage assigns each detection to a named 10\-20 role from facial landmarks. Evaluating under subject\-disjoint leave\-one\-subject\-out \(LOSO\) cross\-validation across five subjects wearing the clinically\-validated Small/Medium/Large caps, the detector attains mAP@.5 = 0.94 \+/\- 0.07 across five held\-out folds \(0.96 pooled\). A dedicated leave\-one\-cap\-out axis, holding out every frame of a cap regardless of subject, leaves Medium and Large mAP@.5 within 0.01 of LOSO \(0.97, 0.97\) while Small drops to 0.72 \+/\- 0.28, a gap confounded with subject familiarity rather than cap style. Geometric augmentation \(rotation, perspective, mixup\) improves in\-plane\-roll robustness and temporal\-electrode recall at no inference cost, and a landmark\-driven head crop extends the usable distance range, lifting mAP@.5 from 0.23 to 0.45 at 0.6 x apparent scale. A compact mobile\-candidate backbone \(YOLOv10n\) keeps the detector at real\-time throughput \(19 FPS\) on a commodity CPU at 640 px.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.20142v1)

---

> ### 3. TargetFinder: Detecting Widgets from Pixels on Desktop Interfaces
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-22 |
> | 👤 作者 | Ahmed Ben Akouche |
>
> **📄 英文摘要：**
> ''Target\-aware'' pointing techniques, like Bubble Cursor or Semantic Pointing, outperform traditional pointing by leveraging knowledge of target locations. Yet the lack of application\-agnostic widget geometry information limits their adoption across the desktop. We present TargetFinder, a computer vision\-based system for real\-time detection of GUI widgets. TargetFinder leverages several fine\-tuned YOLO networks trained on a new dataset of 520 annotated desktop screenshots \(~38,000 annotations\) spanning Windows, macOS, Ubuntu, and web interfaces. TargetFinder uses lightweight screen monitoring and low\-latency detection, achieving millisecond responsiveness suitable for interactive use. Evaluations show that TargetFinder outperforms the baseline methods \(OmniParser and REMAUI\), while system\-wide implementations of Bubble Cursor and Semantic Pointing demonstrate the feasibility of deploying universal target\-aware techniques that work across applications. We release the dataset, models, annotation tool, and an open\-source library for research and applications.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.19907v1)

---

> ### 4. PRISM\-DR: Per\-lesion Retinal Inference with Specialist Models for Diabetic Retinopathy
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-22 |
> | 👤 作者 | Zübeyr Özeren |
>
> **📄 英文摘要：**
> Diabetic retinopathy is a leading cause of preventable blindness; its early lesions are small, low contrast, and easily missed in manual screening. Most automated detectors handle the four non\-proliferative DR lesions: microaneurysms, hemorrhages, hard exudates, and soft exudates, with a single multi\-class model, even though these lesions differ sharply in size, color, morphology, and prevalence, so a shared model favors common, easy classes over rare, difficult ones. We present PRISM\-DR, a lesion\-specific pipeline that trains one single\-class detector per lesion, each with its own configuration. From a raw fundus image, the pipeline applies region of interest cropping, fundus\-specific preprocessing, four parallel YOLO detectors, tiling, per\-lesion ensembling of five cross\-validation folds, and an inter\-lesion suppression step that resolves overlaps by physical lesion size and clinical priority rather than confidence. Per lesion, the best of five YOLO generations is selected, and augmentation is tuned by Bayesian optimization. Trained on IDRiD with stratified five\-fold cross\-validation, the system reaches a test mAP50 of 0.527 and F1 of 0.529, highest AP50 on hard exudates with 0.561. Without fine\-tuning, the models transfer well where the imaging scale is close to IDRiD and degrade as field of view and resolution depart. These modest absolute results reflect a small single\-source training set and a difficult task; however, treating each lesion as a separate detection problem is a practical alternative to a single multi\-class model.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.19864v1)

---

> ### 5. Optimization of sim\-to\-real transfer in the humanoid robot NICO
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-20 |
> | 👤 作者 | Juraj Gavura |
>
> **📄 英文摘要：**
> Robotic grasping requires accurate coordination between visual perception, object localization, inverse kinematics, and hand control. However, when movements planned in simulation are executed on a physical robot, the sim\-to\-real gap can cause small positioning errors that prevent successful grasping. In our previous work, we introduced a low\-cost haptic calibration method that improved 2D reaching accuracy of the humanoid robot NICO. In this paper, we extend this approach from reaching to tabletop object grasping by adding YOLO\-based object and hand detection, stereo vision\-based localization using the robot's built\-in low\-resolution fisheye cameras, and task\-specific corrections for grasp execution. Together, these components form a novel calibration\-based grasping pipeline that does not require RGB\-D cameras, motion capture, or external tracking systems. We also implemented a visual feedback model that aligns the robot hand with the detected object before grasping. Our results show that the fully nonlinear calibration model achieved the best performance inside the calibrated area, while the visual feedback model achieved the highest overall grasping success across the full tabletop workspace.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.18210v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>