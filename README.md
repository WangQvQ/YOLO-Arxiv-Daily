<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Optimization of sim\-to\-real transfer in the humanoid robot NICO
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

> ### 2. Toward Optimal Adenovirus Detection Using YOLO26
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-20 |
> | 👤 作者 | Olivier Rukundo |
>
> **📄 英文摘要：**
> This study systematically benchmarks different data augmentation setups across YOLO26 model size variants to determine the most effective setup for adenovirus detection in TEM images. The benchmarked setups include NAS, GAS, GMAS and DAS, all evaluated under identical training conditions. The adenovirus dataset, selected from the published TEM virus dataset, was re\-annotated by leveraging adenovirus particle positions to generate YOLO\-compatible bounding box annotations. The experimental results demonstrated the impact of the benchmarked data augmentation setups on adenovirus detection with YOLO26 and indicated the most effective data augmentation setup.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.17799v1)

---

> ### 3. Attention from Above: A Multimodal Model for Drone\-Based Object Localization
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-20 |
> | 👤 作者 | Hyun\-Ki Jung |
>
> **📄 英文摘要：**
> Drone\-based object detection technology has advanced rapidly, becoming increasingly sophisticated and efficient. Recently, research trends have expanded beyond the detection of predefined objects toward the identification of specified target objects. For example, desired targets can be specified through textual prompts, enabling accurate detection of objects of interest. To address this demand, this paper proposes an efficient multimodal\-based object detection model aimed at improving small object detection performance. The proposed method is built upon the YOLO\-World framework and replaces the C2f layers used in the YOLOv8 backbone with attention\-based A2C2f layers. This modification enables more precise representation of local features, particularly for small objects or objects with well\-defined boundaries. In addition, the incorporation of attention mechanisms and parallel processing structures significantly enhances the model's computational accuracy. Comparative experiments conducted on the VisDrone dataset demonstrate that the proposed model outperforms the original YOLO\-World model. Specifically, precision increases from 43.0% to 45.1%, recall from 32.8% to 35.0%, the F1 score from 37.2% to 39.4%, mAP@0.5 from 32.5% to 35.2%, and mAP@0.5\-0.95 from 18.5% to 19.9%, confirming a substantial improvement in detection accuracy. These results verify that the proposed approach provides an effective and highly accurate solution for object detection in drone\-based image and video application environments.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.17669v1)

---

> ### 4. AdvSerial: Physical Adversarial Attacks on Infrastructure\-mounted Pedestrian Detectors via Semantic Feature Suppression
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-19 |
> | 👤 作者 | Yuanhao Huang |
>
> **📄 英文摘要：**
> AI\-based visual perception systems are increasingly deployed in infrastructure surveillance, including roadside monitoring units, highway cameras, and smart\-city pedestrian management systems. The security vulnerability of these systems to physical adversarial attacks poses a direct threat to the reliable operation of transportation infrastructure. We propose AdvSerial, a dynamic 2D\-\-3D joint optimization framework for generating continuous high\-angle physical adversarial patches against pedestrian detectors in infrastructure\-based scenarios. We UV\-map a boundary\-aware quilted texture onto 3D garments, combine 2D digital attacks with 3D sparse\- and continuous\-frame rendering, and explicitly suppress person\-specific semantic features while enforcing temporal continuity. A Feature Smooth Quilting strategy reduces visible patch boundaries and bounds cross\-seam feature discontinuities. A serial\-frame loss encourages long uninterrupted sequences of detection failures. In physical world experiments, AdvSerial achieves a 74.8% attack success rate on YOLO\-v5 and degrades mean detection confidence from 84.30% to 39.38%. Experiments spanning eight detectors with different architectures demonstrate strong transferability. Notably, it achieves an $89.71%$ attack success rate on YOLO\-v2 and resists both patch\-detection defenses \(NapGuard\) and 3D\-temporal perception \(Sparse4D\-v3\). The results reveal persistent, temporally consistent failure modes under high\-angle surveillance, and motivate the design of motion\-aware and 3D\-aware defenses for security\-critical infrastructure deployments.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.17069v1)

---

> ### 5. AEGIS: Assay\-Aware Protocol Validation and Runtime Monitoring for Open\-Source Liquid Handling Robots
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-17 |
> | 👤 作者 | Priyanka V. Setty |
>
> **📄 英文摘要：**
> Self\-driving laboratories increasingly rely on low\-cost liquid handlers such as the Opentrons OT\-2, which ship without the pressure\-based aspiration monitoring of Hamilton or Tecan systems and are typically run open\-loop. Two failure modes go undetected: protocols that are syntactically valid but violate assay\-specific invariants \(e.g., tip reuse between a PCR template and a no\-template control\), and physical execution failures \(partial dispense, air bubbles, missing tips\) at runtime. We present AEGIS, a two\-layer guardian for both. Layer 1 pairs a curated machine\-readable assay rule database with an LLM that reasons over OT\-2 Python code, reaching an adjusted F1 of 0.97 on a 24\-protocol benchmark across five assay families and beating rules\-only and LLM\-only ablations across five backends; a free open\-weight model ties the best proprietary one, so no paid API is required. Layer 2 fits a PCA world model to YOLO\-cropped four\-frame pipette trajectories; under a leakage\-free leave\-one\-plate\-out evaluation it reaches average precision 0.89 and operating\-point F1 0.71 \(AUROC 0.80\), a deployment\-faithful number that matches the live demonstration, and we characterize the small\-pipette \(p20\) resolution limit \(F1 0.47\). A live demonstration on a physical OT\-2 \(five replicates per condition\) catches planted no\-tip failures deterministically and partial dispense on coloured dyes, with an always\-VLM self\-vote gate lifting partial\-dispense recall to 5/5; transparent water is a principled limit of any front\-view\-only monitor, which AEGIS surfaces as low\-confidence VLM reasoning rather than a wrong verdict. Cascade triage holds VLM cost near $1.63 per plate versus $10.33 for an always\-VLM baseline. AEGIS is open source and, to our knowledge, the first system to unify pre\-flight assay\-aware validation with runtime visual monitoring for an open\-source liquid handler.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.15620v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>