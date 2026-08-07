<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. From Transparent Labware Segmentation to Collision Avoidance: A Real\-Time Edge\-Aware Perception Pipeline
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-05 |
> | 👤 作者 | Shijun Ding |
>
> **📄 英文摘要：**
> This paper presents an edge\-aware instance segmentation framework that enables real\-time robotic collision avoidance with transparent laboratory glassware using purely visual perception. Transparent vessels defy conventional segmentation due to refraction, specular reflection, and the absence of stable interior texture, yet their boundary contours remain comparatively reliable visual cues. Exploiting this observation, we augment a one\-stage real\-time instance segmentation backbone with a lightweight edge\-detection branch, edge\-guided attention fusion, and a parameter\-free SimAM module, and further construct LabGlass\-IS, a 3485\-image, 21\-category instance segmentation dataset of real laboratory glassware. The enhanced model achieves the highest Boundary F\-score of 97.80 among compared methods, outperforming the YOLO\-prompted FastSAM framework by 18.93 BF points. Furthermore, it maintains an inference speed of 7.1ms per frame and requires only 2.85% of the parameters of the closest accuracy competitor. Multi\-view triangulation of mask centroids further provides 3D positions for conservative bounding\-volume collision constraints. Real\-robot trials achieve a 93.3% collision avoidance success rate, indicating the feasibility of the proposed perception\-to\-action pipeline for robot collision avoidance among fragile transparent objects. Our code is available at https://github.com/havishamy/TransYOLO\_3D. Our video is available at https://havishamy.github.io/paper\-videos/.
>
> **💻 代码链接：** https://github.com/havishamy/TransYOLO_3D.，https://havishamy.github.io/paper-videos/.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.04769v1)

---

> ### 2. Simile Understanding in Text\-to\-Image Models: An Evaluation Framework
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-05 |
> | 👤 作者 | Luecheng Wang |
>
> **📄 英文摘要：**
> Similes provide a compact and expressive way to describe visual characteristics in text prompts. Recent text\-to\-image models \(t2i models\) can produce visually compelling outputs from simile prompts, yet even frontier models frequently misinterpret the metaphorical vehicle and confuse it with the object. These systematic failures reveal a gap between figurative language and object\-level visual grounding in t2i models. To investigate this issue, we propose a scalable evaluation framework for simile understanding. Our framework includes \(1\) a controlled simile dataset in which metaphorical vehicles are drawn from a predefined set of object\-detectable categories and combined with diverse templates, \(2\) automatic grounding metrics based on YOLO \(You Only Look Once\) detection, and \(3\) text encoder layer analysis using Diffusion Lens to track how metaphorical vehicles emerge during generation. Experiments across architecturally diverse t2i models reveal consistent literalization failure patterns. We further discuss potential mitigation strategies for improving simile grounding in t2i models.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.04750v1)

---

> ### 3. NeuroAdaptTrainer: A Fiji/ImageJ Plugin for YOLO\-Based Neuron Segmentation, InteractiveCorrection and Transfer Learning
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-05 |
> | 👤 作者 | Daniela Eraso\-Casas |
>
> **📄 英文摘要：**
> Neuron counting and segmentation in microscopy images of neuronal cultures is a routine and time\-consuming task in neuroscience research, traditionally performed through manual inspection or semi\-automatic tools. We present NeuroAdaptTrainer, an open\-source Fiji/ImageJ plugin that integrates a YOLO instance\-segmentation model directly into the microscopist's workflow. The plugin allows a user to run automatic neuron detection on a single image or a batch of images, manually correct the resulting detections from within Fiji, and use those corrections to adapt the model to new imaging conditions via transfer learning. A built\-in external validation module allows the base and adapted models to be compared quantitatively on a held\-out annotated set. NeuroAdaptTrainer lowers the barrier for non\-specialist users to benefit from deep\-learning\-based segmentation while keeping expert supervision at the center of the workflow.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.05226v1)

---

> ### 4. YOLO\-PVC: 2D\-to\-3D Consolidation of Slice\-wise Detections for Volumetric Liver Tumor Localization in MRI
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-05 |
> | 👤 作者 | Talha Waqas |
>
> **📄 英文摘要：**
> Slice\-wise 2D object detectors are increasingly applied to volumetric data due to their computational efficiency and scalability, yet they often yield fragmented and unstable predictions along the depth axis. We propose YOLO\-PVC, a lightweight and model\-agnostic framework for 2D\-to\-3D consolidation of slice\-wise detections. The method enforces depth continuity, aggregates bounding box coordinates using robust percentile statistics, and further refines axial extent through a lightweight MLP\-based calibration module. Unlike naïve stacking or averaging strategies, YOLO\-PVC explicitly addresses missing detections and outlier slices along the depth dimension. Experiments on 3D liver MRI volumes across three tumor categories demonstrate consistent improvements over multiple aggregation baselines. The heuristic PVC achieves an overall $mathrm\{IoU\}\_\{3D\}$ of $0.665$, while the calibrated variant further improves performance to $0.710$, with high planar overlap \($mathrm\{BEV IoU\} approx 0.78$\). These results demonstrate that structured geometric consolidation provides an effective and practical solution for volumetric liver tumor localization in clinical MRI.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.04642v1)

---

> ### 5. Interpretable Fuzzy Inference for UAV Target Tracking Using Bounding\-Box Geometry
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-04 |
> | 👤 作者 | Reza Ahmari |
>
> **📄 英文摘要：**
> Vision\-based guidance of unmanned aerial vehicles \(UAVs\) toward unmanned ground vehicles \(UGVs\) supports cooperative aerial\-\-ground robotics, but reliable continuous yaw estimation from onboard vision remains challenging because of sensing uncertainty, limited computation, and the need for interpretable control. Existing deep\-learning and geometric\-reconstruction approaches often require large datasets, external localization, or complex modeling assumptions, reducing transparency and deployment suitability on resource\-constrained platforms. We present an interpretable fuzzy\-inference framework that generates continuous yaw commands from low\-dimensional features extracted from YOLO boxes: target centroid location, area, and aspect ratio. No explicit geometric modeling is required. A Mamdani fuzzy system serves as an interpretable baseline using a shoulder\-\-triangle\-\-shoulder input partition. It is followed by a first\-order Takagi\-\-Sugeno model with three antecedent membership terms per input, whose parameters are derived from training\-set quantiles, yielding a compact 27\-rule structure. Evaluation uses 6\{,\}169 labeled samples from a VICON motion\-capture environment. Across five randomized train\-\-test splits, the Takagi\-\-Sugeno model achieves a test\-set mean absolute error of $0.140^circ pm 0.003^circ$, a root mean squared error of $0.200^circ pm 0.008^circ$, and a maximum absolute error of $1.254^circ pm 0.121^circ$. Within\-threshold accuracies are $99.676% pm 0.270%$ for $pm1^circ$ and $100.000% pm 0.000%$ for both $pm3^circ$ and $pm5^circ$. Directional consistency between image\-plane horizontal displacement and predicted yaw sign reaches $90.254% pm 0.612%$. These results show that the framework is transparent, data\-efficient, computationally lightweight, and suitable for real\-time vision\-based UAV guidance toward mobile ground targets.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.04121v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>