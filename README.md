<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Lowering the Barrier to AI\-Driven Inspection: A No\-Code Workflow for Automated Structural Defect Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-25 |
> | 👤 作者 | Michael Holm |
>
> **📄 英文摘要：**
> Structural health monitoring \(SHM\) is essential in modern engineering, providing data for condition\-based maintenance, lifecycle assessment, and predictive decision\-making. Traditionally, SHM relied on visual inspection to detect defects such as cracks and deformations. Early computer vision \(CV\) methods, including thresholding, edge detection, and handcrafted features, aimed to automate this process but were highly sensitive to noise, imaging variations, and multiscale defects, limiting their reliability.   Recent advances in machine learning, particularly convolutional neural networks \(CNNs\) and You Only Look Once \(YOLO\), have improved defect detection accuracy and enabled real\-time analysis. However, adoption in SHM remains limited due to technical barriers such as data labeling, model training, and deployment, which typically require programming expertise.   To address this gap, we introduce YOLOEZ, an open\-source, GUI\-based tool for end\-to\-end YOLO model application. YOLOEZ integrates data labeling, training, and inference into a single interface, enabling high\-performance model development without code while supporting reproducible workflows.   Evaluation against existing software and classical image processing demonstrates that YOLOEZ not only outperforms traditional methods across most detection metrics, but also lowers adoption barriers present in other modern CV tools. By combining accuracy with accessibility, YOLOEZ facilitates wider use of AI\-driven monitoring for predictive maintenance, digital twins, and intelligent structural systems.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.25176v1)

---

> ### 2. Decoupling candidate dual AGN from chance superpositions in the GOTHIC survey via a deep\-learning framework
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-25 |
> | 👤 作者 | Bhavesh Mukheja |
>
> **📄 英文摘要：**
> Dual active galactic nuclei \(DAGN\) mark a critical phase in the evolution of merging galaxies and the pairing of supermassive black holes, yet they remain difficult to identify in large imaging surveys because of projection effects and limited spatial resolution. Compact foreground stars and unresolved substructure can mimic dual nuclei through chance superposition, complicating automated detection. We revisit the 46,061 galaxies flagged but rejected as DAGN candidates by the GOTHIC pipeline, primarily because the two nuclei fell within the SDSS fibre aperture or exceeded its separation threshold. We train a supervised deep\-learning framework based on the YOLOv11 oriented\-bounding\-box architecture on annotated SDSS imaging to separate genuine dual nuclei from foreground stellar contaminants and other spurious alignments. The final model attains a validation precision of 0.919, recall of 0.905, and $F\_1$ of 0.912 for the dual\-nuclei class, and yields 29,605 dual\-nucleus candidates after removing star\-dominated and blended detections. Structured visual inspection indicates that $54.5$\-\-$62%$ are consistent with genuine dual nuclei, implying $sim\(1.4$\-\-$1.8\)times10^\{4\}$ plausible systems. Cross\-calibrating the YOLO separation against the deterministic GOTHIC centroid measurement and restricting to the compact regime \($d le 6.87''$\) gives a conservative subset of $sim 13\{,\}672$ candidates, reaching calibrated separations of $sim 0.56''$. Spectroscopy of the most compact \($le 1$~kpc\) systems shows they are dominated by passive, absorption\-line galaxies with no resolved double\-peaked emission, so confirmation requires higher\-resolution follow\-up. The catalogue is a statistically refined list of candidates, not confirmed DAGN. Nonetheless, deep\-learning detection substantially reduces contamination and expands the plausible DAGN census.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.24164v1)

---

> ### 3. Cross\-Generation Optimization of YOLOv26, YOLOv11, and YOLOv8 for Fine\-Grained Small\-Object Detection and Instance Segmentation in Complex Orchards
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-23 |
> | 👤 作者 | Ranjan Sapkota |
>
> **📄 英文摘要：**
> Small\-object detection and instance segmentation remain challenging in orchard environments because of green\-on\-green similarity, occlusion, and limited pixel representation of fine fruit anatomy. This study presents a cross\-generation benchmark of Ultralytics YOLOv8, YOLOv11, and YOLOv26 for detecting and segmenting apple fruitlet, calyx, and peduncle structures for robotic orchard perception. Five model scales \(n, s, m, l, and x\) were evaluated under conventional 640 x 640 and small\-object focused 960 x 960 training configurations, yielding 30 experiments. Increasing model capacity did not consistently improve accuracy. YOLOv11s\-960 achieved the highest observed mask mAP@50:95 \(0.402\) and box mAP@50:95 \(0.426\), while YOLOv26s\-960 achieved comparable values of 0.397 and 0.425 with only 10.37 M parameters and 34.1 GFLOPs. Peduncle remained the most challenging class. Overall, compact\-to\-moderate YOLO models with small\-object\-focused training provided favorable accuracy efficiency trade\-offs, establishing a practical benchmark for fine\-grained agricultural robotics and orchard perception. Github Link: https://github.com/rnjnspkt/Optimizing\-and\-Comparing\-Ultralytics\-YOLOv26\-YOLOv11\-and\-YOLOv8\-for\-Small\-Object\-Detection\-and\-Seg
>
> **💻 代码链接：** https://github.com/rnjnspkt/Optimizing-and-Comparing-Ultralytics-YOLOv26-YOLOv11-and-YOLOv8-for-Small-Object-Detection-and-Seg
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.23636v1)

---

> ### 4. Spiking Neural Networks for Energy\-Efficient Object Detection in Forward\-Looking Sonar Imagery
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-22 |
> | 👤 作者 | Gwenevere Frank |
>
> **📄 英文摘要：**
> Autonomous underwater vehicles \(AUVs\) are increasingly important tools in industries ranging from research, to energy, to defense. AUVs are power\-constrained platforms operating in remote environments with fixed battery capacities, where propulsion competes with compute and sensors for power over lengthy mission durations. AUVs frequently operate in dark or turbid waters where optical sensing is of limited value, and rely on sonar as their primary sensing modality. Convolutional neural networks \(CNNs\) are the state\-of\-the\-art solution for object detection in forward\-looking sonar imagery, but are energy expensive \(e.g. YOLOv8m: 322 mJ/inference\). Spiking neural networks \(SNNs\) rely on binary spike activations and thus sparse accumulate\-only operations, allowing them to be remarkably energy efficient, particularly when paired with dedicated neuromorphic hardware. The sparse, high\-contrast structure of forward\-looking sonar \(FLS\) returns is structurally matched to spike coding in a way that optical imagery is not. No prior work has assessed the suitability of SNNs for object detection in FLS imagery. SpikeYOLO, a fully spiking network trained with surrogate gradients, was benchmarked against state\-of\-the\-art CNN baselines on three FLS object detection datasets. Key results: SpikeYOLO T=2 achieves 3.3$times$ lower theoretical compute energy on UATD \(97 vs 322 mJ\) at competitive accuracy \(0.529 mAP@0.5:0.95 vs. YOLOv8m's 0.575\); SpikeYOLO matches YOLOv8m on mAP@0.5 and outperforms YOLO\-SONAR and Fast R\-CNN baselines on the sparse Marine\-Debris\-FLS dataset at 4.4$times$ lower energy; SpikeYOLO demonstrates superior robustness to multiplicative speckle noise \(3.0% degradation at $σ\{=\}0.4$ vs. 8.9% for YOLOv8m\), outperforming YOLOv8m outright at $σ\{=\}0.6$, directly relevant to real\-world FLS deployment.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.22072v1)

---

> ### 5. A Modular Agent for Reliable and Auditable Spatial Relation Verification in CT Scans
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-21 |
> | 👤 作者 | Simon Vincent Abel |
>
> **📄 英文摘要：**
> Reliable spatial understanding is an important prerequisite for future medical vision\-language systems that aim to support radiological report generation and structured image understanding. While modern vision\-language models \(VLMs\) show promising performance on many medical imaging tasks, recent evidence suggests they remain weak in controlled spatial reasoning and often fail to reliably ground spatial relations in image evidence. Given that radiological reasoning hinges on understanding the relative positions of anatomical structures and findings, this spatial weakness poses risks to diagnostic accuracy. We present a modular medical imaging agent for binary spatial relation verification in axial CT slices. Instead of directly predicting spatial answers end\-to\-end, the system decomposes the task into explicit stages: language parsing, anatomical localization, and deterministic geometric verification. Natural\-language queries are converted into structured relation tuples, queried organs are localized with a YOLO\-based detector, and the final spatial decision is computed from object centers using deterministic geometric rules. We evaluate the approach on the held\-out MIRP spatial QA benchmark and compare it against representative end\-to\-end VLM baselines. The best\-performing hybrid configuration reaches 94.1% accuracy and 94.2% F1, outperforming direct Qwen2\-VL prompting by 42.5 percentage points in accuracy, while preserving interpretable intermediate representations and auditable reasoning stages. The results suggest that explicit modular spatial verification can serve as a promising building block for future report\-oriented medical imaging agents.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.21140v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>