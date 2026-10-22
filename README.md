<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. TEMPL: A Resource\-Efficient and Transparent Model\-Free Framework for Pseudo\-Label Generation in Material CT Scans
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-20 |
> | 👤 作者 | Ashiqur Rahman |
>
> **📄 英文摘要：**
> The scarcity of large annotated datasets limits deep learning \(DL\) segmentation in material CT analysis. DL\-based pseudo\-labeling can help, but it is often computationally expensive and hard to interpret. We introduce a multi\-stage, model\-free framework that generates pseudo\-labels directly from grayscale CT scans. It combines thresholding, morphological dilation, and a custom breadth\-first search region\-growing algorithm to produce polygonal annotations without pre\-trained weights, labeled data, or a GPU. We compare it against the zero\-shot Segment Anything Model \(SAM\), a fully supervised 3D U\-Net, and classical baselines. On a dendrite dataset with ground truth, our method reaches 63.19% IoU versus 10.74% for zero\-shot SAM, with four times the instance recall at 0.5 IoU \(33.3% versus 8.3%\). The supervised 3D U\-Net serves as a reference ceiling that requires expert labels. On a silica sand specimen, YOLO models trained on our labels performed comparably to models trained on SAM labels in a qualitative comparison. Across 1,053 extracted grains, the median equivalent diameter was 163.8 micrometers, near the specimen's known 106 to 150 micrometers sieve range, compared with 219.5 micrometers for SAM, whose labels merge adjacent particles and add spurious detections. A sensitivity analysis shows a stable operating range for the key threshold and a sharp, predictable failure point linked to phase contrast. Our method needs 15s per image on a standard CPU versus 28s for SAM on a GPU. These results support our transparent approach as a tool for annotation bootstrapping and microstructural quantification in resource\-limited settings.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.28974v1)

---

> ### 2. Sketch2Sim: an end\-to\-end modular framework for hand\-drawn circuit analysis
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-20 |
> | 👤 作者 | Mohammad Kawsar |
>
> **📄 英文摘要：**
> This study proposes a sketch\-to\-simulation system to solve handwritten circuit diagrams. The system converts raw images into a Simulation Program with Integrated Circuit Emphasis \(SPICE\)\-compatible netlist and automatically calculates the circuit's nodal voltages and currents. Another objective is to propose a holistic framework that benefits students and educators by enhancing the teaching and learning experience of electric circuits and expediting engineering design reviews. This article presents a multistage pipeline that converts raw photographs of handwritten electric circuits into executable SPICE\-like netlists, providing a solution. A dataset of 1,000 hand\-drawn Resistor\-Inductor\-Capacitor \(RLC\) circuits was curated and annotated for components, text regions, and character recognition. The proposed system decomposes a circuit diagram into three stages: \(i\) dual You Only Look Once \(YOLO\)\-based detectors for components and text regions, \(ii\) a Convolutional Recurrent Neural Network \(CRNN\) with Connectionist Temporal Classification \(CTC\) and Optical Character Recognition \(OCR\)\-based domain\-aware symbol value extraction, and \(iii\) image processing with thresholding, skeletonization, and Density\-Based Spatial Clustering of Applications with Noise \(DBSCAN\) to automatically identify nodes and map them with corresponding components. The proposed system produces the netlist and solves the circuit. The qualitative and quantitative ablation studies on modular stages, as well as end\-to\-end tests, demonstrate robust performance across various handwriting styles, lighting conditions, and clutter.This study serves as an interactive learning aid and practical tool for professionals, providing a comprehensive user interface \(UI\), enabling rapid digitization of circuit sketches for study, design, prototyping, or troubleshooting.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.28836v1)

---

> ### 3. DocLayout\-YOLO: Axis\-Attentive Document Layout Detector with Mesh\-Guided Synthetic Pretraining
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-20 |
> | 👤 作者 | Zhiyuan Zhao |
>
> **📄 英文摘要：**
> Accurate and efficient document layout analysis is critical for modern document parsing systems. However, although the YOLO framework represents a leading paradigm for the real\-time detector, its adaptation to heterogeneous documents remains underexplored. To address this gap, we propose DocLayout\-YOLO, a document\-specialized real\-time detector that integrates diverse synthetic document pretraining and page\-axis context modeling. From the data perspective, we introduce MeshLayout\-1M, a synthetic pretraining corpus generated by the proposed Mesh\-candidate BestFit algorithm, which provides large\-scale and diverse document layouts for pretraining and improves adaptation to heterogeneous document scenarios. From the model perspective, we propose Page\-Axis Layout Attention \(PALA\), which captures long\-range dependencies along the horizontal and vertical page axes while remaining computationally efficient. Extensive experiments demonstrate that DocLayout\-YOLO achieves state\-of\-the\-art performance on four heterogeneous document benchmarks, obtaining 72.4%/75.5%/96.0%/73.1% mAP50:95 on D4LA, M6Doc, HJDataset, and PRIMA\-LAD, respectively. Compared with existing real\-time detectors, DocLayout\-YOLO establishes a superior accuracy\-efficiency trade\-off in document layout analysis. Code, model, and data are available at https://github.com/opendatalab/DocLayout\-YOLO/tree/release\-v2.
>
> **💻 代码链接：** https://github.com/opendatalab/DocLayout-YOLO/tree/release-v2.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.28171v1)

---

> ### 4. Plate\-YOLO: An End\-to\-End Object Detection Framework for Robust Astrometric Registration of Historical Plates
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-19 |
> | 👤 作者 | Quanfeng Xu |
>
> **📄 英文摘要：**
> Historical astronomical plates provide a long temporal baseline for time\-domain astrophysics, but degradation and structural artifacts can hinder automated astrometric calibration. We present Plate\-YOLO, a framework that combines source detection and stellar classification in one detector and supplies source catalogs to a separate astrometric solver. The detector adapts YOLOv5 with Space\-to\-Depth feature rearrangement, SimAM attention, a high\-resolution P2 head, and a Normalized Wasserstein Distance loss for small bounding boxes. A Gaia\-guided fine\-tuning procedure uses initially solved plates to construct labels for a new archive. On 150 UBAI plates for which the conventional pipeline had failed, the Chinese\-trained model solved 40 plates, and target\-archive fine\-tuning recovered a further 106, giving an overall recovery rate of 146/150 \(97.3%\). In the reported hardware configuration, source extraction took 6.61 s per plate for Plate\-YOLO and 16.73 s for SExtractor. These results demonstrate the practical value of adapting source detection to degraded historical plates.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.25677v1)

---

> ### 5. Precision Cotton Disease Management: Leveraging UAV Imagery and Custom YOLO Deep Learning Framework for Enhanced Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-16 |
> | 👤 作者 | 1. Anwar Iqbal\*1 aiqbal. rime23smme@student. nust. edu. pk 2. Kunwar Faraz Ahmad Khan1 kunwar. faraz@smme. nust. edu. pk 3. Shahbaz Khan1 shahbaz. khan@smme. nust. edu. pk 4. Khawaja Fahad Iqbal1 fahad. iqbal@smme. nust. edu. pk 5. Sara Ali1 |
>
> **📄 英文摘要：**
> Cotton sustains the textile economies of over 80 countries with an annual yield of 113 million bales, yet 30\-60% of yield losses are still attributed to viral, bacterial, and fungal diseases that remain undetected until manual field scouting. To address this critical issue, the integration of Unmanned Aerial Vehicle \(UAV\) imagery with deep learning offers a scalable early\-warning system; however, existing models suffer from low early\-stage accuracy, poor transferability, and high computational cost. In this paper, we propose Cotton\-Vision, a fully convolutional single\-stage detector that synergizes three lightweight architectural innovations: Ghost Bottleneck modules for a 20% parameter reduction, Efficient Channel Attention \(ECA\) for adaptive feature recalibration, and Cross\-Stage Partial \(CSP\) fusion for enriched gradient flow. The network is trained on a novel dataset collected across seven cotton fields in the Punjab province of Pakistan, comprising over 2,700 high\-resolution drone images. Extensive data augmentation \(rotation, shear, exposure, and saturation\) expands the dataset to 11,704 training samples. On a held\-out test set, Cotton\-Vision achieves a precision of 97.7%, a recall of 77.7%, an mAP@50 of 85.6%, and an mAP@50\-95 of 76.6%. These results represent a 10\-20% improvement in precision, recall, and mAP over state\-of\-the\-art baselines, including YOLOv5, YOLOv8, YOLOv11, YOLOv12, RT\-DETR, and RF\-DETR. Through a comprehensive ablation study, we demonstrate that each proposed architectural enhancement contributes to the model's superior performance, yielding a 12\-16% mAP improvement over the baseline architecture. Our findings confirm that this synergy of lightweight and attention\-based modules provides a robust, computationally viable solution for early cotton disease detection.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.24182v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>