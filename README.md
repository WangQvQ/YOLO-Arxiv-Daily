<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Shifted Window Temporal Transformer for YOLO\-Guided Phase Recognition in Laparoscopic Cholecystectomy Surgery
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-26 |
> | 👤 作者 | Sheekar Banerjee |
>
> **📄 英文摘要：**
> Intelligent operating rooms require accurate awareness of surgical workflow for better intra\-operative decisions, teamwork, and surgical documentation. Therefore, accurately recognizing surgical phase during surgery is crucial. However, previous artificial intelligence\-based methods provide fluctuating predictions for the transitions, restricting their clinical use. To address this challenge, we propose a clinically meaningful framework that integrates a Swin Transformer and Temporal Transformer Encoder \(TTE\) for effective surgical phase recognition and a parallel YOLOv11s network for real\-time surgical instrument localization. To avoid phase flickering and to produce stable workflow predictions, a causal temporal smoothing is applied. The proposed framework is tested on the Cholec80 dataset for the simultaneous recognition of the surgical phases and instruments. The overall phase recognition accuracy of this model was 92%, and the macro average F1\-score was 0.86. This performance surpasses CNN based methods such as EndoNet \(81.7%\) and SV\-RCNet \(85.3%\), as well as temporal convolutional and transformer\-based approaches including TeCNO \(88.6%\), Trans\-SVNet \(90.9%\), and LoViT \(91.5%\). Besides, the module of instrument detection achieved mAP@0.5 = 0.863. The reliability of the results is confirmed, obtaining a high recognition rate of the workflows, and a high accuracy of the localization of the instruments. The proposed framework can serve as a foundation for intelligent surgical assistance, enriching situation awareness, and potential robotic surgery applications. Code will be made publicly available at: https://github.com/ac005sheekar/LapChol\-Surgery\-Swin\-Cholec80
>
> **💻 代码链接：** https://github.com/ac005sheekar/LapChol-Surgery-Swin-Cholec80
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.35005v1)

---

> ### 2. AIPC: Evidence\-Guided Agent Workflows for PyTorch Project Deployment
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-26 |
> | 👤 作者 | Zhanwei Wu |
>
> **📄 英文摘要：**
> Deploying and optimizing a PyTorch inference project on a dedicated accelerator, especially an on\-device NPU, requires more than using a framework device API as with CUDA. The project must be converted into hardware\-runtime artifacts and adapted to the target toolchain and application interface. Differences in model formats, quantization flows, operator support, and toolchain interfaces create cross\-layer problems spanning PyTorch/ONNX models, toolchains, runtimes, and device execution. LLM Agents can invoke tools to address such problems, but may drift because of insufficient domain knowledge, large configuration and repair spaces, or poor path selection.   We propose AIPC \(AI Porting & Conversion\), an Agent method for deploying PyTorch inference projects. AIPC combines structured Skills with toolchain, runtime, and profiling evidence to drive cross\-layer diagnosis and repair, organizing model adaptation, artifact building, on\-target execution, accuracy repair, and performance optimization into one deployment flow. We evaluate AIPC on the on\-device NPU path of Qualcomm QAIRT across vision, speech, LLM, and diffusion models. In two optimization cases, 6/7 Skill\-arm Whisper\-Small runs meet predefined accuracy and latency criteria versus 0/6 control runs, while YOLO\-Worldv2 achieves a 2.791x end\-to\-end speedup versus 1.599x for the control. Under a pre\-fixed deployment contract, Skill, tool permissions, and check conditions, some runs achieve zero in\-run human intervention \(0\-HITL\). We also examine Qwen3.8\-27B as a locally deployable model for driving an Agent workflow.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2610.34890v1)

---

> ### 3. TEMPL: A Resource\-Efficient and Transparent Model\-Free Framework for Pseudo\-Label Generation in Material CT Scans
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

> ### 4. Sketch2Sim: an end\-to\-end modular framework for hand\-drawn circuit analysis
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

> ### 5. DocLayout\-YOLO: Axis\-Attentive Document Layout Detector with Mesh\-Guided Synthetic Pretraining
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

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>