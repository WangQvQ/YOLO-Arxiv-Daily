<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. ROD\-Dataset: A Pedestrian\-Perspective Benchmark of Urban Obstacle Classes for YOLO\-Based Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-31 |
> | 👤 作者 | Abtin Zandi |
>
> **📄 英文摘要：**
> Two groups of pedestrians face avoidable risk on the city sidewalk: people living with visual impairment, who reach an obstacle before anything warns them, and people walking with divided attention, who register a hazard late or not at all. A vision system can warn them only about obstacles it has learned to recognize, so its training data limits the protection it offers. No suitable training set exists: public benchmarks are recorded from vehicles, aerial platforms, or fixed rigs, general\-purpose detectors omit the street furniture that causes trips, and pedestrian\-focused collections are confined to single regions with incompatible class schemes. This paper introduces the Real\-time Obstacle Detection \(ROD\) dataset: more than 24,000 images and 40,000 annotated obstacles across 25 classes, from vehicles and people to the stairs, manholes, traffic cones, and benches that general\-purpose detectors overlook. Because street furniture differs between cities, ROD unifies twenty\-nine public collections onto one taxonomy and adds eye\-level imagery from two contrasting cities on two continents, Tehran and Toronto. Every label is human\-verified, and each capture campaign is admitted only after a compact detector shows it can learn the class. Six lightweight YOLO detectors \(nano and tiny variants\) benchmarked on ROD all exceed 0.86 recall; YOLO12n attains the highest recall \(0.889\), YOLO26n the highest precision \(0.925\), and YOLOv9t the best mAP@0.50 \(0.882\) with the fewest parameters. Because the recall and mAP rankings disagree, selecting a model by mAP does not minimize missed obstacles, so per\-class recall is reported for all 25 classes.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.00681v1)

---

> ### 2. Communication\-Efficient Model\-Distributed Inference for CBRS Radar Detection at the Edge
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-31 |
> | 👤 作者 | Teng Li |
>
> **📄 英文摘要：**
> The Citizens Broadband Radio Service \(CBRS\) shares the 3550\-3700 MHz band between incumbent radar systems and commercial LTE/5G users, so radar activity must be detected accurately and promptly. Spectrogram\-based YOLO detectors can localize coexisting radar and cellular signals, but running them on resource\-constrained edge devices is challenging. Model\-distributed inference \(MDI\) can increase throughput by partitioning a model across edge devices. Using an end\-to\-end pipeline from raw I/Q samples to on\-device detection, we show that partitioning recent YOLO generations incurs high activation\-transfer costs due to their multi\-scale structures. To reduce these costs, we propose CEDAR, a communication\-efficient, edge\-distributed architecture for radar detection. Its RF\-aware detector, CEDAR\-YOLO, removes YOLO's multi\-scale neck and two of its three detection scales, leaving a single\-scale path with independently configured frequency and time resolutions, which lowers the communication cost of partitioning. Its compression\-aware MDI framework, CEDAR\-MDI, integrates lightweight activation compressors and selects, for each compressor, the partition that minimizes the pipeline bottleneck. On an NVIDIA Jetson Orin Nano testbed with real\-world over\-the\-air data, CEDAR\-YOLO reduces the parameter count by 35\-57% and model inference time by 16\-33%, and CEDAR\-MDI achieves up to $3.27times$ the throughput of single\-device inference, with detection quality preserved in both cases.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.00625v1)

---

> ### 3. Directional Dual\-Head Supervision for Real\-Time Source\-Free Polyp Segmentation
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-10-30 |
> | 👤 作者 | Hai\-Dang Nguyen |
>
> **📄 英文摘要：**
> Source\-free domain adaptation \(SFDA\) is critical for medical image segmentation as it adapts a model to an unlabeled target domain without accessing source data, thereby preserving data privacy. However, its effectiveness is often limited by confirmation bias: self\-training models tend to miss small polyps, hallucinate foreground, and produce imprecise contours. We propose MedRT\-SFSeg, a real\-time SFDA framework built on a dual\-head one\-to\-one/one\-to\-many YOLO segmentation architecture. Mask\-DHF fuses precision\- and recall\-oriented predictions into stable pseudo\-instances, while DURR uses the sign of pixel\-level dual\-head disagreement as a directional cue to expand or shrink masks, and additionally recovers missed instances and suppresses hallucinations. Seg\-MARD further regularizes feature representations through mask\-guided, region\-stratified sampling. Experiments on bidirectional adaptation between Kvasir\-SEG and CVC\-ClinicDB show that MedRT\-SFSeg achieves 80.43%/80.77% Dice and 5.53/8.44 mm ASD, improving the non\-adapted baseline by 7.7 and 9.8 Dice points while running at more than 62 FPS end\-to\-end. These results suggest that segmentation\-aware dual\-head supervision can improve pseudo\-label quality without sacrificing real\-time inference. Project page: https://danielnguyen\-05.github.io/MedRT\-SFSeg
>
> **💻 代码链接：** https://danielnguyen-05.github.io/MedRT-SFSeg
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.00238v1)

---

> ### 4. Shifted Window Temporal Transformer for YOLO\-Guided Phase Recognition in Laparoscopic Cholecystectomy Surgery
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

> ### 5. AIPC: Evidence\-Guided Agent Workflows for PyTorch Project Deployment
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

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>