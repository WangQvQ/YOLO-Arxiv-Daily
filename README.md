<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. SAT\-Edge\-Agent: Hardware\-in\-the\-Loop Edge\-Agent Orchestration for Onboard Satellite Intelligence
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-04 |
> | 👤 作者 | Longji He |
>
> **📄 英文摘要：**
> Onboard satellite intelligence requires a task layer that translates mission intent into local tool calls, exposes execution state, and returns machine\-consumable artifacts under communication and power constraints. We present SAT\-Edge\-Agent, a hardware\-in\-the\-loop \(HIL\) edge\-agent system deployed on a commercial off\-the\-shelf ARM\-based heterogeneous edge system\-on\-chip. A browser workspace and FastAPI agent coordinate a local OpenAI\-compatible language service with a project\-internal YOLO\-style oriented\-object\-detection endpoint that returns FAIR1M metadata\-backed structured results. Two fixed FAIR1M workloads, one single\-image and one serial two\-image request, were repeated 20 times each and completed 20/20 attempts. Mean Full\-Agent latency was 29.353 s and 60.937 s, with empirical P95 values of 31.166 s and 66.882 s. Mean detector time was 861.386 ms and 1510.920 ms, only 2.93% and 2.48% of the corresponding Full\-Agent means. Profiling indicates that most visible latency occurs outside detector execution. Mean CPU utilization was 20.761% and 20.482%. A 200\-ms NPU\-load field averaged 100% for both workloads, but it represents a shared\-accelerator software field rather than detector\-only occupancy or calibrated utilization. The public evidence package provides sanitized request\-level records, redacted JSON, normalized SSE examples, and scripts reproducing the reported statistics. These results establish a reproducible HIL boundary for observable satellite edge\-agent orchestration, but do not establish detector accuracy, a new geolocation method, calibrated energy efficiency, or flight readiness.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.03728v1)

---

> ### 2. Fast Object Removal Attacks on Safety\-Critical Video\-based Perception Systems
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-03 |
> | 👤 作者 | Mohammad Imtiaz Hasan |
>
> **📄 英文摘要：**
> By leveraging data from video\-based perception systems, intelligent transportation systems \(ITS\) support safety\-critical applications that improve road safety. However, adversaries may manipulate video frames to compromise downstream perception modules, causing failures in safety\-critical functions and increasing risks to vulnerable road users. This paper presents a novel attack model and an end\-to\-end framework for near\-real\-time targeted object removal attack on a video\-based safety\-critical system. The end\-to\-end attack pipeline consists of four stages: localizing targets in each frame, retrieving coherent patches from earlier frames, blending them using context\-aware alpha compositing, and reconstructing attacked frames. Experiments at an intersection on the South Carolina Connected Vehicle Testbed \(SC\-CVT\) show that reconstructed frames have high global similarity to the originals, with frame\-level Peak Signal to Noise Ratio \(PSNR\) above 40 dB and Structural Similarity Index Measure \(SSIM\) above 0.996. Using the YOLO\-based detector, the attack reduces object detections by up to 97.59% and achieves a frame\-level attack success rate of 94.48%. Across the evaluated detectors and frame resolutions, the mean execution time ranges from 0.074 to 0.172 seconds per frame on GPU hardware, indicating near\-real\-time performance in testing. The forensic evaluation using several pretrained tamper\-detection models shows limited ability to distinguish reconstructed from authentic frames. The findings suggest that video\-based perception is vulnerable to stealthy object removal attacks that can degrade the performance of safety\-critical applications by reducing object detectability. These findings can help develop mitigation strategies against adversarial object removal attacks that threaten safety\-critical applications, such as vision\-based pedestrian safety systems.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.02806v1)

---

> ### 3. Extended KAFR: A kinematic\-adaptive paradigm for the efficient analysis of surgical video
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-08-02 |
> | 👤 作者 | Huu Phong Nguyen |
>
> **📄 英文摘要：**
> Artificial Intelligence is increasingly applied to surgical video analysis for phase segmentation, skill assessment, and workflow optimization. A key challenge is the length of surgical recordings, often one to several hours, creating substantial computational burden. We previously developed Kinematics\-Adaptive Frame Recognition \(KAFR\) for robotic surgery, showing that tracking tool motion effectively identifies informative frames while filtering redundant content. However, laparoscopic surgery introduces additional challenges: manual camera control causes frequent motion artifacts, and image quality is generally lower than robotic systems. This study evaluates whether KAFR generalizes to laparoscopic surgery using the Cholec80 benchmark, comprising 80 laparoscopic cholecystectomy procedures annotated for seven surgical phases. KAFR operates in three stages: a fine\-tuned YOLO model detects and segments surgical tools; frames are adaptively selected based on tool displacement or velocity variation; and an X3D model classifies selected frames into surgical phases. KAFR achieved a 91.0% F1 score using only 0.58% of frames for phase classification, representing an approximately seven\-fold reduction compared to typical 4% frame sampling, while maintaining performance comparable to LoViT \(90.2%\) and Trans\-SVNet \(89.7%\). These results demonstrate that kinematics\-based frame selection transfers effectively to the challenging laparoscopic environment.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.01058v1)

---

> ### 4. MDWD: A Street\-Level Dataset for Municipal Solid Waste Detection in Dense Urban Environments
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-31 |
> | 👤 作者 | Andrea Filiberto Lucas |
>
> **📄 英文摘要：**
> Automated visual monitoring of urban environments is a growing Computer Vision research area, but municipal solid waste detection remains under\-represented in dedicated benchmark resources. Existing waste\-related datasets predominantly address individual litter detection, aerial imagery, or image\-level classification, and none simultaneously provide street\-level imagery, instance\-level localization, and categorization of domestic waste streams within a structured municipal collection context. This paper introduces the Maltese Domestic Waste Dataset \(MDWD\), a street\-level benchmark comprising 3,697 high\-resolution images and 11,461 manually annotated instances across five domestic waste categories representative of Malta's municipal collection system. The dataset captures substantial variation in location, illumination, object scale, occlusion, and urban context. To establish reproducible baselines, a cross\-architecture benchmark is conducted across multiple generations of the YOLO family and a transformer\-based detector. On the test set, RF\-DETR\-M achieves the strongest overall performance with an mAP50 of 94.49% and an F1\-score of 93.56%, whilst smaller\-capacity variants maintain competitive accuracy at substantially reduced parameter counts. These results indicate that MDWD supports effective training across both compact real\-time detectors and transformer\-based models, establishing a benchmark for future research in vision\-based municipal waste monitoring.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2608.00257v1)

---

> ### 5. Explainable and Resource\-Efficient Spatial Reasoning in Multimodal LLMs for Decision\-Critical Applications
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-07-29 |
> | 👤 作者 | Piyush Jain |
>
> **📄 英文摘要：**
> As Multimodal Large Language Models \(MLLMs\) are increasingly deployed in decision\-critical pipelines such as robotics, embodied AI, and safety monitoring, the opacity of their spatial judgments limits operator trust and auditability. MLLMs demonstrate strong reasoning but often struggle with fine\-grained spatial understanding and object hallucination. Prior work, ByDeWay, introduced Layered\-Depth\-Based Prompting \(LDP\), a training\-free framework that mitigates hallucinations by structuring prompts using monocular depth estimation. However, coarse depth layering falls short in resolving object\-to\-object spatial relationships within the same geometric plane, such as projective \("left of", "above"\) and topological \("inside", "touching"\) relations. We propose ByDeWay\-V2, which integrates explicit spatial relational context alongside depth cues, expressed as human\-readable predicates that serve as auditable evidence for downstream decision support. Using an open\-vocabulary object detector \(YOLO\-World\-L\), our framework computes pairwise geometric relations between detected objects and injects them as structured spatial predicates into the MLLM prompt, bridging 3D scene depth and 2D spatial semantics without any training. We evaluate ByDeWay\-V2 on the Visual Spatial Reasoning \(VSR\) and BLINK benchmarks across multiple MLLMs, with hallucination grounding assessed via POPE. On the BLINK spatial subset, ByDeWay\-V2 achieves a 46 percent relative F1 improvement over LDP for Qwen2.5\-VL, and recovers BLIP\-Base's spatial reasoning on VSR from near\-random performance to a competitive F1 of 0.53. Our lightest configuration operates under a strict 40\-token context budget on CPU, showing the framework's suitability for resource\-constrained, real\-time decision\-support settings.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2607.27145v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>