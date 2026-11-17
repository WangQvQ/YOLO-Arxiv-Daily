<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. Aerial Person Detection under Edge Constraints: Held\-Out Evaluation and Edge\-Ground Verification
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-15 |
> | 👤 作者 | Raghu Venkat |
>
> **📄 英文摘要：**
> A search drone sees a missing person as a few pixels, carries a small computer, and shares a narrow radio link. We study person detection under those limits; the primary evaluations were declared before they ran, and later analyses are marked exploratory. In an exploratory comparison, holding out recording blocks inferred from thermal frame identifiers cut recall at a fixed false\-alarm target from 32.7% to 12.3%. With nothing tuned on test blocks, a 46,000\-parameter detector gated by a contrast cue finds 10.4% of people against 5.7% for the same network applied everywhere. A standard detector, YOLO11n, trained under the same folds finds 54.5% at a similar false\-alarm rate for about eleven times the arithmetic, and on our 4K field footage at 117 m has a walking person among its top 16 detections in up to 92% of frames against 74% for our board. Run alone, YOLO11n's thresholds set at one location overshot their false\-alarm target 1.7 to 3.4 times at new locations, and its best configuration at altitude cost 4 to 24 standard passes per frame. Re\-scoring its 16 best crops \(154 KB per frame\) with an 842,000\-parameter ground model, thresholds frozen on a validation location, raised recall from 44.9% to 53.7% against the same 16 crops with YOLO's own scores; in a post hoc paired analysis over 15 test locations the gain was \+8.8 points \[\+2.4, \+12.5\]. Pooled false detections fell from 0.86 to 0.64 per frame, but that paired change was not distinguishable from zero. Coverage of a still person in earth tones was very low for both detectors in the clips we flew.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.20127v1)

---

> ### 2. Toward On\-Device Markerless 6DoF Tracking of a Physical Manipulative on a Mixed\-Reality Headset: A Preliminary Study
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-13 |
> | 👤 作者 | Guangxi Feng |
>
> **📄 英文摘要：**
> AR\-Classroom uses physical models and virtual overlays to help learners explore spatial transformations. To align these overlays on a standalone headset, we developed a markerless six\-degree\-of\-freedom\(6DoF\) tracker for a LEGO airplane on the Meta Quest 3. A compact YOLO model trained on synthetic Unity renders predicts 20 keypoints from dynamic RGB crops resized to 128 by 128 pixels. A perspective\-n\-point solver estimates the airplane's pose from these predictions and the corresponding 3D landmarks. In a paired evaluation on 1,000 synthetic validation images, cropping guided by ground\-truth boxes increased the detection rate on annotated images from 32.5 to 46.2 percent. Mean keypoint F1 increased from 42.9 to 44.6 percent, with lower keypoint precision in the cropped view. In preliminary headset tests, 2 Hz was the highest tested tracking target at which the application maintained approximately 72 frames per second. A headset demonstration illustrates virtual overlay alignment with the physical airplane before and after a manual rotation. Quantitative 6DoF pose accuracy on real headset footage will be measured in future work.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.18438v1)

---

> ### 3. What Limits Thermal Infrared Drone Detection? A Cross\-Dataset Benchmark Analysis
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-12 |
> | 👤 作者 | Tamara R. Lenhard |
>
> **📄 英文摘要：**
> Thermal infrared \(IR\) imaging enables drone detection with reduced dependence on ambient illumination, supporting continuous surveillance beyond the limitations of visible\-spectrum sensing. However, substantial and often undocumented variability in camera systems, processing pipelines, and annotation practices across public IR drone datasets complicates the assessment of detector performance and cross\-domain generalization. To examine these sources of variability and their implications for drone detection, we systematically characterize existing IR datasets and establish an artifact taxonomy reflecting their origins across the acquisition and processing pipeline, encompassing optical and sensor effects, tone mapping and image enhancement, and recording\-related artifacts. Focusing on long\-wave IR data, we investigate the impact of training data composition and model characteristics on YOLO\-based detection performance, generalization, and condition\-dependent failure modes. Our results demonstrate that detection performance is governed more strongly by training data composition than by YOLO generation or model capacity, while cross\-dataset generalization exhibits pronounced asymmetries. Even the best\-performing configuration fails to detect over 40% of drones, with elevated miss rates for very small targets, cluttered urban backgrounds, and thermal crossover, highlighting the shortcomings of existing datasets.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.16724v1)

---

> ### 4. When Does Sliced Inference Pay Off? Resolution, Tiling and Dense Fusion for Small Object Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-09 |
> | 👤 作者 | Stefan Becker |
>
> **📄 英文摘要：**
> Sliced inference such as SAHI is the standard remedy for small objects in high\-resolution imagery, but its gains are usually measured against low\-resolution baselines. We revisit sliced inference with scale as the explicit variable and ask when tiling actually pays off. We compare seven dense detectors from four families on VisDrone and DOTA, under pre\-specified hypotheses, calibration\-only tuning and a single evaluation on held\-out test data. Two findings emerge. First, much of the benefit attributed to slicing is a resolution effect: on VisDrone, a single pass trained at a higher resolution outperforms tiling with box fusion \(merging decoded boxes\) for every detector that could be trained at that resolution, and at equal compute a higher resolution helps more than a larger model. Second, tiles carry complementary information that box fusion wastes. We introduce SCAHI \(Slicing with Coverage\-Aware Hyper Inference\), which accumulates the dense outputs of all passes with weights that give every image position the same total coverage, and reads objects off a peak grid fine enough to keep crowded neighbours apart. On identical passes, SCAHI outperforms box fusion for every YOLO detector on both datasets and matches it for FCOS. On DOTA, whose images far exceed the resolution a single pass can be trained at, tiles with SCAHI beat the best single pass for every detector, by 4.3 to 5.6 mAP50 at 2048 px, and replacing only SAHI's merge step by SCAHI adds 2.3 to 4.0 mAP50. An ablation attributes the gain to the border weighting and the resolution of the peak grid. Source code is available at https://github.com/beckerio/scahi.
>
> **💻 代码链接：** https://github.com/beckerio/scahi.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.12657v3)

---

> ### 5. Compute Regeneration through Re\-Productization: Reorganizing Retired Edge AI Silicon into a Delivered Multilingual Service Robot
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-11-08 |
> | 👤 作者 | Hongyu Luo |
>
> **📄 英文摘要：**
> Useful edge hardware can lose product support while retaining substantial computation. We treat this as systems design: compute regeneration here means re\-productization, retaining the compute module while redesigning the surrounding product organization. Our case is GEdgeBot, a multilingual service robot built around a retired 32 GB Jetson AGX Xavier. A source\-grounded inventory shows the box alone supplies inference, while the combined system exposes five on\-device AI services, a seven\-language speech chain, three sensing domains, and one deterministic seven\-byte control boundary. AI and orchestration stay on the Xavier; finite motor\-pulse generation on an STM32F407. Audits cover the six\-mode control state machine, readiness barriers and sound orientation. Historical single\-prototype measurements report 1,782 MB GPU memory with YOLO, Whisper\-small, and NLLB\-200 resident, a direct\-control p99 deviation rising from 0.24 to 49.93 ms under concurrent AI load, and six of six trigger\-to\-follow demonstrations at 4.35 s mean latency, with conversational voice unavailable. A November campaign restores an isolated inference stack, re\-measures AI workloads, and repeats control\-frame runs. Driving two execution organisations with the same 6,000 measured 50 ms frame intervals shows the finite\-pulse boundary removes host jitter from the pulse train but not the command start phase; its stopping bound is host\-enforced, while the MCU 16\-bit pulse field is unclamped. A six\-check static audit exposes a host\-\-MCU queue mismatch, and a 14\-scenario replay executes the audited main\(\) dispatcher across six modes with zero physical device opens and 54 contract\-checked frames. Results support an existence claim and a reproducible architectural hypothesis, not fleet\-level reliability. The 208 direct\-control ratio remains unreplicated; MCU waveform capture and on\-site fault injection remain open.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2611.11698v2)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>