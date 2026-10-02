<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. HandAnthro: Automated Hand Anthropometry from a Single Image
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-29 |
> | 👤 作者 | Fan Zhou |
>
> **📄 英文摘要：**
> Hand anthropometry supports protective\-glove design, but existing measurement methods often require trained operators, specialized hardware, or manual landmarking. We present HandAnthro, which estimates 44 projected hand dimensions from a smartphone photograph of a palm\-up hand on US letter\-size paper. The pipeline reconstructs wrist\-occluded paper boundaries for rectification, whitens non\-hand pixels, and refines 41 anthropometry\-specific landmarks from a fine\-tuned You Only Look Once \(YOLO\) pose model using image\-specific geometry and contours. Controlled evaluation comprised 720 captures from 45 held\-out participants, each contributing 16 images across two smartphones, two backgrounds, two angles, and two nominal illumination settings. HandAnthro produced complete outputs for 704 captures \(97.8%\); among these, mean absolute error \(MAE\) was 3.80 mm per dimension against two trained operators' caliper measurements. Regional MAEs were 2.48 mm for non\-thumb fingers, 6.04 mm for thumbs, and 6.17 mm for palm and wrist. In a researcher\-assisted mobile\-app pilot, automated batch processing returned all 44 dimensions for 260 of 268 retained, researcher\-screened firefighter images \(97.0%\). A descriptive, unpaired comparison with an independent national firefighter reference yielded a mean absolute difference of 2.40 mm across 28 sex\-by\-dimension group\-mean contrasts. These results characterize controlled measurement performance and researcher\-assisted field feasibility for future distributed hand\-anthropometry studies.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.37855v1)

---

> ### 2. ByteTraX: Enhancing the ByteTrack Architecture with Optimised Thresholding
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-29 |
> | 👤 作者 | Thomas A. O'Shea\-Wheller |
>
> **📄 英文摘要：**
> The ByteTrack algorithm is a widely used and computationally efficient multi\-object tracking architecture. Its core innovation lies in the combination of lenient bounding box associations with tracklet similarity matching to robustly deal with object occlusions. However, this strategy is nevertheless vulnerable to erroneous track reclassification and identity switching, as detection confidence scores dictate association priority. To address this, I present a simple enhancement of the ByteTrack architecture\-\-named ByteTraX\-\-that optimises track continuity via a single unified matching threshold, while penalising identity switches through stringent track initiation criteria. This approach achieves consistently improved performance across a range of diverse benchmarks including GMOT\-40, LC\-MOT, SportsMOT, TeamTrack, DAMUNT, and DeepSea\-MOT, while simultaneously increasing processing speed by >10%. Specifically, results demonstrate a >40% reduction in identity switches, accompanied by mean increases in HOTA of 3.6, IDF1 of 5.6, and FPS of 6.3. As such, adoption of the ByteTraX algorithm has the potential to substantially enhance tracking performance over the ByteTrack baseline, while retaining the efficiency needed for real\-time deployment. To facilitate usage, I provide the source code, integration functionality for the YOLO family of object detection models, and deployment instructions via an open source repository.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.37801v2)

---

> ### 3. A Multi\-Dataset Benchmark of YOLO\-Based Weed Detection in Precision Agriculture
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-27 |
> | 👤 作者 | Hristina Zdraveska |
>
> **📄 英文摘要：**
> Weed detection is an important component of precision agriculture, enabling site\-specific weed management and reducing unnecessary herbicide use. Although deep learning methods have achieved strong results for crop and weed detection, many studies rely on single\-dataset evaluation, making it difficult to assess robustness across different agricultural domains. This paper presents a multi\-dataset benchmark of deep object detectors for weed detection in precision agriculture, with a focused evaluation of YOLO26 models. We evaluate nano, small, and medium variants on seven public weed\-detection datasets covering different crops, weed species, field conditions, acquisition setups, and annotation protocols. The models are compared in terms of detection accuracy, model complexity, inference latency, FPS, and model size. In addition to in\-dataset evaluation, we investigate cross\-domain generalization using a unified one\-class weed setup and evaluate multi\-source training using the combined training subsets from all datasets. The results show that YOLO26 achieves strong in\-dataset performance, with YOLO26m obtaining the highest average accuracy and YOLO26s providing the best practical accuracy\-efficiency trade\-off. However, cross\-domain performance decreases substantially, with YOLO26s dropping from an average in\-domain mAP$\_\{50:95\}$ of 0.603 to 0.148 in the off\-domain setting. Multi\-source training improves performance on several datasets, but does not fully eliminate domain shift. Overall, the benchmark highlights the importance of dataset diversity, domain similarity, and target\-domain adaptation for robust weed detection in real\-world precision agriculture applications.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.33991v1)

---

> ### 4. Synthetic Thermal Image Generation for Real\-Time Animal Detection Under Low\-Visibility Conditions
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-26 |
> | 👤 作者 | James Momoh |
>
> **📄 英文摘要：**
> Wildlife\-vehicle collisions remain a significant road safety concern, particularly during nighttime and low\-visibility conditions when RGB\-based perception systems are often unreliable. Thermal imaging offers a promising alternative for detecting animals under poor illumination. However, the limited availability of annotated infrared animal datasets restricts the development of robust deep learning\-based detection models. This paper investigates synthetic thermal image generation as a scalable approach for real\-time animal detection under low\-visibility conditions. A subset of 514 annotated visible\-spectrum animal images from the NTLNP dataset is translated into synthetic thermal representations using CycleGAN\-Turbo, while a limited real thermal dataset of 60 images is expanded through thermal\-focused augmentation. Multiple object detection architectures, including YOLOv8, YOLOv9, YOLOv10, and RT\-DETR, are trained independently on synthetic and real thermal datasets and evaluated using precision, recall, mAP@0.5, mAP@0.5:0.95, model size, and inference latency. Experimental results show that synthetic thermal images provide competitive detection performance, with RT\-DETR achieving the highest synthetic\-data mAP@0.5 of 0.9613. Models trained on augmented real thermal data achieve the strongest overall performance, with YOLOv10s obtaining 0.9879 mAP@0.5 and 0.9571 mAP@0.5:0.95. Computational analysis further indicates that lightweight YOLO variants provide favorable inference latency, supporting their potential for real\-time deployment. These findings demonstrate that synthetic thermal imagery can reduce dependence on scarce infrared datasets and support the development of efficient animal detection systems for future vehicle\-mounted wildlife collision mitigation applications.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.32944v1)

---

> ### 5. Attribution Gaps in Zero\-Training LLM\+OVOD Pipelines: A Fine\-Grained Analysis of the CAAP\-\-SNAP Discrepancy
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-26 |
> | 👤 作者 | Yu\-Feng Yen |
>
> **📄 英文摘要：**
> LAOD and similar zero\-training LLM\+open\-vocabulary\-detector \(OVOD\) pipelines score two things separately: class\-agnostic localization accuracy \(CAAP\) and semantic naming accuracy \(SNAP\). The two consistently diverge, and nobody has asked why. This paper asks why, on the full 5,000\-image COCO\-Val split \(27,273 detections\) rather than the small subset the original work evaluated on. Object visual complexity turns out not to be the driver \-\- small and occluded objects are, if anything, localized better than large ones. Vocabulary novelty is: once the LLM's wording falls outside the detector's native category set, localization accuracy falls from 80.9% to 31.6%. That drop is not spread evenly across unfamiliar phrasing, though. Almost all of it comes from cases where the novel wording actually names a different object than the one COCO annotated \(true synonyms still score 89.3%; semantically unrelated "noise" labels score 12.0%\). A closer look at a further failure subset tells a similar story: 78\-88% of what looks like complete localization failure is really the model correctly finding a real object that COCO's non\-exhaustive 80\-category scheme simply never labeled, not hallucination. Swap the detector backbone \(YOLO\-World for Grounding DINO\) or the LLM \(Gemma\-3 for Qwen2.5\-VL\) and both the effect and its rough size hold up, so this looks like a general property of the pipeline family rather than a quirk of one model pairing. The upshot is that a large share of the apparent CAAP\-\-SNAP gap traces back to closed\-category annotation limits rather than a real grounding failure, which matters for how we detect hallucination, analyze failure modes, and design evaluation for grounded multimodal systems meant to work in the open world.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.32567v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>