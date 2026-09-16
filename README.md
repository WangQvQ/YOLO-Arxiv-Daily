<div align="center">

# YOLO ArXiv Daily

[![Daily Papers](https://img.shields.io/badge/📅-每日更新-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-最新论文-red)](https://arxiv.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)

*自动追踪 YOLO 相关最新论文，提供中英文双语摘要*

</div>

---

## 📑 论文列表

> ### 1. BVB: Benchmarking Agentic Video Understanding via Programmatic Reconstruction in Blender
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-14 |
> | 👤 作者 | Yolo Y. Tang |
>
> **📄 英文摘要：**
> Multimodal agents can create complex videos in software such as Blender by coding without relying on diffusion models. Yet video understanding benchmarks still evaluate models mainly through question answering. If an agent truly understands a video, it can reconstruct it programmatically. We introduce BVB, Blender\-VideoBench, a benchmark that tests this ability by asking agents to reconstruct real\-world videos as animated Blender scenes. To ensure fair comparison, each agent programs the reconstruction through a lightweight harness, Mini\-BVB, in an identical sandbox under a shared cost limit. The benchmark renders each reconstruction from its animated camera and evaluates it on two axes: \(1\) Dual VQA measures how many spatiotemporal facts the reconstruction preserves. \(2\) Latent Similarity measures how closely the reconstruction matches the source video perceptually. Our overall score, a square\-root mean, favors balanced performance. We evaluate 51 configurations from 10 model families and analyze semantic retention, perceptual similarity, reasoning effort, and cost. The best model reaches 88.6 Latent Similarity but retains only 53.7% of the source\-correct spatiotemporal answers. Additional reasoning improves visual similarity but does not close this gap in factual accuracy. In a blind study with 15 raters and five configurations, Latent Similarity correlates strongly with human preference. These results show that programmatic reconstruction is a viable test of agentic video understanding, and that semantic retention remains the main challenge.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.15478v1)

---

> ### 2. SpermYOLO: A Coordinated YOLO\-Based Detector for Accurate and Efficient Sperm and Impurity Detection in Microscopic Images
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-13 |
> | 👤 作者 | Shengqi Chen |
>
> **📄 英文摘要：**
> Accurate sperm detection is essential for computer\-assisted semen analysis, yet it remains challenging in microscopic images due to dense distributions, visually similar artifacts, and sperm\-like impurities. In this paper, we propose SpermYOLO, a coordinated and compact YOLOv11\-derived framework for joint sperm and impurity detection in microscopic images. SpermYOLO introduces four architectural improvements: C3k2\-IDB for channel\-wise discriminative feature extraction, D2SEM for spatial\-\-spectral semantic enhancement, MFM for adaptive multi\-scale feature fusion, and the DESD Head for detail\-enhanced shared prediction. Experiments on the SVIA semen microscopic imaging benchmark show that SpermYOLO achieves 97.2% sperm AP and 75.4% impurity AP, outperforming generic detectors, dedicated sperm detection models, and improved YOLO variants. Compared with the baseline model, SpermYOLO improves sperm AP, impurity AP, $mathrm\{mAP\}\_\{50\}$, and $mathrm\{mAP\}\_\{50:95\}$ by 1.6, 10.0, 5.8, and 2.7 percentage points, respectively, while preserving a lightweight model scale. Cross\-scene evaluation on the SDTB testicular\-biopsy microscopy benchmark shows that SpermYOLO remains effective with extremely small sperm targets and complex tissue backgrounds, achieving the highest $mathrm\{mAP\}\_\{50\}$ and $mathrm\{mAP\}\_\{50:95\}$ of 74.8% and 31.2%, respectively. Ablation studies and qualitative analyses further support these improvements by demonstrating the contributions of the proposed modules and showing more focused feature response patterns than the baseline model. These findings suggest that SpermYOLO is an effective and efficient approach for sperm detection in challenging microscopic imaging scenarios.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.14278v1)

---

> ### 3. Quantum\-Gated LiteSSD: A Parameter\-Efficient Lightweight Hybrid Quantum\-Classical Framework for Forward\-Looking Sonar Object Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-12 |
> | 👤 作者 | Niloy Kumar Mondal |
>
> **📄 英文摘要：**
> Forward\-looking sonar object detection is essential for underwater perception, yet deployment on embedded platforms requires highly compact models. To address this challenge, we explore quantum computing and introduce Quantum\-Gated LiteSSD, a parameter\-efficient hybrid quantum\-\-classical detector that reformulates QuCNet\-style multi\-circuit quantum processing as an identity\-centered channel\-gating mechanism for spatial feature modulation. Experiments on the Marine Debris Watertank dataset and UATD forward\-looking sonar benchmarks demonstrate an effective parameter\-\-accuracy trade\-off. The proposed detector achieves 90.84% $mathrm\{mAP\}\_\{50\}$ on Watertank with approximately $62times$ fewer parameters than YOLO26s and $164.3times$ fewer parameters than SSD\-VGG16. On UATD, the model achieves 70.37% $mathrm\{mAP\}\_\{50\}$ with only 0.150M parameters, making it approximately $4.1times$ smaller than SSGA\-YOLO while retaining meaningful multi\-class detection capability.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.14025v1)

---

> ### 4. ScopeMamba\-YOLO: Widening the Perceptual Scope Inward and Outward for Small Object Detection in Remote Sensing Imagery
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-09 |
> | 👤 作者 | Junjie Fan |
>
> **📄 英文摘要：**
> Small object detection in unmanned aerial vehicle \(UAV\) and remote sensing imagery requires preserving high\-resolution detail while modeling long\-range context. Adding a stride\-4 detection level and removing the stride\-32 stage benefits tiny targets but weakens peripheral spatial support, whereas directly inserting selective scanning into the main feature path can interfere with weak local cues. We propose ScopeMamba\-YOLO, built around an off\-path, zero\-gated selective\-scanning principle that decouples contextual modeling from the convolutional stream. The principle is instantiated by a Cascaded Global\-Context Module \(CGCM\) in the backbone and a Selective\-Scan PAN \(SS\-PAN\) in the neck. An Adaptive Multi\-scale Strip \(AMS\) Block reduces the cost of high\-resolution feature extraction, while a Scale\-Adaptive DFL \(SA\-DFL\) head reallocates distributional support and regression capacity across scales with only 0.008M additional parameters. Controlled experiments show that matched main\-path selective scanning reduces mAP50 by 0.98 pp, whereas off\-path CGCM improves the final configuration by 0.67 pp over the three\-seed no\-CGCM mean; operator controls indicate that this gain is not explained by auxiliary branch capacity alone. ERF analysis further shows that the complete context pathway increases the peripheral energy ratio from 0.008 to 0.090 at stride 8. On VisDrone\-2019, ScopeMamba\-S achieves 50.8% mAP50 with 3.57M parameters, exceeding YOLOv8s by 10.8 pp while using 32% of its parameters; ScopeMamba\-M reaches 52.6% mAP50 with 6.48M parameters. Consistent improvements are also observed on AI\-TOD, especially for very\-tiny and tiny objects.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.10156v1)

---

> ### 5. Vague2Detect: Handling Ambiguous Prompts in Knowledge\-Based Open\-World Detection
>
> | 属性 | 内容 |
> |:---:|:---|
> | 📅 发布日期 | 2026-09-09 |
> | 👤 作者 | Ibrohimjon Muminov |
>
> **📄 英文摘要：**
> Real\-world detectors must often interpret functional or ambiguous prompts, yet conventional models such as YOLO remain restricted to fixed class lists. Even open\-vocabulary models like YOLO\-World frequently misalign vague language with the intended objects. Building on our prior work Commonsense\-Guided Open\-World Object Detection Using LLMs and Visual\-Semantic Matching, we address YOLO\-World's limitations in grounding task\-driven queries. We propose Vague2Detect, a hybrid pipeline in which a fine\-tuned Sentence\-BERT retrieves candidates from a structured household Knowledge Base \(KB\), and YOLO\-World verifies their presence in the image. For prompts outside the KB, a large language model \(GPT\-3.5\-turbo\) generates candidate descriptions, dynamically expanding the KB to cover novel concepts. On a benchmark of household scenes using custom images and an Open Images V7 subset, YOLO\-World alone achieves only 32% Vague Prompt Success Rate \(VPSR\), the ability to map ambiguous queries to correct detections. In contrast, Vague2Detect improves performance to 61% VPSR with high precision, and up to 85% when augmented with GPT fallback.
>
> 🔗 [阅读论文](http://arxiv.org/abs/2609.09949v1)

---

<div align="center">

*由 [YOLO-Arxiv-Daily](https://github.com/WangQvQ/YOLO-Arxiv-Daily) 自动生成*

</div>