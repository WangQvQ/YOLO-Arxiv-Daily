# 每日从arXiv中获取最新YOLO相关论文


## Intelligent Image Search Algorithms Fusing Visual Large Models / 

发布日期：2025-11-25

作者：Kehan Wang

摘要：Fine\-grained image retrieval, which aims to find images containing specific object components and assess their detailed states, is critical in fields like security and industrial inspection. However, conventional methods face significant limitations: manual features \(e.g., SIFT\) lack robustness; deep learning\-based detectors \(e.g., YOLO\) can identify component presence but cannot perform state\-specific retrieval or zero\-shot search; Visual Large Models \(VLMs\) offer semantic and zero\-shot capabilities but suffer from poor spatial grounding and high computational cost, making them inefficient for direct retrieval. To bridge these gaps, this paper proposes DetVLM, a novel intelligent image search framework that synergistically fuses object detection with VLMs. The framework pioneers a search\-enhancement paradigm via a two\-stage pipeline: a YOLO detector first conducts efficient, high\-recall component\-level screening to determine component presence; then, a VLM acts as a recall\-enhancement unit, performing secondary verification for components missed by the detector. This architecture directly enables two advanced capabilities: 1\) State Search: Guided by task\-specific prompts, the VLM refines results by verifying component existence and executing sophisticated state judgments \(e.g., "sun visor lowered"\), allowing retrieval based on component state. 2\) Zero\-shot Search: The framework leverages the VLM's inherent zero\-shot capability to recognize and retrieve images containing unseen components or attributes \(e.g., "driver wearing a mask"\) without any task\-specific training. Experiments on a vehicle component dataset show DetVLM achieves a state\-of\-the\-art overall retrieval accuracy of 94.82%, significantly outperforming detection\-only baselines. It also attains 94.95% accuracy in zero\-shot search for driver mask\-wearing and over 90% average accuracy in state search tasks.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.19920v1)

---


## From Pixels to Posts: Retrieval\-Augmented Fashion Captioning and Hashtag Generation / 

发布日期：2025-11-24

作者：Moazzam Umer Gondal

摘要：This paper introduces the retrieval\-augmented framework for automatic fashion caption and hashtag generation, combining multi\-garment detection, attribute reasoning, and Large Language Model \(LLM\) prompting. The system aims to produce visually grounded, descriptive, and stylistically interesting text for fashion imagery, overcoming the limitations of end\-to\-end captioners that have problems with attribute fidelity and domain generalization. The pipeline combines a YOLO\-based detector for multi\-garment localization, k\-means clustering for dominant color extraction, and a CLIP\-FAISS retrieval module for fabric and gender attribute inference based on a structured product index. These attributes, together with retrieved style examples, create a factual evidence pack that is used to guide an LLM to generate human\-like captions and contextually rich hashtags. A fine\-tuned BLIP model is used as a supervised baseline model for comparison. Experimental results show that the YOLO detector is able to obtain a mean Average Precision \(mAP@0.5\) of 0.71 for nine categories of garments. The RAG\-LLM pipeline generates expressive attribute\-aligned captions and achieves mean attribute coverage of 0.80 with full coverage at the 50% threshold in hashtag generation, whereas BLIP gives higher lexical overlap and lower generalization. The retrieval\-augmented approach exhibits better factual grounding, less hallucination, and great potential for scalable deployment in various clothing domains. These results demonstrate the use of retrieval\-augmented generation as an effective and interpretable paradigm for automated and visually grounded fashion content generation.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.19149v1)

---


## MambaRefine\-YOLO: A Dual\-Modality Small Object Detector for UAV Imagery / 

发布日期：2025-11-24

作者：Shuyu Cao

摘要：Small object detection in Unmanned Aerial Vehicle \(UAV\) imagery is a persistent challenge, hindered by low resolution and background clutter. While fusing RGB and infrared \(IR\) data offers a promising solution, existing methods often struggle with the trade\-off between effective cross\-modal interaction and computational efficiency. In this letter, we introduce MambaRefine\-YOLO. Its core contributions are a Dual\-Gated Complementary Mamba fusion module \(DGC\-MFM\) that adaptively balances RGB and IR modalities through illumination\-aware and difference\-aware gating mechanisms, and a Hierarchical Feature Aggregation Neck \(HFAN\) that uses a \`\`refine\-then\-fuse'' strategy to enhance multi\-scale features. Our comprehensive experiments validate this dual\-pronged approach. On the dual\-modality DroneVehicle dataset, the full model achieves a state\-of\-the\-art mAP of 83.2%, an improvement of 7.9% over the baseline. On the single\-modality VisDrone dataset, a variant using only the HFAN also shows significant gains, demonstrating its general applicability. Our work presents a superior balance between accuracy and speed, making it highly suitable for real\-world UAV applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.19134v1)

---


## Peregrine: One\-Shot Fine\-Tuning for FHE Inference of General Deep CNNs / 

发布日期：2025-11-24

作者：Huaming Ling

摘要：We address two fundamental challenges in adapting general deep CNNs for FHE\-based inference: approximating non\-linear activations such as ReLU with low\-degree polynomials while minimizing accuracy degradation, and overcoming the ciphertext capacity barrier that constrains high\-resolution image processing on FHE inference. Our contributions are twofold: \(1\) a single\-stage fine\-tuning \(SFT\) strategy that directly converts pre\-trained CNNs into FHE\-friendly forms using low\-degree polynomials, achieving competitive accuracy with minimal training overhead; and \(2\) a generalized interleaved packing \(GIP\) scheme that is compatible with feature maps of virtually arbitrary spatial resolutions, accompanied by a suite of carefully designed homomorphic operators that preserve the GIP\-form encryption throughout computation. These advances enable efficient, end\-to\-end FHE inference across diverse CNN architectures. Experiments on CIFAR\-10, ImageNet, and MS COCO demonstrate that the FHE\-friendly CNNs obtained via our SFT strategy achieve accuracy comparable to baselines using ReLU or SiLU activations. Moreover, this work presents the first demonstration of FHE\-based inference for YOLO architectures in object detection leveraging low\-degree polynomial activations.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.18976v1)

---


## AIRHILT: A Human\-in\-the\-Loop Testbed for Multimodal Conflict Detection in Aviation / 

发布日期：2025-11-24

作者：Omar Garib

摘要：We introduce AIRHILT \(Aviation Integrated Reasoning, Human\-in\-the\-Loop Testbed\), a modular and lightweight simulation environment designed to evaluate multimodal pilot and air traffic control \(ATC\) assistance systems for aviation conflict detection. Built on the open\-source Godot engine, AIRHILT synchronizes pilot and ATC radio communications, visual scene understanding from camera streams, and ADS\-B surveillance data within a unified, scalable platform. The environment supports pilot\- and controller\-in\-the\-loop interactions, providing a comprehensive scenario suite covering both terminal area and en route operational conflicts, including communication errors and procedural mistakes. AIRHILT offers standardized JSON\-based interfaces that enable researchers to easily integrate, swap, and evaluate automatic speech recognition \(ASR\), visual detection, decision\-making, and text\-to\-speech \(TTS\) models. We demonstrate AIRHILT through a reference pipeline incorporating fine\-tuned Whisper ASR, YOLO\-based visual detection, ADS\-B\-based conflict logic, and GPT\-OSS\-20B structured reasoning, and present preliminary results from representative runway\-overlap scenarios, where the assistant achieves an average time\-to\-first\-warning of approximately 7.7 s, with average ASR and vision latencies of approximately 5.9 s and 0.4 s, respectively. The AIRHILT environment and scenario suite are openly available, supporting reproducible research on multimodal situational awareness and conflict detection in aviation; code and scenarios are available at https://github.com/ogarib3/airhilt.

中文摘要：


代码链接：https://github.com/ogarib3/airhilt.

论文链接：[阅读更多](http://arxiv.org/abs/2511.18718v1)

---

