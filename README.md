# 每日从arXiv中获取最新YOLO相关论文


## BEM: Training\-Free Background Embedding Memory for False\-Positive Suppression in Real\-Time Fixed\-Background Camera / 

发布日期：2026-04-13

作者：Junwoo Park

摘要：Pretrained detectors perform well on benchmarks but often suffer performance degradation in real\-world deployments due to distribution gaps between training data and target environments. COCO\-like benchmarks emphasize category diversity rather than instance density, causing detectors trained under per\-class sparsity to struggle in dense, single\- or few\-class scenes such as surveillance and traffic monitoring. In fixed\-camera environments, the quasi\-static background provides a stable, label\-free prior that can be exploited at inference to suppress spurious detections. To address the issue, we propose Background Embedding Memory \(BEM\), a lightweight, training\-free, weight\-frozen module that can be attached to pretrained detectors during inference. BEM estimates clean background embeddings, maintains a prototype memory, and re\-scores detection logits with an inverse\-similarity, rank\-weighted penalty, effectively reducing false positives while maintaining recall. Empirically, background\-frame cosine similarity correlates negatively with object count and positively with Precision\-Confidence AUC \(P\-AUC\), motivating its use as a training\-free control signal. Across YOLO and RT\-DETR families on LLVIP and simulated surveillance streams, BEM consistently reduces false positives while preserving real\-time performance. Our code is available at https://github.com/Leo\-Park1214/Background\-Embedding\-Memory.git

中文摘要：


代码链接：https://github.com/Leo-Park1214/Background-Embedding-Memory.git

论文链接：[阅读更多](http://arxiv.org/abs/2604.11714v1)

---


## BLPR: Robust License Plate Recognition under Viewpoint and Illumination Variations via Confidence\-Driven VLM Fallback / 

发布日期：2026-04-10

作者：Guillermo Auza Banegas

摘要：Robust license plate recognition in unconstrained environments remains a significant challenge, particularly in underrepresented regions with limited data availability and unique visual characteristics, such as Bolivia. Recognition accuracy in real\-world conditions is often degraded by factors such as illumination changes and viewpoint distortion. To address these challenges, we introduce BLPR, a novel deep learning\-based License Plate Detection and Recognition \(LPDR\) framework specifically designed for Bolivian license plates. The proposed system follows a two\-stage pipeline where a YOLO\-based detector is pretrained on synthetic data generated in Blender to simulate extreme perspectives and lighting conditions, and subsequently fine\-tuned on street\-level data collected in La Paz, Bolivia. Detected plates are geometrically rectified and passed to a character recognition model. To improve robustness under ambiguous scenarios, a lightweight vision\-language model \(Gemma3 4B\) is selectively triggered as a confidence\-based fallback mechanism. The proposed framework further leverages synthetic\-to\-real domain adaptation to improve robustness under diverse real\-world conditions. We also introduce the first publicly available Bolivian LPDR dataset, enabling evaluation under diverse viewpoint and illumination conditions. The system achieves a character\-level recognition accuracy of 89.6% on real\-world data, demonstrating its effectiveness for deployment in challenging urban environments. Our project is publicly available at https://github.com/EdwinTSalcedo/BLPR.

中文摘要：


代码链接：https://github.com/EdwinTSalcedo/BLPR.

论文链接：[阅读更多](http://arxiv.org/abs/2604.09927v1)

---


## Does Your VFM Speak Plant? The Botanical Grammar of Vision Foundation Models for Object Detection / 

发布日期：2026-04-10

作者：Lars Lundqvist

摘要：Vision foundation models \(VFMs\) offer the promise of zero\-shot object detection without task\-specific training data, yet their performance in complex agricultural scenes remains highly sensitive to text prompt construction. We present a systematic prompt optimization framework evaluating four open\-vocabulary detectors \-\- YOLO World, SAM3, Grounding DINO, and OWLv2 \-\- for cowpea flower and pod detection across synthetic and real field imagery. We decompose prompts into eight axes and conduct one\-factor\-at\-a\-time analysis followed by combinatorial optimization, revealing that models respond divergently to prompt structure: conditions that optimize one architecture can collapse another. Applying model\-specific combinatorial prompts yields substantial gains over a naive species\-name baseline, including \+0.357 mAP@0.5 for YOLO World and \+0.362 mAP@0.5 for OWLv2 on synthetic cowpea flower data. To evaluate cross\-task generalization, we use an LLM to translate the discovered axis structure to a morphologically distinct target \-\- cowpea pods \-\- and compare against prompting using the discovered optimal structures from synthetic flower data. Crucially, prompt structures optimized exclusively on synthetic data transfer effectively to real\-world fields: synthetic\-pipeline prompts match or exceed those discovered on labeled real data for the majority of model\-object combinations \(flower: 0.374 vs. 0.353 for YOLO World; pod: 0.429 vs. 0.371 for SAM3\). Our findings demonstrate that prompt engineering can substantially close the gap between zero\-shot VFMs and supervised detectors without requiring manual annotation, and that optimal prompts are model\-specific, non\-obvious, and transferable across domains.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.09920v1)

---


## AI Driven Soccer Analysis Using Computer Vision / 

发布日期：2026-04-09

作者：Adrian Manchado

摘要：Sport analysis is crucial for team performance since it provides actionable data that can inform coaching decisions, improve player performance, and enhance team strategies. To analyze more complex features from game footage, a computer vision model can be used to identify and track key entities from the field. We propose the use of an object detection and tracking system to predict player positioning throughout the game. To translate this to positioning in relation to the field dimensions, we use a point prediction model to identify key points on the field and combine these with known field dimensions to extract actual distances. For the player\-identification model, object detection models like YOLO and Faster R\-CNN are evaluated on the accuracy of our custom video footage using multiple different evaluation metrics. The goal is to identify the best model for object identification to obtain the most accurate results when paired with SAM2 \(Segment Anything Model 2\) for segmentation and tracking. For the key point detection model, we use a CNN model to find consistent locations in the soccer field. Through homography, the positions of points and objects in the camera perspective will be transformed to a real\-ground perspective. The segmented player masks from SAM2 are transformed from camera perspective to real\-world field coordinates through homography, regardless of camera angle or movement. The transformed real\-world coordinates can be used to calculate valuable tactical insights including player speed, distance covered, positioning heatmaps, and more complex team statistics, providing coaches and players with actionable performance data previously unavailable from standard video analysis.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.08722v1)

---


## Your Agent Is Mine: Measuring Malicious Intermediary Attacks on the LLM Supply Chain / 

发布日期：2026-04-09

作者：Hanzhi Liu

摘要：Large language model \(LLM\) agents increasingly rely on third\-party API routers to dispatch tool\-calling requests across multiple upstream providers. These routers operate as application\-layer proxies with full plaintext access to every in\-flight JSON payload, yet no provider enforces cryptographic integrity between client and upstream model. We present the first systematic study of this attack surface. We formalize a threat model for malicious LLM API routers and define two core attack classes, payload injection \(AC\-1\) and secret exfiltration \(AC\-2\), together with two adaptive evasion variants: dependency\-targeted injection \(AC\-1.a\) and conditional delivery \(AC\-1.b\). Across 28 paid routers purchased from Taobao, Xianyu, and Shopify\-hosted storefronts and 400 free routers collected from public communities, we find 1 paid and 8 free routers actively injecting malicious code, 2 deploying adaptive evasion triggers, 17 touching researcher\-owned AWS canary credentials, and 1 draining ETH from a researcher\-owned private key. Two poisoning studies further show that ostensibly benign routers can be pulled into the same attack surface: a leaked OpenAI key generates 100M GPT\-5.4 tokens and more than seven Codex sessions, while weakly configured decoys yield 2B billed tokens, 99 credentials across 440 Codex sessions, and 401 sessions already running in autonomous YOLO mode. We build Mine, a research proxy that implements all four attack classes against four public agent frameworks, and use it to evaluate three deployable client\-side defenses: a fail\-closed policy gate, response\-side anomaly screening, and append\-only transparency logging.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.08407v1)

---

