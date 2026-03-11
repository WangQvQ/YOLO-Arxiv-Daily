# 每日从arXiv中获取最新YOLO相关论文


## The Patrologia Graeca Corpus: OCR, Annotation, and Open Release of Noisy Nineteenth\-Century Polytonic Greek Editions / 

发布日期：2026-03-10

作者：Chahan Vidal\-Gorène

摘要：We present the Patrologia Graeca Corpus, the first large\-scale open OCR and linguistic resource for nineteenthcentury editions of Ancient Greek. The collection covers the remaining undigitized volumes of the Patrologia Graeca \(PG\), printed in complex bilingual \(Greek\-Latin\) layouts and characterized by highly degraded polytonic Greek typography. Through a dedicated pipeline combining YOLO\-based layout detection and CRNN\-based text recognition, we achieve a character error rate \(CER\) of 1.05% and a word error rate \(WER\) of 4.69%, largely outperforming existing OCR systems for polytonic Greek. The resulting corpus contains around six million lemmatized and part\-of\-speech tagged tokens, aligned with full OCR and layout annotations. Beyond its philological value, this corpus establishes a new benchmark for OCR on noisy polytonic Greek and provides training material for future models, including LLMs.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.09470v1)

---


## YOLO\-NAS\-Bench: A Surrogate Benchmark with Self\-Evolving Predictors for YOLO Architecture Search / 

发布日期：2026-03-10

作者：Zhe Li

摘要：Neural Architecture Search \(NAS\) for object detection is severely bottlenecked by high evaluation cost, as fully training each candidate YOLO architecture on COCO demands days of GPU time. Meanwhile, existing NAS benchmarks largely target image classification, leaving the detection community without a comparable benchmark for NAS evaluation. To address this gap, we introduce YOLO\-NAS\-Bench, the first surrogate benchmark tailored to YOLO\-style detectors. YOLO\-NAS\-Bench defines a search space spanning channel width, block depth, and operator type across both backbone and neck, covering the core modules of YOLOv8 through YOLO12. We sample 1,000 architectures via random, stratified, and Latin Hypercube strategies, train them on COCO\-mini, and build a LightGBM surrogate predictor. To sharpen the predictor in the high\-performance regime most relevant to NAS, we propose a Self\-Evolving Mechanism that progressively aligns the predictor's training distribution with the high\-performance frontier, by using the predictor itself to discover and evaluate informative architectures in each iteration. This method grows the pool to 1,500 architectures and raises the ensemble predictor's R2 from 0.770 to 0.815 and Sparse Kendall Tau from 0.694 to 0.752, demonstrating strong predictive accuracy and ranking consistency. Using the final predictor as the fitness function for evolutionary search, we discover architectures that surpass all official YOLOv8\-YOLO12 baselines at comparable latency on COCO\-mini, confirming the predictor's discriminative power for top\-performing detection architectures.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.09405v1)

---


## ER\-Pose: Rethinking Keypoint\-Driven Representation Learning for Real\-Time Human Pose Estimation / 

发布日期：2026-03-09

作者：Nanjun Li

摘要：Single\-stage multi\-person pose estimation aims to jointly perform human localization and keypoint prediction within a unified framework, offering advantages in inference efficiency and architectural simplicity. Consequently, multi\-scale real\-time detection architectures, such as YOLO\-like models, are widely adopted for real\-time pose estimation. However, these approaches typically inherit a box\-driven modeling paradigm from object detection, in which pose estimation is implicitly constrained by bounding\-box supervision during training. This formulation introduces biases in sample assignment and feature representation, resulting in task misalignment and ultimately limiting pose estimation accuracy. In this work, we revisit box\-driven single\-stage pose estimation from a keypoint\-driven perspective and identify semantic conflicts among parallel objectives as a key source of performance degradation. To address this issue, we propose a keypoint\-driven learning paradigm that elevates pose estimation to a primary prediction objective. Specifically, we remove bounding\-box prediction and redesign the prediction head to better accommodate the high\-dimensional structured representations for pose estimation. We further introduce a keypoint\-driven dynamic sample assignment strategy to align training objectives with pose evaluation metrics, enabling dense supervision during training and efficient NMS\-free inference. In addition, we propose a smooth OKS\-based loss function to stabilize optimization in regression\-based pose estimation. Based on these designs, we develop a single\-stage multi\-person pose estimation framework, termed ER\-Pose. On MS COCO and CrowdPose, ER\-Pose\-n achieves AP improvements of 3.2/6.7 without pre\-training and 7.4/4.9 with pre\-training respectively compared with the baseline YOLO\-Pose. These improvements are achieved with fewer parameters and higher inference efficiency.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.08681v1)

---


## Real\-Time Drone Detection in Event Cameras via Per\-Pixel Frequency Analysis / 

发布日期：2026-03-09

作者：Michael Bezick

摘要：Detecting fast\-moving objects, such as unmanned aerial vehicle \(UAV\), from event camera data is challenging due to the sparse, asynchronous nature of the input. Traditional Discrete Fourier Transforms \(DFT\) are effective at identifying periodic signals, such as spinning rotors, but they assume uniformly sampled data, which event cameras do not provide. We propose a novel per\-pixel temporal analysis framework using the Non\-uniform Discrete Fourier Transform \(NDFT\), which we call Drone Detection via Harmonic Fingerprinting \(DDHF\). Our method uses purely analytical techniques that identify the frequency signature of drone rotors, as characterized by frequency combs in their power spectra, enabling a tunable and generalizable algorithm that achieves accurate real\-time localization of UAV. We compare against a YOLO detector under equivalent conditions, demonstrating improvement in accuracy and latency across a difficult array of drone speeds, distances, and scenarios. DDHF achieves an average localization F1 score of 90.89% and average latency of 2.39ms per frame, while YOLO achieves an F1 score of 66.74% and requires 12.40ms per frame. Through utilization of purely analytic techniques, DDHF is quickly tuned on small data, easily interpretable, and achieves competitive accuracies and latencies to deep learning alternatives.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.08386v1)

---


## Evaluating Synthetic Data for Baggage Trolley Detection in Airport Logistics / 

发布日期：2026-03-08

作者：Abdeldjalil Taibi

摘要：Efficient luggage trolley management is critical for reducing congestion and ensuring asset availability in modern airports. Automated detection systems face two main challenges. First, strict security and privacy regulations limit large\-scale data collection. Second, existing public datasets lack the diversity, scale, and annotation quality needed to handle dense, overlapping trolley arrangements typical of real\-world operations.   To address these limitations, we introduce a synthetic data generation pipeline based on a high\-fidelity Digital Twin of Algiers International Airport using NVIDIA Omniverse. The pipeline produces richly annotated data with oriented bounding boxes, capturing complex trolley formations, including tightly nested chains. We evaluate YOLO\-OBB using five training strategies: real\-only, synthetic\-only, linear probing, full fine\-tuning, and mixed training. This allows us to assess how synthetic data can complement limited real\-world annotations.   Our results show that mixed training with synthetic data and only 40 percent of real annotations matches or exceeds the full real\-data baseline, achieving 0.94 mAP@50 and 0.77 mAP@50\-95, while reducing annotation effort by 25 to 35 percent. Multi\-seed experiments confirm strong reproducibility with a standard deviation below 0.01 on mAP@50, demonstrating the practical effectiveness of synthetic data for automated trolley detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.07645v1)

---

