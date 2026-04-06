# 每日从arXiv中获取最新YOLO相关论文


## Can VLMs Truly Forget? Benchmarking Training\-Free Visual Concept Unlearning / 

发布日期：2026-04-03

作者：Zhangyun Tan

摘要：VLMs trained on web\-scale data retain sensitive and copyrighted visual concepts that deployment may require removing. Training\-based unlearning methods share a structural flaw: fine\-tuning on a narrow forget set degrades general capabilities before unlearning begins, making it impossible to attribute subsequent performance drops to the unlearning procedure itself. Training\-free approaches sidestep this by suppressing concepts through prompts or system instructions, but no rigorous benchmark exists for evaluating them on visual tasks.   We introduce VLM\-UnBench, the first benchmark for training\-free visual concept unlearning in VLMs. It covers four forgetting levels, 7 source datasets, and 11 concept axes, and pairs a three\-level probe taxonomy with five evaluation conditions to separate genuine forgetting from instruction compliance. Across 8 evaluation settings and 13 VLM configurations, realistic unlearning prompts leave forget accuracy near the no\-instruction baseline; meaningful reductions appear only under oracle conditions that disclose the target concept to the model. Object and scene concepts are the most resistant to suppression, and stronger instruction\-tuned models remain capable despite explicit forget instructions. These results expose a clear gap between prompt\-level suppression and true visual concept erasure.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.03114v1)

---


## Deep Neural Network Based Roadwork Detection for Autonomous Driving / 

发布日期：2026-04-02

作者：Sebastian Wullrich

摘要：Road construction sites create major challenges for both autonomous vehicles and human drivers due to their highly dynamic and heterogeneous nature. This paper presents a real\-time system that detects and localizes roadworks by combining a YOLO neural network with LiDAR data. The system identifies individual roadwork objects while driving, merges them into coherent construction sites and records their outlines in world coordinates. The model training was based on an adapted US dataset and a new dataset collected from test drives with a prototype vehicle in Berlin, Germany. Evaluations on real\-world road construction sites showed a localization accuracy below 0.5 m. The system can support traffic authorities with up\-to\-date roadwork data and could enable autonomous vehicles to navigate construction sites more safely in the future.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.02282v1)

---


## Fluently Lying: Adversarial Robustness Can Be Substrate\-Dependent / 

发布日期：2026-04-01

作者：Daye Kang

摘要：The primary tools used to monitor and defend object detectors under adversarial attack assume that when accuracy degrades, detection count drops in tandem. This coupling was assumed, not measured. We report a counterexample observed on a single model: under standard PGD, EMS\-YOLO, a spiking neural network \(SNN\) object detector, retains more than 70% of its detections while mAP collapses from 0.528 to 0.042. We term this count\-preserving accuracy collapse Quality Corruption \(QC\), to distinguish it from the suppression that dominates untargeted evaluation. Across four SNN architectures and two threat models \(l\-infinity and l\-2\), QC appears only in one of the four detectors tested \(EMS\-YOLO\). On this model, all five standard defense components fail to detect or mitigate QC, suggesting the defense ecosystem may rely on a shared assumption calibrated on a single substrate. These results provide, to our knowledge, the first evidence that adversarial failure modes can be substrate\-dependent.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.00605v1)

---


## AutoFormBench: Benchmark Dataset for Automating Form Understanding / 

发布日期：2026-03-31

作者：Gaurab Baral

摘要：Automated processing of structured documents such as government forms, healthcare records, and enterprise invoices remains a persistent challenge due to the high degree of layout variability encountered in real\-world settings. This paper introduces AutoFormBench, a benchmark dataset of 407 annotated real\-world forms spanning government, healthcare, and enterprise domains, designed to train and evaluate form element detection models. We present a systematic comparison of classical OpenCV approaches and four YOLO architectures \(YOLOv8, YOLOv11, YOLOv26\-s, and YOLOv26\-l\) for localizing and classifying fillable form elements. specifically checkboxes, input lines, and text boxes across diverse PDF document types. YOLOv11 demonstrates consistently superior performance in both F1 score and Jaccard accuracy across all element classes and tolerance levels.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.29832v1)

---


## Sim\-to\-Real Fruit Detection Using Synthetic Data: Quantitative Evaluation and Embedded Deployment with Isaac Sim / 

发布日期：2026-03-30

作者：Martina Hutter\-Mironovova

摘要：This study investigates the effectiveness of synthetic data for sim\-to\-real transfer in object detection under constrained data conditions and embedded deployment requirements. Synthetic datasets were generated in NVIDIA Isaac Sim and combined with limited real\-world fruit images to train YOLO\-based detection models under real\-only, synthetic\-only, and hybrid regimes. Performance was evaluated on two test datasets: an in\-domain dataset with conditions matching the training data and a domain shift dataset containing real fruit and different background conditions. Results show that models trained exclusively on real data achieve the highest accuracy, while synthetic\-only models exhibit reduced performance due to a domain gap. Hybrid training strategies significantly improve performance compared to synthetic\-only approaches and achieve results close to real\-only training while reducing the need for manual annotation. Under domain shift conditions, all models show performance degradation, with hybrid models providing improved robustness. The trained models were successfully deployed on a Jetson Orin NX using TensorRT optimization, achieving real\-time inference performance. The findings highlight that synthetic data is most effective when used in combination with real data and that deployment constraints must be considered alongside detection accuracy.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.28670v1)

---

