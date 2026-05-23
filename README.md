# 每日从arXiv中获取最新YOLO相关论文


## Dual\-Integrated Low\-Latency Single\-Lens Infrared Computational Imaging for Object Detection / 

发布日期：2026-05-21

作者：Xuquan Wang

摘要：Computational imaging enables compact infrared systems, but deep\-learning pipelines that combine image reconstruction and object detection often introduce substantial inference latency. Most existing acceleration strategies compress the reconstruction network while overlooking physical priors from the optical path, leaving a trade\-off between accuracy and speed. We present Physics\-aware Dual\-Integrated Network \(PDI\-Net\), a low\-latency framework that integrates infrared reconstruction with object detection and further embeds optical priors into the learning process. PDI\-Net uses a supervised U\-Net during training, while a semi\-U\-Net encoder shares features directly with a YOLO\-based detector during inference, avoiding full image reconstruction. To bridge the gap between fidelity\-oriented reconstruction features and detection\-oriented semantics, we introduce a physics\-aware large\-small bridge \(PALS\-Bridge\), which uses field\-dependent point spread function priors to adaptively modulate multiscale convolutional branches. A physics\-informed optical degradation simulation pipeline is also developed for training and validation. The method is deployed on a single\-lens infrared camera, reducing system weight by about 50% compared with traditional multi\-lens designs. On the M3FD benchmark under low\-SNR conditions, PDI\-Net reduces inference time by 84.06% compared with the Rec\+Det with pruning strategy while improving mAP@0.5:0.95 by 5.07%. These results demonstrate compact, low\-latency computational infrared imaging for real\-time object detection on resource\-constrained platforms.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2605.21964v1)

---


## GSA\-YOLO: A High\-Efficiency Framework via Structured Sparsity and Adaptive Knowledge Distillation for Real\-Time X\-ray Security Inspection / 

发布日期：2026-05-20

作者：Jiahao Kong

摘要：X\-ray security inspection requires accurate real\-time detection of prohibited items, but existing models often struggle to balance the challenges of severe occlusion, complex clutter, and strict speed requirements. To overcome these challenges, this paper proposes GSA\-YOLO, a novel lightweight framework built upon the YOLOv8n architecture, specifically engineered to enhance detection robustness and inference efficiency. GSA\-YOLO strategically integrates structured sparsity and adaptive knowledge transfer through three core components: Group Lasso \(GL\) applied to the network neck for robust feature extraction; Sparse Structure Selection \(SSS\) applied to the detection head for significant model slimming; and an Adaptive Knowledge Distillation \(Ada\-KD\) mechanism for comprehensive accuracy recovery. This integrated approach synergistically enhances feature representation while pruning redundant channels, maximizing model efficiency without sacrificing performance. Rigorous evaluations on the HiXray and PIDray datasets confirm GSA\-YOLO's comprehensive capability, achieving a leading inference speed of 189.62 FPS, accompanied by a reduction in computational cost from 8.7G to 8.0G. Crucially, GSA\-YOLO secures mAP50:95 results of 0.531 and 0.679 on HiXray and PIDray, demonstrating 2.4% and 1.8% improvements over the baseline, respectively. Compared to other models, GSA\-YOLO exhibits enhanced accuracy while maintaining computational efficiency, making it a promising solution for practical X\-ray security inspection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2605.20669v1)

---


## LER\-YOLO: Reliability\-Aware Expert Routing for Misaligned RGB\-Infrared UAV Detection / 

发布日期：2026-05-20

作者：Liming Hou

摘要：Detecting small unmanned aerial vehicles from RGB\-infrared remote\-sensing pairs remains challenging due to tiny target scale, cluttered backgrounds, and spatial misalignment between heterogeneous sensors. Existing bimodal detectors often align or fuse features without assessing the reliability of local cross\-sensor correspondence, allowing mismatch artifacts to propagate into the detection head. To address this issue, we propose LER\-YOLO, a reliability\-aware sparse mixture\-of\-experts framework for misaligned RGB\-infrared UAV detection. LER\-YOLO first introduces an Uncertainty\-Aware Target Alignment module that resamples visible features toward the infrared reference and estimates a spatial reliability map. This reliability prior is then used by a Reliability\-Guided Sparse MoE Fusion module to adaptively select k experts from RGB\-dominant, infrared\-dominant, and interactive fusion experts, enabling trustworthy cross\-modal interaction while suppressing unreliable fusion. Experiments on the public MBU benchmark under a YOLOv5s\-family protocol show that LER\-YOLO achieves 89.7\+/\-0.2% AP50 over three independent seeds, with a best result of 89.9%. Extensive ablations, parameter\-matched comparisons, synthetic\-shift evaluations, and complexity analysis demonstrate that the gains mainly come from reliability\-guided expert routing rather than increased model capacity.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2605.20667v1)

---


## A novel YOLO26\-MoE optimized by an LLM agent for insulator fault detection considering UAV images / 

发布日期：2026-05-19

作者：João Pedro Matos\-Carvalho

摘要：The inspection of electrical power line insulators is essential for ensuring grid reliability and preventing failures caused by damaged or degraded insulation components. In recent years, Unmanned Aerial Vehicles \(UAVs\) combined with deep learning\-based vision systems have emerged as an effective solution for automating this process. However, insulator fault detection remains challenging due to small defect regions, heterogeneous fault patterns, complex backgrounds, and varying imaging conditions. To address these challenges, this paper proposes an optimized YOLO26\-MoE, a novel object detection architecture that integrates a sparse Mixture\-of\-Experts \(MoE\) module into the high\-resolution branch of the YOLO26 detector. The proposed modification enables adaptive feature refinement for subtle and diverse fault patterns while preserving the efficiency of a one\-stage detection framework. Hyperparameter optimization, final training, and evaluation were coordinated through a tool\-augmented Large Language Model \(LLM\) agent. The proposed model achieved 0.9900 mAP@0.5 and 0.9515 mAP@0.5:0.95, outperforming the latest YOLO versions. These results demonstrate that the proposed model provides an effective and reliable solution for UAV\-based insulator fault detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2605.19595v1)

---


## You Only Landmark Once: Lightweight U\-Net Face Super Resolution with YOLO\-World Landmark Heatmaps / 

发布日期：2026-05-13

作者：Riccardo Carraro

摘要：Face image super\-resolution aims to recover high\-resolution facial images from severely degraded inputs. Under extreme upscaling factors, fine facial details are often lost, making accurate reconstruction challenging. Existing methods typically rely on heavy network architectures, adversarial training schemes, or separate alignment networks, increasing model complexity and computational cost. To address these issues, we propose a lightweight U\-Net based\-architecture designed to reconstructs $128\{ times \}128$ facial images from severely degraded $16\{ times \}16$ inputs, achieving an $8 times $ magnification. A key contribution is a novel auxiliary\-training\-free supervision strategy that leverages heatmaps generated by YOLO\-World, an open\-vocabulary object detector, to localize key facial features such as eyes, nose, and mouth. These heatmaps are converted into spatial weights to form a heatmap\-guided loss that emphasizes reconstruction errors in semantically important regions. Unlike prior methods that require dedicated landmark or alignment networks, our approach directly reuses detector outputs as supervision, maintaining an efficient training and inference pipeline. Experiments on the aligned CelebA dataset demonstrate that the proposed loss consistently improves quantitative metrics and produces sharper, more realistic reconstructions. Overall, our results show that lightweight networks can effectively exploit detection\-driven priors for perceptually convincing extreme upscaling, without adversarial training or increased computational cost.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2605.14166v1)

---

