# 每日从arXiv中获取最新YOLO相关论文


## GLANCE: A Global\-Local Coordination Multi\-Agent Framework for Music\-Grounded Non\-Linear Video Editing / 

发布日期：2026-04-06

作者：Zihao Lin

摘要：Music\-grounded mashup video creation is a challenging form of video non\-linear editing, where a system must compose a coherent timeline from large collections of source videos while aligning with music rhythm, user intent, story completeness, and long\-range structural constraints. Existing approaches typically rely on fixed pipelines or simplified retrieval\-and\-concatenation paradigms, limiting their ability to adapt to diverse prompts and heterogeneous source materials. In this paper, we present GLANCE, a global\-local coordination multi\-agent framework for music\-grounded nonlinear video editing. GLANCE adopts a bi\-loop architecture for better editing practice: an outer loop performs long\-horizon planning and task\-graph construction, and an inner loop adopts the "Observe\-Think\-Act\-Verify" flow for segment\-wise editing tasks and their refinements. To address the cross\-segment and global conflict emerging after subtimelines composition, we introduce a dedicated global\-local coordination mechanism with both preventive and corrective components, which includes a novelly designed context controller, conflict region decomposition module, and a bottom\-up dynamic negotiation mechanism. To support rigorous evaluation, we construct MVEBench, a new benchmark that factorizes editing difficulty along task type, prompt specificity, and music length, and propose an agent\-as\-a\-judge evaluation framework for scalable multi\-dimensional assessment. Experimental results show that GLANCE consistently outperforms prior research baselines and open\-source product baselines under the same backbone models. With GPT\-4o\-mini as the backbone, GLANCE improves over the strongest baseline by 33.2% and 15.6% on two task settings, respectively. Human evaluation further confirms the quality of the generated videos and validates the effectiveness of the proposed evaluation framework.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.05076v1)

---


## SARES\-DEIM: Sparse Mixture\-of\-Experts Meets DETR for Robust SAR Ship Detection / 

发布日期：2026-04-05

作者：Fenghao Song

摘要：Ship detection in Synthetic Aperture Radar \(SAR\) imagery is fundamentally challenged by inherent coherent speckle noise, complex coastal clutter, and the prevalence of small\-scale targets. Conventional detectors, primarily designed for optical imagery, often exhibit limited robustness against SAR\-specific degradation and suffer from the loss of fine\-grained ship signatures during spatial downsampling. To address these limitations, we propose SARES\-DEIM, a domain\-aware detection framework grounded in the DEtection TRansformer \(DETR\) paradigm. Central to our approach is SARESMoE \(SAR\-aware Expert Selection Mixture\-of\-Experts\), a module leveraging a sparse gating mechanism to selectively route features toward specialized frequency and wavelet experts. This sparsely\-activated architecture effectively filters speckle noise and semantic clutter while maintaining high computational efficiency. Furthermore, we introduce the Space\-to\-Depth Enhancement Pyramid \(SDEP\) neck to preserve high\-resolution spatial cues from shallow stages, significantly improving the localization of small targets. Extensive experiments on two benchmark datasets demonstrate the superiority of SARES\-DEIM. Notably, on the challenging HRSID dataset, our model achieves a mAP50:95 of 76.4% and a mAP50 of 93.8%, outperforming state\-of\-the\-art YOLO\-series and specialized SAR detectors.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.04127v1)

---


## Can VLMs Truly Forget? Benchmarking Training\-Free Visual Concept Unlearning / 

发布日期：2026-04-03

作者：Zhangyun Tan

摘要：VLMs trained on web\-scale data retain sensitive and copyrighted visual concepts that deployment may require removing. Training\-based unlearning methods share a structural flaw: fine\-tuning on a narrow forget set degrades general capabilities before unlearning begins, making it impossible to attribute subsequent performance drops to the unlearning procedure itself. Training\-free approaches sidestep this by suppressing concepts through prompts or system instructions, but no rigorous benchmark exists for evaluating them on visual tasks.   We introduce VLM\-UnBench, the first benchmark for training\-free visual concept unlearning in VLMs. It covers four forgetting levels, 7 source datasets, and 11 concept axes, and pairs a three\-level probe taxonomy with five evaluation conditions to separate genuine forgetting from instruction compliance. Across 8 evaluation settings and 13 VLM configurations, realistic unlearning prompts leave forget accuracy near the no\-instruction baseline; meaningful reductions appear only under oracle conditions that disclose the target concept to the model. Object and scene concepts are the most resistant to suppression, and stronger instruction\-tuned models remain capable despite explicit forget instructions. These results expose a clear gap between prompt\-level suppression and true visual concept erasure.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.03114v1)

---


## YOLOv11 Demystified: A Practical Guide to High\-Performance Object Detection / 

发布日期：2026-04-03

作者：Nikhileswara Rao Sulake

摘要：YOLOv11 is the latest iteration in the You Only Look Once \(YOLO\) series of real\-time object detectors, introducing novel architectural modules to improve feature extraction and small\-object detection. In this paper, we present a detailed analysis of YOLOv11, including its backbone, neck, and head components. The model key innovations, the C3K2 blocks, Spatial Pyramid Pooling \- Fast \(SPPF\), and C2PSA \(Cross Stage Partial with Spatial Attention\) modules enhance spatial feature processing while preserving speed. We compare YOLOv11 performance to prior YOLO versions on standard benchmarks, highlighting improvements in mean Average Precision \(mAP\) and inference speed. Our results demonstrate that YOLOv11 achieves superior accuracy without sacrificing real\-time capabilities, making it well\-suited for applications in autonomous driving, surveillance, and video analytics.This work formalizes YOLOv11 in a research context, providing a clear reference for future studies.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.03349v1)

---


## Deep Neural Network Based Roadwork Detection for Autonomous Driving / 

发布日期：2026-04-02

作者：Sebastian Wullrich

摘要：Road construction sites create major challenges for both autonomous vehicles and human drivers due to their highly dynamic and heterogeneous nature. This paper presents a real\-time system that detects and localizes roadworks by combining a YOLO neural network with LiDAR data. The system identifies individual roadwork objects while driving, merges them into coherent construction sites and records their outlines in world coordinates. The model training was based on an adapted US dataset and a new dataset collected from test drives with a prototype vehicle in Berlin, Germany. Evaluations on real\-world road construction sites showed a localization accuracy below 0.5 m. The system can support traffic authorities with up\-to\-date roadwork data and could enable autonomous vehicles to navigate construction sites more safely in the future.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2604.02282v1)

---

