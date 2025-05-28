# 每日从arXiv中获取最新YOLO相关论文


## YOLO\-SPCI: Enhancing Remote Sensing Object Detection via Selective\-Perspective\-Class Integration / 

发布日期：2025-05-27

作者：Xinyuan Wang

摘要：Object detection in remote sensing imagery remains a challenging task due to extreme scale variation, dense object distributions, and cluttered backgrounds. While recent detectors such as YOLOv8 have shown promising results, their backbone architectures lack explicit mechanisms to guide multi\-scale feature refinement, limiting performance on high\-resolution aerial data. In this work, we propose YOLO\-SPCI, an attention\-enhanced detection framework that introduces a lightweight Selective\-Perspective\-Class Integration \(SPCI\) module to improve feature representation. The SPCI module integrates three components: a Selective Stream Gate \(SSG\) for adaptive regulation of global feature flow, a Perspective Fusion Module \(PFM\) for context\-aware multi\-scale integration, and a Class Discrimination Module \(CDM\) to enhance inter\-class separability. We embed two SPCI blocks into the P3 and P5 stages of the YOLOv8 backbone, enabling effective refinement while preserving compatibility with the original neck and head. Experiments on the NWPU VHR\-10 dataset demonstrate that YOLO\-SPCI achieves superior performance compared to state\-of\-the\-art detectors.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.21370v1)

---


## YOLO\-FireAD: Efficient Fire Detection via Attention\-Guided Inverted Residual Learning and Dual\-Pooling Feature Preservation / 

发布日期：2025-05-27

作者：Weichao Pan

摘要：Fire detection in dynamic environments faces continuous challenges, including the interference of illumination changes, many false detections or missed detections, and it is difficult to achieve both efficiency and accuracy. To address the problem of feature extraction limitation and information loss in the existing YOLO\-based models, this study propose You Only Look Once for Fire Detection with Attention\-guided Inverted Residual and Dual\-pooling Downscale Fusion \(YOLO\-FireAD\) with two core innovations: \(1\) Attention\-guided Inverted Residual Block \(AIR\) integrates hybrid channel\-spatial attention with inverted residuals to adaptively enhance fire features and suppress environmental noise; \(2\) Dual Pool Downscale Fusion Block \(DPDF\) preserves multi\-scale fire patterns through learnable fusion of max\-average pooling outputs, mitigating small\-fire detection failures. Extensive evaluation on two public datasets shows the efficient performance of our model. Our proposed model keeps the sum amount of parameters \(1.45M, 51.8% lower than YOLOv8n\) \(4.6G, 43.2% lower than YOLOv8n\), and mAP75 is higher than the mainstream real\-time object detection models YOLOv8n, YOL\-Ov9t, YOLOv10n, YOLO11n, YOLOv12n and other YOLOv8 variants 1.3\-5.5%.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.20884v1)

---


## Detailed Evaluation of Modern Machine Learning Approaches for Optic Plastics Sorting / 

发布日期：2025-05-22

作者：Vaishali Maheshkar

摘要：According to the EPA, only 25% of waste is recycled, and just 60% of U.S. municipalities offer curbside recycling. Plastics fare worse, with a recycling rate of only 8%; an additional 16% is incinerated, while the remaining 76% ends up in landfills. The low plastic recycling rate stems from contamination, poor economic incentives, and technical difficulties, making efficient recycling a challenge. To improve recovery, automated sorting plays a critical role. Companies like AMP Robotics and Greyparrot utilize optical systems for sorting, while Materials Recovery Facilities \(MRFs\) employ Near\-Infrared \(NIR\) sensors to detect plastic types.   Modern optical sorting uses advances in computer vision such as object recognition and instance segmentation, powered by machine learning. Two\-stage detectors like Mask R\-CNN use region proposals and classification with deep backbones like ResNet. Single\-stage detectors like YOLO handle detection in one pass, trading some accuracy for speed. While such methods excel under ideal conditions with a large volume of labeled training data, challenges arise in realistic scenarios, emphasizing the need to further examine the efficacy of optic detection for automated sorting.   In this study, we compiled novel datasets totaling 20,000\+ images from varied sources. Using both public and custom machine learning pipelines, we assessed the capabilities and limitations of optical recognition for sorting. Grad\-CAM, saliency maps, and confusion matrices were employed to interpret model behavior. We perform this analysis on our custom trained models from the compiled datasets. To conclude, our findings are that optic recognition methods have limited success in accurate sorting of real\-world plastics at MRFs, primarily because they rely on physical properties such as color and shape.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.16513v1)

---


## ViQAgent: Zero\-Shot Video Question Answering via Agent with Open\-Vocabulary Grounding Validation / 

发布日期：2025-05-21

作者：Tony Montes

摘要：Recent advancements in Video Question Answering \(VideoQA\) have introduced LLM\-based agents, modular frameworks, and procedural solutions, yielding promising results. These systems use dynamic agents and memory\-based mechanisms to break down complex tasks and refine answers. However, significant improvements remain in tracking objects for grounding over time and decision\-making based on reasoning to better align object references with language model outputs, as newer models get better at both tasks. This work presents an LLM\-brained agent for zero\-shot Video Question Answering \(VideoQA\) that combines a Chain\-of\-Thought framework with grounding reasoning alongside YOLO\-World to enhance object tracking and alignment. This approach establishes a new state\-of\-the\-art in VideoQA and Video Understanding, showing enhanced performance on NExT\-QA, iVQA, and ActivityNet\-QA benchmarks. Our framework also enables cross\-checking of grounding timeframes, improving accuracy and providing valuable support for verification and increased output reliability across multiple video domains. The code is available at https://github.com/t\-montes/viqagent.

中文摘要：


代码链接：https://github.com/t-montes/viqagent.

论文链接：[阅读更多](http://arxiv.org/abs/2505.15928v1)

---


## Towards Self\-Improvement of Diffusion Models via Group Preference Optimization / 

发布日期：2025-05-16

作者：Renjie Chen

摘要：Aligning text\-to\-image \(T2I\) diffusion models with Direct Preference Optimization \(DPO\) has shown notable improvements in generation quality. However, applying DPO to T2I faces two challenges: the sensitivity of DPO to preference pairs and the labor\-intensive process of collecting and annotating high\-quality data. In this work, we demonstrate that preference pairs with marginal differences can degrade DPO performance. Since DPO relies exclusively on relative ranking while disregarding the absolute difference of pairs, it may misclassify losing samples as wins, or vice versa. We empirically show that extending the DPO from pairwise to groupwise and incorporating reward standardization for reweighting leads to performance gains without explicit data selection. Furthermore, we propose Group Preference Optimization \(GPO\), an effective self\-improvement method that enhances performance by leveraging the model's own capabilities without requiring external data. Extensive experiments demonstrate that GPO is effective across various diffusion models and tasks. Specifically, combining with widely used computer vision models, such as YOLO and OCR, the GPO improves the accurate counting and text rendering capabilities of the Stable Diffusion 3.5 Medium by 20 percentage points. Notably, as a plug\-and\-play method, no extra overhead is introduced during inference.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.11070v1)

---

