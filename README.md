# 每日从arXiv中获取最新YOLO相关论文


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


## A High\-Performance Thermal Infrared Object Detection Framework with Centralized Regulation / 

发布日期：2025-05-16

作者：Jinke Li

摘要：Thermal Infrared \(TIR\) technology involves the use of sensors to detect and measure infrared radiation emitted by objects, and it is widely utilized across a broad spectrum of applications. The advancements in object detection methods utilizing TIR images have sparked significant research interest. However, most traditional methods lack the capability to effectively extract and fuse local\-global information, which is crucial for TIR\-domain feature attention. In this study, we present a novel and efficient thermal infrared object detection framework, known as CRT\-YOLO, that is based on centralized feature regulation, enabling the establishment of global\-range interaction on TIR information. Our proposed model integrates efficient multi\-scale attention \(EMA\) modules, which adeptly capture long\-range dependencies while incurring minimal computational overhead. Additionally, it leverages the Centralized Feature Pyramid \(CFP\) network, which offers global regulation of TIR features. Extensive experiments conducted on two benchmark datasets demonstrate that our CRT\-YOLO model significantly outperforms conventional methods for TIR image object detection. Furthermore, the ablation study provides compelling evidence of the effectiveness of our proposed modules, reinforcing the potential impact of our approach on advancing the field of thermal infrared object detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.10825v1)

---


## Geofenced Unmanned Aerial Robotic Defender for Deer Detection and Deterrence \(GUARD\) / 

发布日期：2025-05-16

作者：Ebasa Temesgen

摘要：Wildlife\-induced crop damage, particularly from deer, threatens agricultural productivity. Traditional deterrence methods often fall short in scalability, responsiveness, and adaptability to diverse farmland environments. This paper presents an integrated unmanned aerial vehicle \(UAV\) system designed for autonomous wildlife deterrence, developed as part of the Farm Robotics Challenge. Our system combines a YOLO\-based real\-time computer vision module for deer detection, an energy\-efficient coverage path planning algorithm for efficient field monitoring, and an autonomous charging station for continuous operation of the UAV. In collaboration with a local Minnesota farmer, the system is tailored to address practical constraints such as terrain, infrastructure limitations, and animal behavior. The solution is evaluated through a combination of simulation and field testing, demonstrating robust detection accuracy, efficient coverage, and extended operational time. The results highlight the feasibility and effectiveness of drone\-based wildlife deterrence in precision agriculture, offering a scalable framework for future deployment and extension.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.10770v1)

---

