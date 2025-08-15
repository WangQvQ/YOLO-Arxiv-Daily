# 每日从arXiv中获取最新YOLO相关论文


## SynSpill: Improved Industrial Spill Detection With Synthetic Data / 

发布日期：2025-08-13

作者：Aaditya Baranwal

摘要：Large\-scale Vision\-Language Models \(VLMs\) have transformed general\-purpose visual recognition through strong zero\-shot capabilities. However, their performance degrades significantly in niche, safety\-critical domains such as industrial spill detection, where hazardous events are rare, sensitive, and difficult to annotate. This scarcity \-\- driven by privacy concerns, data sensitivity, and the infrequency of real incidents \-\- renders conventional fine\-tuning of detectors infeasible for most industrial settings.   We address this challenge by introducing a scalable framework centered on a high\-quality synthetic data generation pipeline. We demonstrate that this synthetic corpus enables effective Parameter\-Efficient Fine\-Tuning \(PEFT\) of VLMs and substantially boosts the performance of state\-of\-the\-art object detectors such as YOLO and DETR. Notably, in the absence of synthetic data \(SynSpill dataset\), VLMs still generalize better to unseen spill scenarios than these detectors. When SynSpill is used, both VLMs and detectors achieve marked improvements, with their performance becoming comparable.   Our results underscore that high\-fidelity synthetic data is a powerful means to bridge the domain gap in safety\-critical applications. The combination of synthetic generation and lightweight adaptation offers a cost\-effective, scalable pathway for deploying vision systems in industrial environments where real data is scarce/impractical to obtain.   Project Page: https://synspill.vercel.app

中文摘要：


代码链接：https://synspill.vercel.app

论文链接：[阅读更多](http://arxiv.org/abs/2508.10171v1)

---


## Robustness analysis of Deep Sky Objects detection models on HPC / 

发布日期：2025-08-13

作者：Olivier Parisot

摘要：Astronomical surveys and the growing involvement of amateur astronomers are producing more sky images than ever before, and this calls for automated processing methods that are accurate and robust. Detecting Deep Sky Objects \-\- such as galaxies, nebulae, and star clusters \-\- remains challenging because of their faint signals and complex backgrounds. Advances in Computer Vision and Deep Learning now make it possible to improve and automate this process. In this paper, we present the training and comparison of different detection models \(YOLO, RET\-DETR\) on smart telescope images, using High\-Performance Computing \(HPC\) to parallelise computations, in particular for robustness testing.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.09831v1)

---


## SegDAC: Segmentation\-Driven Actor\-Critic for Visual Reinforcement Learning / 

发布日期：2025-08-12

作者：Alexandre Brown

摘要：Visual reinforcement learning \(RL\) is challenging due to the need to learn both perception and actions from high\-dimensional inputs and noisy rewards. Although large perception models exist, integrating them effectively into RL for visual generalization and improved sample efficiency remains unclear. We propose SegDAC, a Segmentation\-Driven Actor\-Critic method. SegDAC uses Segment Anything \(SAM\) for object\-centric decomposition and YOLO\-World to ground segments semantically via text prompts. It includes a novel transformer\-based architecture that supports a dynamic number of segments at each time step and effectively learns which segments to focus on using online RL, without using human labels. By evaluating SegDAC over a challenging visual generalization benchmark using Maniskill3, which covers diverse manipulation tasks under strong visual perturbations, we demonstrate that SegDAC achieves significantly better visual generalization, doubling prior performance on the hardest setting and matching or surpassing prior methods in sample efficiency across all evaluated tasks.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.09325v1)

---


## Grasp\-HGN: Grasping the Unexpected / 

发布日期：2025-08-11

作者：Mehrshad Zandigohar

摘要：For transradial amputees, robotic prosthetic hands promise to regain the capability to perform daily living activities. To advance next\-generation prosthetic hand control design, it is crucial to address current shortcomings in robustness to out of lab artifacts, and generalizability to new environments. Due to the fixed number of object to interact with in existing datasets, contrasted with the virtually infinite variety of objects encountered in the real world, current grasp models perform poorly on unseen objects, negatively affecting users' independence and quality of life.   To address this: \(i\) we define semantic projection, the ability of a model to generalize to unseen object types and show that conventional models like YOLO, despite 80% training accuracy, drop to 15% on unseen objects. \(ii\) we propose Grasp\-LLaVA, a Grasp Vision Language Model enabling human\-like reasoning to infer the suitable grasp type estimate based on the object's physical characteristics resulting in a significant 50.2% accuracy over unseen object types compared to 36.7% accuracy of an SOTA grasp estimation model.   Lastly, to bridge the performance\-latency gap, we propose Hybrid Grasp Network \(HGN\), an edge\-cloud deployment infrastructure enabling fast grasp estimation on edge and accurate cloud inference as a fail\-safe, effectively expanding the latency vs. accuracy Pareto. HGN with confidence calibration \(DC\) enables dynamic switching between edge and cloud models, improving semantic projection accuracy by 5.6% \(to 42.3%\) with 3.5x speedup over the unseen object types. Over a real\-world sample mix, it reaches 86% average accuracy \(12.2% gain over edge\-only\), and 2.2x faster inference than Grasp\-LLaVA alone.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.07648v1)

---


## YOLOv8\-Based Deep Learning Model for Automated Poultry Disease Detection and Health Monitoring paper / 

发布日期：2025-08-06

作者：Akhil Saketh Reddy Sabbella

摘要：In the poultry industry, detecting chicken illnesses is essential to avoid financial losses. Conventional techniques depend on manual observation, which is laborious and prone to mistakes. Using YOLO v8 a deep learning model for real\-time object recognition. This study suggests an AI based approach, by developing a system that analyzes high resolution chicken photos, YOLO v8 detects signs of illness, such as abnormalities in behavior and appearance. A sizable, annotated dataset has been used to train the algorithm, which provides accurate real\-time identification of infected chicken and prompt warnings to farm operators for prompt action. By facilitating early infection identification, eliminating the need for human inspection, and enhancing biosecurity in large\-scale farms, this AI technology improves chicken health management. The real\-time features of YOLO v8 provide a scalable and effective method for improving farm management techniques.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2508.04658v1)

---

