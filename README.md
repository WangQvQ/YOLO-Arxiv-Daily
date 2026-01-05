# 每日从arXiv中获取最新YOLO相关论文


## RoLID\-11K: A Dashcam Dataset for Small\-Object Roadside Litter Detection / 

发布日期：2026-01-01

作者：Tao Wu

摘要：Roadside litter poses environmental, safety and economic challenges, yet current monitoring relies on labour\-intensive surveys and public reporting, providing limited spatial coverage. Existing vision datasets for litter detection focus on street\-level still images, aerial scenes or aquatic environments, and do not reflect the unique characteristics of dashcam footage, where litter appears extremely small, sparse and embedded in cluttered road\-verge backgrounds. We introduce RoLID\-11K, the first large\-scale dataset for roadside litter detection from dashcams, comprising over 11k annotated images spanning diverse UK driving conditions and exhibiting pronounced long\-tail and small\-object distributions. We benchmark a broad spectrum of modern detectors, from accuracy\-oriented transformer architectures to real\-time YOLO models, and analyse their strengths and limitations on this challenging task. Our results show that while CO\-DETR and related transformers achieve the best localisation accuracy, real\-time models remain constrained by coarse feature hierarchies. RoLID\-11K establishes a challenging benchmark for extreme small\-object detection in dynamic driving scenes and aims to support the development of scalable, low\-cost systems for roadside\-litter monitoring. The dataset is available at https://github.com/xq141839/RoLID\-11K.

中文摘要：


代码链接：https://github.com/xq141839/RoLID-11K.

论文链接：[阅读更多](http://arxiv.org/abs/2601.00398v1)

---


## Application Research of a Deep Learning Model Integrating CycleGAN and YOLO in PCB Infrared Defect Detection / 

发布日期：2026-01-01

作者：Chao Yang

摘要：This paper addresses the critical bottleneck of infrared \(IR\) data scarcity in Printed Circuit Board \(PCB\) defect detection by proposing a cross\-modal data augmentation framework integrating CycleGAN and YOLOv8. Unlike conventional methods relying on paired supervision, we leverage CycleGAN to perform unpaired image\-to\-image translation, mapping abundant visible\-light PCB images into the infrared domain. This generative process synthesizes high\-fidelity pseudo\-IR samples that preserve the structural semantics of defects while accurately simulating thermal distribution patterns. Subsequently, we construct a heterogeneous training strategy that fuses generated pseudo\-IR data with limited real IR samples to train a lightweight YOLOv8 detector. Experimental results demonstrate that this method effectively enhances feature learning under low\-data conditions. The augmented detector significantly outperforms models trained on limited real data alone and approaches the performance benchmarks of fully supervised training, proving the efficacy of pseudo\-IR synthesis as a robust augmentation strategy for industrial inspection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.00237v1)

---


## FireRescue: A UAV\-Based Dataset and Enhanced YOLO Model for Object Detection in Fire Rescue Scenes / 

发布日期：2025-12-31

作者：Qingyu Xu

摘要：Object detection in fire rescue scenarios is importance for command and decision\-making in firefighting operations. However, existing research still suffers from two main limitations. First, current work predominantly focuses on environments such as mountainous or forest areas, while paying insufficient attention to urban rescue scenes, which are more frequent and structurally complex. Second, existing detection systems include a limited number of classes, such as flames and smoke, and lack a comprehensive system covering key targets crucial for command decisions, such as fire trucks and firefighters. To address the above issues, this paper first constructs a new dataset named "FireRescue" for rescue command, which covers multiple rescue scenarios, including urban, mountainous, forest, and water areas, and contains eight key categories such as fire trucks and firefighters, with a total of 15,980 images and 32,000 bounding boxes. Secondly, to tackle the problems of inter\-class confusion and missed detection of small targets caused by chaotic scenes, diverse targets, and long\-distance shooting, this paper proposes an improved model named FRS\-YOLO. On the one hand, the model introduces a plug\-and\-play multidi\-mensional collaborative enhancement attention module, which enhances the discriminative representation of easily confused categories \(e.g., fire trucks vs. ordinary trucks\) through cross\-dimensional feature interaction. On the other hand, it integrates a dynamic feature sampler to strengthen high\-response foreground features, thereby mitigating the effects of smoke occlusion and background interference. Experimental results demonstrate that object detection in fire rescue scenarios is highly challenging, and the proposed method effectively improves the detection performance of YOLO series models in this context.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.24622v1)

---


## Using Large Language Models To Translate Machine Results To Human Results / 

发布日期：2025-12-30

作者：Trishna Niraula

摘要：Artificial intelligence \(AI\) has transformed medical imaging, with computer vision \(CV\) systems achieving state\-of\-the\-art performance in classification and detection tasks. However, these systems typically output structured predictions, leaving radiologists responsible for translating results into full narrative reports. Recent advances in large language models \(LLMs\), such as GPT\-4, offer new opportunities to bridge this gap by generating diagnostic narratives from structured findings. This study introduces a pipeline that integrates YOLOv5 and YOLOv8 for anomaly detection in chest X\-ray images with a large language model \(LLM\) to generate natural\-language radiology reports. The YOLO models produce bounding\-box predictions and class labels, which are then passed to the LLM to generate descriptive findings and clinical summaries. YOLOv5 and YOLOv8 are compared in terms of detection accuracy, inference latency, and the quality of generated text, as measured by cosine similarity to ground\-truth reports. Results show strong semantic similarity between AI and human reports, while human evaluation reveals GPT\-4 excels in clarity \(4.88/5\) but exhibits lower scores for natural writing flow \(2.81/5\), indicating that current systems achieve clinical accuracy but remain stylistically distinguishable from radiologist\-authored text.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.24518v1)

---


## AI\-Driven Evaluation of Surgical Skill via Action Recognition / 

发布日期：2025-12-30

作者：Yan Meng

摘要：The development of effective training and evaluation strategies is critical. Conventional methods for assessing surgical proficiency typically rely on expert supervision, either through onsite observation or retrospective analysis of recorded procedures. However, these approaches are inherently subjective, susceptible to inter\-rater variability, and require substantial time and effort from expert surgeons. These demands are often impractical in low\- and middle\-income countries, thereby limiting the scalability and consistency of such methods across training programs. To address these limitations, we propose a novel AI\-driven framework for the automated assessment of microanastomosis performance. The system integrates a video transformer architecture based on TimeSformer, improved with hierarchical temporal attention and weighted spatial attention mechanisms, to achieve accurate action recognition within surgical videos. Fine\-grained motion features are then extracted using a YOLO\-based object detection and tracking method, allowing for detailed analysis of instrument kinematics. Performance is evaluated along five aspects of microanastomosis skill, including overall action execution, motion quality during procedure\-critical actions, and general instrument handling. Experimental validation using a dataset of 58 expert\-annotated videos demonstrates the effectiveness of the system, achieving 87.7% frame\-level accuracy in action segmentation that increased to 93.62% with post\-processing, and an average classification accuracy of 76% in replicating expert assessments across all skill aspects. These findings highlight the system's potential to provide objective, consistent, and interpretable feedback, thereby enabling more standardized, data\-driven training and evaluation in surgical education.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.24411v1)

---

