# 每日从arXiv中获取最新YOLO相关论文


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


## Kinematic\-Based Assessment of Surgical Actions in Microanastomosis / 

发布日期：2025-12-30

作者：Yan Meng

摘要：Proficiency in microanastomosis is a critical surgical skill in neurosurgery, where the ability to precisely manipulate fine instruments is crucial to successful outcomes. These procedures require sustained attention, coordinated hand movements, and highly refined motor skills, underscoring the need for objective and systematic methods to evaluate and enhance microsurgical training. Conventional assessment approaches typically rely on expert raters supervising the procedures or reviewing surgical videos, which is an inherently subjective process prone to inter\-rater variability, inconsistency, and significant time investment. These limitations highlight the necessity for automated and scalable solutions. To address this challenge, we introduce a novel AI\-driven framework for automated action segmentation and performance assessment in microanastomosis procedures, designed to operate efficiently on edge computing platforms. The proposed system comprises three main components: \(1\) an object tip tracking and localization module based on YOLO and DeepSORT; \(2\) an action segmentation module leveraging self\-similarity matrix for action boundary detection and unsupervised clustering; and \(3\) a supervised classification module designed to evaluate surgical gesture proficiency. Experimental validation on a dataset of 58 expert\-rated microanastomosis videos demonstrates the effectiveness of our approach, achieving a frame\-level action segmentation accuracy of 92.4% and an overall skill classification accuracy of 85.5% in replicating expert evaluations. These findings demonstrate the potential of the proposed method to provide objective, real\-time feedback in microsurgical education, thereby enabling more standardized, data\-driven training protocols and advancing competency assessment in high\-stakes surgical environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.23942v1)

---


## Detection Fire in Camera RGB\-NIR / 

发布日期：2025-12-29

作者：Nguyen Truong Khai

摘要：Improving the accuracy of fire detection using infrared night vision cameras remains a challenging task. Previous studies have reported strong performance with popular detection models. For example, YOLOv7 achieved an mAP50\-95 of 0.51 using an input image size of 640 x 1280, RT\-DETR reached an mAP50\-95 of 0.65 with an image size of 640 x 640, and YOLOv9 obtained an mAP50\-95 of 0.598 at the same resolution. Despite these results, limitations in dataset construction continue to cause issues, particularly the frequent misclassification of bright artificial lights as fire.   This report presents three main contributions: an additional NIR dataset, a two\-stage detection model, and Patched\-YOLO. First, to address data scarcity, we explore and apply various data augmentation strategies for both the NIR dataset and the classification dataset. Second, to improve night\-time fire detection accuracy while reducing false positives caused by artificial lights, we propose a two\-stage pipeline combining YOLOv11 and EfficientNetV2\-B0. The proposed approach achieves higher detection accuracy compared to previous methods, particularly for night\-time fire detection. Third, to improve fire detection in RGB images, especially for small and distant objects, we introduce Patched\-YOLO, which enhances the model's detection capability through patch\-based processing. Further details of these contributions are discussed in the following sections.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.23594v1)

---

