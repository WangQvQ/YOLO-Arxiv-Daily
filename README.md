# 每日从arXiv中获取最新YOLO相关论文


## User Prompting Strategies and Prompt Enhancement Methods for Open\-Set Object Detection in XR Environments / 

发布日期：2026-01-30

作者：Junfeng Lin

摘要：Open\-set object detection \(OSOD\) localizes objects while identifying and rejecting unknown classes at inference. While recent OSOD models perform well on benchmarks, their behavior under realistic user prompting remains underexplored. In interactive XR settings, user\-generated prompts are often ambiguous, underspecified, or overly detailed. To study prompt\-conditioned robustness, we evaluate two OSOD models, GroundingDINO and YOLO\-E, on real\-world XR images and simulate diverse user prompting behaviors using vision\-language models. We consider four prompt types: standard, underdetailed, overdetailed, and pragmatically ambiguous, and examine the impact of two enhancement strategies on these prompts. Results show that both models exhibit stable performance under underdetailed and standard prompts, while they suffer degradation under ambiguous prompts. Overdetailed prompts primarily affect GroundingDINO. Prompt enhancement substantially improves robustness under ambiguity, yielding gains exceeding 55% mIoU and 41% average confidence. Based on the findings, we propose several prompting strategies and prompt enhancement methods for OSOD models in XR environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.23281v1)

---


## About an Automating Annotation Method for Robot Markers / 

发布日期：2026-01-30

作者：Wataru Uemura

摘要：Factory automation has become increasingly important due to labor shortages, leading to the introduction of autonomous mobile robots for tasks such as material transportation. Markers are commonly used for robot self\-localization and object identification. In the RoboCup Logistics League \(RCLL\), ArUco markers are employed both for robot localization and for identifying processing modules. Conventional recognition relies on OpenCV\-based image processing, which detects black\-and\-white marker patterns. However, these methods often fail under noise, motion blur, defocus, or varying illumination conditions. Deep\-learning\-based recognition offers improved robustness under such conditions, but requires large amounts of annotated data. Annotation must typically be done manually, as the type and position of objects cannot be detected automatically, making dataset preparation a major bottleneck. In contrast, ArUco markers include built\-in recognition modules that provide both ID and positional information, enabling automatic annotation. This paper proposes an automated annotation method for training deep\-learning models on ArUco marker images. By leveraging marker detection results obtained from the ArUco module, the proposed approach eliminates the need for manual labeling. A YOLO\-based model is trained using the automatically annotated dataset, and its performance is evaluated under various conditions. Experimental results demonstrate that the proposed method improves recognition performance compared with conventional image\-processing techniques, particularly for images affected by blur or defocus. Automatic annotation also reduces human effort and ensures consistent labeling quality. Future work will investigate the relationship between confidence thresholds and recognition performance.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.22982v1)

---


## A Comparative Evaluation of Large Vision\-Language Models for 2D Object Detection under SOTIF Conditions / 

发布日期：2026-01-30

作者：Ji Zhou

摘要：Reliable environmental perception remains one of the main obstacles for safe operation of automated vehicles. Safety of the Intended Functionality \(SOTIF\) concerns safety risks from perception insufficiencies, particularly under adverse conditions where conventional detectors often falter. While Large Vision\-Language Models \(LVLMs\) demonstrate promising semantic reasoning, their quantitative effectiveness for safety\-critical 2D object detection is underexplored. This paper presents a systematic evaluation of ten representative LVLMs using the PeSOTIF dataset, a benchmark specifically curated for long\-tail traffic scenarios and environmental degradations. Performance is quantitatively compared against the classical perception approach, a YOLO\-based detector. Experimental results reveal a critical trade\-off: top\-performing LVLMs \(e.g., Gemini 3, Doubao\) surpass the YOLO baseline in recall by over 25% in complex natural scenarios, exhibiting superior robustness to visual degradation. Conversely, the baseline retains an advantage in geometric precision for synthetic perturbations. These findings highlight the complementary strengths of semantic reasoning versus geometric regression, supporting the use of LVLMs as high\-level safety validators in SOTIF\-oriented automated driving systems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.22830v1)

---


## BLO\-Inst: Bi\-Level Optimization Based Alignment of YOLO and SAM for Robust Instance Segmentation / 

发布日期：2026-01-29

作者：Li Zhang

摘要：The Segment Anything Model has revolutionized image segmentation with its zero\-shot capabilities, yet its reliance on manual prompts hinders fully automated deployment. While integrating object detectors as prompt generators offers a pathway to automation, existing pipelines suffer from two fundamental limitations: objective mismatch, where detectors optimized for geometric localization do not correspond to the optimal prompting context required by SAM, and alignment overfitting in standard joint training, where the detector simply memorizes specific prompt adjustments for training samples rather than learning a generalizable policy. To bridge this gap, we introduce BLO\-Inst, a unified framework that aligns detection and segmentation objectives by bi\-level optimization. We formulate the alignment as a nested optimization problem over disjoint data splits. In the lower level, the SAM is fine\-tuned to maximize segmentation fidelity given the current detection proposals on a subset \($D\_1$\). In the upper level, the detector is updated to generate bounding boxes that explicitly minimize the validation loss of the fine\-tuned SAM on a separate subset \($D\_2$\). This effectively transforms the detector into a segmentation\-aware prompt generator, optimizing the bounding boxes not just for localization accuracy, but for downstream mask quality. Extensive experiments demonstrate that BLO\-Inst achieves superior performance, outperforming standard baselines on tasks in general and biomedical domains.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.22061v1)

---


## An AI Framework for Microanastomosis Motion Assessment / 

发布日期：2026-01-28

作者：Yan Meng

摘要：Proficiency in microanastomosis is a fundamental competency across multiple microsurgical disciplines. These procedures demand exceptional precision and refined technical skills, making effective, standardized assessment methods essential. Traditionally, the evaluation of microsurgical techniques has relied heavily on the subjective judgment of expert raters. They are inherently constrained by limitations such as inter\-rater variability, lack of standardized evaluation criteria, susceptibility to cognitive bias, and the time\-intensive nature of manual review. These shortcomings underscore the urgent need for an objective, reliable, and automated system capable of assessing microsurgical performance with consistency and scalability. To bridge this gap, we propose a novel AI framework for the automated assessment of microanastomosis instrument handling skills. The system integrates four core components: \(1\) an instrument detection module based on the You Only Look Once \(YOLO\) architecture; \(2\) an instrument tracking module developed from Deep Simple Online and Realtime Tracking \(DeepSORT\); \(3\) an instrument tip localization module employing shape descriptors; and \(4\) a supervised classification module trained on expert\-labeled data to evaluate instrument handling proficiency. Experimental results demonstrate the effectiveness of the framework, achieving an instrument detection precision of 97%, with a mean Average Precision \(mAP\) of 96%, measured by Intersection over Union \(IoU\) thresholds ranging from 50% to 95% \(mAP50\-95\).

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.21120v1)

---

