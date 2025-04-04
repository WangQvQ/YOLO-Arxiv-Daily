# 每日从arXiv中获取最新YOLO相关论文


## LLM\-Guided Evolution: An Autonomous Model Optimization for Object Detection / 

发布日期：2025-04-03

作者：YiMing Yu

摘要：In machine learning, Neural Architecture Search \(NAS\) requires domain knowledge of model design and a large amount of trial\-and\-error to achieve promising performance. Meanwhile, evolutionary algorithms have traditionally relied on fixed rules and pre\-defined building blocks. The Large Language Model \(LLM\)\-Guided Evolution \(GE\) framework transformed this approach by incorporating LLMs to directly modify model source code for image classification algorithms on CIFAR data and intelligently guide mutations and crossovers. A key element of LLM\-GE is the "Evolution of Thought" \(EoT\) technique, which establishes feedback loops, allowing LLMs to refine their decisions iteratively based on how previous operations performed. In this study, we perform NAS for object detection by improving LLM\-GE to modify the architecture of You Only Look Once \(YOLO\) models to enhance performance on the KITTI dataset. Our approach intelligently adjusts the design and settings of YOLO to find the optimal algorithms against objective such as detection accuracy and speed. We show that LLM\-GE produced variants with significant performance improvements, such as an increase in Mean Average Precision from 92.5% to 94.5%. This result highlights the flexibility and effectiveness of LLM\-GE on real\-world challenges, offering a novel paradigm for automated machine learning that combines LLM\-driven reasoning with evolutionary strategies.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.02280v1)

---


## A YOLO\-Based Semi\-Automated Labeling Approach to Improve Fault Detection Efficiency in Railroad Videos / 

发布日期：2025-04-01

作者：Dylan Lester

摘要：Manual labeling for large\-scale image and video datasets is often time\-intensive, error\-prone, and costly, posing a significant barrier to efficient machine learning workflows in fault detection from railroad videos. This study introduces a semi\-automated labeling method that utilizes a pre\-trained You Only Look Once \(YOLO\) model to streamline the labeling process and enhance fault detection accuracy in railroad videos. By initiating the process with a small set of manually labeled data, our approach iteratively trains the YOLO model, using each cycle's output to improve model accuracy and progressively reduce the need for human intervention.   To facilitate easy correction of model predictions, we developed a system to export YOLO's detection data as an editable text file, enabling rapid adjustments when detections require refinement. This approach decreases labeling time from an average of 2 to 4 minutes per image to 30 seconds to 2 minutes, effectively minimizing labor costs and labeling errors. Unlike costly AI based labeling solutions on paid platforms, our method provides a cost\-effective alternative for researchers and practitioners handling large datasets in fault detection and other detection based machine learning applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.01010v1)

---


## SU\-YOLO: Spiking Neural Network for Efficient Underwater Object Detection / 

发布日期：2025-03-31

作者：Chenyang Li

摘要：Underwater object detection is critical for oceanic research and industrial safety inspections. However, the complex optical environment and the limited resources of underwater equipment pose significant challenges to achieving high accuracy and low power consumption. To address these issues, we propose Spiking Underwater YOLO \(SU\-YOLO\), a Spiking Neural Network \(SNN\) model. Leveraging the lightweight and energy\-efficient properties of SNNs, SU\-YOLO incorporates a novel spike\-based underwater image denoising method based solely on integer addition, which enhances the quality of feature maps with minimal computational overhead. In addition, we introduce Separated Batch Normalization \(SeBN\), a technique that normalizes feature maps independently across multiple time steps and is optimized for integration with residual structures to capture the temporal dynamics of SNNs more effectively. The redesigned spiking residual blocks integrate the Cross Stage Partial Network \(CSPNet\) with the YOLO architecture to mitigate spike degradation and enhance the model's feature extraction capabilities. Experimental results on URPC2019 underwater dataset demonstrate that SU\-YOLO achieves mAP of 78.8% with 6.97M parameters and an energy consumption of 2.98 mJ, surpassing mainstream SNN models in both detection accuracy and computational efficiency. These results underscore the potential of SNNs for engineering applications. The code is available in https://github.com/lwxfight/snn\-underwater.

中文摘要：


代码链接：https://github.com/lwxfight/snn-underwater.

论文链接：[阅读更多](http://arxiv.org/abs/2503.24389v1)

---


## AI\-Assisted Colonoscopy: Polyp Detection and Segmentation using Foundation Models / 

发布日期：2025-03-31

作者：Uxue Delaquintana\-Aramendi

摘要：In colonoscopy, 80% of the missed polyps could be detected with the help of Deep Learning models. In the search for algorithms capable of addressing this challenge, foundation models emerge as promising candidates. Their zero\-shot or few\-shot learning capabilities, facilitate generalization to new data or tasks without extensive fine\-tuning. A concept that is particularly advantageous in the medical imaging domain, where large annotated datasets for traditional training are scarce. In this context, a comprehensive evaluation of foundation models for polyp segmentation was conducted, assessing both detection and delimitation. For the study, three different colonoscopy datasets have been employed to compare the performance of five different foundation models, DINOv2, YOLO\-World, GroundingDINO, SAM and MedSAM, against two benchmark networks, YOLOv8 and Mask R\-CNN. Results show that the success of foundation models in polyp characterization is highly dependent on domain specialization. For optimal performance in medical applications, domain\-specific models are essential, and generic models require fine\-tuning to achieve effective results. Through this specialization, foundation models demonstrated superior performance compared to state\-of\-the\-art detection and segmentation models, with some models even excelling in zero\-shot evaluation; outperforming fine\-tuned models on unseen data.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.24138v1)

---


## Intelligent Bear Prevention System Based on Computer Vision: An Approach to Reduce Human\-Bear Conflicts in the Tibetan Plateau Area, China / 

发布日期：2025-03-29

作者：Pengyu Chen

摘要：Conflicts between humans and bears on the Tibetan Plateau present substantial threats to local communities and hinder wildlife preservation initiatives. This research introduces a novel strategy that incorporates computer vision alongside Internet of Things \(IoT\) technologies to alleviate these issues. Tailored specifically for the harsh environment of the Tibetan Plateau, the approach utilizes the K210 development board paired with the YOLO object detection framework along with a tailored bear\-deterrent mechanism, offering minimal energy usage and real\-time efficiency in bear identification and deterrence. The model's performance was evaluated experimentally, achieving a mean Average Precision \(mAP\) of 91.4%, demonstrating excellent precision and dependability. By integrating energy\-efficient components, the proposed system effectively surpasses the challenges of remote and off\-grid environments, ensuring uninterrupted operation in secluded locations. This study provides a viable, eco\-friendly, and expandable solution to mitigate human\-bear conflicts, thereby improving human safety and promoting bear conservation in isolated areas like Yushu, China.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2503.23178v1)

---

