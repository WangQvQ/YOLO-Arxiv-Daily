# 每日从arXiv中获取最新YOLO相关论文


## YOLO26: An Analysis of NMS\-Free End to End Framework for Real\-Time Object Detection / 

发布日期：2026-01-19

作者：Sudip Chakrabarty

摘要：The "You Only Look Once" \(YOLO\) framework has long served as the benchmark for real\-time object detection, yet traditional iterations \(YOLOv1 through YOLO11\) remain constrained by the latency and hyperparameter sensitivity of Non\-Maximum Suppression \(NMS\) post\-processing. This paper analyzes a comprehensive analysis of YOLO26, an architecture that fundamentally redefines this paradigm by eliminating NMS in favor of a native end\-to\-end learning strategy. This study examines the critical innovations that enable this transition, specifically the introduction of the MuSGD optimizer for stabilizing lightweight backbones, STAL for small\-target\-aware assignment, and ProgLoss for dynamic supervision. Through a systematic review of official performance benchmarks, the results demonstrate that YOLO26 establishes a new Pareto front, outperforming a comprehensive suite of predecessors and state\-of\-the\-art competitors \(including RTMDet and DAMO\-YOLO\) in both inference speed and detection accuracy. The analysis confirms that by decoupling representation learning from heuristic post\-processing, YOLOv26 successfully resolves the historical trade\-off between latency and precision, signaling the next evolutionary step in edge\-based computer vision.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.12882v1)

---


## A Multimodal Assistive System for Product Localization and Retrieval for People who are Blind or have Low Vision / 

发布日期：2026-01-18

作者：Ligao Ruan

摘要：Shopping is a routine activity for sighted individuals, yet for people who are blind or have low vision \(pBLV\), locating and retrieving products in physical environments remains a challenge. This paper presents a multimodal wearable assistive system that integrates object detection with vision\-language models to support independent product or item retrieval, with the goal of enhancing users'autonomy and sense of agency. The system operates through three phases: product search, which identifies target products using YOLO\-World detection combined with embedding similarity and color histogram matching; product navigation, which provides spatialized sonification and VLM\-generated verbal descriptions to guide users toward the target; and product correction, which verifies whether the user has reached the correct product and provides corrective feedback when necessary. Technical evaluation demonstrated promising performance across all modules, with product detection achieving near\-perfect accuracy at close range and high accuracy when facing shelves within 1.5 m. VLM\-based navigation achieved up to 94.4% accuracy, and correction accuracy exceeded 86% under optimal model configurations. These results demonstrate the system's potential to address the last\-meter problem in assistive shopping. Future work will focus on user studies with pBLV participants and integration with multi\-scale navigation ecosystems.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.12486v1)

---


## SME\-YOLO: A Real\-Time Detector for Tiny Defect Detection on PCB Surfaces / 

发布日期：2026-01-16

作者：Meng Han

摘要：Surface defects on Printed Circuit Boards \(PCBs\) directly compromise product reliability and safety. However, achieving high\-precision detection is challenging because PCB defects are typically characterized by tiny sizes, high texture similarity, and uneven scale distributions. To address these challenges, this paper proposes a novel framework based on YOLOv11n, named SME\-YOLO \(Small\-target Multi\-scale Enhanced YOLO\). First, we employ the Normalized Wasserstein Distance Loss \(NWDLoss\). This metric effectively mitigates the sensitivity of Intersection over Union \(IoU\) to positional deviations in tiny objects. Second, the original upsampling module is replaced by the Efficient Upsampling Convolution Block \(EUCB\). By utilizing multi\-scale convolutions, the EUCB gradually recovers spatial resolution and enhances the preservation of edge and texture details for tiny defects. Finally, this paper proposes the Multi\-Scale Focused Attention \(MSFA\) module. Tailored to the specific spatial distribution of PCB defects, this module adaptively strengthens perception within key scale intervals, achieving efficient fusion of local fine\-grained features and global context information. Experimental results on the PKU\-PCB dataset demonstrate that SME\-YOLO achieves state\-of\-the\-art performance. Specifically, compared to the baseline YOLOv11n, SME\-YOLO improves mAP by 2.2% and Precision by 4%, validating the effectiveness of the proposed method.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.11402v1)

---


## SAMannot: A Memory\-Efficient, Local, Open\-source Framework for Interactive Video Instance Segmentation based on SAM2 / 

发布日期：2026-01-16

作者：Gergely Dinya

摘要：Current research workflows for precise video segmentation are often forced into a compromise between labor\-intensive manual curation, costly commercial platforms, and/or privacy\-compromising cloud\-based services. The demand for high\-fidelity video instance segmentation in research is often hindered by the bottleneck of manual annotation and the privacy concerns of cloud\-based tools. We present SAMannot, an open\-source, local framework that integrates the Segment Anything Model 2 \(SAM2\) into a human\-in\-the\-loop workflow. To address the high resource requirements of foundation models, we modified the SAM2 dependency and implemented a processing layer that minimizes computational overhead and maximizes throughput, ensuring a highly responsive user interface. Key features include persistent instance identity management, an automated \`\`lock\-and\-refine'' workflow with barrier frames, and a mask\-skeletonization\-based auto\-prompting mechanism. SAMannot facilitates the generation of research\-ready datasets in YOLO and PNG formats alongside structured interaction logs. Verified through animal behavior tracking use\-cases and subsets of the LVOS and DAVIS benchmark datasets, the tool provides a scalable, private, and cost\-effective alternative to commercial platforms for complex video annotation tasks.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.11301v2)

---


## UAV\-Based Infrastructure Inspections: A Literature Review and Proposed Framework for AEC\+FM / 

发布日期：2026-01-16

作者：Amir Farzin Nikkhah

摘要：Unmanned Aerial Vehicles \(UAVs\) are transforming infrastructure inspections in the Architecture, Engineering, Construction, and Facility Management \(AEC\+FM\) domain. By synthesizing insights from over 150 studies, this review paper highlights UAV\-based methodologies for data acquisition, photogrammetric modeling, defect detection, and decision\-making support. Key innovations include path optimization, thermal integration, and advanced machine learning \(ML\) models such as YOLO and Faster R\-CNN for anomaly detection. UAVs have demonstrated value in structural health monitoring \(SHM\), disaster response, urban infrastructure management, energy efficiency evaluations, and cultural heritage preservation. Despite these advancements, challenges in real\-time processing, multimodal data fusion, and generalizability remain. A proposed workflow framework, informed by literature and a case study, integrates RGB imagery, LiDAR, and thermal sensing with transformer\-based architectures to improve accuracy and reliability in detecting structural defects, thermal anomalies, and geometric inconsistencies. The proposed framework ensures precise and actionable insights by fusing multimodal data and dynamically adapting path planning for complex environments, presented as a comprehensive step\-by\-step guide to address these challenges effectively. This paper concludes with future research directions emphasizing lightweight AI models, adaptive flight planning, synthetic datasets, and richer modality fusion to streamline modern infrastructure inspections.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.11665v1)

---

