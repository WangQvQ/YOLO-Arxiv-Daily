# 每日从arXiv中获取最新YOLO相关论文


## HideAndSeg: an AI\-based tool with automated prompting for octopus segmentation in natural habitats / 

发布日期：2025-11-06

作者：Alan de Aguiar

摘要：Analyzing octopuses in their natural habitats is challenging due to their camouflage capability, rapid changes in skin texture and color, non\-rigid body deformations, and frequent occlusions, all of which are compounded by variable underwater lighting and turbidity. Addressing the lack of large\-scale annotated datasets, this paper introduces HideAndSeg, a novel, minimally supervised AI\-based tool for segmenting videos of octopuses. It establishes a quantitative baseline for this task. HideAndSeg integrates SAM2 with a custom\-trained YOLOv11 object detector. First, the user provides point coordinates to generate the initial segmentation masks with SAM2. These masks serve as training data for the YOLO model. After that, our approach fully automates the pipeline by providing a bounding box prompt to SAM2, eliminating the need for further manual intervention. We introduce two unsupervised metrics \- temporal consistency $DICE\_t$ and new component count $NC\_t$ \- to quantitatively evaluate segmentation quality and guide mask refinement in the absence of ground\-truth data, i.e., real\-world information that serves to train, validate, and test AI models. Results show that HideAndSeg achieves satisfactory performance, reducing segmentation noise compared to the manually prompted approach. Our method can re\-identify and segment the octopus even after periods of complete occlusion in natural environments, a scenario in which the manually prompted model fails. By reducing the need for manual analysis in real\-world scenarios, this work provides a practical tool that paves the way for more efficient behavioral studies of wild cephalopods.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.04426v1)

---


## Desert Waste Detection and Classification Using Data\-Based and Model\-Based Enhanced YOLOv12 DL Model / 

发布日期：2025-11-05

作者：Abdulmumin Sa'ad

摘要：The global waste crisis is escalating, with solid waste generation expected to increase by 70% by 2050. Traditional waste collection methods, particularly in remote or harsh environments like deserts, are labor\-intensive, inefficient, and often hazardous. Recent advances in computer vision and deep learning have opened the door to automated waste detection systems, yet most research focuses on urban environments and recyclable materials, overlooking organic and hazardous waste and underexplored terrains such as deserts. In this work, we propose an enhanced real\-time object detection framework based on a pruned, lightweight version of YOLOv12 integrated with Self\-Adversarial Training \(SAT\) and specialized data augmentation strategies. Using the DroneTrashNet dataset, we demonstrate significant improvements in precision, recall, and mean average precision \(mAP\), while achieving low latency and compact model size suitable for deployment on resource\-constrained aerial drones. Benchmarking our model against state\-of\-the\-art lightweight YOLO variants further highlights its optimal balance of accuracy and efficiency. Our results validate the effectiveness of combining data\-centric and model\-centric enhancements for robust, real\-time waste detection in desert environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.03888v1)

---


## The Urban Vision Hackathon Dataset and Models: Towards Image Annotations and Accurate Vision Models for Indian Traffic / 

发布日期：2025-11-04

作者：Akash Sharma

摘要：This report describes the UVH\-26 dataset, the first public release by AIM@IISc of a large\-scale dataset of annotated traffic\-camera images from India. The dataset comprises 26,646 high\-resolution \(1080p\) images sampled from 2800 Bengaluru's Safe\-City CCTV cameras over a 4\-week period, and subsequently annotated through a crowdsourced hackathon involving 565 college students from across India. In total, 1.8 million bounding boxes were labeled across 14 vehicle classes specific to India: Cycle, 2\-Wheeler \(Motorcycle\), 3\-Wheeler \(Auto\-rickshaw\), LCV \(Light Commercial Vehicles\), Van, Tempo\-traveller, Hatchback, Sedan, SUV, MUV, Mini\-bus, Bus, Truck and Other. Of these, 283k\-316k consensus ground truth bounding boxes and labels were derived for distinct objects in the 26k images using Majority Voting and STAPLE algorithms. Further, we train multiple contemporary detectors, including YOLO11\-S/X, RT\-DETR\-S/X, and DAMO\-YOLO\-T/L using these datasets, and report accuracy based on mAP50, mAP75 and mAP50:95. Models trained on UVH\-26 achieve 8.4\-31.5% improvements in mAP50:95 over equivalent baseline models trained on COCO dataset, with RT\-DETR\-X showing the best performance at 0.67 \(mAP50:95\) as compared to 0.40 for COCO\-trained weights for common classes \(Car, Bus, and Truck\). This demonstrates the benefits of domain\-specific training data for Indian traffic scenarios. The release package provides the 26k images with consensus annotations based on Majority Voting \(UVH\-26\-MV\) and STAPLE \(UVH\-26\-ST\) and the 6 fine\-tuned YOLO and DETR models on each of these datasets. By capturing the heterogeneity of Indian urban mobility directly from operational traffic\-camera streams, UVH\-26 addresses a critical gap in existing global benchmarks, and offers a foundation for advancing detection, classification, and deployment of intelligent transportation systems in emerging nations with complex traffic conditions.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.02563v1)

---


## ASTROFLOW: A Real\-Time End\-to\-End Pipeline for Radio Single\-Pulse Searches / 

发布日期：2025-11-04

作者：Guanhong Lin

摘要：Fast radio bursts \(FRBs\) are extremely bright, millisecond duration cosmic transients of unknown origin. The growing number of wide\-field and high\-time\-resolution radio surveys, particularly with next\-generation facilities such as the SKA and MeerKAT, will dramatically increase FRB discovery rates, but also produce data volumes that overwhelm conventional search pipelines. Real\-time detection thus demands software that is both algorithmically robust and computationally efficient. We present Astroflow, an end\-to\-end, GPU\-accelerated pipeline for single\-pulse detection in radio time\-frequency data. Built on a unified C\+\+/CUDA core with a Python interface, Astroflow integrates RFI excision, incoherent dedispersion, dynamic\-spectrum tiling, and a YOLO\-based deep detector. Through vectorized memory access, shared\-memory tiling, and OpenMP parallelism, it achieves 10x faster\-than\-real\-time processing on consumer GPUs for a typical 150 s, 2048\-channel observation, while preserving high sensitivity across a wide range of pulse widths and dispersion measures. These results establish the feasibility of a fully integrated, GPU\-accelerated single\-pulse search stack, capable of scaling to the data volumes expected from upcoming large\-scale surveys. Astroflow offers a reusable and deployable solution for real\-time transient discovery, and provides a framework that can be continuously refined with new data and models.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.02328v1)

---


## Autobiasing Event Cameras for Flickering Mitigation / 

发布日期：2025-11-04

作者：Mehdi Sefidgar Dilmaghani

摘要：Understanding and mitigating flicker effects caused by rapid variations in light intensity is critical for enhancing the performance of event cameras in diverse environments. This paper introduces an innovative autonomous mechanism for tuning the biases of event cameras, effectively addressing flicker across a wide frequency range \-25 Hz to 500 Hz. Unlike traditional methods that rely on additional hardware or software for flicker filtering, our approach leverages the event cameras inherent bias settings. Utilizing a simple Convolutional Neural Networks \-CNNs, the system identifies instances of flicker in a spatial space and dynamically adjusts specific biases to minimize its impact. The efficacy of this autobiasing system was robustly tested using a face detector framework under both well\-lit and low\-light conditions, as well as across various frequencies. The results demonstrated significant improvements: enhanced YOLO confidence metrics for face detection, and an increased percentage of frames capturing detected faces. Moreover, the average gradient, which serves as an indicator of flicker presence through edge detection, decreased by 38.2 percent in well\-lit conditions and by 53.6 percent in low\-light conditions. These findings underscore the potential of our approach to significantly improve the functionality of event cameras in a range of adverse lighting scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.02180v1)

---

