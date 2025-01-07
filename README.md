# 每日从arXiv中获取最新YOLO相关论文


## Identifying Surgical Instruments in Pedagogical Cataract Surgery Videos through an Optimized Aggregation Network / 

发布日期：2025-01-05

作者：Sanya Sinha

摘要：Instructional cataract surgery videos are crucial for ophthalmologists and trainees to observe surgical details repeatedly. This paper presents a deep learning model for real\-time identification of surgical instruments in these videos, using a custom dataset scraped from open\-access sources. Inspired by the architecture of YOLOV9, the model employs a Programmable Gradient Information \(PGI\) mechanism and a novel Generally\-Optimized Efficient Layer Aggregation Network \(Go\-ELAN\) to address the information bottleneck problem, enhancing Minimum Average Precision \(mAP\) at higher Non\-Maximum Suppression Intersection over Union \(NMS IoU\) scores. The Go\-ELAN YOLOV9 model, evaluated against YOLO v5, v7, v8, v9 vanilla, Laptool and DETR, achieves a superior mAP of 73.74 at IoU 0.5 on a dataset of 615 images with 10 instrument classes, demonstrating the effectiveness of the proposed model.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.02618v1)

---


## Accurate Crop Yield Estimation of Blueberries using Deep Learning and Smart Drones / 

发布日期：2025-01-04

作者：Hieu D. Nguyen

摘要：We present an AI pipeline that involves using smart drones equipped with computer vision to obtain a more accurate fruit count and yield estimation of the number of blueberries in a field. The core components are two object\-detection models based on the YOLO deep learning architecture: a Bush Model that is able to detect blueberry bushes from images captured at low altitudes and at different angles, and a Berry Model that can detect individual berries that are visible on a bush. Together, both models allow for more accurate crop yield estimation by allowing intelligent control of the drone's position and camera to safely capture side\-view images of bushes up close. In addition to providing experimental results for our models, which show good accuracy in terms of precision and recall when captured images are cropped around the foreground center bush, we also describe how to deploy our models to map out blueberry fields using different sampling strategies, and discuss the challenges of annotating very small objects \(blueberries\) and difficulties in evaluating the effectiveness of our models.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.02344v1)

---


## Efficient Video\-Based ALPR System Using YOLO and Visual Rhythm / 

发布日期：2025-01-04

作者：Victor Nascimento Ribeiro

摘要：Automatic License Plate Recognition \(ALPR\) involves extracting vehicle license plate information from image or a video capture. These systems have gained popularity due to the wide availability of low\-cost surveillance cameras and advances in Deep Learning. Typically, video\-based ALPR systems rely on multiple frames to detect the vehicle and recognize the license plates. Therefore, we propose a system capable of extracting exactly one frame per vehicle and recognizing its license plate characters from this singular image using an Optical Character Recognition \(OCR\) model. Early experiments show that this methodology is viable.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.02270v1)

---


## Lightweight G\-YOLOv11: Advancing Efficient Fracture Detection in Pediatric Wrist X\-rays / 

发布日期：2024-12-31

作者：Abdesselam Ferdi

摘要：Computer\-aided diagnosis \(CAD\) systems have greatly improved the interpretation of medical images by radiologists and surgeons. However, current CAD systems for fracture detection in X\-ray images primarily rely on large, resource\-intensive detectors, which limits their practicality in clinical settings. To address this limitation, we propose a novel lightweight CAD system based on the YOLO detector for fracture detection. This system, named ghost convolution\-based YOLOv11 \(G\-YOLOv11\), builds on the latest version of the YOLO detector family and incorporates the ghost convolution operation for feature extraction. The ghost convolution operation generates the same number of feature maps as traditional convolution but requires fewer linear operations, thereby reducing the detector's computational resource requirements. We evaluated the performance of the proposed G\-YOLOv11 detector on the GRAZPEDWRI\-DX dataset, achieving an mAP@0.5 of 0.535 with an inference time of 2.4 ms on an NVIDIA A10 GPU. Compared to the standard YOLOv11l, G\-YOLOv11l achieved reductions of 13.6% in mAP@0.5 and 68.7% in size. These results establish a new state\-of\-the\-art benchmark in terms of efficiency, outperforming existing detectors. Code and models are available at https://github.com/AbdesselamFerdi/G\-YOLOv11.

中文摘要：


代码链接：https://github.com/AbdesselamFerdi/G-YOLOv11.

论文链接：[阅读更多](http://arxiv.org/abs/2501.00647v1)

---


## YOLO\-UniOW: Efficient Universal Open\-World Object Detection / 

发布日期：2024-12-30

作者：Lihao Liu

摘要：Traditional object detection models are constrained by the limitations of closed\-set datasets, detecting only categories encountered during training. While multimodal models have extended category recognition by aligning text and image modalities, they introduce significant inference overhead due to cross\-modality fusion and still remain restricted by predefined vocabulary, leaving them ineffective at handling unknown objects in open\-world scenarios. In this work, we introduce Universal Open\-World Object Detection \(Uni\-OWD\), a new paradigm that unifies open\-vocabulary and open\-world object detection tasks. To address the challenges of this setting, we propose YOLO\-UniOW, a novel model that advances the boundaries of efficiency, versatility, and performance. YOLO\-UniOW incorporates Adaptive Decision Learning to replace computationally expensive cross\-modality fusion with lightweight alignment in the CLIP latent space, achieving efficient detection without compromising generalization. Additionally, we design a Wildcard Learning strategy that detects out\-of\-distribution objects as "unknown" while enabling dynamic vocabulary expansion without the need for incremental learning. This design empowers YOLO\-UniOW to seamlessly adapt to new categories in open\-world environments. Extensive experiments validate the superiority of YOLO\-UniOW, achieving achieving 34.6 AP and 30.0 APr on LVIS with an inference speed of 69.6 FPS. The model also sets benchmarks on M\-OWODB, S\-OWODB, and nuScenes datasets, showcasing its unmatched performance in open\-world object detection. Code and models are available at https://github.com/THU\-MIG/YOLO\-UniOW.

中文摘要：


代码链接：https://github.com/THU-MIG/YOLO-UniOW.

论文链接：[阅读更多](http://arxiv.org/abs/2412.20645v1)

---

