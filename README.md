# 每日从arXiv中获取最新YOLO相关论文


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


## Differential Evolution Integrated Hybrid Deep Learning Model for Object Detection in Pre\-made Dishes / 

发布日期：2024-12-29

作者：Lujia Lv

摘要：With the continuous improvement of people's living standards and fast\-paced working conditions, pre\-made dishes are becoming increasingly popular among families and restaurants due to their advantages of time\-saving, convenience, variety, cost\-effectiveness, standard quality, etc. Object detection is a key technology for selecting ingredients and evaluating the quality of dishes in the pre\-made dishes industry. To date, many object detection approaches have been proposed. However, accurate object detection of pre\-made dishes is extremely difficult because of overlapping occlusion of ingredients, similarity of ingredients, and insufficient light in the processing environment. As a result, the recognition scene is relatively complex and thus leads to poor object detection by a single model. To address this issue, this paper proposes a Differential Evolution Integrated Hybrid Deep Learning \(DEIHDL\) model. The main idea of DEIHDL is three\-fold: 1\) three YOLO\-based and transformer\-based base models are developed respectively to increase diversity for detecting objects of pre\-made dishes, 2\) the three base models are integrated by differential evolution optimized self\-adjusting weights, and 3\) weighted boxes fusion strategy is employed to score the confidence of the three base models during the integration. As such, DEIHDL possesses the multi\-performance originating from the three base models to achieve accurate object detection in complex pre\-made dish scenes. Extensive experiments on real datasets demonstrate that the proposed DEIHDL model significantly outperforms the base models in detecting objects of pre\-made dishes.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2412.20370v1)

---


## Plastic Waste Classification Using Deep Learning: Insights from the WaDaBa Dataset / 

发布日期：2024-12-28

作者：Suman Kunwar

摘要：With the increasing use of plastic, the challenges associated with managing plastic waste have become more challenging, emphasizing the need of effective solutions for classification and recycling. This study explores the potential of deep learning, focusing on convolutional neural networks \(CNNs\) and object detection models like YOLO \(You Only Look Once\), to tackle this issue using the WaDaBa dataset. The study shows that YOLO\- 11m achieved highest accuracy \(98.03%\) and mAP50 \(0.990\), with YOLO\-11n performing similarly but highest mAP50\(0.992\). Lightweight models like YOLO\-10n trained faster but with lower accuracy, whereas MobileNet V2 showed impressive performance \(97.12% accuracy\) but fell short in object detection. Our study highlights the potential of deep learning models in transforming how we classify plastic waste, with YOLO models proving to be the most effective. By balancing accuracy and computational efficiency, these models can help to create scalable, impactful solutions in waste management and recycling.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2412.20232v1)

---


## YOLO\-MST: Multiscale deep learning method for infrared small target detection based on super\-resolution and YOLO / 

发布日期：2024-12-27

作者：Taoran Yue

摘要：With the advancement of aerospace technology and the increasing demands of military applications, the development of low false\-alarm and high\-precision infrared small target detection algorithms has emerged as a key focus of research globally. However, the traditional model\-driven method is not robust enough when dealing with features such as noise, target size, and contrast. The existing deep\-learning methods have limited ability to extract and fuse key features, and it is difficult to achieve high\-precision detection in complex backgrounds and when target features are not obvious. To solve these problems, this paper proposes a deep\-learning infrared small target detection method that combines image super\-resolution technology with multi\-scale observation. First, the input infrared images are preprocessed with super\-resolution and multiple data enhancements are performed. Secondly, based on the YOLOv5 model, we proposed a new deep\-learning network named YOLO\-MST. This network includes replacing the SPPF module with the self\-designed MSFA module in the backbone, optimizing the neck, and finally adding a multi\-scale dynamic detection head to the prediction head. By dynamically fusing features from different scales, the detection head can better adapt to complex scenes. The mAP@0.5 detection rates of this method on two public datasets, SIRST and IRIS, reached 96.4% and 99.5% respectively, more effectively solving the problems of missed detection, false alarms, and low precision.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2412.19878v1)

---

