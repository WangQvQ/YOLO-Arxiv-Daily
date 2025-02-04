# 每日从arXiv中获取最新YOLO相关论文


## Early Diagnosis and Severity Assessment of Weligama Coconut Leaf Wilt Disease and Coconut Caterpillar Infestation using Deep Learning\-based Image Processing Techniques / 

发布日期：2025-01-31

作者：Samitha Vidhanaarachchi

摘要：Global Coconut \(Cocos nucifera \(L.\)\) cultivation faces significant challenges, including yield loss, due to pest and disease outbreaks. In particular, Weligama Coconut Leaf Wilt Disease \(WCWLD\) and Coconut Caterpillar Infestation \(CCI\) damage coconut trees, causing severe coconut production loss in Sri Lanka and nearby coconut\-producing countries. Currently, both WCWLD and CCI are detected through on\-field human observations, a process that is not only time\-consuming but also limits the early detection of infections. This paper presents a study conducted in Sri Lanka, demonstrating the effectiveness of employing transfer learning\-based Convolutional Neural Network \(CNN\) and Mask Region\-based\-CNN \(Mask R\-CNN\) to identify WCWLD and CCI at their early stages and to assess disease progression. Further, this paper presents the use of the You Only Look Once \(YOLO\) object detection model to count the number of caterpillars distributed on leaves with CCI. The introduced methods were tested and validated using datasets collected from Matara, Puttalam, and Makandura, Sri Lanka. The results show that the proposed methods identify WCWLD and CCI with an accuracy of 90% and 95%, respectively. In addition, the proposed WCWLD disease severity identification method classifies the severity with an accuracy of 97%. Furthermore, the accuracies of the object detection models for calculating the number of caterpillars in the leaflets were: YOLOv5\-96.87%, YOLOv8\-96.1%, and YOLO11\-95.9%.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.18835v1)

---


## Adaptive Object Detection for Indoor Navigation Assistance: A Performance Evaluation of Real\-Time Algorithms / 

发布日期：2025-01-30

作者：Abhinav Pratap

摘要：This study addresses the need for accurate and efficient object detection in assistive technologies for visually impaired individuals. We evaluate four real\-time object detection algorithms YOLO, SSD, Faster R\-CNN, and Mask R\-CNN within the context of indoor navigation assistance. Using the Indoor Objects Detection dataset, we analyze detection accuracy, processing speed, and adaptability to indoor environments. Our findings highlight the trade\-offs between precision and efficiency, offering insights into selecting optimal algorithms for realtime assistive navigation. This research advances adaptive machine learning applications, enhancing indoor navigation solutions for the visually impaired and promoting accessibility.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.18444v1)

---


## Efficient Feature Fusion for UAV Object Detection / 

发布日期：2025-01-29

作者：Xudong Wang

摘要：Object detection in unmanned aerial vehicle \(UAV\) remote sensing images poses significant challenges due to unstable image quality, small object sizes, complex backgrounds, and environmental occlusions. Small objects, in particular, occupy small portions of images, making their accurate detection highly difficult. Existing multi\-scale feature fusion methods address these challenges to some extent by aggregating features across different resolutions. However, they often fail to effectively balance the classification and localization performance for small objects, primarily due to insufficient feature representation and imbalanced network information flow. In this paper, we propose a novel feature fusion framework specifically designed for UAV object detection tasks to enhance both localization accuracy and classification performance. The proposed framework integrates hybrid upsampling and downsampling modules, enabling feature maps from different network depths to be flexibly adjusted to arbitrary resolutions. This design facilitates cross\-layer connections and multi\-scale feature fusion, ensuring improved representation of small objects. Our approach leverages hybrid downsampling to enhance fine\-grained feature representation, improving spatial localization of small targets, even under complex conditions. Simultaneously, the upsampling module aggregates global contextual information, optimizing feature consistency across scales and enhancing classification robustness in cluttered scenes. Experimental results on two public UAV datasets demonstrate the effectiveness of the proposed framework. Integrated into the YOLO\-v10 model, our method achieves a 2% improvement in average precision \(AP\) compared to the baseline YOLO\-v10 model, while maintaining the same number of parameters. These results highlight the potential of our framework for accurate and efficient UAV object detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.17983v2)

---


## Real Time Scheduling Framework for Multi Object Detection via Spiking Neural Networks / 

发布日期：2025-01-29

作者：Donghwa Kang

摘要：Given the energy constraints in autonomous mobile agents \(AMAs\), such as unmanned vehicles, spiking neural networks \(SNNs\) are increasingly favored as a more efficient alternative to traditional artificial neural networks. AMAs employ multi\-object detection \(MOD\) from multiple cameras to identify nearby objects while ensuring two essential objectives, \(R1\) timing guarantee and \(R2\) high accuracy for safety. In this paper, we propose RT\-SNN, the first system design, aiming at achieving R1 and R2 in SNN\-based MOD systems on AMAs. Leveraging the characteristic that SNNs gather feature data of input image termed as membrane potential, through iterative computation over multiple timesteps, RT\-SNN provides multiple execution options with adjustable timesteps and a novel method for reusing membrane potential to support R1. Then, it captures how these execution strategies influence R2 by introducing a novel notion of mean absolute error and membrane confidence. Further, RT\-SNN develops a new scheduling framework consisting of offline schedulability analysis for R1 and a run\-time scheduling algorithm for R2 using the notion of membrane confidence. We deployed RT\-SNN to Spiking\-YOLO, the SNN\-based MOD model derived from ANN\-to\-SNN conversion, and our experimental evaluation confirms its effectiveness in meeting the R1 and R2 requirements while providing significant energy efficiency.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.18412v1)

---


## Assessing the Capability of YOLO\- and Transformer\-based Object Detectors for Real\-time Weed Detection / 

发布日期：2025-01-29

作者：Alicia Allmendinger

摘要：Spot spraying represents an efficient and sustainable method for reducing the amount of pesticides, particularly herbicides, used in agricultural fields. To achieve this, it is of utmost importance to reliably differentiate between crops and weeds, and even between individual weed species in situ and under real\-time conditions. To assess suitability for real\-time application, different object detection models that are currently state\-of\-the\-art are compared. All available models of YOLOv8, YOLOv9, YOLOv10, and RT\-DETR are trained and evaluated with images from a real field situation. The images are separated into two distinct datasets: In the initial data set, each species of plants is trained individually; in the subsequent dataset, a distinction is made between monocotyledonous weeds, dicotyledonous weeds, and three chosen crops. The results demonstrate that while all models perform equally well in the metrics evaluated, the YOLOv9 models, particularly the YOLOv9s and YOLOv9e, stand out in terms of their strong recall scores \(66.58 % and 72.36 %\), as well as mAP50 \(73.52 % and 79.86 %\), and mAP50\-95 \(43.82 % and 47.00 %\) in dataset 2. However, the RT\-DETR models, especially RT\-DETR\-l, excel in precision with reaching 82.44 % on dataset 1 and 81.46 % in dataset 2, making them particularly suitable for scenarios where minimizing false positives is critical. In particular, the smallest variants of the YOLO models \(YOLOv8n, YOLOv9t, and YOLOv10n\) achieve substantially faster inference times down to 7.58 ms for dataset 2 on the NVIDIA GeForce RTX 4090 GPU for analyzing one frame, while maintaining competitive accuracy, highlighting their potential for deployment in resource\-constrained embedded computing devices as typically used in productive setups.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.17387v2)

---

