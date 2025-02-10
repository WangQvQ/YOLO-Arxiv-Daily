# 每日从arXiv中获取最新YOLO相关论文


## MHAF\-YOLO: Multi\-Branch Heterogeneous Auxiliary Fusion YOLO for accurate object detection / 

发布日期：2025-02-07

作者：Zhiqiang Yang

摘要：Due to the effective multi\-scale feature fusion capabilities of the Path Aggregation FPN \(PAFPN\), it has become a widely adopted component in YOLO\-based detectors. However, PAFPN struggles to integrate high\-level semantic cues with low\-level spatial details, limiting its performance in real\-world applications, especially with significant scale variations. In this paper, we propose MHAF\-YOLO, a novel detection framework featuring a versatile neck design called the Multi\-Branch Auxiliary FPN \(MAFPN\), which consists of two key modules: the Superficial Assisted Fusion \(SAF\) and Advanced Assisted Fusion \(AAF\). The SAF bridges the backbone and the neck by fusing shallow features, effectively transferring crucial low\-level spatial information with high fidelity. Meanwhile, the AAF integrates multi\-scale feature information at deeper neck layers, delivering richer gradient information to the output layer and further enhancing the model learning capacity. To complement MAFPN, we introduce the Global Heterogeneous Flexible Kernel Selection \(GHFKS\) mechanism and the Reparameterized Heterogeneous Multi\-Scale \(RepHMS\) module to enhance feature fusion. RepHMS is globally integrated into the network, utilizing GHFKS to select larger convolutional kernels for various feature layers, expanding the vertical receptive field and capturing contextual information across spatial hierarchies. Locally, it optimizes convolution by processing both large and small kernels within the same layer, broadening the lateral receptive field and preserving crucial details for detecting smaller targets. The source code of this work is available at: https://github.com/yang0201/MHAF\-YOLO.

中文摘要：


代码链接：https://github.com/yang0201/MHAF-YOLO.

论文链接：[阅读更多](http://arxiv.org/abs/2502.04656v1)

---


## Brain Tumor Identification using Improved YOLOv8 / 

发布日期：2025-02-06

作者：Rupesh Dulal

摘要：Identifying the extent of brain tumors is a significant challenge in brain cancer treatment. The main difficulty is in the approximate detection of tumor size. Magnetic resonance imaging \(MRI\) has become a critical diagnostic tool. However, manually detecting the boundaries of brain tumors from MRI scans is a labor\-intensive task that requires extensive expertise. Deep learning and computer\-aided detection techniques have led to notable advances in machine learning for this purpose. In this paper, we propose a modified You Only Look Once \(YOLOv8\) model to accurately detect the tumors within the MRI images. The proposed model replaced the Non\-Maximum Suppression \(NMS\) algorithm with a Real\-Time Detection Transformer \(RT\- DETR\) in the detection head. NMS filters out redundant or overlapping bounding boxes in the detected tumors, but they are hand\-designed and pre\-set. RT\-DETR removes hand\-designed components. The second improvement was made by replacing the normal convolution block with ghost convolution. Ghost Convolution reduces computational and memory costs while maintaining high accuracy and enabling faster inference, making it ideal for resource\-constrained environments and real\-time applications. The third improvement was made by introducing a vision transformer block in the backbone of YOLOv8 to extract context\-aware features. We used a publicly available dataset of brain tumors in the proposed model. The proposed model performed better than the original YOLOv8 model and also performed better than other object detectors \(Faster R\- CNN, Mask R\-CNN, YOLO, YOLOv3, YOLOv4, YOLOv5, SSD, RetinaNet, EfficientDet, and DETR\). The proposed model achieved 0.91 mAP \(mean Average Precision\)@0.5.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2502.03746v1)

---


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

