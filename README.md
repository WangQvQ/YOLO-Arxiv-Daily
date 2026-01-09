# 每日从arXiv中获取最新YOLO相关论文


## Character Detection using YOLO for Writer Identification in multiple Medieval books / 

发布日期：2026-01-08

作者：Alessandra Scotto di Freca

摘要：Paleography is the study of ancient and historical handwriting, its key objectives include the dating of manuscripts and understanding the evolution of writing. Estimating when a document was written and tracing the development of scripts and writing styles can be aided by identifying the individual scribes who contributed to a medieval manuscript. Although digital technologies have made significant progress in this field, the general problem remains unsolved and continues to pose open challenges. ... We previously proposed an approach focused on identifying specific letters or abbreviations that characterize each writer. In that study, we considered the letter "a", as it was widely present on all pages of text and highly distinctive, according to the suggestions of expert paleographers. We used template matching techniques to detect the occurrences of the character "a" on each page and the convolutional neural network \(CNN\) to attribute each instance to the correct scribe. Moving from the interesting results achieved from this previous system and being aware of the limitations of the template matching technique, which requires an appropriate threshold to work, we decided to experiment in the same framework with the use of the YOLO object detection model to identify the scribe who contributed to the writing of different medieval books. We considered the fifth version of YOLO to implement the YOLO object detection model, which completely substituted the template matching and CNN used in the previous work. The experimental results demonstrate that YOLO effectively extracts a greater number of letters considered, leading to a more accurate second\-stage classification. Furthermore, the YOLO confidence score provides a foundation for developing a system that applies a rejection threshold, enabling reliable writer identification even in unseen manuscripts.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.04834v1)

---


## CT Scans As Video: Efficient Intracranial Hemorrhage Detection Using Multi\-Object Tracking / 

发布日期：2026-01-05

作者：Amirreza Parvahan

摘要：Automated analysis of volumetric medical imaging on edge devices is severely constrained by the high memory and computational demands of 3D Convolutional Neural Networks \(CNNs\). This paper develops a lightweight computer vision framework that reconciles the efficiency of 2D detection with the necessity of 3D context by reformulating volumetric Computer Tomography \(CT\) data as sequential video streams. This video\-viewpoint paradigm is applied to the time\-sensitive task of Intracranial Hemorrhage \(ICH\) detection using the Hemorica dataset. To ensure operational efficiency, we benchmarked multiple generations of the YOLO architecture \(v8, v10, v11 and v12\) in their Nano configurations, selecting the version with the highest mAP@50 to serve as the slice\-level backbone. A ByteTrack algorithm is then introduced to enforce anatomical consistency across the $z$\-axis. To address the initialization lag inherent in video trackers, a hybrid inference strategy and a spatiotemporal consistency filter are proposed to distinguish true pathology from transient prediction noise. Experimental results on independent test data demonstrate that the proposed framework serves as a rigorous temporal validator, increasing detection Precision from 0.703 to 0.779 compared to the baseline 2D detector, while maintaining high sensitivity. By approximating 3D contextual reasoning at a fraction of the computational cost, this method provides a scalable solution for real\-time patient prioritization in resource\-constrained environments, such as mobile stroke units and IoT\-enabled remote clinics.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.02521v1)

---


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

