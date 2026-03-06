# 每日从arXiv中获取最新YOLO相关论文


## A 360\-degree Multi\-camera System for Blue Emergency Light Detection Using Color Attention RT\-DETR and the ABLDataset / 

发布日期：2026-03-05

作者：Francisco Vacalebri\-Lloret

摘要：This study presents an advanced system for detecting blue lights on emergency vehicles, developed using ABLDataset, a curated dataset that includes images of European emergency vehicles under various climatic and geographic conditions. The system employs a configuration of four fisheye cameras, each with a 180\-degree horizontal field of view, mounted on the sides of the vehicle. A calibration process enables the azimuthal localization of the detections. Additionally, a comparative analysis of major deep neural network algorithms was conducted, including YOLO \(v5, v8, and v10\), RetinaNet, Faster R\-CNN, and RT\-DETR. RT\-DETR was selected as the base model and enhanced through the incorporation of a color attention block, achieving an accuracy of 94.7 percent and a recall of 94.1 percent on the test set, with field test detections reaching up to 70 meters. Furthermore, the system estimates the approach angle of the emergency vehicle relative to the center of the car using geometric transformations. Designed for integration into a multimodal system that combines visual and acoustic data, this system has demonstrated high efficiency, offering a promising approach to enhancing Advanced Driver Assistance Systems \(ADAS\) and road safety.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.05058v1)

---


## Yolo\-Key\-6D: Single Stage Monocular 6D Pose Estimation with Keypoint Enhancements / 

发布日期：2026-03-04

作者：Kemal Alperen Çetiner

摘要：Estimating the 6D pose of objects from a single RGB image is a critical task for robotics and extended reality applications. However, state\-of\-the\-art multi stage methods often suffer from high latency, making them unsuitable for real time use. In this paper, we present Yolo\-Key\-6D, a novel single stage, end\-to\-end framework for monocular 6D pose estimation designed for both speed and accuracy. Our approach enhances a YOLO based architecture by integrating an auxiliary head that regresses the 2D projections of an object's 3D bounding box corners. This keypoint detection task significantly improves the network's understanding of 3D geometry. For stable end\-to\-end training, we directly regress rotation using a continuous 9D representation projected to SO\(3\) via singular value decomposition. On the LINEMOD and LINEMOD\-Occluded benchmarks, YOLO\-Key\-6D achieves competitive accuracy scores of 96.24% and 69.41%, respectively, with the ADD\(\-S\) 0.1d metric, while proving itself to operate in real time. Our results demonstrate that a carefully designed single stage method can provide a practical and effective balance of performance and efficiency for real world deployment.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.03879v1)

---


## Tracking Feral Horses in Aerial Video Using Oriented Bounding Boxes / 

发布日期：2026-03-04

作者：Saeko Takizawa

摘要：The social structures of group\-living animals such as feral horses are diverse and remain insufficiently understood, even within a single species. To investigate group dynamics, aerial videos are often utilized to track individuals and analyze their movement trajectories, which are essential for evaluating inter\-individual interactions and comparing social behaviors. Accurate individual tracking is therefore crucial. In multi\-animal tracking, axis\-aligned bounding boxes \(bboxes\) are widely used; however, for aerial top\-view footage of entire groups, their performance degrades due to complex backgrounds, small target sizes, high animal density, and varying body orientations. To address this issue, we employ oriented bounding boxes \(OBBs\), which include rotation angles and reduce unnecessary background. Nevertheless, current OBB detectors such as YOLO\-OBB restrict angles within a 180$^\{circ\}$ range, making it impossible to distinguish head from tail and often causing sudden 180$^\{circ\}$ flips across frames, which severely disrupts continuous tracking. To overcome this limitation, we propose a head\-orientation estimation method that crops OBB\-centered patches, applies three detectors \(head, tail, and head\-tail\), and determines the final label through IoU\-based majority voting. Experiments using 299 test images show that our method achieves 99.3% accuracy, outperforming individual models, demonstrating its effectiveness for robust OBB\-based tracking.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.03604v1)

---


## SEP\-YOLO: Fourier\-Domain Feature Representation for Transparent Object Instance Segmentation / 

发布日期：2026-03-03

作者：Fengming Zhang

摘要：Transparent object instance segmentation presents significant challenges in computer vision, due to the inherent properties of transparent objects, including boundary blur, low contrast, and high dependence on background context. Existing methods often fail as they depend on strong appearance cues and clear boundaries. To address these limitations, we propose SEP\-YOLO, a novel framework that integrates a dual\-domain collaborative mechanism for transparent object instance segmentation. Our method incorporates a Frequency Domain Detail Enhancement Module, which separates and enhances weak highfrequency boundary components via learnable complex weights. We further design a multi\-scale spatial refinement stream, which consists of a Content\-Aware Alignment Neck and a Multi\-scale Gated Refinement Block, to ensure precise feature alignment and boundary localization in deep semantic features. We also provide high\-quality instance\-level annotations for the Trans10K dataset, filling the critical data gap in transparent object instance segmentation. Extensive experiments on the Trans10K and GVD datasets show that SEP\-YOLO achieves state\-of\-the\-art \(SOTA\) performance.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.02648v1)

---


## Towards Khmer Scene Document Layout Detection / 

发布日期：2026-02-28

作者：Marry Kong

摘要：While document layout analysis for Latin scripts has advanced significantly, driven by the advent of large multimodal models \(LMMs\), progress for the Khmer language remains constrained because of the scarcity of annotated training data. This gap is particularly acute for scene documents, where perspective distortions and complex backgrounds challenge traditional methods. Given the structural complexities of Khmer script, such as diacritics and multi\-layer character stacking, existing Latin\-based layout analysis models fail to accurately delineate semantic layout units, particularly for dense text regions \(e.g., list items\). In this paper, we present the first comprehensive study on Khmer scene document layout detection. We contribute a novel framework comprising three key elements: \(1\) a robust training and benchmarking dataset specifically for Khmer scene layouts; \(2\) an open\-source document augmentation tool capable of synthesizing realistic scene documents to scale training data; and \(3\) layout detection baselines utilizing YOLO\-based architectures with oriented bounding boxes \(OBB\) to handle geometric distortions. To foster further research in the Khmer document analysis and recognition \(DAR\) community, we release our models, code, and datasets in this gated repository \(in review\).

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.00707v1)

---

