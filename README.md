# 每日从arXiv中获取最新YOLO相关论文


## CERBERUS: Crack Evaluation & Recognition Benchmark for Engineering Reliability & Urban Stability / 

发布日期：2025-06-27

作者：Justin Reinman

摘要：CERBERUS is a synthetic benchmark designed to help train and evaluate AI models for detecting cracks and other defects in infrastructure. It includes a crack image generator and realistic 3D inspection scenarios built in Unity. The benchmark features two types of setups: a simple Fly\-By wall inspection and a more complex Underpass scene with lighting and geometry challenges. We tested a popular object detection model \(YOLO\) using different combinations of synthetic and real crack data. Results show that combining synthetic and real data improves performance on real\-world images. CERBERUS provides a flexible, repeatable way to test defect detection systems and supports future research in automated infrastructure inspection. CERBERUS is publicly available at https://github.com/justinreinman/Cerberus\-Defect\-Generator.

中文摘要：


代码链接：https://github.com/justinreinman/Cerberus-Defect-Generator.

论文链接：[阅读更多](http://arxiv.org/abs/2506.21909v1)

---


## Visual Content Detection in Educational Videos with Transfer Learning and Dataset Enrichment / 

发布日期：2025-06-27

作者：Dipayan Biswas

摘要：Video is transforming education with online courses and recorded lectures supplementing and replacing classroom teaching. Recent research has focused on enhancing information retrieval for video lectures with advanced navigation, searchability, summarization, as well as question answering chatbots. Visual elements like tables, charts, and illustrations are central to comprehension, retention, and data presentation in lecture videos, yet their full potential for improving access to video content remains underutilized. A major factor is that accurate automatic detection of visual elements in a lecture video is challenging; reasons include i\) most visual elements, such as charts, graphs, tables, and illustrations, are artificially created and lack any standard structure, and ii\) coherent visual objects may lack clear boundaries and may be composed of connected text and visual components. Despite advancements in deep learning based object detection, current models do not yield satisfactory performance due to the unique nature of visual content in lectures and scarcity of annotated datasets. This paper reports on a transfer learning approach for detecting visual elements in lecture video frames. A suite of state of the art object detection models were evaluated for their performance on lecture video datasets. YOLO emerged as the most promising model for this task. Subsequently YOLO was optimized for lecture video object detection with training on multiple benchmark datasets and deploying a semi\-supervised auto labeling strategy. Results evaluate the success of this approach, also in developing a general solution to the problem of object detection in lecture videos. Paper contributions include a publicly released benchmark of annotated lecture video frames, along with the source code to facilitate future research.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.21903v1)

---


## YOLO\-FDA: Integrating Hierarchical Attention and Detail Enhancement for Surface Defect Detection / 

发布日期：2025-06-26

作者：Jiawei Hu

摘要：Surface defect detection in industrial scenarios is both crucial and technically demanding due to the wide variability in defect types, irregular shapes and sizes, fine\-grained requirements, and complex material textures. Although recent advances in AI\-based detectors have improved performance, existing methods often suffer from redundant features, limited detail sensitivity, and weak robustness under multiscale conditions. To address these challenges, we propose YOLO\-FDA, a novel YOLO\-based detection framework that integrates fine\-grained detail enhancement and attention\-guided feature fusion. Specifically, we adopt a BiFPN\-style architecture to strengthen bidirectional multilevel feature aggregation within the YOLOv5 backbone. To better capture fine structural changes, we introduce a Detail\-directional Fusion Module \(DDFM\) that introduces a directional asymmetric convolution in the second\-lowest layer to enrich spatial details and fuses the second\-lowest layer with low\-level features to enhance semantic consistency. Furthermore, we propose two novel attention\-based fusion strategies, Attention\-weighted Concatenation \(AC\) and Cross\-layer Attention Fusion \(CAF\) to improve contextual representation and reduce feature noise. Extensive experiments on benchmark datasets demonstrate that YOLO\-FDA consistently outperforms existing state\-of\-the\-art methods in terms of both accuracy and robustness across diverse types of defects and scales.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.21135v1)

---


## Lightweight Multi\-Frame Integration for Robust YOLO Object Detection in Videos / 

发布日期：2025-06-25

作者：Yitong Quan

摘要：Modern image\-based object detection models, such as YOLOv7, primarily process individual frames independently, thus ignoring valuable temporal context naturally present in videos. Meanwhile, existing video\-based detection methods often introduce complex temporal modules, significantly increasing model size and computational complexity. In practical applications such as surveillance and autonomous driving, transient challenges including motion blur, occlusions, and abrupt appearance changes can severely degrade single\-frame detection performance. To address these issues, we propose a straightforward yet highly effective strategy: stacking multiple consecutive frames as input to a YOLO\-based detector while supervising only the output corresponding to a single target frame. This approach leverages temporal information with minimal modifications to existing architectures, preserving simplicity, computational efficiency, and real\-time inference capability. Extensive experiments on the challenging MOT20Det and our BOAT360 datasets demonstrate that our method improves detection robustness, especially for lightweight models, effectively narrowing the gap between compact and heavy detection networks. Additionally, we contribute the BOAT360 benchmark dataset, comprising annotated fisheye video sequences captured from a boat, to support future research in multi\-frame video object detection in challenging real\-world scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.20550v1)

---


## From Codicology to Code: A Comparative Study of Transformer and YOLO\-based Detectors for Layout Analysis in Historical Documents / 

发布日期：2025-06-25

作者：Sergio Torres Aguilar

摘要：Robust Document Layout Analysis \(DLA\) is critical for the automated processing and understanding of historical documents with complex page organizations. This paper benchmarks five state\-of\-the\-art object detection architectures on three annotated datasets representing a spectrum of codicological complexity: The e\-NDP, a corpus of Parisian medieval registers \(1326\-1504\); CATMuS, a diverse multiclass dataset derived from various medieval and modern sources \(ca.12th\-17th centuries\) and HORAE, a corpus of decorated books of hours \(ca.13th\-16th centuries\). We evaluate two Transformer\-based models \(Co\-DETR, Grounding DINO\) against three YOLO variants \(AABB, OBB, and YOLO\-World\). Our findings reveal significant performance variations dependent on model architecture, data set characteristics, and bounding box representation. In the e\-NDP dataset, Co\-DETR achieves state\-of\-the\-art results \(0.752 mAP@.50:.95\), closely followed by YOLOv11X\-OBB \(0.721\). Conversely, on the more complex CATMuS and HORAE datasets, the CNN\-based YOLOv11x\-OBB significantly outperforms all other models \(0.564 and 0.568, respectively\). This study unequivocally demonstrates that using Oriented Bounding Boxes \(OBB\) is not a minor refinement but a fundamental requirement for accurately modeling the non\-Cartesian nature of historical manuscripts. We conclude that a key trade\-off exists between the global context awareness of Transformers, ideal for structured layouts, and the superior generalization of CNN\-OBB models for visually diverse and complex documents.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.20326v1)

---

