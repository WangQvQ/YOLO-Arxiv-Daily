# 每日从arXiv中获取最新YOLO相关论文


## Computer vision training dataset generation for robotic environments using Gaussian splatting / 

发布日期：2025-12-15

作者：Patryk Niżeniec

摘要：This paper introduces a novel pipeline for generating large\-scale, highly realistic, and automatically labeled datasets for computer vision tasks in robotic environments. Our approach addresses the critical challenges of the domain gap between synthetic and real\-world imagery and the time\-consuming bottleneck of manual annotation. We leverage 3D Gaussian Splatting \(3DGS\) to create photorealistic representations of the operational environment and objects. These assets are then used in a game engine where physics simulations create natural arrangements. A novel, two\-pass rendering technique combines the realism of splats with a shadow map generated from proxy meshes. This map is then algorithmically composited with the image to add both physically plausible shadows and subtle highlights, significantly enhancing realism. Pixel\-perfect segmentation masks are generated automatically and formatted for direct use with object detection models like YOLO. Our experiments show that a hybrid training strategy, combining a small set of real images with a large volume of our synthetic data, yields the best detection and segmentation performance, confirming this as an optimal strategy for efficiently achieving robust and accurate models.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.13411v1)

---


## FID\-Net: A Feature\-Enhanced Deep Learning Network for Forest Infestation Detection / 

发布日期：2025-12-15

作者：Yan Zhang

摘要：Forest pests threaten ecosystem stability, requiring efficient monitoring. To overcome the limitations of traditional methods in large\-scale, fine\-grained detection, this study focuses on accurately identifying infected trees and analyzing infestation patterns. We propose FID\-Net, a deep learning model that detects pest\-affected trees from UAV visible\-light imagery and enables infestation analysis via three spatial metrics. Based on YOLOv8n, FID\-Net introduces a lightweight Feature Enhancement Module \(FEM\) to extract disease\-sensitive cues, an Adaptive Multi\-scale Feature Fusion Module \(AMFM\) to align and fuse dual\-branch features \(RGB and FEM\-enhanced\), and an Efficient Channel Attention \(ECA\) mechanism to enhance discriminative information efficiently. From detection results, we construct a pest situation analysis framework using: \(1\) Kernel Density Estimation to locate infection hotspots; \(2\) neighborhood evaluation to assess healthy trees' infection risk; \(3\) DBSCAN clustering to identify high\-density healthy clusters as priority protection zones. Experiments on UAV imagery from 32 forest plots in eastern Tianshan, China, show that FID\-Net achieves 86.10% precision, 75.44% recall, 82.29% mAP@0.5, and 64.30% mAP@0.5:0.95, outperforming mainstream YOLO models. Analysis confirms infected trees exhibit clear clustering, supporting targeted forest protection. FID\-Net enables accurate tree health discrimination and, combined with spatial metrics, provides reliable data for intelligent pest monitoring, early warning, and precise management.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.13104v1)

---


## Adaptive Detector\-Verifier Framework for Zero\-Shot Polyp Detection in Open\-World Settings / 

发布日期：2025-12-13

作者：Shengkai Xu

摘要：Polyp detectors trained on clean datasets often underperform in real\-world endoscopy, where illumination changes, motion blur, and occlusions degrade image quality. Existing approaches struggle with the domain gap between controlled laboratory conditions and clinical practice, where adverse imaging conditions are prevalent. In this work, we propose AdaptiveDetector, a novel two\-stage detector\-verifier framework comprising a YOLOv11 detector with a vision\-language model \(VLM\) verifier. The detector adaptively adjusts per\-frame confidence thresholds under VLM guidance, while the verifier is fine\-tuned with Group Relative Policy Optimization \(GRPO\) using an asymmetric, cost\-sensitive reward function specifically designed to discourage missed detections \-\- a critical clinical requirement. To enable realistic assessment under challenging conditions, we construct a comprehensive synthetic testbed by systematically degrading clean datasets with adverse conditions commonly encountered in clinical practice, providing a rigorous benchmark for zero\-shot evaluation. Extensive zero\-shot evaluation on synthetically degraded CVC\-ClinicDB and Kvasir\-SEG images demonstrates that our approach improves recall by 14 to 22 percentage points over YOLO alone, while precision remains within 0.7 points below to 1.7 points above the baseline. This combination of adaptive thresholding and cost\-sensitive reinforcement learning achieves clinically aligned, open\-world polyp detection with substantially fewer false negatives, thereby reducing the risk of missed precancerous polyps and improving patient outcomes.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.12492v1)

---


## TCLeaf\-Net: a transformer\-convolution framework with global\-local attention for robust in\-field lesion\-level plant leaf disease detection / 

发布日期：2025-12-13

作者：Zishen Song

摘要：Timely and accurate detection of foliar diseases is vital for safeguarding crop growth and reducing yield losses. Yet, in real\-field conditions, cluttered backgrounds, domain shifts, and limited lesion\-level datasets hinder robust modeling. To address these challenges, we release Daylily\-Leaf, a paired lesion\-level dataset comprising 1,746 RGB images and 7,839 lesions captured under both ideal and in\-field conditions, and propose TCLeaf\-Net, a transformer\-convolution hybrid detector optimized for real\-field use. TCLeaf\-Net is designed to tackle three major challenges. To mitigate interference from complex backgrounds, the transformer\-convolution module \(TCM\) couples global context with locality\-preserving convolution to suppress non\-leaf regions. To reduce information loss during downsampling, the raw\-scale feature recalling and sampling \(RSFRS\) block combines bilinear resampling and convolution to preserve fine spatial detail. To handle variations in lesion scale and feature shifts, the deformable alignment block with FPN \(DFPN\) employs offset\-based alignment and multi\-receptive\-field perception to strengthen multi\-scale fusion. Experimental results show that on the in\-field split of the Daylily\-Leaf dataset, TCLeaf\-Net improves mAP@50 by 5.4 percentage points over the baseline model, reaching 78.2%, while reducing computation by 7.5 GFLOPs and GPU memory usage by 8.7%. Moreover, the model outperforms recent YOLO and RT\-DETR series in both precision and recall, and demonstrates strong performance on the PlantDoc, Tomato\-Leaf, and Rice\-Leaf datasets, validating its robustness and generalizability to other plant disease detection scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.12357v1)

---


## Cognitive\-YOLO: LLM\-Driven Architecture Synthesis from First Principles of Data for Object Detection / 

发布日期：2025-12-13

作者：Jiahao Zhao

摘要：Designing high\-performance object detection architectures is a complex task, where traditional manual design is time\-consuming and labor\-intensive, and Neural Architecture Search \(NAS\) is computationally prohibitive. While recent approaches using Large Language Models \(LLMs\) show promise, they often function as iterative optimizers within a search loop, rather than generating architectures directly from a holistic understanding of the data. To address this gap, we propose Cognitive\-YOLO, a novel framework for LLM\-driven architecture synthesis that generates network configurations directly from the intrinsic characteristics of the dataset. Our method consists of three stages: first, an analysis module extracts key meta\-features \(e.g., object scale distribution and scene density\) from the target dataset; second, the LLM reasons upon these features, augmented with state\-of\-the\-art components retrieved via Retrieval\-Augmented Generation \(RAG\), to synthesize the architecture into a structured Neural Architecture Description Language \(NADL\); finally, a compiler instantiates this description into a deployable model. Extensive experiments on five diverse object detection datasets demonstrate that our proposed Cognitive\-YOLO consistently generates superior architectures, achieving highly competitive performance and demonstrating a superior performance\-per\-parameter trade\-off compared to strong baseline models across multiple benchmarks. Crucially, our ablation studies prove that the LLM's data\-driven reasoning is the primary driver of performance, demonstrating that a deep understanding of data "first principles" is more critical for achieving a superior architecture than simply retrieving SOTA components.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.12281v1)

---

