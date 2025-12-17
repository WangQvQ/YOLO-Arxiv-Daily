# 每日从arXiv中获取最新YOLO相关论文


## From YOLO to VLMs: Advancing Zero\-Shot and Few\-Shot Detection of Wastewater Treatment Plants Using Satellite Imagery in MENA Region / 

发布日期：2025-12-16

作者：Akila Premarathna

摘要：In regions of the Middle East and North Africa \(MENA\), there is a high demand for wastewater treatment plants \(WWTPs\), crucial for sustainable water management. Precise identification of WWTPs from satellite images enables environmental monitoring. Traditional methods like YOLOv8 segmentation require extensive manual labeling. But studies indicate that vision\-language models \(VLMs\) are an efficient alternative to achieving equivalent or superior results through inherent reasoning and annotation. This study presents a structured methodology for VLM comparison, divided into zero\-shot and few\-shot streams specifically to identify WWTPs. The YOLOv8 was trained on a governmental dataset of 83,566 high\-resolution satellite images from Egypt, Saudi Arabia, and UAE: ~85% WWTPs \(positives\), 15% non\-WWTPs \(negatives\). Evaluated VLMs include LLaMA 3.2 Vision, Qwen 2.5 VL, DeepSeek\-VL2, Gemma 3, Gemini, and Pixtral 12B \(Mistral\), used to identify WWTP components such as circular/rectangular tanks, aeration basins and distinguish confounders via expert prompts producing JSON outputs with confidence and descriptions. The dataset comprises 1,207 validated WWTP locations \(198 UAE, 354 KSA, 655 Egypt\) and equal non\-WWTP sites from field/AI data, as 600mx600m Geo\-TIFF images \(Zoom 18, EPSG:4326\). Zero\-shot evaluations on WWTP images showed several VLMs out\-performing YOLOv8's true positive rate, with Gemma\-3 highest. Results confirm that VLMs, particularly with zero\-shot, can replace YOLOv8 for efficient, annotation\-free WWTP classification, enabling scalable remote sensing.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.14312v1)

---


## VajraV1 \-\- The most accurate Real Time Object Detector of the YOLO family / 

发布日期：2025-12-15

作者：Naman Balbir Singh Makkar

摘要：Recent years have seen significant advances in real\-time object detection, with the release of YOLOv10, YOLO11, YOLOv12, and YOLOv13 between 2024 and 2025. This technical report presents the VajraV1 model architecture, which introduces architectural enhancements over existing YOLO\-based detectors. VajraV1 combines effective design choices from prior YOLO models to achieve state\-of\-the\-art accuracy among real\-time object detectors while maintaining competitive inference speed.   On the COCO validation set, VajraV1\-Nano achieves 44.3% mAP, outperforming YOLOv12\-N by 3.7% and YOLOv13\-N by 2.7% at latency competitive with YOLOv12\-N and YOLOv11\-N. VajraV1\-Small achieves 50.4% mAP, exceeding YOLOv12\-S and YOLOv13\-S by 2.4%. VajraV1\-Medium achieves 52.7% mAP, outperforming YOLOv12\-M by 0.2%. VajraV1\-Large achieves 53.7% mAP, surpassing YOLOv13\-L by 0.3%. VajraV1\-Xlarge achieves 56.2% mAP, outperforming all existing real\-time object detectors.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.13834v1)

---


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


## Human\-AI Collaboration Mechanism Study on AIGC Assisted Image Production for Special Coverage / 

发布日期：2025-12-14

作者：Yajie Yang

摘要：Artificial Intelligence Generated Content \(AIGC\) assisting image production triggers controversy in journalism while attracting attention from media agencies. Key issues involve misinformation, authenticity, semantic fidelity, and interpretability. Most AIGC tools are opaque "black boxes," hindering the dual demands of content accuracy and semantic alignment and creating ethical, sociotechnical, and trust dilemmas. This paper explores pathways for controllable image production in journalism's special coverage and conducts two experiments with projects from China's media agency: \(1\) Experiment 1 tests cross\-platform adaptability via standardized prompts across three scenes, revealing disparities in semantic alignment, cultural specificity, and visual realism driven by training\-corpus bias and platform\-level filtering. \(2\) Experiment 2 builds a human\-in\-the\-loop modular pipeline combining high\-precision segmentation \(SAM, GroundingDINO\), semantic alignment \(BrushNet\), and style regulating \(Style\-LoRA, Prompt\-to\-Prompt\), ensuring editorial fidelity through CLIP\-based semantic scoring, NSFW/OCR/YOLO filtering, and verifiable content credentials. Traceable deployment preserves semantic representation. Consequently, we propose a human\-AI collaboration mechanism for AIGC assisted image production in special coverage and recommend evaluating Character Identity Stability \(CIS\), Cultural Expression Accuracy \(CEA\), and User\-Public Appropriateness \(U\-PA\).

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.13739v1)

---

