# 每日从arXiv中获取最新YOLO相关论文


## YOLO11\-4K: An Efficient Architecture for Real\-Time Small Object Detection in 4K Panoramic Images / 

发布日期：2025-12-18

作者：Huma Hafeez

摘要：The processing of omnidirectional 360\-degree images poses significant challenges for object detection due to inherent spatial distortions, wide fields of view, and ultra\-high\-resolution inputs. Conventional detectors such as YOLO are optimised for standard image sizes \(for example, 640x640 pixels\) and often struggle with the computational demands of 4K or higher\-resolution imagery typical of 360\-degree vision. To address these limitations, we introduce YOLO11\-4K, an efficient real\-time detection framework tailored for 4K panoramic images. The architecture incorporates a novel multi\-scale detection head with a P2 layer to improve sensitivity to small objects often missed at coarser scales, and a GhostConv\-based backbone to reduce computational complexity without sacrificing representational power. To enable evaluation, we manually annotated the CVIP360 dataset, generating 6,876 frame\-level bounding boxes and producing a publicly available, detection\-ready benchmark for 4K panoramic scenes. YOLO11\-4K achieves 0.95 mAP at 0.50 IoU with 28.3 milliseconds inference per frame, representing a 75 percent latency reduction compared to YOLO11 \(112.3 milliseconds\), while also improving accuracy \(mAP at 0.50 of 0.95 versus 0.908\). This balance of efficiency and precision enables robust object detection in expansive 360\-degree environments, making the framework suitable for real\-world high\-resolution panoramic applications. While this work focuses on 4K omnidirectional images, the approach is broadly applicable to high\-resolution detection tasks in autonomous navigation, surveillance, and augmented reality.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.16493v1)

---


## From Words to Wavelengths: VLMs for Few\-Shot Multispectral Object Detection / 

发布日期：2025-12-17

作者：Manuel Nkegoum

摘要：Multispectral object detection is critical for safety\-sensitive applications such as autonomous driving and surveillance, where robust perception under diverse illumination conditions is essential. However, the limited availability of annotated multispectral data severely restricts the training of deep detectors. In such data\-scarce scenarios, textual class information can serve as a valuable source of semantic supervision. Motivated by the recent success of Vision\-Language Models \(VLMs\) in computer vision, we explore their potential for few\-shot multispectral object detection. Specifically, we adapt two representative VLM\-based detectors, Grounding DINO and YOLO\-World, to handle multispectral inputs and propose an effective mechanism to integrate text, visual and thermal modalities. Through extensive experiments on two popular multispectral image benchmarks, FLIR and M3FD, we demonstrate that VLM\-based detectors not only excel in few\-shot regimes, significantly outperforming specialized multispectral models trained with comparable data, but also achieve competitive or superior results under fully supervised settings. Our findings reveal that the semantic priors learned by large\-scale VLMs effectively transfer to unseen spectral modalities, ofFering a powerful pathway toward data\-efficient multispectral perception.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.15971v1)

---


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

