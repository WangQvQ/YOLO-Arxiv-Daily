# 每日从arXiv中获取最新YOLO相关论文


## Retrieving Objects from 3D Scenes with Box\-Guided Open\-Vocabulary Instance Segmentation / 

发布日期：2025-12-22

作者：Khanh Nguyen

摘要：Locating and retrieving objects from scene\-level point clouds is a challenging problem with broad applications in robotics and augmented reality. This task is commonly formulated as open\-vocabulary 3D instance segmentation. Although recent methods demonstrate strong performance, they depend heavily on SAM and CLIP to generate and classify 3D instance masks from images accompanying the point cloud, leading to substantial computational overhead and slow processing that limit their deployment in real\-world settings. Open\-YOLO 3D alleviates this issue by using a real\-time 2D detector to classify class\-agnostic masks produced directly from the point cloud by a pretrained 3D segmenter, eliminating the need for SAM and CLIP and significantly reducing inference time. However, Open\-YOLO 3D often fails to generalize to object categories that appear infrequently in the 3D training data. In this paper, we propose a method that generates 3D instance masks for novel objects from RGB images guided by a 2D open\-vocabulary detector. Our approach inherits the 2D detector's ability to recognize novel objects while maintaining efficient classification, enabling fast and accurate retrieval of rare instances from open\-ended text queries. Our code will be made available at https://github.com/ndkhanh360/BoxOVIS.

中文摘要：


代码链接：https://github.com/ndkhanh360/BoxOVIS.

论文链接：[阅读更多](http://arxiv.org/abs/2512.19088v1)

---


## Building UI/UX Dataset for Dark Pattern Detection and YOLOv12x\-based Real\-Time Object Recognition Detection System / 

发布日期：2025-12-20

作者：Se\-Young Jang

摘要：With the accelerating pace of digital transformation and the widespread adoption of online platforms, both social and technical concerns regarding dark patterns\-user interface designs that undermine users' ability to make informed and rational choices\-have become increasingly prominent. As corporate online platforms grow more sophisticated in their design strategies, there is a pressing need for proactive and real\-time detection technologies that go beyond the predominantly reactive approaches employed by regulatory authorities. In this paper, we propose a visual dark pattern detection framework that improves both detection accuracy and real\-time performance. To this end, we constructed a proprietary visual object detection dataset by manually collecting 4,066 UI/UX screenshots containing dark patterns from 194 websites across six major industrial sectors in South Korea and abroad. The collected images were annotated with five representative UI components commonly associated with dark patterns: Button, Checkbox, Input Field, Pop\-up, and QR Code. This dataset has been publicly released to support further research and development in the field. To enable real\-time detection, this study adopted the YOLOv12x object detection model and applied transfer learning to optimize its performance for visual dark pattern recognition. Experimental results demonstrate that the proposed approach achieves a high detection accuracy of 92.8% in terms of mAP@50, while maintaining a real\-time inference speed of 40.5 frames per second \(FPS\), confirming its effectiveness for practical deployment in online environments. Furthermore, to facilitate future research and contribute to technological advancements, the dataset constructed in this study has been made publicly available at https://github.com/B4E2/B4E2\-DarkPattern\-YOLO\-DataSet.

中文摘要：


代码链接：https://github.com/B4E2/B4E2-DarkPattern-YOLO-DataSet.

论文链接：[阅读更多](http://arxiv.org/abs/2512.18269v1)

---


## YolovN\-CBi: A Lightweight and Efficient Architecture for Real\-Time Detection of Small UAVs / 

发布日期：2025-12-19

作者：Ami Pandat

摘要：Unmanned Aerial Vehicles, commonly known as, drones pose increasing risks in civilian and defense settings, demanding accurate and real\-time drone detection systems. However, detecting drones is challenging because of their small size, rapid movement, and low visual contrast. A modified architecture of YolovN called the YolovN\-CBi is proposed that incorporates the Convolutional Block Attention Module \(CBAM\) and the Bidirectional Feature Pyramid Network \(BiFPN\) to improve sensitivity to small object detections. A curated training dataset consisting of 28K images is created with various flying objects and a local test dataset is collected with 2500 images consisting of very small drone objects. The proposed architecture is evaluated on four benchmark datasets, along with the local test dataset. The baseline Yolov5 and the proposed Yolov5\-CBi architecture outperform newer Yolo versions, including Yolov8 and Yolov12, in the speed\-accuracy trade\-off for small object detection. Four other variants of the proposed CBi architecture are also proposed and evaluated, which vary in the placement and usage of CBAM and BiFPN. These variants are further distilled using knowledge distillation techniques for edge deployment, using a Yolov5m\-CBi teacher and a Yolov5n\-CBi student. The distilled model achieved a mA@P0.5:0.9 of 0.6573, representing a 6.51% improvement over the teacher's score of 0.6171, highlighting the effectiveness of the distillation process. The distilled model is 82.9% faster than the baseline model, making it more suitable for real\-time drone detection. These findings highlight the effectiveness of the proposed CBi architecture, together with the distilled lightweight models in advancing efficient and accurate real\-time detection of small UAVs.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2512.18046v1)

---


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

