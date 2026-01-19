# 每日从arXiv中获取最新YOLO相关论文


## SME\-YOLO: A Real\-Time Detector for Tiny Defect Detection on PCB Surfaces / 

发布日期：2026-01-16

作者：Meng Han

摘要：Surface defects on Printed Circuit Boards \(PCBs\) directly compromise product reliability and safety. However, achieving high\-precision detection is challenging because PCB defects are typically characterized by tiny sizes, high texture similarity, and uneven scale distributions. To address these challenges, this paper proposes a novel framework based on YOLOv11n, named SME\-YOLO \(Small\-target Multi\-scale Enhanced YOLO\). First, we employ the Normalized Wasserstein Distance Loss \(NWDLoss\). This metric effectively mitigates the sensitivity of Intersection over Union \(IoU\) to positional deviations in tiny objects. Second, the original upsampling module is replaced by the Efficient Upsampling Convolution Block \(EUCB\). By utilizing multi\-scale convolutions, the EUCB gradually recovers spatial resolution and enhances the preservation of edge and texture details for tiny defects. Finally, this paper proposes the Multi\-Scale Focused Attention \(MSFA\) module. Tailored to the specific spatial distribution of PCB defects, this module adaptively strengthens perception within key scale intervals, achieving efficient fusion of local fine\-grained features and global context information. Experimental results on the PKU\-PCB dataset demonstrate that SME\-YOLO achieves state\-of\-the\-art performance. Specifically, compared to the baseline YOLOv11n, SME\-YOLO improves mAP by 2.2% and Precision by 4%, validating the effectiveness of the proposed method.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.11402v1)

---


## SAMannot: A Memory\-Efficient, Local, Open\-source Framework for Interactive Video Instance Segmentation based on SAM2 / 

发布日期：2026-01-16

作者：Gergely Dinya

摘要：Current research workflows for precise video segmentation are often forced into a compromise between labor\-intensive manual curation, costly commercial platforms, and/or privacy\-compromising cloud\-based services. The demand for high\-fidelity video instance segmentation in research is often hindered by the bottleneck of manual annotation and the privacy concerns of cloud\-based tools. We present SAMannot, an open\-source, local framework that integrates the Segment Anything Model 2 \(SAM2\) into a human\-in\-the\-loop workflow. To address the high resource requirements of foundation models, we modified the SAM2 dependency and implemented a processing layer that minimizes computational overhead and maximizes throughput, ensuring a highly responsive user interface. Key features include persistent instance identity management, an automated \`\`lock\-and\-refine'' workflow with barrier frames, and a mask\-skeletonization\-based auto\-prompting mechanism. SAMannot facilitates the generation of research\-ready datasets in YOLO and PNG formats alongside structured interaction logs. Verified through animal behavior tracking use\-cases and subsets of the LVOS and DAVIS benchmark datasets, the tool provides a scalable, private, and cost\-effective alternative to commercial platforms for complex video annotation tasks.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.11301v1)

---


## Edge\-Optimized Multimodal Learning for UAV Video Understanding via BLIP\-2 / 

发布日期：2026-01-13

作者：Yizhan Feng

摘要：The demand for real\-time visual understanding and interaction in complex scenarios is increasingly critical for unmanned aerial vehicles. However, a significant challenge arises from the contradiction between the high computational cost of large Vision language models and the limited computing resources available on UAV edge devices. To address this challenge, this paper proposes a lightweight multimodal task platform based on BLIP\-2, integrated with YOLO\-World and YOLOv8\-Seg models. This integration extends the multi\-task capabilities of BLIP\-2 for UAV applications with minimal adaptation and without requiring task\-specific fine\-tuning on drone data. Firstly, the deep integration of BLIP\-2 with YOLO models enables it to leverage the precise perceptual results of YOLO for fundamental tasks like object detection and instance segmentation, thereby facilitating deeper visual\-attention understanding and reasoning. Secondly, a content\-aware key frame sampling mechanism based on K\-Means clustering is designed, which incorporates intelligent frame selection and temporal feature concatenation. This equips the lightweight BLIP\-2 architecture with the capability to handle video\-level interactive tasks effectively. Thirdly, a unified prompt optimization scheme for multi\-task adaptation is implemented. This scheme strategically injects structured event logs from the YOLO models as contextual information into BLIP\-2's input. Combined with output constraints designed to filter out technical details, this approach effectively guides the model to generate accurate and contextually relevant outputs for various tasks.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.08408v1)

---


## YOLOBirDrone: Dataset for Bird vs Drone Detection and Classification and a YOLO based enhanced learning architecture / 

发布日期：2026-01-13

作者：Dapinder Kaur

摘要：The use of aerial drones for commercial and defense applications has benefited in many ways and is therefore utilized in several different application domains. However, they are also increasingly used for targeted attacks, posing a significant safety challenge and necessitating the development of drone detection systems. Vision\-based drone detection systems currently have an accuracy limitation and struggle to distinguish between drones and birds, particularly when the birds are small in size. This research work proposes a novel YOLOBirDrone architecture that improves the detection and classification accuracy of birds and drones. YOLOBirDrone has different components, including an adaptive and extended layer aggregation \(AELAN\), a multi\-scale progressive dual attention module \(MPDA\), and a reverse MPDA \(RMPDA\) to preserve shape information and enrich features with local and global spatial and channel information. A large\-scale dataset, BirDrone, is also introduced in this article, which includes small and challenging objects for robust aerial object identification. Experimental results demonstrate an improvement in performance metrics through the proposed YOLOBirDrone architecture compared to other state\-of\-the\-art algorithms, with detection accuracy reaching approximately 85% across various scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.08319v1)

---


## Sesame Plant Segmentation Dataset: A YOLO Formatted Annotated Dataset / 

发布日期：2026-01-12

作者：Sunusi Ibrahim Muhammad

摘要：This paper presents the Sesame Plant Segmentation Dataset, an open source annotated image dataset designed to support the development of artificial intelligence models for agricultural applications, with a specific focus on sesame plants. The dataset comprises 206 training images, 43 validation images, and 43 test images in YOLO compatible segmentation format, capturing sesame plants at early growth stages under varying environmental conditions. Data were collected using a high resolution mobile camera from farms in Jirdede, Daura Local Government Area, Katsina State, Nigeria, and annotated using the Segment Anything Model version 2 with farmer supervision. Unlike conventional bounding box datasets, this dataset employs pixel level segmentation to enable more precise detection and analysis of sesame plants in real world farm settings. Model evaluation using the Ultralytics YOLOv8 framework demonstrated strong performance for both detection and segmentation tasks. For bounding box detection, the model achieved a recall of 79 percent, precision of 79 percent, mean average precision at IoU 0.50 of 84 percent, and mean average precision from 0.50 to 0.95 of 58 percent. For segmentation, it achieved a recall of 82 percent, precision of 77 percent, mean average precision at IoU 0.50 of 84 percent, and mean average precision from 0.50 to 0.95 of 52 percent. The dataset represents a novel contribution to sesame focused agricultural vision datasets in Nigeria and supports applications such as plant monitoring, yield estimation, and agricultural research.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.07970v1)

---

