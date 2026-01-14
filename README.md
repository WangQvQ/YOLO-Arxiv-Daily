# 每日从arXiv中获取最新YOLO相关论文


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


## Billboard in Focus: Estimating Driver Gaze Duration from a Single Image / 

发布日期：2026-01-11

作者：Carlos Pizarroso

摘要：Roadside billboards represent a central element of outdoor advertising, yet their presence may contribute to driver distraction and accident risk. This study introduces a fully automated pipeline for billboard detection and driver gaze duration estimation, aiming to evaluate billboard relevance without reliance on manual annotations or eye\-tracking devices. Our pipeline operates in two stages: \(1\) a YOLO\-based object detection model trained on Mapillary Vistas and fine\-tuned on BillboardLamac images achieved 94% mAP@50 in the billboard detection task \(2\) a classifier based on the detected bounding box positions and DINOv2 features. The proposed pipeline enables estimation of billboard driver gaze duration from individual frames. We show that our method is able to achieve 68.1% accuracy on BillboardLamac when considering individual frames. These results are further validated using images collected from Google Street View.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.07073v1)

---


## Character Detection using YOLO for Writer Identification in multiple Medieval books / 

发布日期：2026-01-08

作者：Alessandra Scotto di Freca

摘要：Paleography is the study of ancient and historical handwriting, its key objectives include the dating of manuscripts and understanding the evolution of writing. Estimating when a document was written and tracing the development of scripts and writing styles can be aided by identifying the individual scribes who contributed to a medieval manuscript. Although digital technologies have made significant progress in this field, the general problem remains unsolved and continues to pose open challenges. ... We previously proposed an approach focused on identifying specific letters or abbreviations that characterize each writer. In that study, we considered the letter "a", as it was widely present on all pages of text and highly distinctive, according to the suggestions of expert paleographers. We used template matching techniques to detect the occurrences of the character "a" on each page and the convolutional neural network \(CNN\) to attribute each instance to the correct scribe. Moving from the interesting results achieved from this previous system and being aware of the limitations of the template matching technique, which requires an appropriate threshold to work, we decided to experiment in the same framework with the use of the YOLO object detection model to identify the scribe who contributed to the writing of different medieval books. We considered the fifth version of YOLO to implement the YOLO object detection model, which completely substituted the template matching and CNN used in the previous work. The experimental results demonstrate that YOLO effectively extracts a greater number of letters considered, leading to a more accurate second\-stage classification. Furthermore, the YOLO confidence score provides a foundation for developing a system that applies a rejection threshold, enabling reliable writer identification even in unseen manuscripts.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2601.04834v1)

---

