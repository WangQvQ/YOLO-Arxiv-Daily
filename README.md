# 每日从arXiv中获取最新YOLO相关论文


## YOLOv8 to YOLO11: A Comprehensive Architecture In\-depth Comparative Review / 

发布日期：2025-01-23

作者：Priyanto Hidayatullah

摘要：In the field of deep learning\-based computer vision, YOLO is revolutionary. With respect to deep learning models, YOLO is also the one that is evolving the most rapidly. Unfortunately, not every YOLO model possesses scholarly publications. Moreover, there exists a YOLO model that lacks a publicly accessible official architectural diagram. Naturally, this engenders challenges, such as complicating the understanding of how the model operates in practice. Furthermore, the review articles that are presently available do not delve into the specifics of each model. The objective of this study is to present a comprehensive and in\-depth architecture comparison of the four most recent YOLO models, specifically YOLOv8 through YOLO11, thereby enabling readers to quickly grasp not only how each model functions, but also the distinctions between them. To analyze each YOLO version's architecture, we meticulously examined the relevant academic papers, documentation, and scrutinized the source code. The analysis reveals that while each version of YOLO has improvements in architecture and feature extraction, certain blocks remain unchanged. The lack of scholarly publications and official diagrams presents challenges for understanding the model's functionality and future enhancement. Future developers are encouraged to provide these resources.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.13400v1)

---


## YOLOSCM: An improved YOLO algorithm for cars detection / 

发布日期：2025-01-23

作者：Changhui Deng

摘要：Detecting objects in urban traffic images presents considerable difficulties because of the following reasons: 1\) These images are typically immense in size, encompassing millions or even hundreds of millions of pixels, yet computational resources are constrained. 2\) The small size of vehicles in certain scenarios leads to insufficient information for accurate detection. 3\) The uneven distribution of vehicles causes inefficient use of computational resources. To address these issues, we propose YOLOSCM \(You Only Look Once with Segmentation Clustering Module\), an efficient and effective framework. To address the challenges of large\-scale images and the non\-uniform distribution of vehicles, we propose a Segmentation Clustering Module \(SCM\). This module adaptively identifies clustered regions, enabling the model to focus on these areas for more precise detection. Additionally, we propose a new training strategy to optimize the detection of small vehicles and densely packed targets in complex urban traffic scenes. We perform extensive experiments on urban traffic datasets to demonstrate the effectiveness and superiority of our proposed approach.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.13343v1)

---


## Data\-driven Detection and Evaluation of Damages in Concrete Structures: Using Deep Learning and Computer Vision / 

发布日期：2025-01-21

作者：Saeid Ataei

摘要：Structural integrity is vital for maintaining the safety and longevity of concrete infrastructures such as bridges, tunnels, and walls. Traditional methods for detecting damages like cracks and spalls are labor\-intensive, time\-consuming, and prone to human error. To address these challenges, this study explores advanced data\-driven techniques using deep learning for automated damage detection and analysis. Two state\-of\-the\-art instance segmentation models, YOLO\-v7 instance segmentation and Mask R\-CNN, were evaluated using a dataset comprising 400 images, augmented to 10,995 images through geometric and color\-based transformations to enhance robustness. The models were trained and validated using a dataset split into 90% training set, validation and test set 10%. Performance metrics such as precision, recall, mean average precision \(mAP@0.5\), and frames per second \(FPS\) were used for evaluation. YOLO\-v7 achieved a superior mAP@0.5 of 96.1% and processed 40 FPS, outperforming Mask R\-CNN, which achieved a mAP@0.5 of 92.1% with a slower processing speed of 18 FPS. The findings recommend YOLO\-v7 instance segmentation model for real\-time, high\-speed structural health monitoring, while Mask R\-CNN is better suited for detailed offline assessments. This study demonstrates the potential of deep learning to revolutionize infrastructure maintenance, offering a scalable and efficient solution for automated damage detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.11836v1)

---


## Efficient Auto\-Labeling of Large\-Scale Poultry Datasets \(ALPD\) Using Semi\-Supervised Models, Active Learning, and Prompt\-then\-Detect Approach / 

发布日期：2025-01-18

作者：Ramesh Bahadur Bist

摘要：The rapid growth of AI in poultry farming has highlighted the challenge of efficiently labeling large, diverse datasets. Manual annotation is time\-consuming, making it impractical for modern systems that continuously generate data. This study explores semi\-supervised auto\-labeling methods, integrating active learning, and prompt\-then\-detect paradigm to develop an efficient framework for auto\-labeling of large poultry datasets aimed at advancing AI\-driven behavior and health monitoring. Viideo data were collected from broilers and laying hens housed at the University of Arkansas and the University of Georgia. The collected videos were converted into images, pre\-processed, augmented, and labeled. Various machine learning models, including zero\-shot models like Grounding DINO, YOLO\-World, and CLIP, and supervised models like YOLO and Faster\-RCNN, were utilized for broilers, hens, and behavior detection. The results showed that YOLOv8s\-World and YOLOv9s performed better when compared performance metrics for broiler and hen detection under supervised learning, while among the semi\-supervised model, YOLOv8s\-ALPD achieved the highest precision \(96.1%\) and recall \(99.0%\) with an RMSE of 1.9. The hybrid YOLO\-World model, incorporating the optimal YOLOv8s backbone, demonstrated the highest overall performance. It achieved a precision of 99.2%, recall of 99.4%, and an F1 score of 98.7% for breed detection, alongside a precision of 88.4%, recall of 83.1%, and an F1 score of 84.5% for individual behavior detection. Additionally, semi\-supervised models showed significant improvements in behavior detection, achieving up to 31% improvement in precision and 16% in F1\-score. The semi\-supervised models with minimal active learning reduced annotation time by over 80% compared to full manual labeling. Moreover, integrating zero\-shot models enhanced detection and behavior identification.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.10809v1)

---


## Polyp detection in colonoscopy images using YOLOv11 / 

发布日期：2025-01-15

作者：Alok Ranjan Sahoo

摘要：Colorectal cancer \(CRC\) is one of the most commonly diagnosed cancers all over the world. It starts as a polyp in the inner lining of the colon. To prevent CRC, early polyp detection is required. Colonosopy is used for the inspection of the colon. Generally, the images taken by the camera placed at the tip of the endoscope are analyzed by the experts manually. Various traditional machine learning models have been used with the rise of machine learning. Recently, deep learning models have shown more effectiveness in polyp detection due to their superiority in generalizing and learning small features. These deep learning models for object detection can be segregated into two different types: single\-stage and two\-stage. Generally, two stage models have higher accuracy than single stage ones but the single stage models have low inference time. Hence, single stage models are easy to use for quick object detection. YOLO is one of the singlestage models used successfully for polyp detection. It has drawn the attention of researchers because of its lower inference time. The researchers have used Different versions of YOLO so far, and with each newer version, the accuracy of the model is increasing. This paper aims to see the effectiveness of the recently released YOLOv11 to detect polyp. We analyzed the performance for all five models of YOLOv11 \(YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x\) with Kvasir dataset for the training and testing. Two different versions of the dataset were used. The first consisted of the original dataset, and the other was created using augmentation techniques. The performance of all the models with these two versions of the dataset have been analysed.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.09051v1)

---

