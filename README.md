# 每日从arXiv中获取最新YOLO相关论文


## Automated Re\-Identification of Holstein\-Friesian Cattle in Dense Crowds / 

发布日期：2026-02-17

作者：Phoenix Yu

摘要：Holstein\-Friesian detection and re\-identification \(Re\-ID\) methods capture individuals well when targets are spatially separate. However, existing approaches, including YOLO\-based species detection, break down when cows group closely together. This is particularly prevalent for species which have outline\-breaking coat patterns. To boost both effectiveness and transferability in this setting, we propose a new detect\-segment\-identify pipeline that leverages the Open\-Vocabulary Weight\-free Localisation and the Segment Anything models as pre\-processing stages alongside Re\-ID networks. To evaluate our approach, we publish a collection of nine days CCTV data filmed on a working dairy farm. Our methodology overcomes detection breakdown in dense animal groupings, resulting in a 98.93% accuracy. This significantly outperforms current oriented bounding box\-driven, as well as SAM species detection baselines with accuracy improvements of 47.52% and 27.13%, respectively. We show that unsupervised contrastive learning can build on this to yield 94.82% Re\-ID accuracy on our test data. Our work demonstrates that Re\-ID in crowded scenarios is both practical as well as reliable in working farm settings with no manual intervention. Code and dataset are provided for reproducibility.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.15962v1)

---


## A Study on Real\-time Object Detection using Deep Learning / 

发布日期：2026-02-17

作者：Ankita Bose

摘要：Object detection has compelling applications over a range of domains, including human\-computer interfaces, security and video surveillance, navigation and road traffic monitoring, transportation systems, industrial automation healthcare, the world of Augmented Reality \(AR\) and Virtual Reality \(VR\), environment monitoring and activity identification. Applications of real time object detection in all these areas provide dynamic analysis of the visual information that helps in immediate decision making. Furthermore, advanced deep learning algorithms leverage the progress in the field of object detection providing more accurate and efficient solutions. There are some outstanding deep learning algorithms for object detection which includes, Faster R CNN\(Region\-based Convolutional Neural Network\),Mask R\-CNN, Cascade R\-CNN, YOLO \(You Only Look Once\), SSD \(Single Shot Multibox Detector\), RetinaNet etc. This article goes into great detail on how deep learning algorithms are used to enhance real time object recognition. It provides information on the different object detection models available, open benchmark datasets, and studies on the use of object detection models in a range of applications. Additionally, controlled studies are provided to compare various strategies and produce some illuminating findings. Last but not least, a number of encouraging challenges and approaches are offered as suggestions for further investigation in both relevant deep learning approaches and object recognition.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.15926v1)

---


## A Novel Public Dataset for Strawberry \(Fragaria x ananassa\) Ripeness Detection and Comparative Evaluation of YOLO\-Based Models / 

发布日期：2026-02-17

作者：Mustafa Yurdakul

摘要：The strawberry \(Fragaria x ananassa\), known worldwide for its economic value and nutritional richness, is a widely cultivated fruit. Determining the correct ripeness level during the harvest period is crucial for both preventing losses for producers and ensuring consumers receive a quality product. However, traditional methods, i.e., visual assessments alone, can be subjective and have a high margin of error. Therefore, computer\-assisted systems are needed. However, the scarcity of comprehensive datasets accessible to everyone in the literature makes it difficult to compare studies in this field. In this study, a new and publicly available strawberry ripeness dataset, consisting of 566 images and 1,201 labeled objects, prepared under variable light and environmental conditions in two different greenhouses in Turkey, is presented to the literature. Comparative tests conducted on the data set using YOLOv8, YOLOv9, and YOLO11\-based models showed that the highest precision value was 90.94% in the YOLOv9c model, while the highest recall value was 83.74% in the YOLO11s model. In terms of the general performance criterion mAP@50, YOLOv8s was the best performing model with a success rate of 86.09%. The results show that small and medium\-sized models work more balanced and efficiently on this type of dataset, while also establishing a fundamental reference point for smart agriculture applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.15656v2)

---


## Ground\-Truth Depth in Vision Language Models: Spatial Context Understanding in Conversational AI for XR\-Robotic Support in Emergency First Response / 

发布日期：2026-02-16

作者：Rodrigo Gutierrez Maquilon

摘要：Large language models \(LLMs\) are increasingly used in emergency first response \(EFR\) applications to support situational awareness \(SA\) and decision\-making, yet most operate on text or 2D imagery and offer little support for core EFR SA competencies like spatial reasoning. We address this gap by evaluating a prototype that fuses robot\-mounted depth sensing and YOLO detection with a vision language model \(VLM\) capable of verbalizing metrically\-grounded distances of detected objects \(e.g., the chair is 3.02 meters away\). In a mixed\-reality toxic\-smoke scenario, participants estimated distances to a victim and an exit window under three conditions: video\-only, depth\-agnostic VLM, and depth\-augmented VLM. Depth\-augmentation improved objective accuracy and stability, e.g., the victim and window distance estimation error dropped, while raising situational awareness without increasing workload. Conversely, depth\- agnostic assistance increased workload and slightly worsened accuracy. We contribute to human SA augmentation by demonstrating that metrically grounded, object\-centric verbal information supports spatial reasoning in EFR and improves decision\-relevant judgments under time pressure.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.15237v1)

---


## YOLO26: A Comprehensive Architecture Overview and Key Improvements / 

发布日期：2026-02-16

作者：Priyanto Hidayatullah

摘要：You Only Look Once \(YOLO\) has been the prominent model for computer vision in deep learning for a decade. This study explores the novel aspects of YOLO26, the most recent version in the YOLO series. The elimination of Distribution Focal Loss \(DFL\), implementation of End\-to\-End NMS\-Free Inference, introduction of ProgLoss \+ Small\-Target\-Aware Label Assignment \(STAL\), and use of the MuSGD optimizer are the primary enhancements designed to improve inference speed, which is claimed to achieve a 43% boost in CPU mode. This is designed to allow YOLO26 to attain real\-time performance on edge devices or those without GPUs. Additionally, YOLO26 offers improvements in many computer vision tasks, including instance segmentation, pose estimation, and oriented bounding box \(OBB\) decoding. We aim for this effort to provide more value than just consolidating information already included in the existing technical documentation. Therefore, we performed a rigorous architectural investigation into YOLO26, mostly using the source code available in its GitHub repository and its official documentation. The authentic and detailed operational mechanisms of YOLO26 are inside the source code, which is seldom extracted by others. The YOLO26 architectural diagram is shown as the outcome of the investigation. This study is, to our knowledge, the first one presenting the CNN\-based YOLO26 architecture, which is the core of YOLO26. Our objective is to provide a precise architectural comprehension of YOLO26 for researchers and developers aspiring to enhance the YOLO model, ensuring it remains the leading deep learning model in computer vision.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.14582v1)

---

