# 每日从arXiv中获取最新YOLO相关论文


## A Novel Public Dataset for Strawberry \(Fragaria x ananassa\) Ripeness Detection and Comparative Evaluation of YOLO\-Based Models / 

发布日期：2026-02-17

作者：Mustafa Yurdakul

摘要：The strawberry \(Fragaria x ananassa\), known worldwide for its economic value and nutritional richness, is a widely cultivated fruit. Determining the correct ripeness level during the harvest period is crucial for both preventing losses for producers and ensuring consumers receive a quality product. However, traditional methods, i.e., visual assessments alone, can be subjective and have a high margin of error. Therefore, computer\-assisted systems are needed. However, the scarcity of comprehensive datasets accessible to everyone in the literature makes it difficult to compare studies in this field. In this study, a new and publicly available strawberry ripeness dataset, consisting of 566 images and 1,201 labeled objects, prepared under variable light and environmental conditions in two different greenhouses in Turkey, is presented to the literature. Comparative tests conducted on the data set using YOLOv8, YOLOv9, and YOLO11\-based models showed that the highest precision value was 90.94% in the YOLOv9c model, while the highest recall value was 83.74% in the YOLO11s model. In terms of the general performance criterion mAP@50, YOLOv8s was the best performing model with a success rate of 86.09%. The results show that small and medium\-sized models work more balanced and efficiently on this type of dataset, while also establishing a fundamental reference point for smart agriculture applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.15656v1)

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


## Detection of On\-Ground Chestnuts Using Artificial Intelligence Toward Automated Picking / 

发布日期：2026-02-15

作者：Kaixuan Fang

摘要：Traditional mechanized chestnut harvesting is too costly for small producers, non\-selective, and prone to damaging nuts. Accurate, reliable detection of chestnuts on the orchard floor is crucial for developing low\-cost, vision\-guided automated harvesting technology. However, developing a reliable chestnut detection system faces challenges in complex environments with shading, varying natural light conditions, and interference from weeds, fallen leaves, stones, and other foreign on\-ground objects, which have remained unaddressed. This study collected 319 images of chestnuts on the orchard floor, containing 6524 annotated chestnuts. A comprehensive set of 29 state\-of\-the\-art real\-time object detectors, including 14 in the YOLO \(v11\-13\) and 15 in the RT\-DETR \(v1\-v4\) families at varied model scales, was systematically evaluated through replicated modeling experiments for chestnut detection. Experimental results show that the YOLOv12m model achieves the best mAP@0.5 of 95.1% among all the evaluated models, while the RT\-DETRv2\-R101 was the most accurate variant among RT\-DETR models, with mAP@0.5 of 91.1%. In terms of mAP@\[0.5:0.95\], the YOLOv11x model achieved the best accuracy of 80.1%. All models demonstrate significant potential for real\-time chestnut detection, and YOLO models outperformed RT\-DETR models in terms of both detection accuracy and inference, making them better suited for on\-board deployment. Both the dataset and software programs in this study have been made publicly available at https://github.com/AgFood\-Sensing\-and\-Intelligence\-Lab/ChestnutDetection.

中文摘要：


代码链接：https://github.com/AgFood-Sensing-and-Intelligence-Lab/ChestnutDetection.

论文链接：[阅读更多](http://arxiv.org/abs/2602.14140v1)

---


## Robustness of Object Detection of Autonomous Vehicles in Adverse Weather Conditions / 

发布日期：2026-02-13

作者：Fox Pettersen

摘要：As self\-driving technology advances toward widespread adoption, determining safe operational thresholds across varying environmental conditions becomes critical for public safety. This paper proposes a method for evaluating the robustness of object detection ML models in autonomous vehicles under adverse weather conditions. It employs data augmentation operators to generate synthetic data that simulates different severance degrees of the adverse operation conditions at progressive intensity levels to find the lowest intensity of the adverse conditions at which the object detection model fails. The robustness of the object detection model is measured by the average first failure coefficients \(AFFC\) over the input images in the benchmark. The paper reports an experiment with four object detection models: YOLOv5s, YOLOv11s, Faster R\-CNN, and Detectron2, utilising seven data augmentation operators that simulate weather conditions fog, rain, and snow, and lighting conditions of dark, bright, flaring, and shadow. The experiment data show that the method is feasible, effective, and efficient to evaluate and compare the robustness of object detection models in various adverse operation conditions. In particular, the Faster R\-CNN model achieved the highest robustness with an overall average AFFC of 71.9% over all seven adverse conditions, while YOLO variants showed the AFFC values of 43%. The method is also applied to assess the impact of model training that targets adverse operation conditions using synthetic data on model robustness. It is observed that such training can improve robustness in adverse conditions but may suffer from diminishing returns and forgetting phenomena \(i.e., decline in robustness\) if overtrained.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.12902v1)

---

