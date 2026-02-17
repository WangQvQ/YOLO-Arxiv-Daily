# 每日从arXiv中获取最新YOLO相关论文


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


## A Lightweight and Explainable DenseNet\-121 Framework for Grape Leaf Disease Classification / 

发布日期：2026-02-12

作者：Md. Ehsanul Haque

摘要：Grapes are among the most economically and culturally significant fruits on a global scale, and table grapes and wine are produced in significant quantities in Europe and Asia. The production and quality of grapes are significantly impacted by grape diseases such as Bacterial Rot, Downy Mildew, and Powdery Mildew. Consequently, the sustainable management of a vineyard necessitates the early and precise identification of these diseases. Current automated methods, particularly those that are based on the YOLO framework, are often computationally costly and lack interpretability that makes them unsuitable for real\-world scenarios. This study proposes grape leaf disease classification using Optimized DenseNet 121. Domain\-specific preprocessing and extensive connectivity reveal disease\-relevant characteristics, including veins, edges, and lesions. An extensive comparison with baseline CNN models, including ResNet18, VGG16, AlexNet, and SqueezeNet, demonstrates that the proposed model exhibits superior performance. It achieves an accuracy of 99.27%, an F1 score of 99.28%, a specificity of 99.71%, and a Kappa of 98.86%, with an inference time of 9 seconds. The cross\-validation findings show a mean accuracy of 99.12%, indicating strength and generalizability across all classes. We also employ Grad\-CAM to highlight disease\-related regions to guarantee the model is highlighting physiologically relevant aspects and increase transparency and confidence. Model optimization reduces processing requirements for real\-time deployment, while transfer learning ensures consistency on smaller and unbalanced samples. An effective architecture, domain\-specific preprocessing, and interpretable outputs make the proposed framework scalable, precise, and computationally inexpensive for detecting grape leaf diseases.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.12484v1)

---


## A Scoping Review of Deep Learning for Urban Visual Pollution and Proposal of a Real\-Time Monitoring Framework with a Visual Pollution Index / 

发布日期：2026-02-10

作者：Mohammad Masudur Rahman

摘要：Urban Visual Pollution \(UVP\) has emerged as a critical concern, yet research on automatic detection and application remains fragmented. This scoping review maps the existing deep learning\-based approaches for detecting, classifying, and designing a comprehensive application framework for visual pollution management. Following the PRISMA\-ScR guidelines, seven academic databases \(Scopus, Web of Science, IEEE Xplore, ACM DL, ScienceDirect, SpringerNatureLink, and Wiley\) were systematically searched and reviewed, and 26 articles were found. Most research focuses on specific pollutant categories and employs variations of YOLO, Faster R\-CNN, and EfficientDet architectures. Although several datasets exist, they are limited to specific areas and lack standardized taxonomies. Few studies integrate detection into real\-time application systems, yet they tend to be geographically skewed. We proposed a framework for monitoring visual pollution that integrates a visual pollution index to assess the severity of visual pollution for a certain area. This review highlights the need for a unified UVP management system that incorporates pollutant taxonomy, a cross\-city benchmark dataset, a generalized deep learning model, and an assessment index that supports sustainable urban aesthetics and enhances the well\-being of urban dwellers.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.09446v1)

---

