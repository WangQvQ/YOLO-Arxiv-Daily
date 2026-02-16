# 每日从arXiv中获取最新YOLO相关论文


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


## Weak to Strong: VLM\-Based Pseudo\-Labeling as a Weakly Supervised Training Strategy in Multimodal Video\-based Hidden Emotion Understanding Tasks / 

发布日期：2026-02-08

作者：Yufei Wang

摘要：To tackle the automatic recognition of "concealed emotions" in videos, this paper proposes a multimodal weak\-supervision framework and achieves state\-of\-the\-art results on the iMiGUE tennis\-interview dataset. First, YOLO 11x detects and crops human portraits frame\-by\-frame, and DINOv2\-Base extracts visual features from the cropped regions. Next, by integrating Chain\-of\-Thought and Reflection prompting \(CoT \+ Reflection\), Gemini 2.5 Pro automatically generates pseudo\-labels and reasoning texts that serve as weak supervision for downstream models. Subsequently, OpenPose produces 137\-dimensional key\-point sequences, augmented with inter\-frame offset features; the usual graph neural network backbone is simplified to an MLP to efficiently model the spatiotemporal relationships of the three key\-point streams. An ultra\-long\-sequence Transformer independently encodes both the image and key\-point sequences, and their representations are concatenated with BERT\-encoded interview transcripts. Each modality is first pre\-trained in isolation, then fine\-tuned jointly, with pseudo\-labeled samples merged into the training set for further gains. Experiments demonstrate that, despite severe class imbalance, the proposed approach lifts accuracy from under 0.6 in prior work to over 0.69, establishing a new public benchmark. The study also validates that an "MLP\-ified" key\-point backbone can match \- or even surpass \- GCN\-based counterparts in this task.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.08057v1)

---


## CA\-YOLO: Cross Attention Empowered YOLO for Biomimetic Localization / 

发布日期：2026-02-07

作者：Zhen Zhang

摘要：In modern complex environments, achieving accurate and efficient target localization is essential in numerous fields. However, existing systems often face limitations in both accuracy and the ability to recognize small targets. In this study, we propose a bionic stabilized localization system based on CA\-YOLO, designed to enhance both target localization accuracy and small target recognition capabilities. Acting as the "brain" of the system, the target detection algorithm emulates the visual focusing mechanism of animals by integrating bionic modules into the YOLO backbone network. These modules include the introduction of a small target detection head and the development of a Characteristic Fusion Attention Mechanism \(CFAM\). Furthermore, drawing inspiration from the human Vestibulo\-Ocular Reflex \(VOR\), a bionic pan\-tilt tracking control strategy is developed, which incorporates central positioning, stability optimization, adaptive control coefficient adjustment, and an intelligent recapture function. The experimental results show that CA\-YOLO outperforms the original model on standard datasets \(COCO and VisDrone\), with average accuracy metrics improved by 3.94%and 4.90%, respectively.Further time\-sensitive target localization experiments validate the effectiveness and practicality of this bionic stabilized localization system.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.07523v1)

---

