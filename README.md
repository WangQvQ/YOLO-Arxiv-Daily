# 每日从arXiv中获取最新YOLO相关论文


## Cranio\-ID: Graph\-Based Craniofacial Identification via Automatic Landmark Annotation in 2D Multi\-View X\-rays / 

发布日期：2025-11-18

作者：Ravi Shankar Prasad

摘要：In forensic craniofacial identification and in many biomedical applications, craniometric landmarks are important. Traditional methods for locating landmarks are time\-consuming and require specialized knowledge and expertise. Current methods utilize superimposition and deep learning\-based methods that employ automatic annotation of landmarks. However, these methods are not reliable due to insufficient large\-scale validation studies. In this paper, we proposed a novel framework Cranio\-ID: First, an automatic annotation of landmarks on 2D skulls \(which are X\-ray scans of faces\) with their respective optical images using our trained YOLO\-pose models. Second, cross\-modal matching by formulating these landmarks into graph representations and then finding semantic correspondence between graphs of these two modalities using cross\-attention and optimal transport framework. Our proposed framework is validated on the S2F and CUHK datasets \(CUHK dataset resembles with S2F dataset\). Extensive experiments have been conducted to evaluate the performance of our proposed framework, which demonstrates significant improvements in both reliability and accuracy, as well as its effectiveness in cross\-domain skull\-to\-face and sketch\-to\-face matching in forensic science.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.14411v1)

---


## LSP\-YOLO: A Lightweight Single\-Stage Network for Sitting Posture Recognition on Embedded Devices / 

发布日期：2025-11-18

作者：Nanjun Li

摘要：With the rise in sedentary behavior, health problems caused by poor sitting posture have drawn increasing attention. Most existing methods, whether using invasive sensors or computer vision, rely on two\-stage pipelines, which result in high intrusiveness, intensive computation, and poor real\-time performance on embedded edge devices. Inspired by YOLOv11\-Pose, a lightweight single\-stage network for sitting posture recognition on embedded edge devices termed LSP\-YOLO was proposed. By integrating partial convolution\(PConv\) and Similarity\-Aware Activation Module\(SimAM\), a lightweight module, Light\-C3k2, was designed to reduce computational cost while maintaining feature extraction capability. In the recognition head, keypoints were directly mapped to posture classes through pointwise convolution, and intermediate supervision was employed to enable efficient fusion of pose estimation and classification. Furthermore, a dataset containing 5,000 images across six posture categories was constructed for model training and testing. The smallest trained model, LSP\-YOLO\-n, achieved 94.2% accuracy and 251 Fps on personal computer\(PC\) with a model size of only 1.9 MB. Meanwhile, real\-time and high\-accuracy inference under constrained computational resources was demonstrated on the SV830C \+ GC030A platform. The proposed approach is characterized by high efficiency, lightweight design and deployability, making it suitable for smart classrooms, rehabilitation, and human\-computer interaction applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.14322v1)

---


## Hardware optimization on Android for inference of AI models / 

发布日期：2025-11-17

作者：Iulius Gherasim

摘要：The pervasive integration of Artificial Intelligence models into contemporary mobile computing is notable across numerous use cases, from virtual assistants to advanced image processing. Optimizing the mobile user experience involves minimal latency and high responsiveness from deployed AI models with challenges from execution strategies that fully leverage real time constraints to the exploitation of heterogeneous hardware architecture. In this paper, we research and propose the optimal execution configurations for AI models on an Android system, focusing on two critical tasks: object detection \(YOLO family\) and image classification \(ResNet\). These configurations evaluate various model quantization schemes and the utilization of on device accelerators, specifically the GPU and NPU. Our core objective is to empirically determine the combination that achieves the best trade\-off between minimal accuracy degradation and maximal inference speed\-up.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.13453v1)

---


## YOLO Meets Mixture\-of\-Experts: Adaptive Expert Routing for Robust Object Detection / 

发布日期：2025-11-17

作者：Ori Meiraz

摘要：This paper presents a novel Mixture\-of\-Experts framework for object detection, incorporating adaptive routing among multiple YOLOv9\-T experts to enable dynamic feature specialization and achieve higher mean Average Precision \(mAP\) and Average Recall \(AR\) compared to a single YOLOv9\-T model.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.13344v2)

---


## MCAQ\-YOLO: Morphological Complexity\-Aware Quantization for Efficient Object Detection with Curriculum Learning / 

发布日期：2025-11-17

作者：Yoonjae Seo

摘要：Most neural network quantization methods apply uniform bit precision across spatial regions, ignoring the heterogeneous structural and textural complexity of visual data. This paper introduces MCAQ\-YOLO, a morphological complexity\-aware quantization framework for object detection. The framework employs five morphological metrics \- fractal dimension, texture entropy, gradient variance, edge density, and contour complexity \- to characterize local visual morphology and guide spatially adaptive bit allocation. By correlating these metrics with quantization sensitivity, MCAQ\-YOLO dynamically adjusts bit precision according to spatial complexity. In addition, a curriculum\-based quantization\-aware training scheme progressively increases quantization difficulty to stabilize optimization and accelerate convergence. Experimental results demonstrate a strong correlation between morphological complexity and quantization sensitivity and show that MCAQ\-YOLO achieves superior detection accuracy and convergence efficiency compared with uniform quantization. On a safety equipment dataset, MCAQ\-YOLO attains 85.6 percent mAP@0.5 with an average of 4.2 bits and a 7.6x compression ratio, yielding 3.5 percentage points higher mAP than uniform 4\-bit quantization while introducing only 1.8 ms of additional runtime overhead per image. Cross\-dataset validation on COCO and Pascal VOC further confirms consistent performance gains, indicating that morphology\-driven spatial quantization can enhance efficiency and robustness for computationally constrained, safety\-critical visual recognition tasks.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.12976v1)

---

