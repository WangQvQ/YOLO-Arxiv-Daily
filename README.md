# 每日从arXiv中获取最新YOLO相关论文


## An Enhanced YOLOv8 Model for Real\-Time and Accurate Pothole Detection and Measurement / 

发布日期：2025-05-07

作者：Mustafa Yurdakul

摘要：Potholes cause vehicle damage and traffic accidents, creating serious safety and economic problems. Therefore, early and accurate detection of potholes is crucial. Existing detection methods are usually only based on 2D RGB images and cannot accurately analyze the physical characteristics of potholes. In this paper, a publicly available dataset of RGB\-D images \(PothRGBD\) is created and an improved YOLOv8\-based model is proposed for both pothole detection and pothole physical features analysis. The Intel RealSense D415 depth camera was used to collect RGB and depth data from the road surfaces, resulting in a PothRGBD dataset of 1000 images. The data was labeled in YOLO format suitable for segmentation. A novel YOLO model is proposed based on the YOLOv8n\-seg architecture, which is structurally improved with Dynamic Snake Convolution \(DSConv\), Simple Attention Module \(SimAM\) and Gaussian Error Linear Unit \(GELU\). The proposed model segmented potholes with irregular edge structure more accurately, and performed perimeter and depth measurements on depth maps with high accuracy. The standard YOLOv8n\-seg model achieved 91.9% precision, 85.2% recall and 91.9% mAP@50. With the proposed model, the values increased to 93.7%, 90.4% and 93.8% respectively. Thus, an improvement of 1.96% in precision, 6.13% in recall and 2.07% in mAP was achieved. The proposed model performs pothole detection as well as perimeter and depth measurement with high accuracy and is suitable for real\-time applications due to its low model complexity. In this way, a lightweight and effective model that can be used in deep learning\-based intelligent transportation solutions has been acquired.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.04207v1)

---


## Sim2Real Transfer for Vision\-Based Grasp Verification / 

发布日期：2025-05-05

作者：Pau Amargant

摘要：The verification of successful grasps is a crucial aspect of robot manipulation, particularly when handling deformable objects. Traditional methods relying on force and tactile sensors often struggle with deformable and non\-rigid objects. In this work, we present a vision\-based approach for grasp verification to determine whether the robotic gripper has successfully grasped an object. Our method employs a two\-stage architecture; first YOLO\-based object detection model to detect and locate the robot's gripper and then a ResNet\-based classifier determines the presence of an object. To address the limitations of real\-world data capture, we introduce HSR\-GraspSynth, a synthetic dataset designed to simulate diverse grasping scenarios. Furthermore, we explore the use of Visual Question Answering capabilities as a zero\-shot baseline to which we compare our model. Experimental results demonstrate that our approach achieves high accuracy in real\-world environments, with potential for integration into grasping pipelines. Code and datasets are publicly available at https://github.com/pauamargant/HSR\-GraspSynth .

中文摘要：


代码链接：https://github.com/pauamargant/HSR-GraspSynth

论文链接：[阅读更多](http://arxiv.org/abs/2505.03046v1)

---


## Design description of Wisdom Computing Persperctive / 

发布日期：2025-05-02

作者：TianYi Yu

摘要：This course design aims to develop and research a handwriting matrix recognition and step\-by\-step visual calculation process display system, addressing the issue of abstract formulas and complex calculation steps that students find difficult to understand when learning mathematics. By integrating artificial intelligence with visualization animation technology, the system enhances precise recognition of handwritten matrix content through the introduction of Mamba backbone networks, completes digital extraction and matrix reconstruction using the YOLO model, and simultaneously combines CoordAttention coordinate attention mechanisms to improve the accurate grasp of character spatial positions. The calculation process is demonstrated frame by frame through the Manim animation engine, vividly showcasing each mathematical calculation step, helping students intuitively understand the intrinsic logic of mathematical operations. Through dynamically generating animation processes for different computational tasks, the system exhibits high modularity and flexibility, capable of generating various mathematical operation examples in real\-time according to student needs. By innovating human\-computer interaction methods, it brings mathematical calculation processes to life, helping students bridge the gap between knowledge and understanding on a deeper level, ultimately achieving a learning experience where "every step is understood." The system's scalability and interactivity make it an intuitive, user\-friendly, and efficient auxiliary tool in education.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.03800v1)

---


## FBRT\-YOLO: Faster and Better for Real\-Time Aerial Image Detection / 

发布日期：2025-04-29

作者：Yao Xiao

摘要：Embedded flight devices with visual capabilities have become essential for a wide range of applications. In aerial image detection, while many existing methods have partially addressed the issue of small target detection, challenges remain in optimizing small target detection and balancing detection accuracy with efficiency. These issues are key obstacles to the advancement of real\-time aerial image detection. In this paper, we propose a new family of real\-time detectors for aerial image detection, named FBRT\-YOLO, to address the imbalance between detection accuracy and efficiency. Our method comprises two lightweight modules: Feature Complementary Mapping Module \(FCM\) and Multi\-Kernel Perception Unit\(MKP\), designed to enhance object perception for small targets in aerial images. FCM focuses on alleviating the problem of information imbalance caused by the loss of small target information in deep networks. It aims to integrate spatial positional information of targets more deeply into the network,better aligning with semantic information in the deeper layers to improve the localization of small targets. We introduce MKP, which leverages convolutions with kernels of different sizes to enhance the relationships between targets of various scales and improve the perception of targets at different scales. Extensive experimental results on three major aerial image datasets, including Visdrone, UAVDT, and AI\-TOD,demonstrate that FBRT\-YOLO outperforms various real\-time detectors in terms of performance and speed.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.20670v1)

---


## MASF\-YOLO: An Improved YOLOv11 Network for Small Object Detection on Drone View / 

发布日期：2025-04-25

作者：Liugang Lu

摘要：With the rapid advancement of Unmanned Aerial Vehicle \(UAV\) and computer vision technologies, object detection from UAV perspectives has emerged as a prominent research area. However, challenges for detection brought by the extremely small proportion of target pixels, significant scale variations of objects, and complex background information in UAV images have greatly limited the practical applications of UAV. To address these challenges, we propose a novel object detection network Multi\-scale Context Aggregation and Scale\-adaptive Fusion YOLO \(MASF\-YOLO\), which is developed based on YOLOv11. Firstly, to tackle the difficulty of detecting small objects in UAV images, we design a Multi\-scale Feature Aggregation Module \(MFAM\), which significantly improves the detection accuracy of small objects through parallel multi\-scale convolutions and feature fusion. Secondly, to mitigate the interference of background noise, we propose an Improved Efficient Multi\-scale Attention Module \(IEMA\), which enhances the focus on target regions through feature grouping, parallel sub\-networks, and cross\-spatial learning. Thirdly, we introduce a Dimension\-Aware Selective Integration Module \(DASI\), which further enhances multi\-scale feature fusion capabilities by adaptively weighting and fusing low\-dimensional features and high\-dimensional features. Finally, we conducted extensive performance evaluations of our proposed method on the VisDrone2019 dataset. Compared to YOLOv11\-s, MASFYOLO\-s achieves improvements of 4.6% in mAP@0.5 and 3.5% in mAP@0.5:0.95 on the VisDrone2019 validation set. Remarkably, MASF\-YOLO\-s outperforms YOLOv11\-m while requiring only approximately 60% of its parameters and 65% of its computational cost. Furthermore, comparative experiments with state\-of\-the\-art detectors confirm that MASF\-YOLO\-s maintains a clear competitive advantage in both detection accuracy and model efficiency.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2504.18136v1)

---

