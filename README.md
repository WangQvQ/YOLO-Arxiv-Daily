# 每日从arXiv中获取最新YOLO相关论文


## PaniCar: Securing the Perception of Advanced Driving Assistance Systems Against Emergency Vehicle Lighting / 

发布日期：2025-05-08

作者：Elad Feldman

摘要：The safety of autonomous cars has come under scrutiny in recent years, especially after 16 documented incidents involving Teslas \(with autopilot engaged\) crashing into parked emergency vehicles \(police cars, ambulances, and firetrucks\). While previous studies have revealed that strong light sources often introduce flare artifacts in the captured image, which degrade the image quality, the impact of flare on object detection performance remains unclear. In this research, we unveil PaniCar, a digital phenomenon that causes an object detector's confidence score to fluctuate below detection thresholds when exposed to activated emergency vehicle lighting. This vulnerability poses a significant safety risk, and can cause autonomous vehicles to fail to detect objects near emergency vehicles. In addition, this vulnerability could be exploited by adversaries to compromise the security of advanced driving assistance systems \(ADASs\). We assess seven commercial ADASs \(Tesla Model 3, "manufacturer C", HP, Pelsee, AZDOME, Imagebon, Rexing\), four object detectors \(YOLO, SSD, RetinaNet, Faster R\-CNN\), and 14 patterns of emergency vehicle lighting to understand the influence of various technical and environmental factors. We also evaluate four SOTA flare removal methods and show that their performance and latency are insufficient for real\-time driving constraints. To mitigate this risk, we propose Caracetamol, a robust framework designed to enhance the resilience of object detectors against the effects of activated emergency vehicle lighting. Our evaluation shows that on YOLOv3 and Faster RCNN, Caracetamol improves the models' average confidence of car detection by 0.20, the lower confidence bound by 0.33, and reduces the fluctuation range by 0.33. In addition, Caracetamol is capable of processing frames at a rate of between 30\-50 FPS, enabling real\-time ADAS car detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.05183v1)

---


## Adaptive Contextual Embedding for Robust Far\-View Borehole Detection / 

发布日期：2025-05-08

作者：Xuesong Liu

摘要：In controlled blasting operations, accurately detecting densely distributed tiny boreholes from far\-view imagery is critical for operational safety and efficiency. However, existing detection methods often struggle due to small object scales, highly dense arrangements, and limited distinctive visual features of boreholes. To address these challenges, we propose an adaptive detection approach that builds upon existing architectures \(e.g., YOLO\) by explicitly leveraging consistent embedding representations derived through exponential moving average \(EMA\)\-based statistical updates.   Our method introduces three synergistic components: \(1\) adaptive augmentation utilizing dynamically updated image statistics to robustly handle illumination and texture variations; \(2\) embedding stabilization to ensure consistent and reliable feature extraction; and \(3\) contextual refinement leveraging spatial context for improved detection accuracy. The pervasive use of EMA in our method is particularly advantageous given the limited visual complexity and small scale of boreholes, allowing stable and robust representation learning even under challenging visual conditions. Experiments on a challenging proprietary quarry\-site dataset demonstrate substantial improvements over baseline YOLO\-based architectures, highlighting our method's effectiveness in realistic and complex industrial scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2505.05008v1)

---


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

