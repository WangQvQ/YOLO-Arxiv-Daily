# 每日从arXiv中获取最新YOLO相关论文


## EA: An Event Autoencoder for High\-Speed Vision Sensing / 

发布日期：2025-07-09

作者：Riadul Islam

摘要：High\-speed vision sensing is essential for real\-time perception in applications such as robotics, autonomous vehicles, and industrial automation. Traditional frame\-based vision systems suffer from motion blur, high latency, and redundant data processing, limiting their performance in dynamic environments. Event cameras, which capture asynchronous brightness changes at the pixel level, offer a promising alternative but pose challenges in object detection due to sparse and noisy event streams. To address this, we propose an event autoencoder architecture that efficiently compresses and reconstructs event data while preserving critical spatial and temporal features. The proposed model employs convolutional encoding and incorporates adaptive threshold selection and a lightweight classifier to enhance recognition accuracy while reducing computational complexity. Experimental results on the existing Smart Event Face Dataset \(SEFD\) demonstrate that our approach achieves comparable accuracy to the YOLO\-v4 model while utilizing up to $35.5times$ fewer parameters. Implementations on embedded platforms, including Raspberry Pi 4B and NVIDIA Jetson Nano, show high frame rates ranging from 8 FPS up to 44.8 FPS. The proposed classifier exhibits up to 87.84x better FPS than the state\-of\-the\-art and significantly improves event\-based vision performance, making it ideal for low\-power, high\-speed applications in real\-time edge computing.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.06459v1)

---


## ECORE: Energy\-Conscious Optimized Routing for Deep Learning Models at the Edge / 

发布日期：2025-07-08

作者：Daghash K. Alqahtani

摘要：Edge computing enables data processing closer to the source, significantly reducing latency an essential requirement for real\-time vision\-based analytics such as object detection in surveillance and smart city environments. However, these tasks place substantial demands on resource constrained edge devices, making the joint optimization of energy consumption and detection accuracy critical. To address this challenge, we propose ECORE, a framework that integrates multiple dynamic routing strategies including estimation based techniques and a greedy selection algorithm to direct image processing requests to the most suitable edge device\-model pair. ECORE dynamically balances energy efficiency and detection performance based on object characteristics. We evaluate our approach through extensive experiments on real\-world datasets, comparing the proposed routers against widely used baseline techniques. The evaluation leverages established object detection models \(YOLO, SSD, EfficientDet\) and diverse edge platforms, including Jetson Orin Nano, Raspberry Pi 4 and 5, and TPU accelerators. Results demonstrate that our proposed context\-aware routing strategies can reduce energy consumption and latency by 45% and 49%, respectively, while incurring only a 2% loss in detection accuracy compared to accuracy\-centric methods.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.06011v1)

---


## YOLO\-APD: Enhancing YOLOv8 for Robust Pedestrian Detection on Complex Road Geometries / 

发布日期：2025-07-07

作者：Aquino Joctum

摘要：Autonomous vehicle perception systems require robust pedestrian detection, particularly on geometrically complex roadways like Type\-S curved surfaces, where standard RGB camera\-based methods face limitations. This paper introduces YOLO\-APD, a novel deep learning architecture enhancing the YOLOv8 framework specifically for this challenge. YOLO\-APD integrates several key architectural modifications: a parameter\-free SimAM attention mechanism, computationally efficient C3Ghost modules, a novel SimSPPF module for enhanced multi\-scale feature pooling, the Mish activation function for improved optimization, and an Intelligent Gather & Distribute \(IGD\) module for superior feature fusion in the network's neck. The concept of leveraging vehicle steering dynamics for adaptive region\-of\-interest processing is also presented. Comprehensive evaluations on a custom CARLA dataset simulating complex scenarios demonstrate that YOLO\-APD achieves state\-of\-the\-art detection accuracy, reaching 77.7% mAP@0.5:0.95 and exceptional pedestrian recall exceeding 96%, significantly outperforming baseline models, including YOLOv8. Furthermore, it maintains real\-time processing capabilities at 100 FPS, showcasing a superior balance between accuracy and efficiency. Ablation studies validate the synergistic contribution of each integrated component. Evaluation on the KITTI dataset confirms the architecture's potential while highlighting the need for domain adaptation. This research advances the development of highly accurate, efficient, and adaptable perception systems based on cost\-effective sensors, contributing to enhanced safety and reliability for autonomous navigation in challenging, less\-structured driving environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.05376v1)

---


## Development of an Improved Capsule\-Yolo Network for Automatic Tomato Plant Disease Early Detection and Diagnosis / 

发布日期：2025-07-03

作者：Idris Ochijenu

摘要：Like many countries, Nigeria is naturally endowed with fertile agricultural soil that supports large\-scale tomato production. However, the prevalence of disease causing pathogens poses a significant threat to tomato health, often leading to reduced yields and, in severe cases, the extinction of certain species. These diseases jeopardise both the quality and quantity of tomato harvests, contributing to food insecurity. Fortunately, tomato diseases can often be visually identified through distinct forms, appearances, or textures, typically first visible on leaves and fruits. This study presents an enhanced Capsule\-YOLO network architecture designed to automatically segment overlapping and occluded tomato leaf images from complex backgrounds using the YOLO framework. It identifies disease symptoms with impressive performance metrics: 99.31% accuracy, 98.78% recall, and 99.09% precision, and a 98.93% F1\-score representing improvements of 2.91%, 1.84%, 5.64%, and 4.12% over existing state\-of\-the\-art methods. Additionally, a user\-friendly interface was developed to allow farmers and users to upload images of affected tomato plants and detect early disease symptoms. The system also provides recommendations for appropriate diagnosis and treatment. The effectiveness of this approach promises significant benefits for the agricultural sector by enhancing crop yields and strengthening food security.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.03219v1)

---


## Lightweight Shrimp Disease Detection Research Based on YOLOv8n / 

发布日期：2025-07-03

作者：Fei Yuhuan

摘要：Shrimp diseases are one of the primary causes of economic losses in shrimp aquaculture. To prevent disease transmission and enhance intelligent detection efficiency in shrimp farming, this paper proposes a lightweight network architecture based on YOLOv8n. First, by designing the RLDD detection head and C2f\-EMCM module, the model reduces computational complexity while maintaining detection accuracy, improving computational efficiency. Subsequently, an improved SegNext\_Attention self\-attention mechanism is introduced to further enhance the model's feature extraction capability, enabling more precise identification of disease characteristics. Extensive experiments, including ablation studies and comparative evaluations, are conducted on a self\-constructed shrimp disease dataset, with generalization tests extended to the URPC2020 dataset. Results demonstrate that the proposed model achieves a 32.3% reduction in parameters compared to the original YOLOv8n, with a mAP@0.5 of 92.7% \(3% improvement over YOLOv8n\). Additionally, the model outperforms other lightweight YOLO\-series models in mAP@0.5, parameter count, and model size. Generalization experiments on the URPC2020 dataset further validate the model's robustness, showing a 4.1% increase in mAP@0.5 compared to YOLOv8n. The proposed method achieves an optimal balance between accuracy and efficiency, providing reliable technical support for intelligent disease detection in shrimp aquaculture.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.02354v1)

---

