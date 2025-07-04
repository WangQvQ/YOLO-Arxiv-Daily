# 每日从arXiv中获取最新YOLO相关论文


## Lightweight Shrimp Disease Detection Research Based on YOLOv8n / 

发布日期：2025-07-03

作者：Fei Yuhuan

摘要：Shrimp diseases are one of the primary causes of economic losses in shrimp aquaculture. To prevent disease transmission and enhance intelligent detection efficiency in shrimp farming, this paper proposes a lightweight network architecture based on YOLOv8n. First, by designing the RLDD detection head and C2f\-EMCM module, the model reduces computational complexity while maintaining detection accuracy, improving computational efficiency. Subsequently, an improved SegNext\_Attention self\-attention mechanism is introduced to further enhance the model's feature extraction capability, enabling more precise identification of disease characteristics. Extensive experiments, including ablation studies and comparative evaluations, are conducted on a self\-constructed shrimp disease dataset, with generalization tests extended to the URPC2020 dataset. Results demonstrate that the proposed model achieves a 32.3% reduction in parameters compared to the original YOLOv8n, with a mAP@0.5 of 92.7% \(3% improvement over YOLOv8n\). Additionally, the model outperforms other lightweight YOLO\-series models in mAP@0.5, parameter count, and model size. Generalization experiments on the URPC2020 dataset further validate the model's robustness, showing a 4.1% increase in mAP@0.5 compared to YOLOv8n. The proposed method achieves an optimal balance between accuracy and efficiency, providing reliable technical support for intelligent disease detection in shrimp aquaculture.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.02354v1)

---


## Autonomous AI Surveillance: Multimodal Deep Learning for Cognitive and Behavioral Monitoring / 

发布日期：2025-07-02

作者：Ameer Hamza

摘要：This study presents a novel classroom surveillance system that integrates multiple modalities, including drowsiness, tracking of mobile phone usage, and face recognition,to assess student attentiveness with enhanced precision.The system leverages the YOLOv8 model to detect both mobile phone and sleep usage,\(Ghatge et al., 2024\) while facial recognition is achieved through LResNet Occ FC body tracking using YOLO and MTCNN.\(Durai et al., 2024\) These models work in synergy to provide comprehensive, real\-time monitoring, offering insights into student engagement and behavior.\(S et al., 2023\) The framework is trained on specialized datasets, such as the RMFD dataset for face recognition and a Roboflow dataset for mobile phone detection. The extensive evaluation of the system shows promising results. Sleep detection achieves 97. 42% mAP@50, face recognition achieves 86. 45% validation accuracy and mobile phone detection reach 85. 89% mAP@50. The system is implemented within a core PHP web application and utilizes ESP32\-CAM hardware for seamless data capture.\(Neto et al., 2024\) This integrated approach not only enhances classroom monitoring, but also ensures automatic attendance recording via face recognition as students remain seated in the classroom, offering scalability for diverse educational environments.\(Banada,2025\)

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.01590v1)

---


## UAVD\-Mamba: Deformable Token Fusion Vision Mamba for Multimodal UAV Detection / 

发布日期：2025-07-01

作者：Wei Li

摘要：Unmanned Aerial Vehicle \(UAV\) object detection has been widely used in traffic management, agriculture, emergency rescue, etc. However, it faces significant challenges, including occlusions, small object sizes, and irregular shapes. These challenges highlight the necessity for a robust and efficient multimodal UAV object detection method. Mamba has demonstrated considerable potential in multimodal image fusion. Leveraging this, we propose UAVD\-Mamba, a multimodal UAV object detection framework based on Mamba architectures. To improve geometric adaptability, we propose the Deformable Token Mamba Block \(DTMB\) to generate deformable tokens by incorporating adaptive patches from deformable convolutions alongside normal patches from normal convolutions, which serve as the inputs to the Mamba Block. To optimize the multimodal feature complementarity, we design two separate DTMBs for the RGB and infrared \(IR\) modalities, with the outputs from both DTMBs integrated into the Mamba Block for feature extraction and into the Fusion Mamba Block for feature fusion. Additionally, to improve multiscale object detection, especially for small objects, we stack four DTMBs at different scales to produce multiscale feature representations, which are then sent to the Detection Neck for Mamba \(DNM\). The DNM module, inspired by the YOLO series, includes modifications to the SPPF and C3K2 of YOLOv11 to better handle the multiscale features. In particular, we employ cross\-enhanced spatial attention before the DTMB and cross\-channel attention after the Fusion Mamba Block to extract more discriminative features. Experimental results on the DroneVehicle dataset show that our method outperforms the baseline OAFA method by 3.6% in the mAP metric. Codes will be released at https://github.com/GreatPlum\-hnu/UAVD\-Mamba.git.

中文摘要：


代码链接：https://github.com/GreatPlum-hnu/UAVD-Mamba.git.

论文链接：[阅读更多](http://arxiv.org/abs/2507.00849v1)

---


## Research on Improving the High Precision and Lightweight Diabetic Retinopathy Detection of YOLOv8n / 

发布日期：2025-07-01

作者：Fei Yuhuan

摘要：Early detection and diagnosis of diabetic retinopathy is one of the current research focuses in ophthalmology. However, due to the subtle features of micro\-lesions and their susceptibility to background interference, ex\-isting detection methods still face many challenges in terms of accuracy and robustness. To address these issues, a lightweight and high\-precision detection model based on the improved YOLOv8n, named YOLO\-KFG, is proposed. Firstly, a new dynamic convolution KWConv and C2f\-KW module are designed to improve the backbone network, enhancing the model's ability to perceive micro\-lesions. Secondly, a fea\-ture\-focused diffusion pyramid network FDPN is designed to fully integrate multi\-scale context information, further improving the model's ability to perceive micro\-lesions. Finally, a lightweight shared detection head GSDHead is designed to reduce the model's parameter count, making it more deployable on re\-source\-constrained devices. Experimental results show that compared with the base model YOLOv8n, the improved model reduces the parameter count by 20.7%, increases mAP@0.5 by 4.1%, and improves the recall rate by 7.9%. Compared with single\-stage mainstream algorithms such as YOLOv5n and YOLOv10n, YOLO\-KFG demonstrates significant advantages in both detection accuracy and efficiency.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2507.00780v1)

---


## Deep Learning\-Based Semantic Segmentation for Real\-Time Kidney Imaging and Measurements with Augmented Reality\-Assisted Ultrasound / 

发布日期：2025-06-30

作者：Gijs Luijten

摘要：Ultrasound \(US\) is widely accessible and radiation\-free but has a steep learning curve due to its dynamic nature and non\-standard imaging planes. Additionally, the constant need to shift focus between the US screen and the patient poses a challenge. To address these issues, we integrate deep learning \(DL\)\-based semantic segmentation for real\-time \(RT\) automated kidney volumetric measurements, which are essential for clinical assessment but are traditionally time\-consuming and prone to fatigue. This automation allows clinicians to concentrate on image interpretation rather than manual measurements. Complementing DL, augmented reality \(AR\) enhances the usability of US by projecting the display directly into the clinician's field of view, improving ergonomics and reducing the cognitive load associated with screen\-to\-patient transitions. Two AR\-DL\-assisted US pipelines on HoloLens\-2 are proposed: one streams directly via the application programming interface for a wireless setup, while the other supports any US device with video output for broader accessibility. We evaluate RT feasibility and accuracy using the Open Kidney Dataset and open\-source segmentation models \(nnU\-Net, Segmenter, YOLO with MedSAM and LiteMedSAM\). Our open\-source GitHub pipeline includes model implementations, measurement algorithms, and a Wi\-Fi\-based streaming solution, enhancing US training and diagnostics, especially in point\-of\-care settings.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2506.23721v1)

---

