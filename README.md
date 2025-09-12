# 每日从arXiv中获取最新YOLO相关论文


## Classification of Driver Behaviour Using External Observation Techniques for Autonomous Vehicles / 

发布日期：2025-09-11

作者：Ian Nell

摘要：Road traffic accidents remain a significant global concern, with human error, particularly distracted and impaired driving, among the leading causes. This study introduces a novel driver behavior classification system that uses external observation techniques to detect indicators of distraction and impairment. The proposed framework employs advanced computer vision methodologies, including real\-time object tracking, lateral displacement analysis, and lane position monitoring. The system identifies unsafe driving behaviors such as excessive lateral movement and erratic trajectory patterns by implementing the YOLO object detection model and custom lane estimation algorithms. Unlike systems reliant on inter\-vehicular communication, this vision\-based approach enables behavioral analysis of non\-connected vehicles. Experimental evaluations on diverse video datasets demonstrate the framework's reliability and adaptability across varying road and environmental conditions.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.09349v1)

---


## Model\-Agnostic Open\-Set Air\-to\-Air Visual Object Detection for Reliable UAV Perception / 

发布日期：2025-09-11

作者：Spyridon Loukovitis

摘要：Open\-set detection is crucial for robust UAV autonomy in air\-to\-air object detection under real\-world conditions. Traditional closed\-set detectors degrade significantly under domain shifts and flight data corruption, posing risks to safety\-critical applications. We propose a novel, model\-agnostic open\-set detection framework designed specifically for embedding\-based detectors. The method explicitly handles unknown object rejection while maintaining robustness against corrupted flight data. It estimates semantic uncertainty via entropy modeling in the embedding space and incorporates spectral normalization and temperature scaling to enhance open\-set discrimination. We validate our approach on the challenging AOT aerial benchmark and through extensive real\-world flight tests. Comprehensive ablation studies demonstrate consistent improvements over baseline methods, achieving up to a 10% relative AUROC gain compared to standard YOLO\-based detectors. Additionally, we show that background rejection further strengthens robustness without compromising detection accuracy, making our solution particularly well\-suited for reliable UAV perception in dynamic air\-to\-air environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.09297v1)

---


## FPI\-Det: a face\-\-phone Interaction Dataset for phone\-use detection and understanding / 

发布日期：2025-09-11

作者：Jianqin Gao

摘要：The widespread use of mobile devices has created new challenges for vision systems in safety monitoring, workplace productivity assessment, and attention management. Detecting whether a person is using a phone requires not only object recognition but also an understanding of behavioral context, which involves reasoning about the relationship between faces, hands, and devices under diverse conditions. Existing generic benchmarks do not fully capture such fine\-grained human\-\-device interactions. To address this gap, we introduce the FPI\-Det, containing 22\{,\}879 images with synchronized annotations for faces and phones across workplace, education, transportation, and public scenarios. The dataset features extreme scale variation, frequent occlusions, and varied capture conditions. We evaluate representative YOLO and DETR detectors, providing baseline results and an analysis of performance across object sizes, occlusion levels, and environments. Source code and dataset is available at https://github.com/KvCgRv/FPI\-Det.

中文摘要：


代码链接：https://github.com/KvCgRv/FPI-Det.

论文链接：[阅读更多](http://arxiv.org/abs/2509.09111v1)

---


## A New Hybrid Model of Generative Adversarial Network and You Only Look Once Algorithm for Automatic License\-Plate Recognition / 

发布日期：2025-09-08

作者：Behnoud Shafiezadeh

摘要：Automatic License\-Plate Recognition \(ALPR\) plays a pivotal role in Intelligent Transportation Systems \(ITS\) as a fundamental element of Smart Cities. However, due to its high variability, ALPR faces challenging issues more efficiently addressed by deep learning techniques. In this paper, a selective Generative Adversarial Network \(GAN\) is proposed for deblurring in the preprocessing step, coupled with the state\-of\-the\-art You\-Only\-Look\-Once \(YOLO\)v5 object detection architectures for License\-Plate Detection \(LPD\), and the integrated Character Segmentation \(CS\) and Character Recognition \(CR\) steps. The selective preprocessing bypasses unnecessary and sometimes counter\-productive input manipulations, while YOLOv5 LPD/CS\+CR delivers high accuracy and low computing cost. As a result, YOLOv5 achieves a detection time of 0.026 seconds for both LP and CR detection stages, facilitating real\-time applications with exceptionally rapid responsiveness. Moreover, the proposed model achieves accuracy rates of 95% and 97% in the LPD and CR detection phases, respectively. Furthermore, the inclusion of the Deblur\-GAN pre\-processor significantly improves detection accuracy by nearly 40%, especially when encountering blurred License Plates \(LPs\).To train and test the learning components, we generated and publicly released our blur and ALPR datasets \(using Iranian license plates as a use\-case\), which are more representative of close\-to\-real\-life ad\-hoc situations. The findings demonstrate that employing the state\-of\-the\-art YOLO model results in excellent overall precision and detection time, making it well\-suited for portable applications. Additionally, integrating the Deblur\-GAN model as a preliminary processing step enhances the overall effectiveness of our comprehensive model, particularly when confronted with blurred scenes captured by the camera as input.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.06868v1)

---


## When Language Model Guides Vision: Grounding DINO for Cattle Muzzle Detection / 

发布日期：2025-09-08

作者：Rabin Dulal

摘要：Muzzle patterns are among the most effective biometric traits for cattle identification. Fast and accurate detection of the muzzle region as the region of interest is critical to automatic visual cattle identification.. Earlier approaches relied on manual detection, which is labor\-intensive and inconsistent. Recently, automated methods using supervised models like YOLO have become popular for muzzle detection. Although effective, these methods require extensive annotated datasets and tend to be trained data\-dependent, limiting their performance on new or unseen cattle. To address these limitations, this study proposes a zero\-shot muzzle detection framework based on Grounding DINO, a vision\-language model capable of detecting muzzles without any task\-specific training or annotated data. This approach leverages natural language prompts to guide detection, enabling scalable and flexible muzzle localization across diverse breeds and environments. Our model achieves a mean Average Precision \(mAP\)@0.5 of 76.8%, demonstrating promising performance without requiring annotated data. To our knowledge, this is the first research to provide a real\-world, industry\-oriented, and annotation\-free solution for cattle muzzle detection. The framework offers a practical alternative to supervised methods, promising improved adaptability and ease of deployment in livestock monitoring applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.06427v1)

---

