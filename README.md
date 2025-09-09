# 每日从arXiv中获取最新YOLO相关论文


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


## Cross\-Modal Enhancement and Benchmark for UAV\-based Open\-Vocabulary Object Detection / 

发布日期：2025-09-07

作者：Zhenhai Weng

摘要：Open\-Vocabulary Object Detection \(OVD\) has emerged as a pivotal technology for applications involving Unmanned Aerial Vehicles \(UAVs\). However, the prevailing large\-scale datasets for OVD pre\-training are predominantly composed of ground\-level, natural images. This creates a significant domain gap, causing models trained on them to exhibit a substantial drop in performance on UAV imagery. To address this limitation, we first propose a refined UAV\-Label engine. Then we construct and introduce UAVDE\-2M\(contains over 2,000,000 instances and 1800 categories\) and UAVCAP\-15k\(contains over 15,000 images\). Furthermore, we propose a novel Cross\-Attention Gated Enhancement Fusion \(CAGE\) module and integrate it into the YOLO\-World\-v2 architecture. Finally, extensive experiments on the VisDrone and SIMD datasets verify the effectiveness of our proposed method for applications in UAV\-based imagery and remote sensing.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.06011v1)

---


## Evaluating YOLO Architectures: Implications for Real\-Time Vehicle Detection in Urban Environments of Bangladesh / 

发布日期：2025-09-06

作者：Ha Meem Hossain

摘要：Vehicle detection systems trained on Non\-Bangladeshi datasets struggle to accurately identify local vehicle types in Bangladesh's unique road environments, creating critical gaps in autonomous driving technology for developing regions. This study evaluates six YOLO model variants on a custom dataset featuring 29 distinct vehicle classes, including region\-specific vehicles such as \`\`Desi Nosimon'', \`\`Leguna'', \`\`Battery Rickshaw'', and \`\`CNG''. The dataset comprises high\-resolution images \(1920x1080\) captured across various Bangladeshi roads using mobile phone cameras and manually annotated using LabelImg with YOLO format bounding boxes. Performance evaluation revealed YOLOv11x as the top performer, achieving 63.7% mAP@0.5, 43.8% mAP@0.5:0.95, 61.4% recall, and 61.6% F1\-score, though requiring 45.8 milliseconds per image for inference. Medium variants \(YOLOv8m, YOLOv11m\) struck an optimal balance, delivering robust detection performance with mAP@0.5 values of 62.5% and 61.8% respectively, while maintaining moderate inference times around 14\-15 milliseconds. The study identified significant detection challenges for rare vehicle classes, with Construction Vehicles and Desi Nosimons showing near\-zero accuracy due to dataset imbalances and insufficient training samples. Confusion matrices revealed frequent misclassifications between visually similar vehicles, particularly Mini Trucks versus Mini Covered Vans. This research provides a foundation for developing robust object detection systems specifically adapted to Bangladesh traffic conditions, addressing critical needs in autonomous vehicle technology advancement for developing regions where conventional generic\-trained models fail to perform adequately.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.05652v1)

---


## An Analysis of Layer\-Freezing Strategies for Enhanced Transfer Learning in YOLO Architectures / 

发布日期：2025-09-05

作者：Andrzej D. Dobrzycki

摘要：The You Only Look Once \(YOLO\) architecture is crucial for real\-time object detection. However, deploying it in resource\-constrained environments such as unmanned aerial vehicles \(UAVs\) requires efficient transfer learning. Although layer freezing is a common technique, the specific impact of various freezing configurations on contemporary YOLOv8 and YOLOv10 architectures remains unexplored, particularly with regard to the interplay between freezing depth, dataset characteristics, and training dynamics. This research addresses this gap by presenting a detailed analysis of layer\-freezing strategies. We systematically investigate multiple freezing configurations across YOLOv8 and YOLOv10 variants using four challenging datasets that represent critical infrastructure monitoring. Our methodology integrates a gradient behavior analysis \(L2 norm\) and visual explanations \(Grad\-CAM\) to provide deeper insights into training dynamics under different freezing strategies. Our results reveal that there is no universal optimal freezing strategy but, rather, one that depends on the properties of the data. For example, freezing the backbone is effective for preserving general\-purpose features, while a shallower freeze is better suited to handling extreme class imbalance. These configurations reduce graphics processing unit \(GPU\) memory consumption by up to 28% compared to full fine\-tuning and, in some cases, achieve mean average precision \(mAP@50\) scores that surpass those of full fine\-tuning. Gradient analysis corroborates these findings, showing distinct convergence patterns for moderately frozen models. Ultimately, this work provides empirical findings and practical guidelines for selecting freezing strategies. It offers a practical, evidence\-based approach to balanced transfer learning for object detection in scenarios with limited resources.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2509.05490v1)

---

