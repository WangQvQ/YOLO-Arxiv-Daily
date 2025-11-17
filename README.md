# 每日从arXiv中获取最新YOLO相关论文


## Facial Expression Recognition with YOLOv11 and YOLOv12: A Comparative Study / 

发布日期：2025-11-14

作者：Umma Aymon

摘要：Facial Expression Recognition remains a challenging task, especially in unconstrained, real\-world environments. This study investigates the performance of two lightweight models, YOLOv11n and YOLOv12n, which are the nano variants of the latest official YOLO series, within a unified detection and classification framework for FER. Two benchmark classification datasets, FER2013 and KDEF, are converted into object detection format and model performance is evaluated using mAP 0.5, precision, recall, and confusion matrices. Results show that YOLOv12n achieves the highest overall performance on the clean KDEF dataset with a mAP 0.5 of 95.6, and also outperforms YOLOv11n on the FER2013 dataset in terms of mAP 63.8, reflecting stronger sensitivity to varied expressions. In contrast, YOLOv11n demonstrates higher precision 65.2 on FER2013, indicating fewer false positives and better reliability in noisy, real\-world conditions. On FER2013, both models show more confusion between visually similar expressions, while clearer class separation is observed on the cleaner KDEF dataset. These findings underscore the trade\-off between sensitivity and precision, illustrating how lightweight YOLO models can effectively balance performance and efficiency. The results demonstrate adaptability across both controlled and real\-world conditions, establishing these models as strong candidates for real\-time, resource\-constrained emotion\-aware AI applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.10940v1)

---


## YOLO\-Drone: An Efficient Object Detection Approach Using the GhostHead Network for Drone Images / 

发布日期：2025-11-14

作者：Hyun\-Ki Jung

摘要：Object detection using images or videos captured by drones is a promising technology with significant potential across various industries. However, a major challenge is that drone images are typically taken from high altitudes, making object identification difficult. This paper proposes an effective solution to address this issue. The base model used in the experiments is YOLOv11, the latest object detection model, with a specific implementation based on YOLOv11n. The experimental data were sourced from the widely used and reliable VisDrone dataset, a standard benchmark in drone\-based object detection. This paper introduces an enhancement to the Head network of the YOLOv11 algorithm, called the GhostHead Network. The model incorporating this improvement is named YOLO\-Drone. Experimental results demonstrate that YOLO\-Drone achieves significant improvements in key detection accuracy metrics, including Precision, Recall, F1\-Score, and mAP \(0.5\), compared to the original YOLOv11. Specifically, the proposed model recorded a 0.4% increase in Precision, a 0.6% increase in Recall, a 0.5% increase in F1\-Score, and a 0.5% increase in mAP \(0.5\). Additionally, the Inference Speed metric, which measures image processing speed, also showed a notable improvement. These results indicate that YOLO\-Drone is a high\-performance model with enhanced accuracy and speed compared to YOLOv11. To further validate its reliability, comparative experiments were conducted against other high\-performance object detection models, including YOLOv8, YOLOv9, and YOLOv10. The results confirmed that the proposed model outperformed YOLOv8 by 0.1% in mAP \(0.5\) and surpassed YOLOv9 and YOLOv10 by 0.3% and 0.6%, respectively.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.10905v1)

---


## Robust Object Detection with Pseudo Labels from VLMs using Per\-Object Co\-teaching / 

发布日期：2025-11-13

作者：Uday Bhaskar

摘要：Foundation models, especially vision\-language models \(VLMs\), offer compelling zero\-shot object detection for applications like autonomous driving, a domain where manual labelling is prohibitively expensive. However, their detection latency and tendency to hallucinate predictions render them unsuitable for direct deployment. This work introduces a novel pipeline that addresses this challenge by leveraging VLMs to automatically generate pseudo\-labels for training efficient, real\-time object detectors. Our key innovation is a per\-object co\-teaching\-based training strategy that mitigates the inherent noise in VLM\-generated labels. The proposed per\-object coteaching approach filters noisy bounding boxes from training instead of filtering the entire image. Specifically, two YOLO models learn collaboratively, filtering out unreliable boxes from each mini\-batch based on their peers' per\-object loss values. Overall, our pipeline provides an efficient, robust, and scalable approach to train high\-performance object detectors for autonomous driving, significantly reducing reliance on costly human annotation. Experimental results on the KITTI dataset demonstrate that our method outperforms a baseline YOLOv5m model, achieving a significant mAP@0.5 boost \($31.12%$ to $46.61%$\) while maintaining real\-time detection latency. Furthermore, we show that supplementing our pseudo\-labelled data with a small fraction of ground truth labels \($10%$\) leads to further performance gains, reaching $57.97%$ mAP@0.5 on the KITTI dataset. We observe similar performance improvements for the ACDC and BDD100k datasets.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.09955v1)

---


## DKDS: A Benchmark Dataset of Degraded Kuzushiji Documents with Seals for Detection and Binarization / 

发布日期：2025-11-12

作者：Rui\-Yang Ju

摘要：Kuzushiji, a pre\-modern Japanese cursive script, can currently be read and understood by only a few thousand trained experts in Japan. With the rapid development of deep learning, researchers have begun applying Optical Character Recognition \(OCR\) techniques to transcribe Kuzushiji into modern Japanese. Although existing OCR methods perform well on clean pre\-modern Japanese documents written in Kuzushiji, they often fail to consider various types of noise, such as document degradation and seals, which significantly affect recognition accuracy. To the best of our knowledge, no existing dataset specifically addresses these challenges. To address this gap, we introduce the Degraded Kuzushiji Documents with Seals \(DKDS\) dataset as a new benchmark for related tasks. We describe the dataset construction process, which required the assistance of a trained Kuzushiji expert, and define two benchmark tracks: \(1\) text and seal detection and \(2\) document binarization. For the text and seal detection track, we provide baseline results using multiple versions of the You Only Look Once \(YOLO\) models for detecting Kuzushiji characters and seals. For the document binarization track, we present baseline results from traditional binarization algorithms, traditional algorithms combined with K\-means clustering, and Generative Adversarial Network \(GAN\)\-based methods. The DKDS dataset and the implementation code for baseline methods are available at https://ruiyangju.github.io/DKDS.

中文摘要：


代码链接：https://ruiyangju.github.io/DKDS.

论文链接：[阅读更多](http://arxiv.org/abs/2511.09117v1)

---


## Hardware\-Aware YOLO Compression for Low\-Power Edge AI on STM32U5 for Weeds Detection in Digital Agriculture / 

发布日期：2025-11-11

作者：Charalampos S. Kouzinopoulos

摘要：Weeds significantly reduce crop yields worldwide and pose major challenges to sustainable agriculture. Traditional weed management methods, primarily relying on chemical herbicides, risk environmental contamination and lead to the emergence of herbicide\-resistant species. Precision weeding, leveraging computer vision and machine learning methods, offers a promising eco\-friendly alternative but is often limited by reliance on high\-power computational platforms. This work presents an optimized, low\-power edge AI system for weeds detection based on the YOLOv8n object detector deployed on the STM32U575ZI microcontroller. Several compression techniques are applied to the detection model, including structured pruning, integer quantization and input image resolution scaling in order to meet strict hardware constraints. The model is trained and evaluated on the CropAndWeed dataset with 74 plant species, achieving a balanced trade\-off between detection accuracy and efficiency. Our system supports real\-time, in\-situ weeds detection with a minimal energy consumption of 51.8mJ per inference, enabling scalable deployment in power\-constrained agricultural environments.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2511.07990v1)

---

