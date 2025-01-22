# 每日从arXiv中获取最新YOLO相关论文


## Data\-driven Detection and Evaluation of Damages in Concrete Structures: Using Deep Learning and Computer Vision / 

发布日期：2025-01-21

作者：Saeid Ataei

摘要：Structural integrity is vital for maintaining the safety and longevity of concrete infrastructures such as bridges, tunnels, and walls. Traditional methods for detecting damages like cracks and spalls are labor\-intensive, time\-consuming, and prone to human error. To address these challenges, this study explores advanced data\-driven techniques using deep learning for automated damage detection and analysis. Two state\-of\-the\-art instance segmentation models, YOLO\-v7 instance segmentation and Mask R\-CNN, were evaluated using a dataset comprising 400 images, augmented to 10,995 images through geometric and color\-based transformations to enhance robustness. The models were trained and validated using a dataset split into 90% training set, validation and test set 10%. Performance metrics such as precision, recall, mean average precision \(mAP@0.5\), and frames per second \(FPS\) were used for evaluation. YOLO\-v7 achieved a superior mAP@0.5 of 96.1% and processed 40 FPS, outperforming Mask R\-CNN, which achieved a mAP@0.5 of 92.1% with a slower processing speed of 18 FPS. The findings recommend YOLO\-v7 instance segmentation model for real\-time, high\-speed structural health monitoring, while Mask R\-CNN is better suited for detailed offline assessments. This study demonstrates the potential of deep learning to revolutionize infrastructure maintenance, offering a scalable and efficient solution for automated damage detection.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.11836v1)

---


## Efficient Auto\-Labeling of Large\-Scale Poultry Datasets \(ALPD\) Using Semi\-Supervised Models, Active Learning, and Prompt\-then\-Detect Approach / 

发布日期：2025-01-18

作者：Ramesh Bahadur Bist

摘要：The rapid growth of AI in poultry farming has highlighted the challenge of efficiently labeling large, diverse datasets. Manual annotation is time\-consuming, making it impractical for modern systems that continuously generate data. This study explores semi\-supervised auto\-labeling methods, integrating active learning, and prompt\-then\-detect paradigm to develop an efficient framework for auto\-labeling of large poultry datasets aimed at advancing AI\-driven behavior and health monitoring. Viideo data were collected from broilers and laying hens housed at the University of Arkansas and the University of Georgia. The collected videos were converted into images, pre\-processed, augmented, and labeled. Various machine learning models, including zero\-shot models like Grounding DINO, YOLO\-World, and CLIP, and supervised models like YOLO and Faster\-RCNN, were utilized for broilers, hens, and behavior detection. The results showed that YOLOv8s\-World and YOLOv9s performed better when compared performance metrics for broiler and hen detection under supervised learning, while among the semi\-supervised model, YOLOv8s\-ALPD achieved the highest precision \(96.1%\) and recall \(99.0%\) with an RMSE of 1.9. The hybrid YOLO\-World model, incorporating the optimal YOLOv8s backbone, demonstrated the highest overall performance. It achieved a precision of 99.2%, recall of 99.4%, and an F1 score of 98.7% for breed detection, alongside a precision of 88.4%, recall of 83.1%, and an F1 score of 84.5% for individual behavior detection. Additionally, semi\-supervised models showed significant improvements in behavior detection, achieving up to 31% improvement in precision and 16% in F1\-score. The semi\-supervised models with minimal active learning reduced annotation time by over 80% compared to full manual labeling. Moreover, integrating zero\-shot models enhanced detection and behavior identification.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.10809v1)

---


## Polyp detection in colonoscopy images using YOLOv11 / 

发布日期：2025-01-15

作者：Alok Ranjan Sahoo

摘要：Colorectal cancer \(CRC\) is one of the most commonly diagnosed cancers all over the world. It starts as a polyp in the inner lining of the colon. To prevent CRC, early polyp detection is required. Colonosopy is used for the inspection of the colon. Generally, the images taken by the camera placed at the tip of the endoscope are analyzed by the experts manually. Various traditional machine learning models have been used with the rise of machine learning. Recently, deep learning models have shown more effectiveness in polyp detection due to their superiority in generalizing and learning small features. These deep learning models for object detection can be segregated into two different types: single\-stage and two\-stage. Generally, two stage models have higher accuracy than single stage ones but the single stage models have low inference time. Hence, single stage models are easy to use for quick object detection. YOLO is one of the singlestage models used successfully for polyp detection. It has drawn the attention of researchers because of its lower inference time. The researchers have used Different versions of YOLO so far, and with each newer version, the accuracy of the model is increasing. This paper aims to see the effectiveness of the recently released YOLOv11 to detect polyp. We analyzed the performance for all five models of YOLOv11 \(YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x\) with Kvasir dataset for the training and testing. Two different versions of the dataset were used. The first consisted of the original dataset, and the other was created using augmentation techniques. The performance of all the models with these two versions of the dataset have been analysed.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.09051v1)

---


## Detecting Wildfire Flame and Smoke through Edge Computing using Transfer Learning Enhanced Deep Learning Models / 

发布日期：2025-01-15

作者：Giovanny Vazquez

摘要：Autonomous unmanned aerial vehicles \(UAVs\) integrated with edge computing capabilities empower real\-time data processing directly on the device, dramatically reducing latency in critical scenarios such as wildfire detection. This study underscores Transfer Learning's \(TL\) significance in boosting the performance of object detectors for identifying wildfire smoke and flames, especially when trained on limited datasets, and investigates the impact TL has on edge computing metrics. With the latter focusing how TL\-enhanced You Only Look Once \(YOLO\) models perform in terms of inference time, power usage, and energy consumption when using edge computing devices. This study utilizes the Aerial Fire and Smoke Essential \(AFSE\) dataset as the target, with the Flame and Smoke Detection Dataset \(FASDD\) and the Microsoft Common Objects in Context \(COCO\) dataset serving as source datasets. We explore a two\-stage cascaded TL method, utilizing D\-Fire or FASDD as initial stage target datasets and AFSE as the subsequent stage. Through fine\-tuning, TL significantly enhances detection precision, achieving up to 79.2% mean Average Precision \(mAP@0.5\), reduces training time, and increases model generalizability across the AFSE dataset. However, cascaded TL yielded no notable improvements and TL alone did not benefit the edge computing metrics evaluated. Lastly, this work found that YOLOv5n remains a powerful model when lacking hardware acceleration, finding that YOLOv5n can process images nearly twice as fast as its newer counterpart, YOLO11n. Overall, the results affirm TL's role in augmenting the accuracy of object detectors while also illustrating that additional enhancements are needed to improve edge computing performance.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.08639v1)

---


## Benchmarking YOLOv8 for Optimal Crack Detection in Civil Infrastructure / 

发布日期：2025-01-12

作者：Woubishet Zewdu Taffese

摘要：Ensuring the structural integrity and safety of bridges is crucial for the reliability of transportation networks and public safety. Traditional crack detection methods are increasingly being supplemented or replaced by advanced artificial intelligence \(AI\) techniques. However, most of the models rely on two\-stage target detection algorithms, which pose concerns for real\-time applications due to their lower speed. While models such as YOLO \(You Only Look Once\) have emerged as transformative tools due to their remarkable speed and accuracy. However, the potential of the latest YOLOv8 framework in this domain remains underexplored. This study bridges that gap by rigorously evaluating YOLOv8's performance across five model scales \(nano, small, medium, large, and extra\-large\) using a high\-quality Roboflow dataset. A comprehensive hyperparameter optimization was performed, testing six state\-of\-the\-art optimizers\-Stochastic Gradient Descent, Adaptive Moment Estimation, Adam with Decoupled Weight Decay, Root Mean Square Propagation, Rectified Adam, and Nesterov\-accelerated Adam. Results revealed that YOLOv8, optimized with Stochastic Gradient Descent, delivered exceptional accuracy and speed, setting a new benchmark for real\-time crack detection. Beyond its immediate application, this research positions YOLOv8 as a foundational approach for integrating advanced computer vision techniques into infrastructure monitoring. By enabling more reliable and proactive maintenance of aging bridge networks, this work paves the way for safer, more efficient transportation systems worldwide.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2501.06922v1)

---

