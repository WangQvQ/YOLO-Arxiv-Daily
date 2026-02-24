# 每日从arXiv中获取最新YOLO相关论文


## A Text\-Guided Vision Model for Enhanced Recognition of Small Instances / 

发布日期：2026-02-23

作者：Hyun\-Ki Jung

摘要：As drone\-based object detection technology continues to evolve, the demand is shifting from merely detecting objects to enabling users to accurately identify specific targets. For example, users can input particular targets as prompts to precisely detect desired objects. To address this need, an efficient text\-guided object detection model has been developed to enhance the detection of small objects. Specifically, an improved version of the existing YOLO\-World model is introduced. The proposed method replaces the C2f layer in the YOLOv8 backbone with a C3k2 layer, enabling more precise representation of local features, particularly for small objects or those with clearly defined boundaries. Additionally, the proposed architecture improves processing speed and efficiency through parallel processing optimization, while also contributing to a more lightweight model design. Comparative experiments on the VisDrone dataset show that the proposed model outperforms the original YOLO\-World model, with precision increasing from 40.6% to 41.6%, recall from 30.8% to 31%, F1 score from 35% to 35.5%, and mAP@0.5 from 30.4% to 30.7%, confirming its enhanced accuracy. Furthermore, the model demonstrates superior lightweight performance, with the parameter count reduced from 4 million to 3.8 million and FLOPs decreasing from 15.7 billion to 15.2 billion. These results indicate that the proposed approach provides a practical and effective solution for precise object detection in drone\-based applications.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.19503v1)

---


## TactEx: An Explainable Multimodal Robotic Interaction Framework for Human\-Like Touch and Hardness Estimation / 

发布日期：2026-02-21

作者：Felix Verstraete

摘要：Accurate perception of object hardness is essential for safe and dexterous contact\-rich robotic manipulation. Here, we present TactEx, an explainable multimodal robotic interaction framework that unifies vision, touch, and language for human\-like hardness estimation and interactive guidance. We evaluate TactEx on fruit\-ripeness assessment, a representative task that requires both tactile sensing and contextual understanding. The system fuses GelSight\-Mini tactile streams with RGB observations and language prompts. A ResNet50\+LSTM model estimates hardness from sequential tactile data, while a cross\-modal alignment module combines visual cues with guidance from a large language model \(LLM\). This explainable multimodal interface allows users to distinguish ripeness levels with statistically significant class separation \(p < 0.01 for all fruit pairs\). For touch placement, we compare YOLO with Grounded\-SAM \(GSAM\) and find GSAM to be more robust for fine\-grained segmentation and contact\-site selection. A lightweight LLM parses user instructions and produces grounded natural\-language explanations linked to the tactile outputs. In end\-to\-end evaluations, TactEx attains 90% task success on simple user queries and generalises to novel tasks without large\-scale tuning. These results highlight the promise of combining pretrained visual and tactile models with language grounding to advance explainable, human\-like touch perception and decision\-making in robotics.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.18967v1)

---


## Depth\-Enhanced YOLO\-SAM2 Detection for Reliable Ballast Insufficiency Identification / 

发布日期：2026-02-21

作者：Shiyu Liu

摘要：This paper presents a depth\-enhanced YOLO\-SAM2 framework for detecting ballast insufficiency in railway tracks using RGB\-D data. Although YOLOv8 provides reliable localization, the RGB\-only model shows limited safety performance, achieving high precision \(0.99\) but low recall \(0.49\) due to insufficient ballast, as it tends to over\-predict the sufficient class. To improve reliability, we incorporate depth\-based geometric analysis enabled by a sleeper\-aligned depth\-correction pipeline that compensates for RealSense spatial distortion using polynomial modeling, RANSAC, and temporal smoothing. SAM2 segmentation further refines region\-of\-interest masks, enabling accurate extraction of sleeper and ballast profiles for geometric classification.   Experiments on field\-collected top\-down RGB\-D data show that depth\-enhanced configurations substantially improve the detection of insufficient ballast. Depending on bounding\-box sampling \(AABB or RBB\) and geometric criteria, recall increases from 0.49 to as high as 0.80, and F1\-score improves from 0.66 to over 0.80. These results demonstrate that integrating depth correction with YOLO\-SAM2 yields a more robust and reliable approach for automated railway ballast inspection, particularly in visually ambiguous or safety\-critical scenarios.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.18961v1)

---


## BloomNet: Exploring Single vs. Multiple Object Annotation for Flower Recognition Using YOLO Variants / 

发布日期：2026-02-20

作者：Safwat Nusrat

摘要：Precise localization and recognition of flowers are crucial for advancing automated agriculture, particularly in plant phenotyping, crop estimation, and yield monitoring. This paper benchmarks several YOLO architectures such as YOLOv5s, YOLOv8n/s/m, and YOLOv12n for flower object detection under two annotation regimes: single\-image single\-bounding box \(SISBB\) and single\-image multiple\-bounding box \(SIMBB\). The FloralSix dataset, comprising 2,816 high\-resolution photos of six different flower species, is also introduced. It is annotated for both dense \(clustered\) and sparse \(isolated\) scenarios. The models were evaluated using Precision, Recall, and Mean Average Precision \(mAP\) at IoU thresholds of 0.5 \(mAP@0.5\) and 0.5\-0.95 \(mAP@0.5:0.95\). In SISBB, YOLOv8m \(SGD\) achieved the best results with Precision 0.956, Recall 0.951, mAP@0.5 0.978, and mAP@0.5:0.95 0.865, illustrating strong accuracy in detecting isolated flowers. With mAP@0.5 0.934 and mAP@0.5:0.95 0.752, YOLOv12n \(SGD\) outperformed the more complicated SIMBB scenario, proving robustness in dense, multi\-object detection. Results show how annotation density, IoU thresholds, and model size interact: recall\-optimized models perform better in crowded environments, whereas precision\-oriented models perform best in sparse scenarios. In both cases, the Stochastic Gradient Descent \(SGD\) optimizer consistently performed better than alternatives. These density\-sensitive sensors are helpful for non\-destructive crop analysis, growth tracking, robotic pollination, and stress evaluation.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.18585v1)

---


## Do Generative Metrics Predict YOLO Performance? An Evaluation Across Models, Augmentation Ratios, and Dataset Complexity / 

发布日期：2026-02-20

作者：Vasile Marian

摘要：Synthetic images are increasingly used to augment object\-detection training sets, but reliably evaluating a synthetic dataset before training remains difficult: standard global generative metrics \(e.g., FID\) often do not predict downstream detection mAP. We present a controlled evaluation of synthetic augmentation for YOLOv11 across three single\-class detection regimes \-\- Traffic Signs \(sparse/near\-saturated\), Cityscapes Pedestrian \(dense/occlusion\-heavy\), and COCO PottedPlant \(multi\-instance/high\-variability\). We benchmark six GAN\-, diffusion\-, and hybrid\-based generators over augmentation ratios from 10% to 150% of the real training split, and train YOLOv11 both from scratch and with COCO\-pretrained initialization, evaluating on held\-out real test splits \(mAP@0.50:0.95\). For each dataset\-generator\-augmentation configuration, we compute pre\-training dataset metrics under a matched\-size bootstrap protocol, including \(i\) global feature\-space metrics in both Inception\-v3 and DINOv2 embeddings and \(ii\) object\-centric distribution distances over bounding\-box statistics. Synthetic augmentation yields substantial gains in the more challenging regimes \(up to \+7.6% and \+30.6% relative mAP in Pedestrian and PottedPlant, respectively\) but is marginal in Traffic Signs and under pretrained fine\-tuning. To separate metric signal from augmentation quantity, we report both raw and augmentation\-controlled \(residualized\) correlations with multiple\-testing correction, showing that metric\-performance alignment is strongly regime\-dependent and that many apparent raw associations weaken after controlling for augmentation level.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2602.18525v1)

---

