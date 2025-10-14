# 每日从arXiv中获取最新YOLO相关论文


## Enhancing Maritime Domain Awareness on Inland Waterways: A YOLO\-Based Fusion of Satellite and AIS for Vessel Characterization / 

发布日期：2025-10-13

作者：Geoffery Agorku

摘要：Maritime Domain Awareness \(MDA\) for inland waterways remains challenged by cooperative system vulnerabilities. This paper presents a novel framework that fuses high\-resolution satellite imagery with vessel trajectory data from the Automatic Identification System \(AIS\). This work addresses the limitations of AIS\-based monitoring by leveraging non\-cooperative satellite imagery and implementing a fusion approach that links visual detections with AIS data to identify dark vessels, validate cooperative traffic, and support advanced MDA. The You Only Look Once \(YOLO\) v11 object detection model is used to detect and characterize vessels and barges by vessel type, barge cover, operational status, barge count, and direction of travel. An annotated data set of 4,550 instances was developed from $5\{,\}973~mathrm\{mi\}^2$ of Lower Mississippi River imagery. Evaluation on a held\-out test set demonstrated vessel classification \(tugboat, crane barge, bulk carrier, cargo ship, and hopper barge\) with an F1 score of 95.8%; barge cover \(covered or uncovered\) detection yielded an F1 score of 91.6%; operational status \(staged or in motion\) classification reached an F1 score of 99.4%. Directionality \(upstream, downstream\) yielded 93.8% accuracy. The barge count estimation resulted in a mean absolute error \(MAE\) of 2.4 barges. Spatial transferability analysis across geographically disjoint river segments showed accuracy was maintained as high as 98%. These results underscore the viability of integrating non\-cooperative satellite sensing with AIS fusion. This approach enables near\-real\-time fleet inventories, supports anomaly detection, and generates high\-quality data for inland waterway surveillance. Future work will expand annotated datasets, incorporate temporal tracking, and explore multi\-modal deep learning to further enhance operational scalability.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.11449v1)

---


## When Does Supervised Training Pay Off? The Hidden Economics of Object Detection in the Era of Vision\-Language Models / 

发布日期：2025-10-13

作者：Samer Al\-Hamadani

摘要：Object detection systems have traditionally relied on supervised learning with manually annotated bounding boxes, achieving high accuracy at the cost of substantial annotation investment. The emergence of Vision\-Language Models \(VLMs\) offers an alternative paradigm enabling zero\-shot detection through natural language queries, eliminating annotation requirements but operating with reduced accuracy. This paper presents the first comprehensive cost\-effectiveness analysis comparing supervised detection \(YOLO\) with zero\-shot VLM inference \(Gemini Flash 2.5\). Through systematic evaluation on 1,000 stratified COCO images and 200 diverse product images spanning consumer electronics and rare categories, combined with detailed Total Cost of Ownership modeling, we establish quantitative break\-even thresholds governing architecture selection. Our findings reveal that supervised YOLO achieves 91.2% accuracy versus 68.5% for zero\-shot Gemini on standard categories, representing a 22.7 percentage point advantage that costs $10,800 in annotation for 100\-category systems. However, this advantage justifies investment only beyond 55 million inferences, equivalent to 151,000 images daily for one year. Zero\-shot Gemini demonstrates 52.3% accuracy on diverse product categories \(ranging from highly web\-prevalent consumer electronics at 75\-85% to rare specialized equipment at 25\-40%\) where supervised YOLO achieves 0% due to architectural constraints preventing detection of untrained classes. Cost per Correct Detection analysis reveals substantially lower per\-detection costs for Gemini \($0.00050 vs $0.143\) at 100,000 inferences despite accuracy deficits. We develop decision frameworks demonstrating that optimal architecture selection depends critically on deployment volume, category stability, budget constraints, and accuracy requirements rather than purely technical performance metrics.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.11302v1)

---


## Slitless Spectroscopy Source Detection Using YOLO Deep Neural Network / 

发布日期：2025-10-13

作者：Xiaohan Chen

摘要：Slitless spectroscopy eliminates the need for slits, allowing light to pass directly through a prism or grism to generate a spectral dispersion image that encompasses all celestial objects within a specified area. This technique enables highly efficient spectral acquisition. However, when processing CSST slitless spectroscopy data, the unique design of its focal plane introduces a challenge: photometric and slitless spectroscopic images do not have a one\-to\-one correspondence. As a result, it becomes essential to first identify and count the sources in the slitless spectroscopic images before extracting spectra. To address this challenge, we employed the You Only Look Once \(YOLO\) object detection algorithm to develop a model for detecting targets in slitless spectroscopy images. This model was trained on 1,560 simulated CSST slitless spectroscopic images. These simulations were generated from the CSST Cycle 6 and Cycle 9 main survey data products, representing the Galactic and nearby galaxy regions and the high galactic latitude regions, respectively. On the validation set, the model achieved a precision of 88.6% and recall of 90.4% for spectral lines, and 87.0% and 80.8% for zeroth\-order images. In testing, it maintained a detection rate >80% for targets brighter than 21 mag \(medium\-density regions\) and 20 mag \(low\-density regions\) in the Galactic and nearby galaxies regions, and >70% for targets brighter than 18 mag in high galactic latitude regions.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.10922v1)

---


## MRS\-YOLO Railroad Transmission Line Foreign Object Detection Based on Improved YOLO11 and Channel Pruning / 

发布日期：2025-10-12

作者：Siyuan Liu

摘要：Aiming at the problems of missed detection, false detection and low detection efficiency in transmission line foreign object detection under railway environment, we proposed an improved algorithm MRS\-YOLO based on YOLO11. Firstly, a multi\-scale Adaptive Kernel Depth Feature Fusion \(MAKDF\) module is proposed and fused with the C3k2 module to form C3k2\_MAKDF, which enhances the model's feature extraction capability for foreign objects of different sizes and shapes. Secondly, a novel Re\-calibration Feature Fusion Pyramid Network \(RCFPN\) is designed as a neck structure to enhance the model's ability to integrate and utilize multi\-level features effectively. Then, Spatial and Channel Reconstruction Detect Head \(SC\_Detect\) based on spatial and channel preprocessing is designed to enhance the model's overall detection performance. Finally, the channel pruning technique is used to reduce the redundancy of the improved model, drastically reduce Parameters and Giga Floating Point Operations Per Second \(GFLOPs\), and improve the detection efficiency. The experimental results show that the mAP50 and mAP50:95 of the MRS\-YOLO algorithm proposed in this paper are improved to 94.8% and 86.4%, respectively, which are 0.7 and 2.3 percentage points higher compared to the baseline, while Parameters and GFLOPs are reduced by 44.2% and 17.5%, respectively. It is demonstrated that the improved algorithm can be better applied to the task of foreign object detection in railroad transmission lines.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.10553v1)

---


## Ordinal Scale Traffic Congestion Classification with Multi\-Modal Vision\-Language and Motion Analysis / 

发布日期：2025-10-11

作者：Yu\-Hsuan Lin

摘要：Accurate traffic congestion classification is essential for intelligent transportation systems and real\-time urban traffic management. This paper presents a multimodal framework combining open\-vocabulary visual\-language reasoning \(CLIP\), object detection \(YOLO\-World\), and motion analysis via MOG2\-based background subtraction. The system predicts congestion levels on an ordinal scale from 1 \(free flow\) to 5 \(severe congestion\), enabling semantically aligned and temporally consistent classification. To enhance interpretability, we incorporate motion\-based confidence weighting and generate annotated visual outputs. Experimental results show the model achieves 76.7 percent accuracy, an F1 score of 0.752, and a Quadratic Weighted Kappa \(QWK\) of 0.684, significantly outperforming unimodal baselines. These results demonstrate the framework's effectiveness in preserving ordinal structure and leveraging visual\-language and motion modalities. Future enhancements include incorporating vehicle sizing and refined density metrics.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2510.10342v1)

---

