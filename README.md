# 每日从arXiv中获取最新YOLO相关论文


## Does YOLO Really Need to See Every Training Image in Every Epoch? / 

发布日期：2026-03-18

作者：Xingxing Xie

摘要：YOLO detectors are known for their fast inference speed, yet training them remains unexpectedly time\-consuming due to their exhaustive pipeline that processes every training image in every epoch, even when many images have already been sufficiently learned. This stands in clear contrast to the efficiency suggested by the \`\`You Only Look Once'' philosophy. This naturally raises an important question: textit\{Does YOLO really need to see every training image in every epoch?\} To explore this, we propose an Anti\-Forgetting Sampling Strategy \(AFSS\) that dynamically determines which images should be used and which can be skipped during each epoch, allowing the detector to learn more effectively and efficiently. Specifically, AFSS measures the learning sufficiency of each training image as the minimum of its detection recall and precision, and dynamically categorizes training images into easy, medium, or hard levels accordingly. Easy training images are sparsely resampled during training in a continuous review manner, with priority given to those that have not been used for a long time to reduce redundancy and prevent forgetting. Moderate training images are partially selected, prioritizing recently unused ones and randomly choosing the rest from unselected images to ensure coverage and prevent forgetting. Hard training images are fully sampled in every epoch to ensure sufficient learning. The learning sufficiency of each training image is periodically updated, enabling detectors to adaptively shift its focus toward the informative training images over time while progressively discarding redundant ones. On widely used natural image detection benchmarks \(MS COCO 2017 and PASCAL VOC 2007\) and remote sensing detection datasets \(DOTA\-v1.0 and DIOR\-R\), AFSS achieves more than $1.43times$ training speedup for YOLO\-series detectors while also improving accuracy.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.17684v1)

---


## Automated identification of Ichneumonoidea wasps via YOLO\-based deep learning: Integrating HiresCam for Explainable AI / 

发布日期：2026-03-17

作者：Joao Manoel Herrera Pinheiro

摘要：Accurate taxonomic identification of parasitoid wasps within the superfamily Ichneumonoidea is essential for biodiversity assessment, ecological monitoring, and biological control programs. However, morphological similarity, small body size, and fine\-grained interspecific variation make manual identification labor\-intensive and expertise\-dependent. This study proposes a deep learning\-based framework for the automated identification of Ichneumonoidea wasps using a YOLO\-based architecture integrated with High\-Resolution Class Activation Mapping \(HiResCAM\) to enhance interpretability. The proposed system simultaneously identifies wasp families from high\-resolution images. The dataset comprises 3556 high\-resolution images of Hymenoptera specimens. The taxonomic distribution is primarily concentrated among the families Ichneumonidae \(n = 786\), Braconidae \(n = 648\), Apidae \(n = 466\), and Vespidae \(n = 460\). Extensive experiments were conducted using a curated dataset, with model performance evaluated through precision, recall, F1 score, and accuracy. The results demonstrate high accuracy of over 96 % and robust generalization across morphological variations. HiResCAM visualizations confirm that the model focuses on taxonomically relevant anatomical regions, such as wing venation, antennae segmentation, and metasomal structures, thereby validating the biological plausibility of the learned features. The integration of explainable AI techniques improves transparency and trustworthiness, making the system suitable for entomological research to accelerate biodiversity characterization in an under\-described parasitoid superfamily.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.16351v1)

---


## Automatic Characterization of Mid\-latitude Multiple Ionospheric Plasma Structures from All\-sky Airglow Images using Deep Learning Technique / 

发布日期：2026-03-16

作者：Jeevan Upadhyaya

摘要：The F\-region ionospheric plasma structures are propagating high and or low electron density regions in the Earth ionosphere. These plasma structures can be observed using ground based all\-sky airglow imagers which can capture faint airglow emissions originating from the F\-region of ionosphere. This study introduces a novel automatic method for determining the propagation parameters \(horizontal velocity and orientation\) of these multiple ionospheric plasma structures observed in O\(1D\) 630.0 nm all\-sky airglow images from Hanle, India located in the mid\-latitude region. We have used a deep learning\-based segmentation model called YOLOv8 \(You Only Look Once\) to localize and BoT\-SORT tracker to track individual mid\-latitude ionospheric plasma structures. Three different automatic algorithms are used to characterize the observed plasma structures utilizing the segmented outputs from the YOLO model. Finally, an additional quality control step is introduced that filters the results from the three automatic algorithms and generates a flag to retain the most reliable estimate. The results of the proposed fully automated pipeline are systematically compared with a previously developed semi\-automatic approach to assess the estimation efficacy. The automatic technique developed in this study is particularly valuable for all\-sky airglow imaging systems having large datasets, where manual intervention or semi\-automatic analysis is impractical.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.15333v1)

---


## Direct Object\-Level Reconstruction via Probabilistic Gaussian Splatting / 

发布日期：2026-03-15

作者：Shuai Guo

摘要：Object\-level 3D reconstruction play important roles across domains such as cultural heritage digitization, industrial manufacturing, and virtual reality. However, existing Gaussian Splatting\-based approaches generally rely on full\-scene reconstruction, in which substantial redundant background information is introduced, leading to increased computational and storage overhead. To address this limitation, we propose an efficient single\-object 3D reconstruction method based on 2D Gaussian Splatting. By directly integrating foreground\-background probability cues into Gaussian primitives and dynamically pruning low\-probability Gaussians during training, the proposed method fundamentally focuses on an object of interest and improves the memory and computational efficiency. Our pipeline leverages probability masks generated by YOLO and SAM to supervise probabilistic Gaussian attributes, replacing binary masks with continuous probability values to mitigate boundary ambiguity. Additionally, we propose a dual\-stage filtering strategy for training's startup to suppress background Gaussians. And, during training, rendered probability masks are conversely employed to refine supervision and enhance boundary consistency across views. Experiments conducted on the MIP\-360, T&T, and NVOS datasets demonstrate that our method exhibits strong self\-correction capability in the presence of mask errors and achieves reconstruction quality comparable to standard 3DGS approaches, while requiring only approximately 1/10 of their Gaussian amount. These results validate the efficiency and robustness of our method for single\-object reconstruction and highlight its potential for applications requiring both high fidelity and computational efficiency.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.14316v1)

---


## TDMM\-LM: Bridging Facial Understanding and Animation via Language Models / 

发布日期：2026-03-14

作者：Luchuan Song

摘要：Text\-guided human body animation has advanced rapidly, yet facial animation lags due to the scarcity of well\-annotated, text\-paired facial corpora. To close this gap, we leverage foundation generative models to synthesize a large, balanced corpus of facial behavior. We design prompts suite covering emotions and head motions, generate about 80 hours of facial videos with multiple generators, and fit per\-frame 3D facial parameters, yielding large\-scale \(prompt and parameter\) pairs for training. Building on this dataset, we probe language models for bidirectional competence over facial motion via two complementary tasks: \(1\) Motion2Language: given a sequence of 3D facial parameters, the model produces natural\-language descriptions capturing content, style, and dynamics; and \(2\) Language2Motion: given a prompt, the model synthesizes the corresponding sequence of 3D facial parameters via quantized motion tokens for downstream animation. Extensive experiments show that in this setting language models can both interpret and synthesize facial motion with strong generalization. To best of our knowledge, this is the first work to cast facial\-parameter modeling as a language problem, establishing a unified path for text\-conditioned facial animation and motion understanding.

中文摘要：


代码链接：摘要中未找到代码链接。

论文链接：[阅读更多](http://arxiv.org/abs/2603.16936v1)

---

