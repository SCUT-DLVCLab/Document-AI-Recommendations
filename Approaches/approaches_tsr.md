<h3 align="center"> Deep Learning Approaches for </h3>
<h1 align="center"> Table Structure Recognition </h1>

<h2> Introduction </h2>

Information in tabular format is prevalent in all sorts of documents. Compared to natural language, tables provide a way to summarize large quantities of data in a more compact and structured format. Tables provide as well a format to assist readers with finding and comparing information.

Tables in documents are typically formatted for human understanding, and humans are generally adept at parsing table structure, identifying table headers, and interpreting relations between table cells. However, it is challenging for a machine to understand tabular data in unstructured formats (e.g. PDF, images) due to the large variability in their layout and style. The key step of table understanding is to represent the unstructured tables in a machine-readable format, where the structure of the table and the content within each cell are encoded according to a pre-defined standard. This is often referred as table stucture recognition(TSR).

---

<h2> Table of Contents </h2>

- [Component-based Methods](#Component-based-Methods)
  - [TableGraph](#TableGraph)
  - [NCGM](#NCGM)
  - [FLAG-Net](#FLAG-Net)
  - [TGRNet](#TGRNet)
  - [ReS2TIM](#ReS2TIM)
  - [GFTE](#GFTE)
  - [Shah Rukh Qasim DGCNN](#Shah-Rukh-Qasim-DGCNN)
  - [GraphTSR](#GraphTSR)
  - [Cycle-CenterNet](#Cycle-CenterNet)
  - [LGPMA](#LGPMA)
  - [CascadeTabNet](#CascadeTabNet)
- [Sequence-based Methods](#Sequence-based-Methods)
  - [TableFormer](#TableFormer)
  - [TableMaster](#TableMaster)
  - [EDD](#EDD)
- [Splitting-based Methods](#Splitting-based-Methods)
  - [TSRFormer](#TSRFormer)
  - [RobusTabNet](#RobusTabNet)
  - [TRUST](#TRUST)
  - [SEM](#SEM)
  - [Yibo Li BiLSTM](#Yibo-Li-BiLSTM)
  - [SPLERGE](#SPLERGE)

---
---

<br>

# Component-based Methods

## TableGraph 

*Hainan Chen et al. TableGraph: An Image Segmentation–Based Table Knowledge Interpretation Model for Civil and Construction Inspection Documentation. ASCE, 2022*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://ascelibrary.org/doi/abs/10.1061/%28ASCE%29CO.1943-7862.0002346?af=R">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ASCE-brightgreen"></img>
  </a> 
</p>

- **Highlights**: Solid Work
- **Abstract**: There are many manuals and codes to normalize each procedure in civil and construction engineering projects. Data tables in the codes offer various references and are playing a more and more valuable role in knowledge management. However, research has focused on regular table structure detection. For nonconventional tables— especially for nested tables—there is no efficient way to conduct automatic interpretation. In this paper, an automatic table knowledge interpretation model (TableGraph) is proposed to automatically extract table data from table images and then transform the table data into table cell graphs to facilitate table information querying. TableGraph considers that a table image is composed of three types of semantic pixel classes: background, table border, and table cell contents. Because TableGraph only considers pixel semantic meaning rather than structural rules or form features, it can handle nonconventional and complex nested table situations.
In addition, a cross-hit algorithm was designed to enable fast content queries on the generated table cell graphs. Validation of a real case of automatic interpretation of inspection manual table data is presented. The results show that the proposed TableGraph model can interpret the structure and contents of table images. 

---

##  NCGM
*Hao Liu et al. Neural Collaborative Graph Machines for Table Structure Recognition. CVPR, 2022*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Neural_Collaborative_Graph_Machines_for_Table_Structure_Recognition_CVPR_2022_paper.html">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-CVF-brightgreen"></img>
  </a>
</p>

- **Highlights**: Multi-modality
- **Abstract**: Recently, table structure recognition has achieved impressive progress with the help of deep graph models. Most of them exploit single visual cues of tabular elements or simply combine visual cues with other modalities via early fusion to reason their graph relationships. However, neither early fusion nor individually reasoning in terms of multiple modalities can be appropriate for all varieties of table structures with great diversity. Instead, different modalities are expected to collaborate with each other in different patterns for different table cases. In the community, the importance of intra-inter modality interactions for table structure reasoning is still unexplored. In this paper, we define it as heterogeneous table structure recognition (Hetero-TSR) problem. With the aim of filling this gap, we present a novel Neural Collaborative Graph Machines (NCGM) equipped with stacked collaborative blocks, which alternatively extracts intra-modality context and models inter-modality interactions in a hierarchical way. It can represent the intrainter modality relationships of tabular elements more robustly, which significantly improves the recognition performance. We also show that the proposed NCGM can modulate collaborative pattern of different modalities conditioned on the context of intra-modality cues, which is vital for diversified table cases. Experimental results on benchmarks demonstrate our proposed NCGM achieves state-ofthe-art performance and beats other contemporary methods by a large margin especially under challenging scenarios.

---

##  FLAG-Net

*Hao Liu et al. Show, Read and Reason: Table Structure Recognition with Flexible Context Aggregator. ACMMM, 2021*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://dl.acm.org/doi/abs/10.1145/3474085.3481534">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACM-brightgreen"></img>
  </a>
</p>

- **Highlights**: Flexible Context Aggregation
- **Abstract**: We investigate the challenging problem of table structure recognition in this work. Many recent methods adopt graph-based context aggregator with strong inductive bias to reason sparse contextual relationships of table elements. However, the strong constraints may be too restrictive to represent the complicated table relationships.
In order to learn more appropriate inductive bias from data, we try to introduce Transformer as context aggregator in this work. Nevertheless, Transformer taking dense context as input requires larger scale data and may suffer from unstable training procedure due to the weakening of inductive bias. To overcome the above limitations, we in this paper design a FLAG (FLexible context AGgregator), which marries Transformer with graph-based context aggregator in an adaptive way. Based on FLAG, an end-to-end framework requiring no extra meta-data or OCR information, termed FLAG-Net, is proposed to flexibly modulate the aggregation of dense context and sparse one for the relational reasoning of table elements. We investigate the modulation pattern in FLAG and show what contextual information is focused, which is vital for recognizing table structure. Extensive experimental results on benchmarks demonstrate the performance of our proposed FLAG-Net surpasses other compared methods by a large margin.

---

##  TGRNet

*Wenyuan Xue et al. TGRNet: A Table Graph Reconstruction Network for Table Structure Recognition. ICCV, 2021*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://openaccess.thecvf.com/content/ICCV2021/html/Xue_TGRNet_A_Table_Graph_Reconstruction_Network_for_Table_Structure_Recognition_ICCV_2021_paper.html">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-CVF-brightgreen"></img>
  </a>
  <a href="https://github.com/xuewenyuan/TGRNet">
  <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Solid Work
- **Abstract**: A table arranging data in rows and columns is a very effective data structure, which has been widely used in business and scientific research. Considering large-scale tabular data in online and offline documents, automatic table recognition has attracted increasing attention from the document analysis community. Though human can easily understand the structure of tables, it remains a challenge for machines to understand that, especially due to a variety of different table layouts and styles. Existing methods usually model a table as either the markup sequence or the adjacency matrix between different table cells, failing to address the importance of the logical location of table cells, e.g., a cell is located in the first row and the second column of the table. In this paper, we reformulate the problem of table structure recognition as the table graph reconstruction, and propose an end-to-end trainable table graph reconstruction network (TGRNet) for table structure recognition.
Specifically, the proposed method has two main branches, a cell detection branch and a cell logical location branch, to jointly predict the spatial location and the logical location of different cells. Experimental results on three popular table recognition datasets and a new dataset with table graph annotations (TableGraph-350K) demonstrate the effectiveness of the proposed TGRNet for table structure recognition. 

---

##  ReS2TIM

*Wenyuan Xue et al. ReS2TIM: Reconstruct Syntactic Structures from Table Images. ICDAR, 2019*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2019-orange"></img>
  <a href="https://ieeexplore.ieee.org/abstract/document/8978027">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-IEEE-brightgreen"></img>
  </a>
</p>

- **Highlights**: Syntactic Representation; Distance-based Sample Weight
- **Abstract**: Tables often represent densely packed but structured data. Understanding table semantics is vital for effective information retrieval and data mining. Unlike web tables, whose semantics are readable directly from markup language and contents, the full analysis of tables published as images requires the conversion of discrete data into structured information. This paper presents a novel framework to convert a table image into its syntactic representation through the relationships between its cells. In order to reconstruct the syntactic structures of a table, we build a cell relationship network to predict the neighbors of each cell in four directions. During the training stage, a distance-based sample weight is proposed to handle the class imbalance problem. According to the detected relationships, the table is represented by a weighted graph that is then employed to infer the basic syntactic table structure.
Experimental evaluation of the proposed framework using two datasets demonstrates the effectiveness of our model for cell relationship detection and table structure inference.

---

##  GFTE

*Yiren Li et al. GFTE: Graph-based Financial Table Extraction. ICPR, 2021*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://arxiv.org/abs/2003.07560">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/Irene323/GFTE">
  <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Strong Baseline
- **Abstract**: Tabular data is a crucial form of information expression, which can organize data in a standard structure for easy information retrieval and comparison. However, in financial industry and many other fields tables are often disclosed in unstructured digital files, e.g. Portable Document Format (PDF) and images, which are difficult to be extracted directly. In this paper, to facilitate deep learning based table extraction from unstructured digital files, we publish a standard Chinese dataset named FinTab, which contains more than 1,600 financial tables of diverse kinds and their corresponding structure representation in JSON. In addition, we propose a novel graph-based convolutional neural network model named GFTE as a baseline for future comparison. GFTE integrates image feature, position feature and textual feature together for precise edge prediction and reaches overall good results.

---

##  Shah Rukh Qasim DGCNN

*Shah Rukh Qasim et al. Rethinking Table Recognition using Graph Neural Networks. ICDAR, 2019*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2019-orange"></img>
  <a href="https://ieeexplore.ieee.org/abstract/document/8978070">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-IEEE-brightgreen"></img>
  </a>
  <a href="https://github.com/shahrukhqasim/TIES-2.0">
  <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Graph Neural Networks
- **Abstract**: Document structure analysis, such as zone segmentation and table recognition, is a complex problem in document processing and is an active area of research. The recent success of deep learning in solving various computer vision and machine learning problems has not been reflected in document structure analysis since conventional neural networks are not well suited to the input structure of the problem. In this paper, we propose an architecture based on graph networks as a better alternative to standard neural networks for table recognition. We argue that graph networks are a more natural choice for these problems, and explore two gradient-based graph neural networks. Our proposed architecture combines the benefits of convolutional neural networks for visual feature extraction and graph networks for dealing with the problem structure. We empirically demonstrate that our method outperforms the baseline by a significant margin.
In addition, we identify the lack of large scale datasets as a major hindrance for deep learning research for structure analysis and present a new large scale synthetic dataset for the problem of table recognition. Finally, we open-source our implementation of dataset generation and the training framework of our graph networks to promote reproducible research in this direction.

---

##  GraphTSR

*Zewen Chi et al. Complicated Table Structure Recognition. arXiv preprint, 2019*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2019-orange"></img>
  <a href="https://arxiv.org/abs/1908.04729">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Strong Baseline
- **Abstract**: The task of table structure recognition aims to recognize the internal structure of a table, which is a key step to make machines understand tables. Currently, there are lots of studies on this task for different file formats such as ASCII text and HTML. It also attracts lots of attention to recognize the table structures in PDF files. However, it is hard for the existing methods to accurately recognize the structure of complicated tables in PDF files. The complicated tables contain spanning cells which occupy at least two columns or rows. To address the issue, we propose a novel graph neural network for recognizing the table structure in PDF files, named GraphTSR. Specifically, it takes table cells as input, and then recognizes the table structures by predicting relations among cells. Moreover, to evaluate the task better, we construct a large-scale table structure recognition dataset from scientific papers, named SciTSR, which contains 15,000 tables from PDF files and their corresponding structure labels. Extensive experiments demonstrate that our proposed model is highly effective for complicated tables and outperforms state-of-the-art baselines over a benchmark dataset and our new constructed dataset.

---

##  Cycle-CenterNet

*Rujiao Long et al. Parsing Table Structures in the Wild. ICCV, 2021*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="http://openaccess.thecvf.com/content/ICCV2021/html/Long_Parsing_Table_Structures_in_the_Wild_ICCV_2021_paper.html">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-CVF-brightgreen"></img>
  </a>
</p>

- **Highlights**: Strong Baseline
- **Abstract**: This paper tackles the problem of table structure parsing (TSP) from images in the wild. In contrast to existing studies that mainly focus on parsing well-aligned tabular images with simple layouts from scanned PDF documents, we aim to establish a practical table structure parsing system for real-world scenarios where tabular input images are taken or scanned with severe deformation, bending or occlusions. For designing such a system, we propose an approach named Cycle-CenterNet on the top of CenterNet with a novel cycle-pairing module to simultaneously detect and group tabular cells into structured tables. In the cyclepairing module, a new pairing loss function is proposed for the network training. Alongside with our Cycle-CenterNet, we also present a large-scale dataset, named Wired Table in the Wild (WTW), which includes well-annotated structure parsing of multiple style tables in several scenes like photo, scanning files, web pages, etc.. In experiments, we demonstrate that our Cycle-CenterNet consistently achieves the best accuracy of table structure parsing on the new WTW dataset by 24.6% absolute improvement evaluated by the TEDS metric. A more comprehensive experimental analysis also validates the advantages of our proposed methods for the TSP task.

---

##  LGPMA

*Liang Qiao et al. LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask Alignment. ICDAR, 2021*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://arxiv.org/abs/2105.06224">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/hikopensource/DAVAR-Lab-OCR/tree/main/demo/table_recognition/lgpma">
  <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**:  Soft Pyramid Mask
- **Abstract**: Table structure recognition is a challenging task due to the various structures and complicated cell spanning relations. Previous methods handled the problem starting from elements in different granularities (rows/columns, text regions), which somehow fell into the issues like lossy heuristic rules or neglect of empty cell division. Based on table structure characteristics, we find that obtaining the aligned bounding boxes of text region can effectively maintain the entire relevant range of different cells. However, the aligned bounding boxes are hard to be accurately predicted due to the visual ambiguities. In this paper, we aim to obtain more reliable aligned bounding boxes by fully utilizing the visual information from both text regions in proposed local features and cell relations in global features. Specifically, we propose the framework of Local and Global Pyramid Mask Alignment, which adopts the soft pyramid mask learning mechanism in both the local and global feature maps.
It allows the predicted boundaries of bounding boxes to break through the limitation of original proposals. A pyramid mask re-scoring module is then integrated to compromise the local and global information and refine the predicted boundaries. Finally, we propose a robust table structure recovery pipeline to obtain the final structure, in which we also effectively solve the problems of empty cells locating and division.
Experimental results show that the proposed method achieves competitive and even new state-of-the-art performance on several public benchmarks.

---

##  CascadeTabNet

*Devashish Prasad et al. CascadeTabNet: An approach for end to end table detection and structure recognition from image-based documents. CVPR, 2020*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2020-orange"></img>
  <a href="http://openaccess.thecvf.com/content_CVPRW_2020/html/w34/Prasad_CascadeTabNet_An_Approach_for_End_to_End_Table_Detection_and_CVPRW_2020_paper.html">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-CVF-brightgreen"></img>
  </a>
  <a href="https://github.com/DevashishPrasad/CascadeTabNet">
  <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: End-to-End; Transfer Learning; Image Augmentation 
- **Abstract**: An automatic table recognition method for interpretation of tabular data in document images majorly involves solving two problems of table detection and table structure recognition. The prior work involved solving both problems independently using two separate approaches. More recent works signify the use of deep learning-based solutions while also attempting to design an end to end solution. In this paper, we present an improved deep learning-based end to end approach for solving both problems of table detection and structure recognition using a single Convolution Neural Network (CNN) model. We propose CascadeTabNet: a Cascade mask Region-based CNN High-Resolution Network (Cascade mask R-CNN HRNet) based model that detects the regions of tables and recognizes the structural body cells from the detected tables at the same time. We evaluate our results on ICDAR 2013, ICDAR 2019 and TableBank public datasets. We achieved 3rd rank in ICDAR 2019 post-competition results for table detection while attaining the best accuracy results for the ICDAR 2013 and TableBank dataset. We also attain the highest accuracy results on the ICDAR 2019 table structure recognition dataset. Additionally, we demonstrate effective transfer learning and image augmentation techniques that enable CNNs to achieve very accurate table detection results.

---

<br>

# Sequence-based Methods

##  TableFormer

*Ahmed Nassar et al. TableFormer: Table Structure Understanding with Transformers. CVPR, 2022*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://openaccess.thecvf.com/content/CVPR2022/html/Nassar_TableFormer_Table_Structure_Understanding_With_Transformers_CVPR_2022_paper.html">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-CVF-brightgreen"></img>
  </a>
</p>

- **Highlights**: Strong Baseline; 
- **Abstract**: Tables organize valuable content in a concise and compact representation. This content is extremely valuable for systems such as search engines, Knowledge Graph’s, etc, since they enhance their predictive capabilities. Unfortunately, tables come in a large variety of shapes and sizes.
Furthermore, they can have complex column/row-header configurations, multiline rows, different variety of separation lines, missing entries, etc. As such, the correct identification of the table-structure from an image is a nontrivial task. In this paper, we present a new table-structure identification model. The latter improves the latest end-toend deep learning model (i.e. encoder-dual-decoder from PubTabNet) in two significant ways. First, we introduce a new object detection decoder for table-cells. In this way, we can obtain the content of the table-cells from programmatic PDF’s directly from the PDF source and avoid the training of the custom OCR decoders. This architectural change leads to more accurate table-content extraction and allows us to tackle non-english tables. Second, we replace the LSTM decoders with transformer based decoders. This upgrade improves significantly the previous state-of-the-art tree-editing-distance-score (TEDS) from 91% to 98.5% on simple tables and from 88.7% to 95% on complex tables.

---

##  TableMaster

*Jiaquan Ye et al. PINGAN-VCGROUP’S SOLUTION FOR ICDAR 2021 COMPETITION ON SCIENTIFIC LITERATURE PARSING TASK B: TABLE RECOGNITION TO HTML. ICDAR, 2021*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://arxiv.org/abs/2105.01848">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
    <a href="https://github.com/JiaquanYe/TableMASTER-mmocr">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Solid Work; Introduce Transformer to Sequence-based Method
- **Abstract**: This paper presents our solution for ICDAR 2021 competition on scientific literature parsing task B: table recognition to HTML. In our method, we divide the table content recognition task into four sub-tasks: table structure recognition, text line detection, text line recognition, and box assignment.
Our table structure recognition algorithm is customized based on MASTER, a robust image text recognition algorithm. PSENet is used to detect each text line in the table image. For text line recognition, our model is also built on MASTER. Finally, in the box assignment phase, we associated the text boxes detected by PSENet with the structure item reconstructed by table structure prediction, and fill the recognized content of the text line into the corresponding item. Our proposed method achieves a 96.84% TEDS score on 9,115 validation samples in the development phase, and a 96.32% TEDS score on 9,064 samples in the final evaluation phase.

---

##  EDD

*Xu Zhong et al. Image-based table recognition: data, model, and evaluation. ECCV, 2020*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2020-orange"></img>
  <a href="https://link.springer.com/chapter/10.1007/978-3-030-58589-1_34">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-Springer-brightgreen"></img>
  </a>
  </a>
    <a href="https://github.com/Line290/EDD-third-party">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-UnOfficial-blue"></img>
  </a>
</p>

- **Highlights**: First Sequence-based Method; New Metric
- **Abstract**: Important information that relates to a specific topic in a document is often organized in tabular format to assist readers with information retrieval and comparison, which may be difficult to provide in natural language. However, tabular data in unstructured digital documents, e.g. Portable Document Format (PDF) and images, are difficult to parse into structured machine-readable format, due to complexity and diversity in their structure and style. To facilitate image-based table recognition with deep learning, we develop and release the largest publicly available table recognition dataset PubTabNet1, containing 568k table images with corresponding structured HTML representation. PubTabNet is automatically generated by matching the XML and PDF representations of the scientific articles in PubMed CentralTM Open Access Subset (PMCOA). We also propose a novel attention-based encoder-dual-decoder (EDD) architecture that converts images of tables into HTML code.
The model has a structure decoder which reconstructs the table structure and helps the cell decoder to recognize cell content. In addition, we propose a new Tree-Edit-Distance-based Similarity (TEDS) metric for table recognition, which more appropriately captures multi-hop cell misalignment and OCR errors than the pre-established metric. The experiments demonstrate that the EDD model can accurately recognize complex tables solely relying on the image representation, outperforming the state-of-the-art by 9.7% absolute TEDS score.

---

<br>

# Splitting-based Methods

##  TSRFormer

*Weihong Lin et al. TSRFormer: Table Structure Recognition with Transformers. ACMMM, 2022*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://dl.acm.org/doi/abs/10.1145/3503161.3548038">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACM-brightgreen"></img>
  </a>
</p>


- **Highlights**: Formulate Table Separation Line Prediction As Line Regression Problem
- **Abstract**: We present a new table structure recognition (TSR) approach, called TSRFormer, to robustly recognizing the structures of complex tables with geometrical distortions from various table images. Unlike previous methods, we formulate table separation line prediction as a line regression problem instead of an image segmentation problem and propose a new two-stage DETR based separator prediction approach, dubbed Separator REgression TRansformer (SepRETR), to predict separation lines from table images directly. To make the two-stage DETR framework work efficiently and effectively for the separation line prediction task, we propose two improvements: 1) A prior-enhanced matching strategy to solve the slow convergence issue of DETR; 2) A new cross attention module to sample features from a high-resolution convolutional feature map directly so that high localization accuracy is achieved with low computational cost.
After separation line prediction, a simple relation network based cell merging module is used to recover spanning cells. With these new techniques, our TSRFormer achieves state-of-the-art performance on several benchmark datasets, including SciTSR, PubTabNet and WTW. Furthermore, we have validated the robustness of our approach to tables with complex structures, borderless cells, large blank spaces, empty or spanning cells as well as distorted or even curved shapes on a more challenging real-world in-house dataset.

---

##  RobusTabNet

*Chixiang Ma et al. Robust Table Detection and Structure Recognition from Heterogeneous Document Images. PR, 2023*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://www.sciencedirect.com/science/article/abs/pii/S0031320322004861">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-Elsevier-brightgreen"></img>
  </a>
</p>


- **Highlights**:  Spatial CNN; Grid CNN
- **Abstract**: We introduce a new table detection and structure recognition approach named RobusTabNet to detect the boundaries of tables and reconstruct the cellular structure of each table from heterogeneous document images. For table detection, we propose to use CornerNet as a new region proposal network to generate higher quality table proposals for Faster RCNN, which has significantly improved the localization accuracy of Faster R-CNN for table detection. Consequently, our table detection approach achieves state-of-the-art performance on three public table detection benchmarks, namely cTDaR TrackA, PubLayNet and IIIT-AR-13K, by only using a lightweight ResNet-18 backbone network.
Furthermore, we propose a new split-and-merge based table structure recognition approach, in which a novel spatial CNN based separation line prediction module is proposed to split each detected table into a grid of cells, and a Grid CNN based cell merging module is applied to recover the spanning cells. As the spatial CNN module can effectively propagate contextual information across the whole table image, our table structure recognizer can robustly recognize tables with large blank spaces and geometrically distorted (even curved) tables. Thanks to these two techniques, our table structure recognition approach achieves state-of-the-art performance on three public benchmarks, including SciTSR, PubTabNet and cTDaR TrackB2-Modern. Moreover, we have further demonstrated the advantages of our approach in recognizing tables with complex structures, large blank spaces, as well as geometrically distorted or even curved shapes on a more challenging in-house dataset.

---

##  TRUST

*Zengyuan guo et al. TRUST: An Accurate and End-to-End Table structure Recognizer Using Splitting-based Transformers. preprint arXiv, 2022*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://arxiv.org/abs/2208.14687">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>


- **Highlights**: Query-based Splitting Module; Vertex-based Merging Module
- **Abstract**: Table structure recognition is a crucial part of document image analysis domain. Its difficulty lies in the need to parse the physical coordinates and logical indices of each cell at the same time. However, the existing methods are difficult to achieve both these goals, especially when the table splitting lines are blurred or tilted. In this paper, we propose an accurate and end-to-end transformer-based table structure recognition method, referred to as TRUST. Transformers are suitable for table structure recognition because of their global computations, perfect memory, and parallel computation. By introducing novel Transformer-based Query-based Splitting Module and Vertexbased Merging Module, the table structure recognition problem is decoupled into two joint optimization sub-tasks: multi-oriented table row/column splitting and table grid merging. The Query-based Splitting Module learns strong context information from long dependencies via Transformer networks, accurately predicts the multi-oriented table row/column separators, and obtains the basic grids of the table accordingly. The Vertex-based Merging Module is capable of aggregating local contextual information between adjacent basic grids, providing the ability to merge basic girds that belong to the same spanning cell accurately. We conduct experiments on several popular benchmarks including PubTabNet and SynthTable, our method achieves new state-of-the-art results. In particular, TRUST runs at 10 FPS on PubTabNet, surpassing the previous methods by a large margin.

---

##  SEM

*Zhenrong Zhang et al. Split, Embed and Merge: An accurate table structure recognizer. preprint arXiv, 2022*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://www.sciencedirect.com/science/article/abs/pii/S0031320322000462">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-Elsevier-brightgreen"></img>
  </a>
</p>


- **Highlights**:  Jointly Modeling Visual and Textual Information; 
- **Abstract**: Table structure recognition is an essential part for making machines understand tables. Its main task is to recognize the internal structure of a table. However, due to the complexity and diversity in their structure and style, it is very difficult to parse the tabular data into the structured format which machines can understand, especially for complex tables. In this paper, we introduce Split, Embed and Merge (SEM), an accurate table structure recognizer. SEM is mainly composed of three parts, splitter, embedder and merger. In the first stage, we apply the splitter to predict the potential regions of the table row/column separators, and obtain the fine grid structure of the table. In the second stage, by taking a full consideration of the textual information in the table, we fuse the output features for each table grid from both vision and text modalities.
Moreover, we achieve a higher precision in our experiments through providing additional textual features. Finally, we process the merging of these basic table grids in a self-regression manner. The corresponding merging results are learned through the attention mechanism. In our experiments, SEM achieves an average F1-Measure of 97:11% on the SciTSR dataset which outperforms other methods by a large margin. We also won the first place of complex tables and third place of all tables in Task-B of ICDAR 2021 Competition on Scientific Literature Parsing. Extensive experiments on other publicly available datasets further demonstrate the effectiveness of our proposed approach.

---

##  Yibo Li BiLSTM

*Yibo Li et al. Rethinking Table Structure Recognition Using Sequence Labeling Methods. ICDAR, 2021*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://link.springer.com/chapter/10.1007/978-3-030-86331-9_35">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-Springer-brightgreen"></img>
  </a>
  </a>
    <a href="https://github.com/L597383845/row-col-table-recognition">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>


- **Highlights**:  Sequence Labeling Model
- **Abstract**: Table structure recognition is an important task in document analysis and attracts the attention of many researchers. However, due to the diversity of table types and the complexity of table structure, the performances of table structure recognition methods are still not well enough in practice. Row and column separators play a significant role in the two-stage table structure recognition and a better row and column separator segmentation result can improve the final recognition results.
Therefore, in this paper, we present a novel deep learning model to detect row and column separators. This model contains a convolution encoder and two parallel row and column decoders. The encoder can extract the visual features by using convolution blocks; the decoder formulates the feature map as a sequence and uses a sequence labeling model, bidirectional long short-term memory networks (BiLSTM) to detect row and column separators. Experiments have been conducted on PubTabNet and the model is benchmarked on several available datasets, including PubTabNet, UNLV ICDAR13, ICDAR19. The results show that our model has a state-of-the-art performance than other strong models. In addition, our model shows a better generalization ability. 

---

##  SPLERGE

*Chris Tensmeyer et al. Deep Splitting and Merging for Table Structure Decomposition. ICDAR, 2019*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2019-orange"></img>
  <a href="https://ieeexplore.ieee.org/abstract/document/8977975">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-IEEE-brightgreen"></img>
  </a>
</p>

- **Highlights**: First Splitting-based Method
- **Abstract**: Given the large variety and complexity of tables, table structure extraction is a challenging task in automated document analysis systems. We present a pair of novel deep learning models (Split and Merge models) that given an input image, 1) predicts the basic table grid pattern and 2) predicts which grid elements should be merged to recover cells that span multiple rows or columns. We propose projection pooling as a novel component of the Split model and grid pooling as a novel part of the Merge model. While most Fully Convolutional Networks rely on local evidence, these unique pooling regions allow our models to take advantage of the global table structure. We achieve state-of-the-art performance on the public ICDAR 2013 Table Competition dataset of PDF documents. On a much larger private dataset which we used to train the models, we significantly outperform both a state-ofthe-art deep model and a major commercial software system.
