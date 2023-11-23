<h1 align="center"> Document AI Recommendations </h1>

<p align="center"><font size=4>Everything about Document AI</font></p>

<p align="center">
   <strong><a href="./Approaches/approaches.md"> Approaches </a></strong> •
   <strong><a href="./Datasets/datasets.md"> Datasets </a></strong> •
   <strong><a href="./SOTAs/sotas.md"> SOTAs </a></strong>
</p>

<p align="center">
   <strong><a href="#vie">Visual Information Extraction</a></strong> •
   <strong><a href="#tsr">Table Structure Recognition</a></strong> •
   <strong>Layout Analysis</a></strong>
</p>


<br>
<br>

<h2> Introduction </h2>

This repository collects all information in the field of Document AI, including [algorithms](Approaches/approaches.md), [datasets](Datasets/datasets.md), [performance comparisons](SOTAs/sotas.md), and so on.

Document AI algorithms aims an reading, analyzing and understanding documents. Research fields include Visual Information Extraction, Table Structure Recognition, Layout Analysis, Document Classification, Document VQA, etc. In recent years, researchers have proposed many methods based on deep learning with outstanding performance and high efficiency, which have been widely applied in real scenarios. We extensively survey the deep-learning-based algorithms and public datasets proposed in recent years and build this reposity. Hope our work helps.


> If you found errors or want to add more related papers, please feel free to contact us. Any pull requests are welcome.

<br>
<br>


<h2 id="vie" align="center"> Visual Information Extraction </h2>

<p align="center">
  <a href="./Approaches/approaches_vie.md">
    <img alt="approaches-vie" src="https://img.shields.io/badge/Approaches-purple"></img>
  </a> 
  <a href="./Datasets/datasets_vie.md">
    <img alt="datasets-vie" src="https://img.shields.io/badge/Datasets-blue"></img>
  </a>
  <a href="./SOTAs/sotas_vie.md">
    <img alt="sotas-vie" src="https://img.shields.io/badge/SOTAs-green"></img>
  </a>
</p>

With the rapid development of Internet technology and the increasing needs of information exchange, quantities of documents are digitalized, stored and distributed in the form of images. Numerous application scenarios, such as receipt understanding, card recognition, automatic paper scoring and document matching, are concerned on how to obtain key information from document images. The process is called visual information extraction (VIE), which aims at **mining, analyzing, and extracting key entities contained in visually rich documents**. For example, given an image of a receipt, the VIE algorithms will tell information such as store name, product details, price, etc. For documents like forms, VIE algorithms will tell the key-value pairs contained.

Visual information extraction can be seen as the promotion and extension of named entity recognition(NER) and entity linking(EL) to the field of visually rich documents. Different from the information extraction in traditional natural language processing, results of VIE not only depends on texts, but also closely relates to the doucment layout, font style, block color, figures, charts and other components.

<br>

<h2 id="tsr" align="center"> Table Structure Recognition </h2>

<p align="center">
  <a href="./Approaches/approaches_tsr.md">
    <img alt="approaches-tsr" src="https://img.shields.io/badge/Approaches-purple"></img>
  </a> 
  <a href="./Datasets/datasets_tsr.md">
    <img alt="datasets-tsr" src="https://img.shields.io/badge/Datasets-blue"></img>
  </a>
  <a href="./SOTAs/sotas_tsr.md">
    <img alt="sotas-tsr" src="https://img.shields.io/badge/SOTAs-green"></img>
  </a>
</p>

Tabular data have been widely used to help people manage and extract important information in many real-world scenarios, including the analysis of financial documents, air pollution indices, and electronic medical records. Though human can easily understand tables with different layouts and styles, it remains a great challenge for machines to automatically recognize the structure of various tables. Considering the massive amount of tabular data presented in unstructured formats (e.g., image and PDF files) and that most table analysis methods focus on semi-structured tables (e.g., CSV files), the community will significantly benefit from an automatic table recognition system, facilitating large-scale tabular data analysis such as table parsing, patient treatment prediction, and credit card fraud detection.

TSR aims to **recognize the cellular structures of tables from table images by extracting the coordinates of cell boxes and row/column spanning information.** This task is very challenging since tables may have complex structures, diverse styles and contents, and become geometrically distorted or even curved during an image capturing process.

<br>
<br>

<h2> Acknowledgment </h2>

Some of the contents in this repository refers to the paper ***Lei Cui, Yiheng Xu, Tengchao Lv and Furu Wei. Document AI: Benchmarks, Models and Applications. DIL workshop, ICDAR 2021.*** It provides comprehensive reference for researchers in related fields. Much thanks for their excellent work.

