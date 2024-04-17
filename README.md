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

This repository is a comprehensive collection of information in the field of Document AI. It includes various resources like [algorithms](Approaches/approaches.md), [datasets](Datasets/datasets.md), and [performance comparisons](SOTAs/sotas.md).

Document AI algorithms are designed to read, analyze, and understand documents. The research fields encompass Visual Information Extraction, Table Structure Recognition, Layout Analysis, Document Classification, Document VQA, etc. In recent years, researchers have proposed numerous deep learning-based methods that have demonstrated outstanding performance and efficiency, making them widely applicable in real-world scenarios. We have conducted an extensive survey of these algorithms/benchmarks and create this repository. We sincerely hope that our efforts contribute to the advancement of the field and prove beneficial to others.


> If you come across any errors or have suggestions for additional related papers, please do not hesitate to contact us. We highly encourage and welcome pull requests to enhance the repository. Your contributions are greatly appreciated.

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

With the rapid development of Internet technology and the growing demands of information exchange, a vast amount of documents are being digitalized, stored, and distributed in the form of images. Various application scenarios, including receipt understanding, card recognition, automatic paper scoring, and document matching, are concerned on how to extract key information from document images. The process, known as visual information extraction (VIE), focuses on **mining, analyzing, and extracting key entities present in visually-rich documents**. For instance, when provided with an image of a receipt, VIE algorithms can identify and extract information such as store name, product details, prices, and more. Similarly, for documents like forms, VIE algorithms can identify and extract the key-value pairs.

Visual information extraction can be seen as the promotion and extension of named entity recognition(NER) and entity linking(EL) to the field of visually rich documents. Different from the information extraction in plain texts, results of VIE not only depends on texts, but also closely relates to the document layout, font style, block color, figures, charts and other components.

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

Tabular data have been widely used to help people manage and extract important information in many real-world scenarios, including the analysis of financial documents, air pollution indices, and electronic medical records. While humans can easily comprehend tables with different layouts and styles, it remains a significant challenge for machines to automatically identify the structure of diverse tables. Considering the massive amount of tabular data presented in unstructured formats (e.g., image and PDF files) and that most table analysis methods focus on semi-structured tables (e.g., CSV files), the community will significantly benefit from an automatic table recognition system, facilitating large-scale tabular data analysis such as table parsing, patient treatment prediction, and credit card fraud detection.

TSR aims to **recognize the cellular structures of tables from table images by extracting the coordinates of cell boxes and row/column spanning information.** This task poses significant difficulties due to the complexity, diverse styles, and contents of tables, as well as potential geometric distortions or curvature that may occur during the image capturing process.

<br>
<br>

<h2> Acknowledgment </h2>

Some of the contents in this repository refers to the paper ***Lei Cui, Yiheng Xu, Tengchao Lv and Furu Wei. Document AI: Benchmarks, Models and Applications. DIL workshop, ICDAR 2021.*** It provides comprehensive reference for researchers in related fields. Much thanks for their excellent work.

