<h3 align="center"> Datasets for </h3>
<h1 align="center"> Visual Information Extraction </h1>

<h2> 🗒️List of Index </h2>

- [SROIE](#sroie)
- [CORD](#cord)
- [FUNSD](#funsd)
- [XFUND](#xfund)
- [EPHOIE](#ephoie)
- [EATEN](#eaten)
- [WildReceipt](#wildreceipt)
- [Kleister](#kleister)
- [CER-VIR](#cer-vir)
- [DeepForm](#deepform)
- [CER-VIR](#cer-vir-1)
- [SIBR](#sibr)


<br>

# SROIE

<p>
    <a href="https://guillaumejaume.github.io/FUNSD/work/">
        <img alt="License" src="https://img.shields.io/badge/License-CC BY 4.0-c1c1c1"></img>
    </a>
    <img src="https://i.creativecommons.org/l/by/4.0/88x31.png">
    <img align=right alt="Commercial" src="https://img.shields.io/badge/Commercial-✔️-brightgreen"></img>
    <img align=right alt="Adapt" src="https://img.shields.io/badge/Adapt-✔️-brightgreen"></img>
    <img align=right alt="Share" src="https://img.shields.io/badge/Share-✔️-brightgreen"></img>
</p>
<br>

<table align=center>
    <th colspan=3>Number of Samples</th>
    <th>Type</th>
    <th>Language</th>
    <th>Access Link</th>
    <th>Task</th>
    <th>Evaluation Metric</th>
    <tr>
        <td>Train</td>
        <td>Validate</td>
        <td>Test</td>
        <td rowspan=2>Receipt</td>
        <td rowspan=2>English</td>
        <td rowspan=2>
            <p>
                <a href="https://rrc.cvc.uab.es/?ch=13">
                    <img alt="Link2" src="https://img.shields.io/badge/Official-2e8b57"></img>
                </a>
                <br>
                <a href="https://github.com/zzzDavid/ICDAR-2019-SROIE">
                    <img alt="Link2" src="https://img.shields.io/badge/UnOfficial-pink"></img>
                </a> 
            </p>
        </td>
        <td rowspan=2>Entity Extraction</td>
        <td rowspan=2>Entity F1-score</td>
    </tr>
    <tr>
        <td>626</td>
        <td>-</td>
        <td>347</td>
    </tr>
</table>

SROIE is a dataset for the 2019 ICDAR Robust Reading Challenge on Scanned Receipts OCR and Information Extraction competition. It contains 973 samples, 626 for training and 347 for testing. Each receipt contains four kinds of key entities: Company, Address, Date, and Total. 

OCR results and strings of key entities for each sample are provided. To do VIE with token tagging approaches like LayoutLM, the category of each word is required. You may use [rule-based methods](https://github.com/antoinedelplace/Chargrid) or re-label the data manually.

<p align=center>
    <img src="../img/dataset_img/SROIE_1.jpg" width=200>
</p>

It is worth noting that the quality of the data annotation will greatly affect the results of VIE. We launched experiments with [ViBERTgrid](https://github.com/ZeningLin/ViBERTgrid-PyTorch). When the model is trained with high quality annotations (re-labelled manually), the entity F1 can reach 97+. When poor quality annotations (rule-based matching) are used to train the model, the entity F1 is only 60.

<br>

# CORD

<p>
    <a href="https://guillaumejaume.github.io/FUNSD/work/">
        <img alt="License" src="https://img.shields.io/badge/License-CC BY 4.0-c1c1c1"></img>
    </a>
    <img src="https://i.creativecommons.org/l/by/4.0/88x31.png">
    <img align=right alt="Commercial" src="https://img.shields.io/badge/Commercial-✔️-brightgreen"></img>
    <img align=right  alt="Adapt" src="https://img.shields.io/badge/Adapt-✔️-brightgreen"></img>
    <img align=right alt="Share" src="https://img.shields.io/badge/Share-✔️-brightgreen"></img>
</p>

<table align=center>
    <th colspan=3>Number of Samples</th>
    <th>Type</th>
    <th>Language</th>
    <th>Access Link</th>
    <th>Task</th>
    <th>Evaluation Metric</th>
    <tr>
        <td>Train</td>
        <td>Validate</td>
        <td>Test</td>
        <td rowspan=2>Receipt</td>
        <td rowspan=2>English</td>
        <td rowspan=2>
            <p>
                <a href="https://github.com/clovaai/cord">
                    <img alt="Link" src="https://img.shields.io/badge/Official-2e8b57"></img>
                </a>
            </p>
        </td>
        <td rowspan=1>Entity Extraction</td>
        <td rowspan=1>Entity F1-score</td>
    </tr>
    <tr>
        <td>800</td>
        <td>100</td>
        <td>100</td>
        <td rowspan=1>Structure Parsing</td>
        <td rowspan=1>Structured Field F1-score </td>
    </tr>
</table>

CORD is an English receipt dataset proposed by Colva AI. 1000 samples are currently publicly available, 800 for training, 100 for validation, and 100 for testing. The receipt images are obtained through cameras, hence inteference like paper bending and background noise may inevitably occur. The data contains high-quality annotations, key labels for each words and linking between entities are provided. The dataset contains a total of four main key information categories such as payment information, and each main category can be further divided into 30 sub-key fields. Unlike other datasets, entities in CORD are hierarchically related. Models should be able to extract all the structured fields, which makes the task challenging.

<p align=center>
    <img src="../img/dataset_img/CORD_1.png" width=600>
</p>

<br>

# FUNSD

<p>
    <a href="https://guillaumejaume.github.io/FUNSD/work/">
        <img alt="License" src="https://img.shields.io/badge/License-Customized-c1c1c1"></img>
    </a>
    <img align=right alt="Commercial" src="https://img.shields.io/badge/Commercial-✖️-ff0000"></img>
    <img align=right alt="Research" src="https://img.shields.io/badge/Research-✔️-brightgreen"></img>
</p>

<table align=center>
    <th colspan=3>Number of Samples</th>
    <th>Type</th>
    <th>Language</th>
    <th>Access Link</th>
    <th>Task</th>
    <th>Evaluation Metric</th>
    <tr>
        <td>Train</td>
        <td>Validate</td>
        <td>Test</td>
        <td rowspan=2>Forms</td>
        <td rowspan=2>English</td>
        <td rowspan=2>
            <p>
                <a href="https://guillaumejaume.github.io/FUNSD/">
                    <img alt="Link" src="https://img.shields.io/badge/Official-2e8b57"></img>
                </a>
            </p>
        </td>
        <td rowspan=1>Entity Extraction</td>
        <td rowspan=1>Entity F1-score</td>
    </tr>
    <tr>
        <td>149</td>
        <td>-</td>
        <td>50</td>
        <td rowspan=1>Entity Linking</td>
        <td rowspan=1>Pair F1-score </td>
    </tr>
</table>

A dataset for Text Detection, Optical Character Recognition, Spatial Layout Analysis and Form Understanding. Contains 199 fully annotated forms, with 31485 words, 9707 semantic entities and 5304 relations. The OCR result of each text segment and word are given, and the category of each paragraph and linkings between entities are included in the annotations.

<p align=center>
    <img src="../img/dataset_img/FUNSD_1.jpg" width=500>
</p>

<br>

# XFUND

<p>
    <a href="">
        <img alt="License" src="https://img.shields.io/badge/License-CC BY NC SA 4.0-c1c1c1"></img>
    </a>
    <img align=right alt="Commercial" src="https://img.shields.io/badge/Commercial-✖️-ff0000"></img>
    <img align=right alt="Adapt" src="https://img.shields.io/badge/Adapt-✔️-brightgreen"></img>
    <img align=right alt="Share" src="https://img.shields.io/badge/Share-✔️-brightgreen"></img>
</p>

<table align=center>
    <th colspan=3>Number of Samples</th>
    <th>Type</th>
    <th>Language</th>
    <th>Access Link</th>
    <th>Task</th>
    <th>Evaluation Metric</th>
    <tr>
        <td>Train</td>
        <td>Validate</td>
        <td>Test</td>
        <td rowspan=2>Forms</td>
        <td rowspan=2>Chinese, Japanese, Spanish, French, Italian, German, Portuguese</td>
        <td rowspan=2>
            <p>
                <a href="https://github.com/doc-analysis/XFUND">
                    <img alt="Link" src="https://img.shields.io/badge/Official-2e8b57"></img>
                </a>
            </p>
        </td>
        <td rowspan=1>Entity Extraction</td>
        <td rowspan=1>Entity F1-score</td>
    </tr>
    <tr>
        <td>149*7</td>
        <td>-</td>
        <td>50*7</td>
        <td rowspan=1>Entity Linking</td>
        <td rowspan=1>Pair F1-score </td>
    </tr>
</table>

XFUND is a multilingual form understanding benchmark dataset that includes human-labeled forms with key-value pairs in 7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese). It is an extension of the FUNSD dataset, the annotations and evaluation metric are the same as FUNSD.

<p align=center>
    <img src="../img/dataset_img/XFUND_1.jpg" height=300>
</p>

<br>

# EPHOIE

<br>

# EATEN

<br>

# WildReceipt

<br>

# Kleister

<br>

# CER-VIR

<br>

# DeepForm

<br>

# CER-VIR

<br>

# SIBR

<br>