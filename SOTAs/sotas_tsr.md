<h3 align="center"> Table Structure Recognition </h3>
<h1 align="center"> SOTAs </h1>

This page contains performance on public benchmarks of visual information extraction alogorithms. Data are collected from papers & official code repositories.

<h2>🎖️Commonly Used Metrics</h2>

<h3><b>F1-score</b></h3>

For comparing two cell structures, we use a method inspired by Hurst’s proto-links: for each table region we generate a list of adjacency relations between each content cell and its nearest neighbour in horizontal and vertical directions. No adjacency relations are generated between blank cells or a blank cell and a content cell. This 1-D list of adjacency relations can be compared to the ground truth by using precision and recall measures. If both cells are identical and the direction matches, then it is marked as correctly retrieved; otherwise it is marked as incorrect. Using neighbourhoods makes the comparison invariant to the absolute position of the table (e.g. if everything is shifted by one cell) and also avoids ambiguities arising with dealing with different types of errors (merged/split cells, inserted empty column, etc.).

$$
precision = \frac{correct adjacency relations}{total adjacency relations}
$$

$$
recall = \frac{correct adjacency relations}{detected adjacency relations}
$$

$$
F1 = \frac{2 \times precision \times recall}{precision + recall}
$$

The [SciTSR](https://github.com/Academic-Hammer/SciTSR) library can be used to calculate the F1-score between tables.

<h3><b>TREE-EDIT-DISTANCE-BASED SIMILARITY</b></h3>

Tables are presented as a tree structure in the HTML format.The root has two children thead and tbody, which group table headers and table body cells, respectively. The children of thead and tbody nodes are table rows (tr). The leaves of the tree are table cells (td). Each cell node has three attributes, i.e. ‘colspan’, ‘rowspan’, and ‘content’. We measure the similarity between two tables using the tree-edit distance proposed by Pawlik and Augsten. The cost of insertion and deletion operations is 1. When the edit is substituting a node n<sub>o</sub> with n<sub>s</sub>, the cost is 1 if either n<sub>o</sub> or n<sub>s</sub> is not td.
When both n<sub>o</sub> and n<sub>s</sub> are td, the substitution cost is 1 if the column span or the row span of n<sub>o</sub> and n<sub>s</sub> is different. Otherwise, the substitution cost is the normalized Levenshtein similarity (∈ [0, 1]) between the content of n<sub>o</sub> and n<sub>s</sub>.Finally, TEDS between two trees is computed as

$$
TEDS(T~a~, T~b~) = 1 - \frac{EditDist(T~a~, T~b~)}{max(|T~a~|, |T~b~|)}
$$

where $EditDist$ denotes tree-edit distance, and $|T|$ is the number of nodes in $T$. The table recognition performance of a method on a set of test samples is defined as the mean of the TEDS score between the recognition result and ground truth of each sample

The [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) library can be used to calculate the TEDS between tables.

<br>

<h2>🗒️List of Index</h2>

- [PubTabNet](#pubtabnet)
- [SciTSR](#scitsr)
- [ICDAR2013](#icdar2013)
- [ICDAR2019](#icdar2019)
- [WTW](#wtw)

---

## PubTabNet

[PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) is automatically generated by matching the XML and PDF representations of the scientific articles in PubMed CentralTM Open Access Subset (PMCOA). It takes TEDS and TEDS-S as the evaluation metric, where TEDS-S refers to the TEDS result that ignoring the text contents.

<table align="center">
<tr>
    <th > Approach </th>
    <th > Training Dataset </th>
    <th> TEDS(%) </th>
    <th> TEDS-S(%) </th>
</tr>
<tr>
    <td><a href="../Approaches/approaches_tsr.md/#tsrformer">TSRFormer</td>
    <td>PubTabNet</a></td>
    <td>-</td>
    <td>97.5</td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_tsr.md/#robustabnet">RobusTabNet</td>
    <td>PubTabNet</td>
    <td>-</td>
    <td>97.0</td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_tsr.md/#trust">TRUST</td>
    <td>PubTabNet</td>
    <td>96.2</td>
    <td>97.1</td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_tsr.md/#tableformer">TableFormer</a></td>
    <td>PubTabNet</td>
    <td>93.6</td>
    <td>96.75</td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_tsr.md/#tablemaster">TableMaster</a></td>
    <td>PubTabNet</td>
    <td>-</td>
    <td>96.76</td>
</tr>
</tr>
    <td><a href="../Approaches/approaches_tsr.md/#edd">EDD</a></td>
    <td>PubTabNet</td>
    <td>88.3</td>
    <td>-</td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_tsr.md/#lgpma">LGPMA</td>
    <td>PubTabNet</td>
    <td>94.6</td>
    <td>96.7</td>
</tr>
</table>

<br>

## SciTSR

[SciTSR](https://github.com/Academic-Hammer/SciTSR) is a large-scale table structure recognition dataset, which contains 15,000 tables in PDF format and their corresponding high quality structure labels obtained from LaTeX source files.


<table align="center">
<tr>
    <th> Approach </th>
    <th> Training Dataset </th>
    <th> Precision </th>
    <th> Recall </th>
    <th> F1 </th>
</tr>
<tr>
    <td><a href="../Approaches/approaches_tsr.md/#tsrformer">TSRFormer</td>
    <td>SciTSR</td>
    <td>99.5</td>
    <td>99.4</td>
    <td>99.4</td>
</tr>
</tr>
    <td><a href="../Approaches/approaches_tsr.md/#robustabnet">RobusTabNet</td>
    <td>SciTSR</td>
    <td>99.4</td>
    <td>99.1</td>
    <td>99.3</td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_tsr.md/#sem">SEM</td>
    <td>SciTSR</td>
    <td>97.70</td>
    <td>96.52</td>
    <td>97.11</td>
</tr>
</tr>
    <td><a href="../Approaches/approaches_tsr.md/#ncgm">NCGM</td>
    <td>SciTSR</td>
    <td>99.7</td>
    <td>99.6</td>
    <td>99.6</td>
</tr>
</tr>
    <td><a href="../Approaches/approaches_tsr.md/#flagnet">FLAGNet</td>
    <td>SciTSR</td>
    <td>99.7</td>
    <td>99.3</td>
    <td>99.5</td>
</tr>
</tr>
    <td><a href="../Approaches/approaches_tsr.md/#lgpma">LGPMA</td>
    <td>SciTSR</td>
    <td>98.2</td>
    <td>99.3</td>
    <td>98.8</td>
</tr>
</table>

<br>

## ICDAR2013

These documents have been collected systematically from the European Union and US Government websites, and we therefore expect them to have public domain status. Each PDF document is accompanied by three XML (or CSV) file containing its ground truth in the following models:

table regions (for evaluating table location)
cell structures (for evaluating table structure recognition)
functional representation (for evaluating table interpretation)

The dataset can be downloaded from [here](http://www.tamirhassan.com/dataset/).

<table align="center">
<tr>
    <th> Approach </th>
    <th> Training Dataset </th>
    <th> Precision </th>
    <th> Recall </th>
    <th> F1 </th>
</tr>
</tr>
    <td><a href="../Approaches/approaches_tsr.md/#splerge">SPLERGE</td>
    <td>ICDAR2013</td>
    <td>94.64</td>
    <td>95.89</td>
    <td>95.26</td>
</tr>
</tr>
    <td><a href="../Approaches/approaches_tsr.md/#lgpma">LGPMA</td>
    <td>SciTSR+ICDAR2013</td>
    <td>96.7</td>
    <td>99.1</td>
    <td>97.9</td>
</tr>
</table>

<br>

## ICDAR2019

Two new datasets consisting of modern and archival documents have been prepared for [cTDaR 2019](http://sac.founderit.com/). The historical dataset contains contributions from more than 23 institutions around the world. The images show a great variety of tables from hand-drawn accounting books to stock exchange lists and train timetables, from record books to prisoner lists, simple tabular prints in books, production census and many more. The modern dataset comes from different kinds of PDF documents such as scientific journals, forms, financial statements, etc. The dataset contains of Chinese and English documents with various formats, including document images and born-digital format. The annotated contents contain the table entities and cell entities in a document.

<br>

## WTW

[WTW](https://github.com/wangwen-whu/WTW-Dataset), which has a total of 14581 images in a wide range of real business scenarios and the corresponding full annotation (including cell coordinates and row/column information) of tables.
The images in the WTW dataset are mainly collected from the natural images that contain at least one table. As our purpose is to parsing table structures without considering the image source, we additionally add the archival document images and the printed document images. Statically, the portion of images from natural scenes, archival, and printed document images are 50%, 30%, and 20%. After obtaining all the images, we statically found 7 challenging cases.WTW dataset covers all challenging cases with a reasonable proportion of each case.

<table align="center">
<tr>
    <th> Approach </th>
    <th> Training Dataset </th>
    <th> Precision </th>
    <th> Recall </th>
    <th> F1 </th>
</tr>
</tr>
    <td><a href="../Approaches/approaches_tsr.md/#tsrformer">TSRFormer</td>
    <td>WTW</td>
    <td>93.7</td>
    <td>93.2</td>
    <td>93.4</td>
</tr>
</tr>
    <td><a href="../Approaches/approaches_tsr.md/#ncgm">NCGM</td>
    <td>WTW</td>
    <td>94.7</td>
    <td>95.5</td>
    <td>95.1</td>
</tr>
</table>