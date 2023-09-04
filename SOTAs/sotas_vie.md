<h3 align="center"> Visual Information Extraction </h3>
<h1 align="center"> SOTAs </h1>

This page contains performance on public benchmarks of visual information extraction alogorithms. Data are collected from papers & official code repositories.

<h2>🎖️Commonly Used Metrics</h2>

<h3><b>F1-score</b></h3>

Given the prediction of the model and the ground-truth, if the predicted string of a key category is completely consistent with the ground-truth, then it will be recorded as a true positive(TP) sample. Let $N_p$ denotes the number of predicted string, $N_g$ for the number of ground-truth entities, $N_t$ for the number of TP samples, then we have

$$
precision = \frac{N_t}{N_p}
$$

$$
recall = \frac{N_t}{N_g}
$$

$$
F1 = \frac{2 \times precision \times recall}{precision + recall}
$$

When using the BIO-tagging schema, the [seqeval](https://github.com/chakki-works/seqeval) library is a good choice for calculating the F1-score.

<h3><b>Edit Distance Score</b></h3>

The edit distance between the prediction string and ground-truth of a key category is calculated as follow

$$
score = 1 - \frac{i + d + m}{N}
$$

where $i$, $d$, $m$, $N$ denotes the number of insertions, number of deletions, number of modifications and the total number of instances occurring in the ground truth, respectively.

The [zhang-shasha](https://github.com/timtadh/zhang-shasha) library can be used to calculate the edit distance between strings.

<br>

<h2>🗒️List of Index</h2>

- [SROIE](#sroie)
- [CORD](#cord)
- [FUNSD](#funsd)
- [XFUND](#xfund)
- [EPHOIE](#ephoie)

---

## SROIE

The SROIE dataset takes the micro-F1-score as the evaluation metric. The dataset contains 4 key categories, each category contains one or no entity. If the predicted string of a key category is consistant with the ground-truth string, it will be recorded as a TP sample. The total number of TP, total number of predictions and total number of ground-truth strings will be used to calculate the score. Evaluation scripts can be found at the [ICDAR2019 SROIE official page](https://rrc.cvc.uab.es/?ch=13&com=downloads) (Download tab, Task 3 Evaluation script).

<table align="center">
<tr>
    <th > Type </th>
    <th colspan=2> Approach </th>
    <th> Precision </th>
    <th> Recall </th>
    <th> F1 </th>
</tr>
<tr>
    <td rowspan=2>Grid-based</td>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#vibertgrid">ViBERTgrid</a></td>
    <td>BERT-base</td>
    <td>-</td>
    <td>-</td>
    <td>96.25</td>
</tr>
<tr>
    <td>RoBERTa-base</td>
    <td>-</td>
    <td>-</td>
    <td>96.40</td>
</tr>
<tr>
    <td rowspan=4>GNN-based</td>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#pick">PICK</a></td>
    <td>-</td>
    <td>-</td>
    <td>96.12</td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#matchvie">MatchVIE</a></td>
    <td>-</td>
    <td>-</td>
    <td>96.57</td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>-</td>
    <td>-</td>
    <td>98.45</td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#formnetv2">FormNetV2</a></td>
    <td>-</td>
    <td>-</td>
    <td>98.31</td>
</tr>
</tr>
    <td rowspan=14>Large Scale Pre-trained</td>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#layoutlm">LayoutLM</a></td>
    <td>base</td>
    <td>94.38</td>
    <td>94.38</td>
    <td>94.38</td>
</tr>
<tr>
    <td>large</td>
    <td>95.24</td>
    <td>95.24</td>
    <td>95.24</td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#layoutlmv2">LayoutLMv2</a></td>
    <td>base</td>
    <td>96.25</td>
    <td>96.25</td>
    <td>96.25</td>
</tr>
<tr>
    <td>large</td>
    <td>99.04</td>
    <td>96.61</td>
    <td>97.81</td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#tilt">TILT</a></td>
    <td>base</td>
    <td>-</td>
    <td>-</td>
    <td>97.65</td>
</tr>
<tr>
    <td>large</td>
    <td>-</td>
    <td>-</td>
    <td>98.10</td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#bros">BROS</a></td>
    <td>base</td>
    <td>-</td>
    <td>-</td>
    <td>95.91</td>
</tr>
<tr>
    <td>large</td>
    <td>-</td>
    <td>-</td>
    <td>96.62</td>
</tr>
</tr>
    <td rowspan=3><a href="../Approaches/approaches_vie.md/#structext">StrucTexT</a></td>
    <td>eng-base</td>
    <td>-</td>
    <td>-</td>
    <td>96.88</td>
</tr>
<tr>
    <td>chn&eng-base</td>
    <td>-</td>
    <td>-</td>
    <td>98.27</td>
</tr>
<tr>
    <td>chn&eng-large</td>
    <td>-</td>
    <td>-</td>
    <td>98.70</td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#wukong-reader">WUKONG-READER</a></td>
    <td>base</td>
    <td>-</td>
    <td>-</td>
    <td>96.88</td>
</tr>
<tr>
    <td>large</td>
    <td>-</td>
    <td>-</td>
    <td>98.15</td>
</tr>
</tr>
    <td rowspan=1><a href="../Approaches/approaches_vie.md/#ernie-layout">ERNIE-layout</a></td>
    <td>large</td>
    <td>-</td>
    <td>-</td>
    <td>97.55</td>
</tr>
</tr>
    <td rowspan=5>End-to-End</td>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#trie">TRIE</a></td>
    <td>ground-truth</td>
    <td>-</td>
    <td>-</td>
    <td>96.18</td>
</tr>
<tr>
    <td>end-to-end</td>
    <td>-</td>
    <td>-</td>
    <td>82.06</td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#vies">VIES</a></td>
    <td>ground-truth</td>
    <td>-</td>
    <td>-</td>
    <td>96.12</td>
</tr>
<tr>
    <td>end-to-end</td>
    <td>-</td>
    <td>-</td>
    <td>91.07</td>
</tr>
</tr>
    <td rowspan=1><a href="../Approaches/approaches_vie.md/#kuang-cfam">Kuang CFAM</a></td>
    <td>end-to-end</td>
    <td>-</td>
    <td>-</td>
    <td>85.87</td>
</tr>
</tr>
    <td rowspan=4>Other Methods</td>
    <td rowspan=4><a href="../Approaches/approaches_vie.md/#tcpn">TCPN</a></td>
    <td>TextLattice</td>
    <td>-</td>
    <td>-</td>
    <td>96.54</td>
</tr>
<tr>
    <td>Tag, ground-truth</td>
    <td>-</td>
    <td>-</td>
    <td>95.46</td>
</tr>
</tr>
    <td>Tag, end-to-end</td>
    <td>-</td>
    <td>-</td>
    <td>91.21</td>
</tr>
<tr>
    <td>Tag&Copy, end-to-end</td>
    <td>-</td>
    <td>-</td>
    <td>91.93</td>
</tr>

</table>

<br>

## CORD

Many mainstream SOTAs treat CORD as an Entity Extraction dataset and follow the calculation protocol of SROIE. These practices are in fact controversial. Authors of the CORD dataset, the clovaai team, do not explicitly specify how to compute metrics for this dataset, but when we browse the [source code](https://github.com/clovaai/donut/blob/217cffb111a57ebce1025ff84a722a8d9914e05b/donut/util.py#L242) of their work (e.g. Donut, SPADE), we can see that they compute the F1-scores for Document Structure Parsing. For example, a receipt usually contains information about the items purchased, including its name, count, and unit price. These entities are hierarchically related, and an item can be represented by a python List in forms of `[item_name, item_count, item_price]`. The algorithm should extract all of the item information List as well as other information like changes, total price from a given document. The prediction will be counted as TP when a same information List exists in the ground-truth.

Scores reported on both Entity Extraction and Document Structure Parsing are shown below. 


<table align="center">
<tr>
    <th rowspan=2> Type </th>
    <th rowspan=2 colspan=2> Approach </th>
    <th colspan=3> Entity Extraction </th>
    <th colspan=3> Document Structure Parsing </th>
</tr>
<tr>
    <th> Precision </th>
    <th> Recall </th>
    <th> F1 </th>
    <th> Precision </th>
    <th> Recall </th>
    <th> F1 </th>
</tr>
<tr>
    <td rowspan=3>GNN-based</td>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>-</td>
    <td>-</td>
    <td>96.93</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#formnet">FormNet</a></td>
    <td>98.02</td>
    <td>96.55</td>
    <td>97.28</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#formnetv2">FomNetV2</a></td>
    <td>-</td>
    <td>-</td>
    <td>97.70</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</tr>
    <td rowspan=21>Large Scale Pre-trained</td>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#layoutlm">LayoutLM</a></td>
    <td>base</td>
    <td>94.37</td>
    <td>95.08</td>
    <td>94.72</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>large</td>
    <td>94.32</td>
    <td>95.54</td>
    <td>94.93</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#layoutlmv2">LayoutLMv2</a></td>
    <td>base</td>
    <td>94.53</td>
    <td>95.39</td>
    <td>94.95</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>large</td>
    <td>95.65</td>
    <td>96.37</td>
    <td>96.01</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#layoutlmv3">LayoutLMv3</a></td>
    <td>base</td>
    <td>-</td>
    <td>-</td>
    <td>96.56</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>large</td>
    <td>-</td>
    <td>-</td>
    <td>97.46</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#docformer">DocFormer</a></td>
    <td>base</td>
    <td>96.52</td>
    <td>96.14</td>
    <td>96.33</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>large</td>
    <td>97.25</td>
    <td>96.74</td>
    <td>96.99</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#tilt">TILT</a></td>
    <td>base</td>
    <td>-</td>
    <td>-</td>
    <td>95.11</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>large</td>
    <td>-</td>
    <td>-</td>
    <td>96.33</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#docformer">BROS</a></td>
    <td>base</td>
    <td>-</td>
    <td>-</td>
    <td>96.50</td>
    <td>-</td>
    <td>-</td>
    <td>95.73</td>
</tr>
<tr>
    <td>large</td>
    <td>-</td>
    <td>-</td>
    <td>97.28</td>
    <td>-</td>
    <td>-</td>
    <td>97.40</td>
</tr>
<tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#udoc">UDoc</a></td>
    <td>UDoc</td>
    <td>-</td>
    <td>-</td>
    <td>96.64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>UDoc*</td>
    <td>-</td>
    <td>-</td>
    <td>96.86</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#lilt">LiLT</a></td>
    <td>[EN-RoBERTa]base</td>
    <td>-</td>
    <td>-</td>
    <td>96.07</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>[InfoXLM]base</td>
    <td>-</td>
    <td>-</td>
    <td>95.77</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#docrel">DocReL</a></td>
    <td>-</td>
    <td>-</td>
    <td>97.00</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#wukong-reader">WUKONG-READER</a></td>
    <td>base</td>
    <td>-</td>
    <td>-</td>
    <td>96.54</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>large</td>
    <td>-</td>
    <td>-</td>
    <td>97.27</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</tr>
    <td rowspan=1><a href="../Approaches/approaches_vie.md/#ernie-layout">ERNIE-layout</a></td>
    <td>large</td>
    <td>-</td>
    <td>-</td>
    <td>96.99</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#geolayoutlm">GeoLayoutLM</a></td>
    <td>-</td>
    <td>-</td>
    <td>97.97</td>
    <td>-</td>
    <td>-</td>
    <td>99.45</td>
</tr>
<tr>
    <td rowspan=2>End-to-End</td>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#donut">Donut</a></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>91.60</td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#esp">ESP</a></td>
    <td>-</td>
    <td>-</td>
    <td>95.65</td>
    <td>-</td>
    <td>-</td>
    <td>98.80</td>
</tr>
<tr>
    <td rowspan=8>Other Methods</td>
    <td rowspan=8><a href="../Approaches/approaches_vie.md/#spade">SPADE</a></td>
    <td>♠ CORD, oracle input</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>92.50</td>
</tr>
<tr>
    <td>♠ CORD</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>88.20</td>
</tr>
<tr>
    <td>♠ CORD+</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>87.40</td>
</tr>
<tr>
    <td>♠ CORD++</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>83.10</td>
</tr>
<tr>
    <td>♠ w/o TCM, CORD, oracle input</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>91.50</td>
</tr>
<tr>
    <td>♠ w/o TCM, CORD</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>87.40</td>
</tr>
<tr>
    <td>♠ w/o TCM, CORD+</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>86.10</td>
</tr>
<tr>
    <td>♠ w/o TCM, CORD++</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>82.60</td>
</tr>

</table>

<br>

## FUNSD

FUNSD requires extracting key entities (Entity Extraction) and key-value pairs (Entity Linking). Micro-F1 is taken as the evaluation metric. Each document contains two kinds of key information: Question and Answer, and each key category has multiple instances. In Entity Extraction task, the predicted entity will be considered as TP if and only if its content and category are consistent with the ground-truth. In Entity Linking task, the prediction will be considered as TP if and only if the predicted pair exists in the ground-truth pairs. 

It is noticable that the two subtasks are independent in most of the mainstream approaches' settings. Take LayoutLM as an example, in Entity Linking task, the [official implementation](https://github.com/microsoft/unilm) takes the groud-truth of Entity Extraction as input and predict the linkings only, the performance of entity extraction is not considered in this case. 

<table align="center">
<tr>
    <th rowspan=2> Type </th>
    <th rowspan=2 colspan=2> Approach </th>
    <th colspan=3> Entity Extraction </th>
    <th colspan=3> Entity Linking </th>
</tr>
<tr>
    <th> Precision </th>
    <th> Recall </th>
    <th> F1 </th>
    <th> Precision </th>
    <th> Recall </th>
    <th> F1 </th>
</tr>
<tr>
    <td rowspan=4>GNN-based</td>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>-</td>
    <td>-</td>
    <td>87.77</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#matchvie">MatchVIE</a></td>
    <td>-</td>
    <td>-</td>
    <td>81.33</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#formnet">FormNet</a></td>
    <td>-</td>
    <td>-</td>
    <td>84.69</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#formnetv2">FormNetV2</a></td>
    <td>-</td>
    <td>-</td>
    <td>92.51</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
    <td rowspan=32>Large Scale Pre-trained</td>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#layoutlm">LayoutLM</a></td>
        <td>base</td>
        <td>75.97</td>
        <td>81.55</td>
        <td>78.66</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
</tr>
<tr>
    <td>large</td>
    <td>75.96</td>
    <td>82.19</td>
    <td>78.95</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#layoutlmv2">LayoutLMv2</a></td>
        <td>base</td>
        <td>80.29</td>
        <td>85.39</td>
        <td>82.76</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
</tr>
<tr>
    <td>large</td>
    <td>83.24</td>
    <td>85.19</td>
    <td>84.20</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</tr>
    <td rowspan=4><a href="../Approaches/approaches_vie.md/#layoutlxlm">LayoutXLM</a></td>
        <td>base, Language Specific Fine-tuning</td>
        <td>-</td>
        <td>-</td>
        <td>79.40</td>
        <td>-</td>
        <td>-</td>
        <td>54.83</td>
</tr>
<tr>
    <td>large, Language Specific Fine-tuning</td>
    <td>-</td>
    <td>-</td>
    <td>82.25</td>
    <td>-</td>
    <td>-</td>
    <td>64.04</td>
</tr>
<tr>
    <td>base, Multitask Fine-tuning</td>
    <td>-</td>
    <td>-</td>
    <td>79.24</td>
    <td>-</td>
    <td>-</td>
    <td>66.71</td>
</tr>
<tr>
    <td>large, Multitask Fine-tuning</td>
    <td>-</td>
    <td>-</td>
    <td>80.68</td>
    <td>-</td>
    <td>-</td>
    <td>76.83</td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#layoutlmv3">LayoutLMv3</a></td>
        <td>base</td>
        <td>-</td>
        <td>-</td>
        <td>90.29</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
</tr>
<tr>
    <td>large</td>
    <td>-</td>
    <td>-</td>
    <td>92.08</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#xylayoutlm">XYLayoutLM</a></td>
    <td>-</td>
    <td>-</td>
    <td>83.35</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#selfdoc">SelfDoc</a></td>
    <td>-</td>
    <td>-</td>
    <td>83.36</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#DocFormer">DocFormer</a></td>
        <td>base</td>
        <td>80.76</td>
        <td>86.09</td>
        <td>83.34</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
</tr>
<tr>
    <td>large</td>
    <td>82.29</td>
    <td>86.94</td>
    <td>84.55</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#structurallm">StructuralLM-large</a></td>
    <td>83.52</td>
    <td>-</td>
    <td>85.14</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#bros">BROS</a></td>
    <td>base</td>
    <td>81.16</td>
    <td>85.02</td>
    <td>83.05</td>
    <td>-</td>
    <td>-</td>
    <td>71.46</td>
</tr>
<tr>
    <td>large</td>
    <td>82.81</td>
    <td>86.31</td>
    <td>84.52</td>
    <td>-</td>
    <td>-</td>
    <td>77.01</td>
</tr>
</tr>
    <td rowspan=3><a href="../Approaches/approaches_vie.md/#structext">StrucTexT</a></td>
    <td>eng-base</td>
    <td>-</td>
    <td>-</td>
    <td>83.09</td>
    <td>-</td>
    <td>-</td>
    <td>44.10</td>
</tr>
<tr>
    <td>chn&eng-base</td>
    <td>-</td>
    <td>-</td>
    <td>84.83</td>
    <td>-</td>
    <td>-</td>
    <td>70.45</td>
</tr>
<tr>
    <td>chn&eng-large</td>
    <td>-</td>
    <td>-</td>
    <td>87.56</td>
    <td>-</td>
    <td>-</td>
    <td>74.21</td>
</tr>
<tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#udoc">UDoc</a></td>
    <td>UDoc</td>
    <td>-</td>
    <td>-</td>
    <td>87.96</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>UDoc*</td>
    <td>-</td>
    <td>-</td>
    <td>87.93</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td rowspan=4><a href="../Approaches/approaches_vie.md/#lilt">LiLT</a></td>
    <td>[En RoBERTa]base</td>
    <td>87.21</td>
    <td>89.65</td>
    <td>88.41</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>[InfoXLM]base</td>
    <td>84.67</td>
    <td>87.09</td>
    <td>85.86</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>[InfoXLM]base, Language Specific Fine-tuning</td>
    <td>-</td>
    <td>-</td>
    <td>84.15</td>
    <td>-</td>
    <td>-</td>
    <td>62.76</td>
</tr>
<tr>
    <td>[InfoXLM]base, Multitask Fine-tuning</td>
    <td>-</td>
    <td>-</td>
    <td>85.74</td>
    <td>-</td>
    <td>-</td>
    <td>74.07</td>
</tr>

<tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#docrel">DocReL</a></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>46.10</td>
</tr>
</tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#wukong-reader">WUKONG-READER</a></td>
    <td>base</td>
    <td>-</td>
    <td>-</td>
    <td>91.52</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>large</td>
    <td>-</td>
    <td>-</td>
    <td>93.62</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</tr>
    <td rowspan=1><a href="../Approaches/approaches_vie.md/#ernie-layout">ERNIE-layout</a></td>
    <td>large</td>
    <td>-</td>
    <td>-</td>
    <td>93.12</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#geolayoutlm">GeoLayoutLM</a></td>
    <td>-</td>
    <td>-</td>
    <td>92.86</td>
    <td>-</td>
    <td>-</td>
    <td>89.45</td>
</tr>
</tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#kvpformer">KVPFormer</a></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>90.86</td>
</tr>
<tr>
    <td rowspan=1>End-to-End</td>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#esp">ESP</a></td>
    <td>-</td>
    <td>-</td>
    <td>91.12</td>
    <td>-</td>
    <td>-</td>
    <td>88.88</td>
</tr>
<tr>
    <td rowspan=1>Other Methods</td>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#spade">SPADE</a></td>
    <td>-</td>
    <td>-</td>
    <td>71.60</td>
    <td>-</td>
    <td>-</td>
    <td>41.30</td>
</tr>
</table>

<br>

## XFUND

XFUND is an extension of the FUNSD dataset. It covers 7 languages, including Chinese, Japanese, Spanish, French, Italian, German, and Portuguese. It contains 1,393 fully annotated forms, and each language includes 199 forms, where the training set includes 149 forms, and the test set includes 50 forms. XFUND also has two subtasks, and its evaluation protocol is consistent with the one in FUNSD.

Note: In the following charts, only the scores reproted on XFUND is counted when calculating the Average F1-score. Hence the values of the Avg. score may be different from the ones reported in papers.

<table align="center">
<tr>
    <th rowspan=2> Type </th>
    <th rowspan=2 colspan=2> Approach </th>
    <th colspan=8> Entity Extraction </th>
    <th colspan=8> Entity Linking </th>
</tr>
<tr>
    <td>ZH</td>
    <td>JA</td>
    <td>ES</td>
    <td>FR</td>
    <td>IT</td>
    <td>DE</td>
    <td>PT</td>
    <td>Avg.</td>
    <td>ZH</td>
    <td>JA</td>
    <td>ES</td>
    <td>FR</td>
    <td>IT</td>
    <td>DE</td>
    <td>PT</td>
    <td>Avg.</td>
</tr>
</tr>
    <td rowspan=11>Large Scale Pre-trained</td>
    <td rowspan=6><a href="../Approaches/approaches_vie.md/#layoutlxlm">LayoutXLM</a></td>
    <td>base, Language Specific Fine-tuning</td>
    <td>89.24</td>
    <td>79.21</td>
    <td>75.50</td>
    <td>79.02</td>
    <td>80.02</td>
    <td>82.22</td>
    <td>79.03</td>
    <td>82.40</td>
    <td>70.73</td>
    <td>69.63</td>
    <td>68.96</td>
    <td>63.53</td>
    <td>64.15</td>
    <td>65.51</td>
    <td>57.18</td>
    <td>65.67</td>
</tr>
<tr>
    <td>large, Language Specific Fine-tuning</td>
    <td>91.61</td>
    <td>80.33</td>
    <td>78.30</td>
    <td>80.98</td>
    <td>82.75</td>
    <td>83.61</td>
    <td>82.73</td>
    <td>82.90</td>
    <td>78.88</td>
    <td>72.25</td>
    <td>76.66</td>
    <td>71.02</td>
    <td>76.91</td>
    <td>68.43</td>
    <td>67.96</td>
    <td>73.16</td>
</tr>
<tr>
    <td>base, Zero-shot transfer</td>
    <td>60.19</td>
    <td>47.15</td>
    <td>45.65</td>
    <td>57.57</td>
    <td>48.46</td>
    <td>52.52</td>
    <td>53.90</td>
    <td>52.21</td>
    <td>44.94</td>
    <td>44.08</td>
    <td>47.08</td>
    <td>44.16</td>
    <td>40.90</td>
    <td>38.20</td>
    <td>36.85</td>
    <td>42.31</td>
</tr>
<tr>
    <td>large, Zero-shot transfer</td>
    <td>68.96</td>
    <td>51.90</td>
    <td>49.76</td>
    <td>61.35</td>
    <td>55.17</td>
    <td>59.05</td>
    <td>60.77</td>
    <td>58.14</td>
    <td>55.31</td>
    <td>56.96</td>
    <td>57.80</td>
    <td>56.15</td>
    <td>51.84</td>
    <td>48.90</td>
    <td>47.95</td>
    <td>53.56</td>
</tr>
<tr>
    <td>base, Multitask Fine-tuning</td>
    <td>89.73</td>
    <td>79.64</td>
    <td>77.98</td>
    <td>81.73</td>
    <td>82.10</td>
    <td>83.22</td>
    <td>82.41</td>
    <td>82.40</td>
    <td>82.41</td>
    <td>81.42</td>
    <td>81.04</td>
    <td>82.21</td>
    <td>83.10</td>
    <td>78.54</td>
    <td>70.44</td>
    <td>79.88</td>
</tr>
<tr>
    <td>large, Multitask Fine-tuning</td>
    <td>91.55</td>
    <td>82.16</td>
    <td>80.55</td>
    <td>83.84</td>
    <td>83.72</td>
    <td>85.30</td>
    <td>86.50</td>
    <td>84.80</td>
    <td>90.00</td>
    <td>86.21</td>
    <td>85.92</td>
    <td>86.69</td>
    <td>86.75</td>
    <td>82.63</td>
    <td>81.60</td>
    <td>85.69</td>
</tr>
</tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#xylayoutlm">XYLayoutLM</a></td>
    <td>91.76</td>
    <td>80.57</td>
    <td>76.87</td>
    <td>79.97</td>
    <td>81.75</td>
    <td>83.35</td>
    <td>80.01</td>
    <td>82.04</td>
    <td>74.45</td>
    <td>70.59</td>
    <td>72.59</td>
    <td>65.21</td>
    <td>65.72</td>
    <td>67.03</td>
    <td>58.98</td>
    <td>67.79</td>
</tr>
</tr>
    <td rowspan=3><a href="../Approaches/approaches_vie.md/#lilt">LiLT</a></td>
    <td>[InfoXLM] base, Language Specific Fine-tuning</td>
    <td>89.38</td>
    <td>79.64</td>
    <td>79.11</td>
    <td>79.53</td>
    <td>83.76</td>
    <td>82.31</td>
    <td>82.20</td>
    <td>82.27</td>
    <td>72.97</td>
    <td>70.37</td>
    <td>71.95</td>
    <td>69.65</td>
    <td>70.43</td>
    <td>65.58</td>
    <td>58.74</td>
    <td>68.53</td>
</tr>
<tr>
    <td>[InfoXLM] base, Zero-shot transfer</td>
    <td>61.52</td>
    <td>51.84</td>
    <td>51.01</td>
    <td>59.23</td>
    <td>53.71</td>
    <td>60.13</td>
    <td>63.25</td>
    <td>57.24</td>
    <td>47.64</td>
    <td>50.81</td>
    <td>49.68</td>
    <td>52.09</td>
    <td>46.97</td>
    <td>41.69</td>
    <td>42.72</td>
    <td>47.37</td>
</tr>
<tr>
    <td>[InfoXLM] base, Multi-task Fine-tuning</td>
    <td>90.47</td>
    <td>80.88</td>
    <td>83.40</td>
    <td>85.77</td>
    <td>87.92</td>
    <td>87.69</td>
    <td>84.93</td>
    <td>85.86</td>
    <td>84.71</td>
    <td>83.45</td>
    <td>83.35</td>
    <td>84.66</td>
    <td>84.58</td>
    <td>78.78</td>
    <td>76.43</td>
    <td>82.28</td>
</tr>
</tr>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#kvpformer">KVPFormer</a></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>94.27</td>
    <td>94.23</td>
    <td>95.23</td>
    <td>97.19</td>
    <td>94.11</td>
    <td>92.41</td>
    <td>92.19</td>
    <td>94.23</td>
</tr>
</tr>
    <td rowspan=2>End-to-End</td>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#esp">ESP</a></td>
    <td>Language Specific Fine-tuning</td>
    <td>90.30</td>
    <td>81.10</td>
    <td>85.40</td>
    <td>90.50</td>
    <td>88.90</td>
    <td>87.20</td>
    <td>87.50</td>
    <td>87.30</td>
    <td>90.80</td>
    <td>88.30</td>
    <td>85.20</td>
    <td>90.90</td>
    <td>90.00</td>
    <td>85.20</td>
    <td>86.20</td>
    <td>88.10</td>
</tr>
<tr>
    <td>Multitask Fine-tuning</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>89.13</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>92.31</td>
</tr>
</table>

<br>

## EPHOIE

EPHOIE contains 11 key categories. It takes the micro-F1 as the evaluation metric. If the predicted string of a key category is consistant with the ground-truth string and not empty, it will be recorded as a TP sample.

<table align="center">
<tr>
    <th > Type </th>
    <th colspan=2> Approach </th>
    <th> Precision </th>
    <th> Recall </th>
    <th> F1 </th>
</tr>
<tr>
    <td rowspan=1>Grid-based</td>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#matchvie">MathcVIE</a></td>
    <td>-</td>
    <td>-</td>
    <td>96.87</td>
</tr>
<tr>
    <td rowspan=4>Large-Scale Pre-trained</td>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#structext">StrucTexT</a></td>
    <td>chn&eng-base</td>
    <td>-</td>
    <td>-</td>
    <td>98.84</td>
</tr>
<tr>
    <td>chn&eng-large</td>
    <td>-</td>
    <td>-</td>
    <td>99.30</td>
</tr>
<tr>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#lilt">LiLT</a></td>
    <td>[InfoXLM]base</td>
    <td>96.99</td>
    <td>98.20</td>
    <td>97.59</td>
</tr>
<tr>
    <td>[ZH-RoBERTa]base</td>
    <td>97.62</td>
    <td>98.33</td>
    <td>97.97</td>
</tr>
<tr>
    <td rowspan=2>End-to-End</td>
    <td rowspan=2><a href="../Approaches/approaches_vie.md/#vies">VIES</a></td>
    <td>ground-truth</td>
    <td>-</td>
    <td>-</td>
    <td>95.23</td>
</tr>
<tr>
    <td>end-to-end</td>
    <td>-</td>
    <td>-</td>
    <td>83.81</td>
</tr>
<tr>
    <td rowspan=4>Other Methods</td>
    <td rowspan=4><a href="../Approaches/approaches_vie.md/#tcpn">TCPN</a></td>
    <td>TextLattice</td>
    <td>-</td>
    <td>-</td>
    <td>98.06</td>
</tr>
<tr>
    <td>Copy Mode, end-to-end</td>
    <td>-</td>
    <td>-</td>
    <td>84.67</td>
</tr>
<tr>
    <td>Tag Mode, end-to-end</td>
    <td>-</td>
    <td>-</td>
    <td>86.19</td>
</tr>
<tr>
    <td>Tag Mode, ground-truth</td>
    <td>-</td>
    <td>-</td>
    <td>97.59</td>
</tr>