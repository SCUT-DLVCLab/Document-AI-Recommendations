<h3 align="center"> Visual Information Extraction </h3>
<h1 align="center"> SOTAs </h1>

This page contains performance on public benchmarks of visual information extraction alogorithms. Data are collected from papers & official code repositories.

<h2>🎖️Commonly Used Metrics</h2>

<h3><b>F1-score</b></h3>

Given the prediction of the model and the ground-truth, if the predicted string of a key category is completely consistent with the ground-truth, then it will be recorded as a true positive(TP) sample. Let $N_p$ denotes the number of predicted string, $N_g$ for the number of ground-truth entities, $N_t$ for the number of TP samples, then we have

$$
precision = \frac{N_t}{N_p} \\
recall = \frac{N_t}{N_g} \\
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

The SROIE dataset takes the micro-F1-score as the evaluation metric. The dataset contains 4 key categories, each category contains one or no entity. If the predicted string of a key category is consistant with the ground-truth string, it will be recorded as a TP sample. The total number of TP, total number of predictions and total number of ground-truth strings will be used to calculate the score. Evaluation scripts can be found at the [ICDAR2019 SROIE official page](https://rrc.cvc.uab.es/?ch=13&com=downloads)(Download page, Task 3 Evaluation script).

| Approach                                                                | Type                    | Precision | Recall | F1    |
| ----------------------------------------------------------------------- | ----------------------- | --------- | ------ | ----- |
| [ViBERTgrid(BERT-base)](../Approaches/approaches_vie.md/#vibertgrid)    | Grid-based              | -         | -      | 96.25 |
| [ViBERTgrid(RoBERTa-base)](../Approaches/approaches_vie.md/#vibertgrid) | Grid-based              | -         | -      | 96.40 |
| [PICK](../Approaches/approaches_vie.md/#pick)                           | GNN-based               | -         | -      | 96.12 |
| [MatchVIE](../Approaches/approaches_vie.md/#matchvie)                   | GNN-based               | -         | -      | 96.57 |
| [GraphDoc](../Approaches/approaches_vie.md/#graphdoc)                   | GNN-based               | -         | -      | 98.45 |
| [LayoutLM-base](../Approaches/approaches_vie.md/#layoutlm)              | Large Scale Pre-trained | 94.38     | 94.38  | 94.38 |
| [LayoutLM-large](../Approaches/approaches_vie.md/#layoutlm)             | Large Scale Pre-trained | 95.24     | 95.24  | 95.24 |
| [LayoutLMv2-base](../Approaches/approaches_vie.md/#layoutlmv2)          | Large Scale Pre-trained | 96.25     | 96.25  | 96.25 |
| [LayoutLMv2-large](../Approaches/approaches_vie.md/#layoutlmv2)         | Large Scale Pre-trained | 99.04     | 96.61  | 97.81 |
| [TILT-base](../Approaches/approaches_vie.md/#tilt)                      | Large Scale Pre-trained | -         | -      | 97.65 |
| [TILT-large](../Approaches/approaches_vie.md/#tilt)                     | Large Scale Pre-trained | -         | -      | 98.10 |
| [BROS-base](../Approaches/approaches_vie.md/#bros)                      | Large Scale Pre-trained | -         | -      | 95.91 |
| [BROS-large](../Approaches/approaches_vie.md/#bros)                     | Large Scale Pre-trained | -         | -      | 96.62 |
| [StrucTexT(eng-base)](../Approaches/approaches_vie.md/#structext)       | Large Scale Pre-trained | -         | -      | 96.88 |
| [StrucTexT(chn&eng-base)](../Approaches/approaches_vie.md/#structext)   | Large Scale Pre-trained | -         | -      | 98.27 |
| [StrucTexT(chn&eng-large)](../Approaches/approaches_vie.md/#structext)  | Large Scale Pre-trained | -         | -      | 98.70 |
| [TRIE(ground-truth)](../Approaches/approaches_vie.md/#trie)             | End to End              | -         | -      | 96.18 |
| [TRIE(end-to-end)](../Approaches/approaches_vie.md/#trie)               | End to End              | -         | -      | 82.06 |
| [VIES(ground-truth)](../Approaches/approaches_vie.md/#vies)             | End to End              | -         | -      | 96.12 |
| [VIES(end-to-end)](../Approaches/approaches_vie.md/#vies)               | End to End              | -         | -      | 91.07 |
| [TCPN(TextLattice)](../Approaches/approaches_vie.md/#tcpn)              | End to End              | -         | -      | 96.54 |
| [TCPN(Tag, end-to-end)](../Approaches/approaches_vie.md/#tcpn)          | End to End              | -         | -      | 95.46 |
| [TCPN(Tag, end-to-end)](../Approaches/approaches_vie.md/#tcpn)          | End to End              | -         | -      | 91.21 |
| [TCPN(Copy&Tag, end-to-end)](../Approaches/approaches_vie.md/#tcpn)     | End to End              | -         | -      | 91.93 |

<br>

## CORD

| Approach                                                        | Type                    | Precision | Recall | F1    |
| --------------------------------------------------------------- | ----------------------- | --------- | ------ | ----- |
| [GraphDoc](../Approaches/approaches_vie.md/#graphdoc)           | Grid-based              | -         | -      | 96.93 |
| [LayoutLM-base](../Approaches/approaches_vie.md/#layoutlm)      | Large Scale Pre-trained | 94.37     | 95.08  | 94.72 |
| [LayoutLM-large](../Approaches/approaches_vie.md/#layoutlm)     | Large Scale Pre-trained | 94.32     | 95.54  | 94.93 |
| [LayoutLMv2-base](../Approaches/approaches_vie.md/#layoutlmv2)  | Large Scale Pre-trained | 94.53     | 95.39  | 94.95 |
| [LayoutLMv2-large](../Approaches/approaches_vie.md/#layoutlmv2) | Large Scale Pre-trained | 95.65     | 96.37  | 96.01 |
| [LayoutLMv3-base](../Approaches/approaches_vie.md/#layoutlmv3)  | Large Scale Pre-trained | -         | -      | 96.56 |
| [LayoutLMv3-large](../Approaches/approaches_vie.md/#layoutlmv3) | Large Scale Pre-trained | -         | -      | 97.46 |
| [DocFormer-base](../Approaches/approaches_vie.md/#docformer)    | Large Scale Pre-trained | 96.52     | 96.14  | 96.33 |
| [DocFormer-large](../Approaches/approaches_vie.md/#docformer)   | Large Scale Pre-trained | 97.25     | 96.74  | 96.99 |
| [TILT-base](../Approaches/approaches_vie.md/#tilt)              | Large Scale Pre-trained | -         | -      | 95.11 |
| [TILT-base](../Approaches/approaches_vie.md/#tilt)              | Large Scale Pre-trained | -         | -      | 95.11 |
| [BROS-base](../Approaches/approaches_vie.md/#tilt)              | Large Scale Pre-trained | -         | -      | 96.50 |
| [BROS-base](../Approaches/approaches_vie.md/#tilt)              | Large Scale Pre-trained | -         | -      | 97.28 |
| [UDoc](../Approaches/approaches_vie.md/#udoc)                   | Large Scale Pre-trained | -         | -      | 96.64 |
| [UDoc*](../Approaches/approaches_vie.md/#udoc)                  | Large Scale Pre-trained | -         | -      | 96.86 |
| [LiLT[EN-R]base](../Approaches/approaches_vie.md/#lilt)         | Large Scale Pre-trained | -         | -      | 96.07 |
| [LILT[InfoXLM]base](../Approaches/approaches_vie.md/#lilt)      | Large Scale Pre-trained | -         | -      | 95.77 |
| [DocReL](../Approaches/approaches_vie.md/#docrel)               | Large Scale Pre-trained | -         | -      | 97.00 |
| [Donut](../Approaches/approaches_vie.md/#donut)                 | End to End              | -         | -      | 91.60 |
| [SPADE](../Approaches/approaches_vie.md/#spade)                 | Others                  | -         | -      | 92.50 |


<br>

## FUNSD

<table align="center">
<tr>
    <th rowspan="2"> Approach </th>
    <th rowspan="2"> Type </th>
    <th colspan="3"> Entity Extraction </th>
    <th colspan="3"> Entity Linking </th>
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
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#matchvie">MatchVIE</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#layoutlm">LayoutLM</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>
<tr>
    <td><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>Graph-based</td>
    <td>-</td>
    <td>-</td>
    <td>82.77</td>
    <td>-</td>
    <td>-</td>
    <td></td>
</tr>

</table>

<br>

## XFUND

<br>

## EPHOIE