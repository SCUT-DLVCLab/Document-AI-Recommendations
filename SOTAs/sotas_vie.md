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
    <td rowspan=3>GNN-based</td>
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
</tr>
    <td rowspan=11>Large Scale Pre-trained</td>
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
    <td rowspan=4>End-to-End</td>
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
    <td rowspan=4>Other Methods</td>
    <td rowspan=4><a href="../Approaches/approaches_vie.md/#vies">VIES</a></td>
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

CORD is an English receipt dataset proposed by Colva AI, and it has two subtasks: Entity Extraction and Entity Linking. It contains 1,000 samples, 800 of which are used for training, 100 for validation, and 100 for testing. The annotations include segment-level and word-level OCR results, key information categories of each segments, and entity linkings. The dataset contains 4 main key information categories, each of them can be further divided into sub-categories, forming a total of 30 key information classes. Different from other datasets, the key information of CORD has a hierarchical relationship, which increases the difficulty of information extraction.

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
    <td rowspan=1>GNN-based</td>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#graphdoc">GraphDoc</a></td>
    <td>-</td>
    <td>-</td>
    <td>96.93</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</tr>
    <td rowspan=17>Large Scale Pre-trained</td>
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
<tr>
    <td rowspan=2>End-to-End</td>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#donut">Donut</a></td>
    <td>-</td>
    <td>-</td>
    <td>91.60</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
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
    <td rowspan=1>Other Methods</td>
    <td colspan=2><a href="../Approaches/approaches_vie.md/#spade">SPADE</a></td>
    <td>-</td>
    <td>-</td>
    <td>92.50</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
</tr>
</table>

<br>

## FUNSD

FUNSD contains 199 form data sampled from the RVL-CDIP dataset, 149 of which are used for training and 50 for testing. The images of the dataset have low resolution, complex layouts, and varied styles. Each form contains three kinds of information including Header, Question, and Answer. The pair linkings between Questions and Answers are given, and the task requires extracting key entities (Entity Extraction) and question-answer pairs (Entity Linking). The dataset contains OCR results at word-level and segment-level. 

It is noticable that the two subtasks are independent in most of the mainstream approaches' settings. Take LayoutLM as an example, in Entity Linking task, the [official implementation](https://github.com/microsoft/unilm) takes groud-truth of keys and values as input and predict the linkings only, the performance of entity extraction is not considered when calculating the metric. 

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
    <td rowspan=2>GNN-based</td>
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
    <td rowspan=27>Large Scale Pre-trained</td>
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

<br>

## EPHOIE