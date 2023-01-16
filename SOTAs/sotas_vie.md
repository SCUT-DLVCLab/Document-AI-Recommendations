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


## CORD


## FUNSD


## XFUND


## EPHOIE