<h3 align="center"> Deep Learning Approaches for </h3>
<h1 align="center"> Visual Information Extraction </h1>

<h2> Introduction </h2>

With the rapid development of Internet technology and the increasing needs of information exchange, quantities of documents are digitalized, stored and distributed in the form of images. Numerous application scenarios, such as receipt understanding, card recognition, automatic paper scoring and document matching, are concerned on how to obtain key information from document images. The process is called visual information extraction (VIE), which aims at **mining, analyzing, and extracting information contained in visually rich documents**. Take receipt understanding as an example, given an image of a receipt, the VIE algorithms will tell information such as store name, product details, price, etc.

Different from the information extraction in traditional natural language processing, results of VIE is not only determined by texts, but also closely related to the doucment layout, font style, block color, figures, charts and other components. The analysis and processing of visually rich documents is a challenging task.

---

<h2> Table of Contents </h2>

- [Grid-based Methods](#grid-based-methods)
  - [ Chargrid ](#-chargrid-)
  - [ BERTgrid](#-bertgrid)
  - [ ViBERTgrid](#-vibertgrid)
- [GNN-based Methods](#gnn-based-methods)
  - [ Liu GNN](#-liu-gnn)
  - [ PICK](#-pick)
  - [ MatchVIE](#-matchvie)
  - [ GraphDoc](#-graphdoc)
- [Large Scale Pre-trained Methods](#large-scale-pre-trained-methods)
  - [ LayoutLM](#-layoutlm)
  - [ LayoutLMv2](#-layoutlmv2)
  - [ LayoutLMv3](#-layoutlmv3)

---
---

<br>

# Grid-based Methods

## <center> Chargrid </center>

*<center> Katti et al. Chargrid: Towards Understanding 2D Documents. EMNLP, 2018. </center>*

<p align="center">
  <img alt="year" src="https://img.shields.io/badge/Year-2018-orange"></img>
  <a href="https://aclanthology.org/D18-1476/">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACL-brightgreen"></img>
  </a> 
  <a href="https://github.com/antoinedelplace/Chargrid">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Unofficial 1-blue"></img>
  </a>
  <a href="https://github.com/sciencefictionlab/chargrid-pytorch">
    <img alt="Code 2" src="https://img.shields.io/badge/Code-Unofficial 2-blue"></img>
  </a>
  <a href="https://github.com/thanhhau097/chargrid2d">
    <img alt="Code 3" src="https://img.shields.io/badge/Code-Unofficial 3-blue"></img>
  </a>
</p>

- **Features**: Grid-based
- **Abstract**: We introduce a novel type of text representation that preserves the 2D layout of a document. This is achieved by encoding each document page as a two-dimensional grid of characters. Based on this representation, we present a generic document understanding pipeline for structured documents. This pipeline makes use of a fully convolutional encoder-decoder network that predicts a segmentation mask and bounding boxes. We demonstrate its capabilities on an information extraction task from invoices and show that it significantly outperforms approaches based on sequential text or document images.

---

## <center> BERTgrid
*<center> Timo I. Denk and Christian Reisswig. BERTgrid: Contextualized Embedding for 2D Document Representation and Understanding. Document Intelligence Workshop at NeurIPS, 2019. </center>*

<p align="center">
  <img alt="year" src="https://img.shields.io/badge/Year-2019-orange"></img>
  <a href="https://arxiv.org/abs/1909.04948">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Features**: Grid-based; Transformer
- **Abstract**: For understanding generic documents, information like font sizes, column layout, and generally the positioning of words may carry semantic information that is crucial for solving a downstream document intelligence task. Our novel BERTgrid, which is based on Chargrid by Katti et al. (2018), represents a document as a grid of contextualized word piece embedding vectors, thereby making its spatial structure and semantics accessible to the processing neural network. The contextualized embedding vectors are retrieved from a BERT language model. We use BERTgrid in combination with a fully convolutional network on a semantic instance segmentation task for extracting fields from invoices. We demonstrate its performance on tabulated line item and document header field extraction.

---

## <center> ViBERTgrid

*<center> Lin et al. ViBERTgrid: A Jointly Trained Multi-Modal 2D Document Representation for Key Information Extraction from Documents. ICDAR, 2021 </center>*

<p align="center">
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://arxiv.org/abs/2105.11672">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/ZeningLin/ViBERTgrid-PyTorch">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Unofficial-blue"></img>
  </a>
</p>

- **Features**: Grid-based; Transformer
- **Abstract**: Recent grid-based document representations like BERTgrid allow the simultaneous encoding of the textual and layout information of a document in a 2D feature map so that state-of-the-art image segmentation and/or object detection models can be straightforwardly leveraged to extract key information from documents. However, such methods have not achieved comparable performance to state-of-the-art sequence- and graph-based methods such as LayoutLM and PICK yet. In this paper, we propose a new multi-modal backbone network by concatenating a BERTgrid to an intermediate layer of a CNN model, where the input of CNN is a document image and the BERTgrid is a grid of word embeddings, to generate a more powerful grid-based document representation, named ViBERTgrid. Unlike BERTgrid, the parameters of BERT and CNN in our multimodal backbone network are trained jointly. Our experimental results demonstrate that this joint training strategy improves significantly the representation ability of ViBERTgrid. Consequently, our ViBERTgrid-based key information extraction approach has achieved state-of-the-art performance on real-world datasets.


---

<br>

# GNN-based Methods

## <center> Liu GNN

*<center> Liu et al. Graph Convolution for Multimodal Information Extraction from Visually Rich Documents. NAACL, 2019. </center>*

<p align="center">
  <img alt="year" src="https://img.shields.io/badge/Year-2019-orange"></img>
  <a href="https://aclanthology.org/N19-2005/">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACL-brightgreen"></img>
  </a>
</p>

- **Features**: GNN-based
- **Abstract**: Visually rich documents (VRDs) are ubiquitous in daily business and life. Examples are purchase receipts, insurance policy documents, custom declaration forms and so on. In VRDs, visual and layout information is critical for document understanding, and texts in such documents cannot be serialized into the one-dimensional sequence without losing information. Classic information extraction models such as BiLSTM-CRF typically operate on text sequences and do not incorporate visual features. In this paper, we introduce a graph convolution based model to combine textual and visual information presented in VRDs. Graph embeddings are trained to summarize the context of a text segment in the document, and further combined with text embeddings for entity extraction. Extensive experiments have been conducted to show that our method outperforms BiLSTM-CRF baselines by significant margins, on two real-world datasets. Additionally, ablation studies are also performed to evaluate the effectiveness of each component of our model.

---

## <center> PICK

*<center> Yu et al. PICK: Processing Key Information Extraction from Documents using Improved Graph Learning-Convolutional Networks. ICPR, 2020. </center>*

<p align="center">
  <img alt="year" src="https://img.shields.io/badge/Year-2020-orange"></img>
  <a href="https://arxiv.org/abs/2004.07464">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/wenwenyu/PICK-pytorch">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Features**: GNN-based
- **Abstract**: Computer vision with state-of-the-art deep learning models has achieved huge success in the field of Optical Character Recognition (OCR) including text detection and recognition tasks recently. However, Key Information Extraction (KIE) from documents as the downstream task of OCR, having a large number of use scenarios in real-world, remains a challenge because documents not only have textual features extracting from OCR systems but also have semantic visual features that are not fully exploited and play a critical role in KIE. Too little work has been devoted to efficiently make full use of both textual and visual features of the documents. In this paper, we introduce PICK, a framework that is effective and robust in handling complex documents layout for KIE by combining graph learning with graph convolution operation, yielding a richer semantic representation containing the textual and visual features and global layout without ambiguity. Extensive experiments on real-world datasets have been conducted to show that our method outperforms baselines methods by significant margins.

---

## <center> MatchVIE

*<center> Tang et al. MatchVIE: Exploiting Match Relevancy between Entities for Visual Information Extraction. IJCAI, 2021. </center>*

<p align="center">
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://www.ijcai.org/proceedings/2021/0144">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-IJCAI-brightgreen"></img>
  </a>
</p>

- **Features**: GNN-based; Entity Linking
- **Abstract**: Visual Information Extraction (VIE) task aims to extract key information from multifarious document images (e.g., invoices and purchase receipts). Most previous methods treat the VIE task simply as a sequence labeling problem or classification problem, which requires models to carefully identify each kind of semantics by introducing multimodal features, such as font, color, layout. But simply introducing multimodal features can't work well when faced with numeric semantic categories or some ambiguous texts. To address this issue, in this paper we propose a novel key-value matching model based on a graph neural network for VIE (MatchVIE). Through key-value matching based on relevancy evaluation, the proposed MatchVIE can bypass the recognitions to various semantics, and simply focuses on the strong relevancy between entities. Besides, we introduce a simple but effective operation, Num2Vec, to tackle the instability of encoded values, which helps model converge more smoothly. Comprehensive experiments demonstrate that the proposed MatchVIE can significantly outperform previous methods. Notably, to the best of our knowledge, MatchVIE may be the first attempt to tackle the VIE task by modeling the relevancy between keys and values and it is a good complement to the existing methods.

---

## <center> GraphDoc

*<center> Zhang et al. Multimodal Pre-training Based on Graph Attention Network for Document Understanding. arXiv preprint, 2022. </center>*

<p align="center">
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://arxiv.org/abs/2203.13530">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/ZZR8066/GraphDoc">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Features**: GNN-based; Pre-trained
- **Abstract**: Document intelligence as a relatively new research topic supports many business applications. Its main task is to automatically read, understand, and analyze documents. However, due to the diversity of formats (invoices, reports, forms, etc.) and layouts in documents, it is difficult to make machines understand documents. In this paper, we present the GraphDoc, a multimodal graph attention-based model for various document understanding tasks. GraphDoc is pre-trained in a multimodal framework by utilizing text, layout, and image information simultaneously. In a document, a text block relies heavily on its surrounding contexts, accordingly we inject the graph structure into the attention mechanism to form a graph attention layer so that each input node can only attend to its neighborhoods. The input nodes of each graph attention layer are composed of textual, visual, and positional features from semantically meaningful regions in a document image. We do the multimodal feature fusion of each node by the gate fusion layer. The contextualization between each node is modeled by the graph attention layer. GraphDoc learns a generic representation from only 320k unlabeled documents via the Masked Sentence Modeling task. Extensive experimental results on the publicly available datasets show that GraphDoc achieves state-of-the-art performance, which demonstrates the effectiveness of our proposed method.


---

<br>

# Large Scale Pre-trained Methods

## <center> LayoutLM

*<center> Xu et al. LayoutLM: Pre-training of Text and Layout for Document Image Understanding. SIGKDD, 2020. </center>*

<p align="center">
  <img alt="year" src="https://img.shields.io/badge/Year-2020-orange"></img>
  <a href="https://arxiv.org/abs/1912.13318">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/microsoft/unilm/tree/master/layoutlm">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>


- **Features**: Pre-trained
- **Abstract**: Pre-training techniques have been verified successfully in a variety of NLP tasks in recent years. Despite the widespread use of pre-training models for NLP applications, they almost exclusively focus on text-level manipulation, while neglecting layout and style information that is vital for document image understanding. In this paper, we propose the LayoutLM to jointly model interactions between text and layout information across scanned document images, which is beneficial for a great number of real-world document image understanding tasks such as information extraction from scanned documents. Furthermore, we also leverage image features to incorporate words' visual information into LayoutLM. To the best of our knowledge, this is the first time that text and layout are jointly learned in a single framework for document-level pre-training. It achieves new state-of-the-art results in several downstream tasks, including form understanding (from 70.72 to 79.27), receipt understanding (from 94.02 to 95.24) and document image classification (from 93.07 to 94.42).

---

## <center> LayoutLMv2

*<center> Xu et al. LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding. ACL, 2021. </center>*

<p align="center">
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://aclanthology.org/2021.acl-long.201/">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACL-brightgreen"></img>
  </a>
  <a href="https://github.com/microsoft/unilm/tree/master/layoutlmv2">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>


- **Features**: Pre-trained
- **Abstract**: Pre-training of text and layout has proved effective in a variety of visually-rich document understanding tasks due to its effective model architecture and the advantage of large-scale unlabeled scanned/digital-born documents. We propose LayoutLMv2 architecture with new pre-training tasks to model the interaction among text, layout, and image in a single multi-modal framework. Specifically, with a two-stream multi-modal Transformer encoder, LayoutLMv2 uses not only the existing masked visual-language modeling task but also the new text-image alignment and text-image matching tasks, which make it better capture the cross-modality interaction in the pre-training stage. Meanwhile, it also integrates a spatial-aware self-attention mechanism into the Transformer architecture so that the model can fully understand the relative positional relationship among different text blocks. Experiment results show that LayoutLMv2 outperforms LayoutLM by a large margin and achieves new state-of-the-art results on a wide variety of downstream visually-rich document understanding tasks, including FUNSD (0.7895 to 0.8420), CORD (0.9493 to 0.9601), SROIE (0.9524 to 0.9781), Kleister-NDA (0.8340 to 0.8520), RVL-CDIP (0.9443 to 0.9564), and DocVQA (0.7295 to 0.8672).

---

## <center> LayoutLMv3

*<center> Huang et al. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. ACMMM, 2022. </center>*

<p align="center">
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://arxiv.org/abs/2204.08387">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/microsoft/unilm/tree/master/layoutlmv3">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>


- **Features**: Pre-trained
- **Abstract**: Self-supervised pre-training techniques have achieved remarkable progress in Document AI. Most multimodal pre-trained models use a masked language modeling objective to learn bidirectional representations on the text modality, but they differ in pre-training objectives for the image modality. This discrepancy adds difficulty to multimodal representation learning. In this paper, we propose LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-patch alignment objective to learn cross-modal alignment by predicting whether the corresponding image patch of a text word is masked. The simple unified architecture and training objectives make LayoutLMv3 a general-purpose pre-trained model for both text-centric and image-centric Document AI tasks. Experimental results show that LayoutLMv3 achieves state-of-the-art performance not only in text-centric tasks, including form understanding, receipt understanding, and document visual question answering, but also in image-centric tasks such as document image classification and document layout analysis.

---
