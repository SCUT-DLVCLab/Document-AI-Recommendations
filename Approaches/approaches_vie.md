<h3 align="center"> Deep Learning Approaches for </h3>
<h1 align="center"> Visual Information Extraction </h1>

<h2> Introduction </h2>

With the rapid advancement of Internet technology and the increasing demand for information exchange, a large number of documents are now digitized, stored, and distributed as images. In various application scenarios, such as receipt understanding, card recognition, automatic paper scoring, and document matching, there is a need to extract key information from document images. This process is known as visual information extraction (VIE), which involves mining, analyzing, and extracting information from visually rich documents. For example, in receipt understanding, VIE algorithms can extract information such as store name, product details, and price from an image of a receipt.

Unlike information extraction from plain text, VIE results are not solely determined by the text content. They are also influenced by factors such as document layout, font style, block color, figures, charts, and other visual components. Analyzing and processing visually rich documents present a challenging task.

Recently proposed deep-learning-based VIE methods can be roughly categorized into six types, namely the **grid-based methods**, the **graph-neural-network-based (GNN-based) methods**, the **Large Scale Pre-trained Models**, the **end-to-end methods**, the **few-shot methods**, **LLM-based methods**,and **other methods**. 

- `Grid-based Methods` treat the document image as a two-dimensional matrix, with text embeddings filling the pixels inside the text bounding box. This grid representation allows for deep feature processing. While grid-based methods are simple and computationally efficient, they may not fully exploit the features of text regions in small sizes.
- `GNN-based Methods` These methods consider text segments as graph nodes and encode the relations between segment coordinates for edge representations. Graph convolution and similar operations are used for feature extraction. GNN-based schemes strike a good balance between cost and performance. However, inherent characteristics of GNN, such as over-smoothing and gradient vanishing, often make the training process challenging.
- `Large Scale Pre-trained Models` These methods leverage pre-training with a large amount of data to obtain effective generic models. They exhibit strong generalizability and can be applied to a wide range of scenarios and other document understanding tasks. However, these models tend to be computationally expensive and require significant computing resources. Finding more efficient architectures and pre-training strategies is an ongoing challenge. 
- VIE is not a standalone process, results from text detection and recognition (optical character recognition, OCR) are required as prerequisites. Issues in OCR results, such as coordinate mismatches and text recognition errors, can impact subsequent steps. Researchers have explored `End-to-end` paradigms to reduce OCR error accumulation, but there is still progress to be made compared to state-of-the-art methods.
- `Few-shot methods` propose efficient structures to enhance model generalization and fully explore intrinsic features with only a small number of samples. While some progress has been made, there is still room for improvement in overall model accuracy for practical applications.
- Large Language Models (LLMs) have demonstrated exceptional performance in various NLP tasks. `LLM-based methods` take advantage of the powerful capabilities of the LLMs to handle VIE by designing appropriate prompt inputs or applying finetuning with task-specific instructions. These approaches offer a new solution to the field of Document Understanding.

---

<h2> 🗒️List of Index </h2>

- [Grid-based Methods](#grid-based-methods)
  - [Chargrid](#chargrid)
  - [BERTgrid](#bertgrid)
  - [ViBERTgrid](#vibertgrid)
  - [VisualWordgrid](#visualwordgrid)
  - [MSAU-PAF](#msau-paf)
- [GNN-based Methods](#gnn-based-methods)
  - [Liu GNN](#liu-gnn)
  - [PICK](#pick)
  - [MatchVIE](#matchvie)
  - [GraphDoc](#graphdoc)
  - [FormNet](#formnet)
  - [FormNetV2](#formnetv2)
  - [GeoContrastNet](#geocontrastnet)
- [Large Scale Pre-trained Models](#large-scale-pre-trained-models)
  - [LayoutLM](#layoutlm)
  - [LayoutLMv2](#layoutlmv2)
  - [LayoutXLM](#layoutxlm)
  - [LayoutLMv3](#layoutlmv3)
  - [LiLT](#lilt)
  - [StrucTexT](#structext)
  - [XYLayoutLM](#xylayoutlm)
  - [SelfDoc](#selfdoc)
  - [DocFormer](#docformer)
  - [TILT](#tilt)
  - [UDoc](#udoc)
  - [DocReL](#docrel)
  - [StructuralLM](#structurallm)
  - [BROS](#bros)
  - [Wei Robust Layout-aware IE](#wei-robust-layout-aware-ie)
  - [WUKONG-READER](#wukong-reader)
  - [ERNIE-Layout](#ernie-layout)
  - [QGN](#qgn)
  - [GeoLayoutLM](#geolayoutlm)
  - [KVPFormer](#kvpformer)
  - [TPP](#tpp)
  - [GraphLayoutLM](#graphlayoutlm)
  - [HGALayoutLM](#hgalayoutlm)
  - [DocFormerv2](#docformerv2)
  - [DocTr](#doctr)
  - [LayoutMask](#layoutmask)
- [End-to-End Methods](#end-to-end-methods)
  - [EATEN](#eaten)
  - [TRIE](#trie)
  - [VIES](#vies)
  - [Donut🍩](#donut)
  - [Dessurt](#dessurt)
  - [StrucTexTv2](#structextv2)
  - [ESP](#esp)
  - [Kuang CFAM](#kuang-cfam)
  - [Pix2Struct](#pix2struct)
  - [SeRum](#serum)
  - [UDOP](#udop)
  - [CREPE](#crepe)
- [Few-shot Methods](#few-shot-methods)
  - [LASER](#laser)
  - [LF-Attn](#lf-attn)
  - [Deep Partial Graph Matching](#deep-partial-graph-matching)
- [LLM-based Methods](#llm-based-methods)
  - [ICL-D3IE](#icl-d3ie)
  - [mPLUG-DocOwl](#mplug-docowl)
  - [mPLUG-DocOwl 1.5](#mplug-docowl-15)
  - [UReader](#ureader)
  - [LLaVAR](#llavar)
  - [UniDoc](#unidoc)
  - [DocPedia](#docpedia)
  - [Monkey](#monkey)
  - [HRVDA](#hrvda)
  - [DoCo](#doco)
  - [OmniParser](#omniparser)
  - [LayoutLLM](#layoutllm)
- [Other Methods](#other-methods)
  - [TCPN](#tcpn)
  - [SPADE♠](#spade)

---
---

<br>

# Grid-based Methods

## Chargrid 

*Katti et al. Chargrid: Towards Understanding 2D Documents. EMNLP, 2018.*

<p>
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
  <a href="https://github.com/hikopensource/DAVAR-Lab-OCR/tree/main/demo/text_ie/chargrid">
    <img alt="Code 4" src="https://img.shields.io/badge/Code-Unofficial 4-blue"></img> 
  </a>
</p>

- **Highlights**: Seminal Work
- **Modalities**: Semantic; Layout
- **Abstract**: We introduce a novel type of text representation that preserves the 2D layout of a document. This is achieved by encoding each document page as a two-dimensional grid of characters. Based on this representation, we present a generic document understanding pipeline for structured documents. This pipeline makes use of a fully convolutional encoder-decoder network that predicts a segmentation mask and bounding boxes. We demonstrate its capabilities on an information extraction task from invoices and show that it significantly outperforms approaches based on sequential text or document images.

---

##  BERTgrid
*Timo I. Denk and Christian Reisswig. BERTgrid: Contextualized Embedding for 2D Document Representation and Understanding. Document Intelligence Workshop at NeurIPS, 2019.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2019-orange"></img>
  <a href="https://arxiv.org/abs/1909.04948">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Introduce Large LM to VIE 
- **Modalities**: Semantic; Layout
- **Abstract**: For understanding generic documents, information like font sizes, column layout, and generally the positioning of words may carry semantic information that is crucial for solving a downstream document intelligence task. Our novel BERTgrid, which is based on Chargrid by Katti et al. (2018), represents a document as a grid of contextualized word piece embedding vectors, thereby making its spatial structure and semantics accessible to the processing neural network. The contextualized embedding vectors are retrieved from a BERT language model. We use BERTgrid in combination with a fully convolutional network on a semantic instance segmentation task for extracting fields from invoices. We demonstrate its performance on tabulated line item and document header field extraction.

---

##  ViBERTgrid

*Lin et al. ViBERTgrid: A Jointly Trained Multi-Modal 2D Document Representation for Key Information Extraction from Documents. ICDAR, 2021*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://arxiv.org/abs/2105.11672">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/ZeningLin/ViBERTgrid-PyTorch">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Unofficial-blue"></img>
  </a>
</p>

- **Highlights**: Flexible Modeling Level; Joint Training
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Recent grid-based document representations like BERTgrid allow the simultaneous encoding of the textual and layout information of a document in a 2D feature map so that state-of-the-art image segmentation and/or object detection models can be straightforwardly leveraged to extract key information from documents. However, such methods have not achieved comparable performance to state-of-the-art sequence- and graph-based methods such as LayoutLM and PICK yet. In this paper, we propose a new multi-modal backbone network by concatenating a BERTgrid to an intermediate layer of a CNN model, where the input of CNN is a document image and the BERTgrid is a grid of word embeddings, to generate a more powerful grid-based document representation, named ViBERTgrid. Unlike BERTgrid, the parameters of BERT and CNN in our multimodal backbone network are trained jointly. Our experimental results demonstrate that this joint training strategy improves significantly the representation ability of ViBERTgrid. Consequently, our ViBERTgrid-based key information extraction approach has achieved state-of-the-art performance on real-world datasets.

---

## VisualWordgrid 

*Mohamed Kerroumi, Othmane Sayem and Aymen Shabou. VisualWordGrid: Information Extraction From Scanned Documents Using A Multimodal Approach. ICDAR, 2021*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://arxiv.org/abs/2010.02358">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a> 
</p>

- **Highlights**: Introduce visual modality to grid
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: We introduce a novel approach for scanned document representation to perform field extraction. It allows the simultaneous encoding of the textual, visual and layout information in a 3-axis tensor used as an input to a segmentation model. We improve the recent Chargrid and Wordgrid \cite{chargrid} models in several ways, first by taking into account the visual modality, then by boosting its robustness in regards to small datasets while keeping the inference time low. Our approach is tested on public and private document-image datasets, showing higher performances compared to the recent state-of-the-art methods.


## MSAU-PAF

*Tuan An Nguyen Dang, Duc Thanh Hoang, Quang Bach Tran, et al. End-to-End Hierarchical Relation Extraction for Generic Form Understanding. ICPR, 2020.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2020-orange"></img>
  <a href="https://arxiv.org/abs/2106.00980">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a> 
</p>

- **Highlights**: End-to-End document pair extraction
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Form understanding is a challenging problem which aims to recognize semantic entities from the input document and their hierarchical relations. Previous approaches face significant difficulty dealing with the complexity of the task, thus treat these objectives separately. To this end, we present a novel deep neural network to jointly perform both entity detection and link prediction in an end-to-end fashion. Our model extends the Multi-stage Attentional U-Net architecture with the Part-Intensity Fields and Part-Association Fields for link prediction, enriching the spatial information flow with the additional supervision from entity linking. We demonstrate the effectiveness of the model on the Form Understanding in Noisy Scanned Documents (FUNSD) dataset, where our method substantially outperforms the original model and state-of-the-art baselines in both Entity Labeling and Entity Linking task.

<br>
<br>

# GNN-based Methods

##  Liu GNN

*Liu et al. Graph Convolution for Multimodal Information Extraction from Visually Rich Documents. NAACL, 2019*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2019-orange"></img>
  <a href="https://aclanthology.org/N19-2005/">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACL-brightgreen"></img>
  </a>
</p>

- **Highlights**: Introduce GCN to VIE
- **Modalities**: Semantic; Layout
- **Abstract**: Visually rich documents (VRDs) are ubiquitous in daily business and life. Examples are purchase receipts, insurance policy documents, custom declaration forms and so on. In VRDs, visual and layout information is critical for document understanding, and texts in such documents cannot be serialized into the one-dimensional sequence without losing information. Classic information extraction models such as BiLSTM-CRF typically operate on text sequences and do not incorporate visual features. In this paper, we introduce a graph convolution based model to combine textual and visual information presented in VRDs. Graph embeddings are trained to summarize the context of a text segment in the document, and further combined with text embeddings for entity extraction. Extensive experiments have been conducted to show that our method outperforms BiLSTM-CRF baselines by significant margins, on two real-world datasets. Additionally, ablation studies are also performed to evaluate the effectiveness of each component of our model.

---

##  PICK

*Yu et al. PICK: Processing Key Information Extraction from Documents using Improved Graph Learning-Convolutional Networks. ICPR, 2020*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2020-orange"></img>
  <a href="https://arxiv.org/abs/2004.07464">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/wenwenyu/PICK-pytorch">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Strong Baseline
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Computer vision with state-of-the-art deep learning models has achieved huge success in the field of Optical Character Recognition (OCR) including text detection and recognition tasks recently. However, Key Information Extraction (KIE) from documents as the downstream task of OCR, having a large number of use scenarios in real-world, remains a challenge because documents not only have textual features extracting from OCR systems but also have semantic visual features that are not fully exploited and play a critical role in KIE. Too little work has been devoted to efficiently make full use of both textual and visual features of the documents. In this paper, we introduce PICK, a framework that is effective and robust in handling complex documents layout for KIE by combining graph learning with graph convolution operation, yielding a richer semantic representation containing the textual and visual features and global layout without ambiguity. Extensive experiments on real-world datasets have been conducted to show that our method outperforms baselines methods by significant margins.

---

##  MatchVIE

*Tang et al. MatchVIE: Exploiting Match Relevancy between Entities for Visual Information Extraction. IJCAI, 2021*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://www.ijcai.org/proceedings/2021/0144">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-IJCAI-brightgreen"></img>
  </a>
</p>

- **Highlights**: Entity Linking
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Visual Information Extraction (VIE) task aims to extract key information from multifarious document images (e.g., invoices and purchase receipts). Most previous methods treat the VIE task simply as a sequence labeling problem or classification problem, which requires models to carefully identify each kind of semantics by introducing multimodal features, such as font, color, layout. But simply introducing multimodal features can't work well when faced with numeric semantic categories or some ambiguous texts. To address this issue, in this paper we propose a novel key-value matching model based on a graph neural network for VIE (MatchVIE). Through key-value matching based on relevancy evaluation, the proposed MatchVIE can bypass the recognitions to various semantics, and simply focuses on the strong relevancy between entities. Besides, we introduce a simple but effective operation, Num2Vec, to tackle the instability of encoded values, which helps model converge more smoothly. Comprehensive experiments demonstrate that the proposed MatchVIE can significantly outperform previous methods. Notably, to the best of our knowledge, MatchVIE may be the first attempt to tackle the VIE task by modeling the relevancy between keys and values and it is a good complement to the existing methods.

---

##  GraphDoc

*Zhang et al. Multimodal Pre-training Based on Graph Attention Network for Document Understanding. arXiv preprint, 2022*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://arxiv.org/abs/2203.13530">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/ZZR8066/GraphDoc">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Low Data Consumption
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Document intelligence as a relatively new research topic supports many business applications. Its main task is to automatically read, understand, and analyze documents. However, due to the diversity of formats (invoices, reports, forms, etc.) and layouts in documents, it is difficult to make machines understand documents. In this paper, we present the GraphDoc, a multimodal graph attention-based model for various document understanding tasks. GraphDoc is pre-trained in a multimodal framework by utilizing text, layout, and image information simultaneously. In a document, a text block relies heavily on its surrounding contexts, accordingly we inject the graph structure into the attention mechanism to form a graph attention layer so that each input node can only attend to its neighborhoods. The input nodes of each graph attention layer are composed of textual, visual, and positional features from semantically meaningful regions in a document image. We do the multimodal feature fusion of each node by the gate fusion layer. The contextualization between each node is modeled by the graph attention layer. GraphDoc learns a generic representation from only 320k unlabeled documents via the Masked Sentence Modeling task. Extensive experimental results on the publicly available datasets show that GraphDoc achieves state-of-the-art performance, which demonstrates the effectiveness of our proposed method.

---

##  FormNet

*Lee et al. FormNet: Structural Encoding beyond Sequential Modeling in Form Document Information Extraction. ACL, 2022*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://arxiv.org/abs/2203.08411">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Low Data Consumption
- **Modalities**: Semantic; Layout
- **Abstract**: Sequence modeling has demonstrated state-of-the-art performance on natural language and document understanding tasks. However, it is challenging to correctly serialize tokens in form-like documents in practice due to their variety of layout patterns. We propose Form Net, a structure-aware sequence model to mitigate the suboptimal serialization of forms. First, we design Rich Attention that leverages the spatial relationship between tokens in a form for more precise attention score calculation. Second, we construct Super-Tokens for each word by embedding representations from their neighboring tokens through graph convolutions. FormNet therefore explicitly recovers local syntactic information that may have been lost during serialization. In experiments, FormNet outperforms existing methods with a more compact model size and less pre training data, establishing new state-of-the-art performance on CORD, FUNSD and Payment benchmarks.

---

##  FormNetV2

*Lee et al. FormNetV2: Multimodal Graph Contrastive Learning for Form Document Information Extraction. ACL, 2023*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2305.02549">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Excellent performance; Contrastive learning
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: The recent advent of self-supervised pre-training techniques has led to a surge in the use of multimodal learning in form document understanding. However, existing approaches that extend the mask language modeling to other modalities require careful multi-task tuning, complex reconstruction target designs, or additional pre-training data. In FormNetV2, we introduce a centralized multimodal graph contrastive learning strategy to unify self-supervised pre-training for all modalities in one loss. The graph contrastive objective maximizes the agreement of multimodal representations, providing a natural interplay for all modalities without special customization. In addition, we extract image features within the bounding box that joins a pair of tokens connected by a graph edge, capturing more targeted visual cues without loading a sophisticated and separately pre-trained image embedder. FormNetV2 establishes new state-of-the-art performance on FUNSD, CORD, SROIE and Payment benchmarks with a more compact model size.

---

## GeoContrastNet

*Nil Biescas, et al. GeoContrastnet: Contrastive Key-value Edge learning for Luanguage-Agnostic Document Understanding. ICDAR, 2024.*


<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2024-orange"></img>
  <a href="https://arxiv.org/pdf/2405.03104">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/NilBiescas/GeoContrastNet">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Entity Linking; Contrastive learning
- **Modalities**: Layout; Visual
- **Abstract**: This paper presents GeoContrastNet, a language-agnostic framework to structured document understanding (DU) by integrating a contrastive learning objective with graph attention networks (GATs), emphasizing the significant role of geometric features. We propose a novel methodology that combines geometric edge features with visual features within an overall two-staged GAT-based framework, demonstrating promising results in both link prediction and semantic entity recognition performance. Our findings reveal that combining both geometric and visual features could match the capabilities of large DU models that rely heavily on Optical Character Recognition (OCR) features in terms of performance accuracy and efficiency. This approach underscores the critical importance of relational layout information between the named text entities in a semi-structured layout of a page. Specifically, our results highlight the model's proficiency in identifying key-value relationships within the FUNSD dataset for forms and also discovering the spatial relationships in table-structured layouts for RVLCDIP business invoices


<br>
<br>

# Large Scale Pre-trained Models

##  LayoutLM

*Xu et al. LayoutLM: Pre-training of Text and Layout for Document Image Understanding. SIGKDD, 2020*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2020-orange"></img>
  <a href="https://arxiv.org/abs/1912.13318">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/microsoft/unilm/tree/master/layoutlm">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>


- **Highlights**: Seminal Work; Strong Baseline
- **Modalities**: Semantic; Layout; Visual(fine-tuning)
- **Abstract**: Pre-training techniques have been verified successfully in a variety of NLP tasks in recent years. Despite the widespread use of pre-training models for NLP applications, they almost exclusively focus on text-level manipulation, while neglecting layout and style information that is vital for document image understanding. In this paper, we propose the LayoutLM to jointly model interactions between text and layout information across scanned document images, which is beneficial for a great number of real-world document image understanding tasks such as information extraction from scanned documents. Furthermore, we also leverage image features to incorporate words' visual information into LayoutLM. To the best of our knowledge, this is the first time that text and layout are jointly learned in a single framework for document-level pre-training. It achieves new state-of-the-art results in several downstream tasks, including form understanding (from 70.72 to 79.27), receipt understanding (from 94.02 to 95.24) and document image classification (from 93.07 to 94.42).

---

##  LayoutLMv2

*Xu et al. LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding. ACL, 2021*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://aclanthology.org/2021.acl-long.201/">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACL-brightgreen"></img>
  </a>
  <a href="https://github.com/microsoft/unilm/tree/master/layoutlmv2">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>


- **Highlights**: Introduce Visual Modality
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Pre-training of text and layout has proved effective in a variety of visually-rich document understanding tasks due to its effective model architecture and the advantage of large-scale unlabeled scanned/digital-born documents. We propose LayoutLMv2 architecture with new pre-training tasks to model the interaction among text, layout, and image in a single multi-modal framework. Specifically, with a two-stream multi-modal Transformer encoder, LayoutLMv2 uses not only the existing masked visual-language modeling task but also the new text-image alignment and text-image matching tasks, which make it better capture the cross-modality interaction in the pre-training stage. Meanwhile, it also integrates a spatial-aware self-attention mechanism into the Transformer architecture so that the model can fully understand the relative positional relationship among different text blocks. Experiment results show that LayoutLMv2 outperforms LayoutLM by a large margin and achieves new state-of-the-art results on a wide variety of downstream visually-rich document understanding tasks, including FUNSD (0.7895 to 0.8420), CORD (0.9493 to 0.9601), SROIE (0.9524 to 0.9781), Kleister-NDA (0.8340 to 0.8520), RVL-CDIP (0.9443 to 0.9564), and DocVQA (0.7295 to 0.8672).

---

##  LayoutXLM

*Xu et al. LayoutXLM: Multimodal Pre-Training for Multilingual Visually-Rich Document Understanding. arXiv preprint, 2021.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://arxiv.org/abs/2104.08836">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/microsoft/unilm/tree/master/layoutxlm">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Multi-language
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Multimodal pre-training with text, layout, and image has achieved SOTA performance for visually-rich document understanding tasks recently, which demonstrates the great potential for joint learning across different modalities. In this paper, we present LayoutXLM, a multimodal pre-trained model for multilingual document understanding, which aims to bridge the language barriers for visually-rich document understanding. To accurately evaluate LayoutXLM, we also introduce a multilingual form understanding benchmark dataset named XFUND, which includes form understanding samples in 7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese), and key-value pairs are manually labeled for each language. Experiment results show that the LayoutXLM model has significantly outperformed the existing SOTA cross-lingual pre-trained models on the XFUND dataset. 

---

##  LayoutLMv3

*Huang et al. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. ACMMM, 2022.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://arxiv.org/abs/2204.08387">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/microsoft/unilm/tree/master/layoutlmv3">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>


- **Highlights**: Image Patch Encoding; Excellent Performance
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Self-supervised pre-training techniques have achieved remarkable progress in Document AI. Most multimodal pre-trained models use a masked language modeling objective to learn bidirectional representations on the text modality, but they differ in pre-training objectives for the image modality. This discrepancy adds difficulty to multimodal representation learning. In this paper, we propose LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-patch alignment objective to learn cross-modal alignment by predicting whether the corresponding image patch of a text word is masked. The simple unified architecture and training objectives make LayoutLMv3 a general-purpose pre-trained model for both text-centric and image-centric Document AI tasks. Experimental results show that LayoutLMv3 achieves state-of-the-art performance not only in text-centric tasks, including form understanding, receipt understanding, and document visual question answering, but also in image-centric tasks such as document image classification and document layout analysis.

---

## LiLT

*Jiapeng Wang, Lianwen Jin and Kai Ding. LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding. ACL, 2022.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://aclanthology.org/2022.acl-long.534/">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACL-brightgreen"></img>
  </a>
  <a href="https://github.com/jpWang/LiLT">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Multi-language; Flexible LM backbone
- **Modalities**: Semantic; Layout
- **Abstract**: Structured document understanding has attracted considerable attention and made significant progress recently, owing to its crucial role in intelligent document processing. However, most existing related models can only deal with the document data of specific language(s) (typically English) included in the pre-training collection, which is extremely limited. To address this issue, we propose a simple yet effective Language-independent Layout Transformer (LiLT) for structured document understanding. LiLT can be pre-trained on the structured documents of a single language and then directly fine-tuned on other languages with the corresponding off-the-shelf monolingual/multilingual pre-trained textual models. Experimental results on eight languages have shown that LiLT can achieve competitive or even superior performance on diverse widely-used downstream benchmarks, which enables language-independent benefit from the pre-training of document layout structure.

---

## StrucTexT

*Li et al. StrucTexT: Structured Text Understanding with Multi-Modal Transformers. ACMMM, 2021.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://arxiv.org/abs/2108.02923">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/PaddlePaddle/VIMER/tree/main/StrucTexT/v1">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Entity Linking; Multi-language
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Structured text understanding on Visually Rich Documents (VRDs) is a crucial part of Document Intelligence. Due to the complexity of content and layout in VRDs, structured text understanding has been a challenging task. Most existing studies decoupled this problem into two sub-tasks: entity labeling and entity linking, which require an entire understanding of the context of documents at both token and segment levels. However, little work has been concerned with the solutions that efficiently extract the structured data from different levels. This paper proposes a unified framework named StrucTexT, which is flexible and effective for handling both sub-tasks. Specifically, based on the transformer, we introduce a segment-token aligned encoder to deal with the entity labeling and entity linking tasks at different levels of granularity. Moreover, we design a novel pre-training strategy with three self-supervised tasks to learn a richer representation. StrucTexT uses the existing Masked Visual Language Modeling task and the new Sentence Length Prediction and Paired Boxes Direction tasks to incorporate the multi-modal information across text, image, and layout. We evaluate our method for structured text understanding at segment-level and token-level and show it outperforms the state-of-the-art counterparts with significantly superior performance on the FUNSD, SROIE, and EPHOIE datasets.


---

## XYLayoutLM

*Gu et al. XYLayoutLM: Towards Layout-Aware Multimodal Networks for Visually-Rich Document Understanding. CVPR, 2022.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://openaccess.thecvf.com/content/CVPR2022/html/Gu_XYLayoutLM_Towards_Layout-Aware_Multimodal_Networks_for_Visually-Rich_Document_Understanding_CVPR_2022_paper.html">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-CVF-brightgreen"></img>
  </a>
</p>

- **Highlights**: Reading Order Sort; Varial Length Input
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Recently, various multimodal networks for Visually-Rich Document Understanding(VRDU) have been proposed, showing the promotion of transformers by integrating visual and layout information with the text embeddings. However, most existing approaches utilize the position embeddings to incorporate the sequence information, neglecting the noisy improper reading order obtained by OCR tools. In this paper, we propose a robust layout-aware multimodal network named XYLayoutLM to capture and leverage rich layout information from proper reading orders produced by our Augmented XY Cut. Moreover, a Dilated Conditional Position Encoding module is proposed to deal with the input sequence of variable lengths, and it additionally extracts local layout information from both textual and visual modalities while generating position embeddings. Experiment results show that our XYLayoutLM achieves competitive results on document understanding tasks.

---

## SelfDoc

*Li et al. SelfDoc: Self-supervised Document Representation Learning. CVPR, 2021.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://openaccess.thecvf.com/content/CVPR2021/html/Li_SelfDoc_Self-Supervised_Document_Representation_Learning_CVPR_2021_paper.html">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-CVF-brightgreen"></img>
  </a>
</p>

- **Highlights**: Cross-modality Encoder
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: We propose SelfDoc, a task-agnostic pre-training framework for document image understanding. Because documents are multimodal and are intended for sequential reading, our framework exploits the positional, textual, and visual information of every semantically meaningful component in a document, and it models the contextualization between each block of content. Unlike existing document pre-training models, our model is coarse-grained instead of treating individual words as input, therefore avoiding an overly fine-grained with excessive contextualization. Beyond that, we introduce cross-modal learning in the model pre-training phase to fully leverage multimodal information from unlabeled documents. For downstream usage, we propose a novel modality-adaptive attention mechanism for multimodal feature fusion by adaptively emphasizing language and vision signals. Our framework benefits from self-supervised pre-training on documents without requiring annotations by a feature masking training strategy. It achieves superior performance on multiple downstream tasks with significantly fewer document images used in the pre-training stage compared to previous works.

---

## DocFormer

*Appalaraju et al. DocFormer: End-to-End Transformer for Document Understanding. ICCV, 2021.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://openaccess.thecvf.com/content/ICCV2021/html/Appalaraju_DocFormer_End-to-End_Transformer_for_Document_Understanding_ICCV_2021_paper.html">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-CVF-brightgreen"></img>
  </a>
  <a href="https://github.com/shabie/docformer">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Unofficial-blue"></img>
  </a>
</p>

- **Highlights**: Novel Multi-modal Attention Layer
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: We present DocFormer - a multi-modal transformer based architecture for the task of Visual Document Understanding (VDU). VDU is a challenging problem which aims to understand documents in their varied formats(forms, receipts etc.) and layouts. In addition, DocFormer is pre-trained in an unsupervised fashion using carefully designed tasks which encourage multi-modal interaction. DocFormer uses text, vision and spatial features and combines them using a novel multi-modal self-attention layer. DocFormer also shares learned spatial embeddings across modalities which makes it easy for the model to correlate text to visual tokens and vice versa. DocFormer is evaluated on 4 different datasets each with strong baselines. DocFormer achieves state-of-the-art results on all of them, sometimes beating models 4x its size (in no. of parameters)


---

## TILT

*Powalski et al. Going Full-TILT Boogie on Document Understanding with Text-Image-Layout Transformer. ICDAR, 2021.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://arxiv.org/abs/2102.09550">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Novel Attention Mechanism
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: We address the challenging problem of Natural Language Comprehension beyond plain-text documents by introducing the TILT neural network architecture which simultaneously learns layout information, visual features, and textual semantics. Contrary to previous approaches, we rely on a decoder capable of unifying a variety of problems involving natural language. The layout is represented as an attention bias and complemented with contextualized visual information, while the core of our model is a pretrained encoder-decoder Transformer. Our novel approach achieves state-of-the-art results in extracting information from documents and answering questions which demand layout understanding (DocVQA, CORD, SROIE). At the same time, we simplify the process by employing an end-to-end model.


---

## UDoc

*Gu et al. UniDoc: Unified Pretraining Framework for Document Understanding. NeurIPS, 2021.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://proceedings.neurips.cc/paper/2021/hash/0084ae4bc24c0795d1e6a4f58444d39b-Abstract.html">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-NeruIPS-brightgreen"></img>
  </a>
</p>

- **Highlights**: Gated Cross-Attention
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Document intelligence automates the extraction of information from documents and supports many business applications. Recent self-supervised learning methods on large-scale unlabeled document datasets have opened up promising directions towards reducing annotation efforts by training models with self-supervised objectives. However, most of the existing document pretraining methods are still language-dominated. We present UDoc, a new unified pretraining framework for document understanding. UDoc is designed to support most document understanding tasks, extending the Transformer to take multimodal embeddings as input. Each input element is composed of words and visual features from a semantic region of the input document image. An important feature of UDoc is that it learns a generic representation by making use of three self-supervised losses, encouraging the representation to model sentences, learn similarities, and align modalities. Extensive empirical analysis demonstrates that the pretraining procedure learns better joint representations and leads to improvements in downstream tasks.


---

## DocReL

*Li et al. Relational Representation Learning in Visually-Rich Documents. ACMMM, 2022.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://arxiv.org/abs/2205.02411">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Contrastive Learning; Global/Local Relational Consistency Modeling
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Relational understanding is critical for a number of visually-rich documents (VRDs) understanding tasks. Through multi-modal pre-training, recent studies provide comprehensive contextual representations and exploit them as prior knowledge for downstream tasks. In spite of their impressive results, we observe that the widespread relational hints (e.g., relation of key/value fields on receipts) built upon contextual knowledge are not excavated yet. To mitigate this gap, we propose DocReL, a Document Relational Representation Learning framework. The major challenge of DocReL roots in the variety of relations. From the simplest pairwise relation to the complex global structure, it is infeasible to conduct supervised training due to the definition of relation varies and even conflicts in different tasks. To deal with the unpredictable definition of relations, we propose a novel contrastive learning task named Relational Consistency Modeling (RCM), which harnesses the fact that existing relations should be consistent in differently augmented positive views. RCM provides relational representations which are more compatible to the urgent need of downstream tasks, even without any knowledge about the exact definition of relation. DocReL achieves better performance on a wide variety of VRD relational understanding tasks, including table structure recognition, key information extraction and reading order detection.


---

## StructuralLM

*Li et al. StructuralLM: Structural Pre-training for Form Understanding. ACL, 2021.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://aclanthology.org/2021.acl-long.493/">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACL-brightgreen"></img>
  </a>
  <a href="https://github.com/alibaba/AliceMind/tree/main/StructuralLM">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Cell-level Modeling
- **Modalities**: Semantic; Layout
- **Abstract**: Large pre-trained language models achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, they almost exclusively focus on text-only representation, while neglecting cell-level layout information that is important for form image understanding. In this paper, we propose a new pre-training approach, StructuralLM, to jointly leverage cell and layout information from scanned documents. Specifically, we pre-train StructuralLM with two new designs to make the most of the interactions of cell and layout information: 1) each cell as a semantic unit; 2) classification of cell positions. The pre-trained StructuralLM achieves new state-of-the-art results in different types of downstream tasks, including form understanding (from 78.95 to 85.14), document visual question answering (from 72.59 to 83.94) and document image classification (from 94.43 to 96.08).


---

## BROS

*Hong et al. BROS: A Pre-Trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents. AAAI, 2022.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/21322">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-AAAI-brightgreen"></img>
  </a>
  <a href="https://github.com/clovaai/bros">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Out-of-Order OCR Results Compatible
- **Modalities**: Semantic; Layout
- **Abstract**: Key information extraction (KIE) from document images requires understanding the contextual and spatial semantics of texts in two-dimensional (2D) space. Many recent studies try to solve the task by developing pre-trained language models focusing on combining visual features from document images with texts and their layout. On the other hand, this paper tackles the problem by going back to the basic: effective combination of text and layout. Specifically, we propose a pre-trained language model, named BROS (BERT Relying On Spatiality), that encodes relative positions of texts in 2D space and learns from unlabeled documents with area-masking strategy. With this optimized training scheme for understanding texts in 2D space, BROS shows comparable or better performance compared to previous methods on four KIE benchmarks (FUNSD, SROIE*, CORD, and SciTSR) without relying on visual features. This paper also reveals two real-world challenges in KIE tasks-(1) minimizing the error from incorrect text ordering and (2) efficient learning from fewer downstream examples-and demonstrates the superiority of BROS over previous methods.


---

## Wei Robust Layout-aware IE

*Wei et al. Robust layout-aware IE for visually rich documents with pre-trained language models. SIGIR, 2020.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2020-orange"></img>
  <a href="https://arxiv.org/abs/2005.11017">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Font-type Encoding
- **Modalities**: Semantic; Layout
- **Abstract**: Many business documents processed in modern NLP and IR pipelines are visually rich: in addition to text, their semantics can also be captured by visual traits such as layout, format, and fonts. We study the problem of information extraction from visually rich documents (VRDs) and present a model that combines the power of large pre-trained language models and graph neural networks to efficiently encode both textual and visual information in business documents. We further introduce new fine-tuning objectives to improve in-domain unsupervised fine-tuning to better utilize large amount of unlabeled in-domain data. We experiment on real world invoice and resume data sets and show that the proposed method outperforms strong text-based RoBERTa baselines by 6.3% absolute F1 on invoices and 4.7% absolute F1 on resumes. When evaluated in a few-shot setting, our method requires up to 30x less annotation data than the baseline to achieve the same level of performance at ~90% F1.

---

## WUKONG-READER

*Bai et al. WUKONG-READER: Multi-modal Pre-training for Fine-grained Visual Document Understanding. arXiv preprint, 2022.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://arxiv.org/abs/2212.09621">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Segment-level Modeling; Excellent Performance
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Unsupervised pre-training on millions of digital-born or scanned documents has shown promising advances in visual document understanding~(VDU). While various vision-language pre-training objectives are studied in existing solutions, the document textline, as an intrinsic granularity in VDU, has seldom been explored so far. A document textline usually contains words that are spatially and semantically correlated, which can be easily obtained from OCR engines. In this paper, we propose Wukong-Reader, trained with new pre-training objectives to leverage the structural knowledge nested in document textlines. We introduce textline-region contrastive learning to achieve fine-grained alignment between the visual regions and texts of document textlines. Furthermore, masked region modeling and textline-grid matching are also designed to enhance the visual and layout representations of textlines. Experiments show that our Wukong-Reader has superior performance on various VDU tasks such as information extraction. The fine-grained alignment over textlines also empowers Wukong-Reader with promising localization ability.

---

## ERNIE-Layout

*Peng et al. ERNIE-Layout: Layout Knowledge Enhanced Pre-training for Visually-rich Document Understanding. EMNLP Findings, 2022.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://aclanthology.org/2022.findings-emnlp.274/">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACL-brightgreen"></img>
  </a>
  <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.4/model_zoo/ernie-layout">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Reading Order Serialization
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Recent years have witnessed the rise and success of pre-training techniques in visually-rich document understanding. However, most existing methods lack the systematic mining and utilization of layout-centered knowledge, leading to sub-optimal performances. In this paper, we propose ERNIE-Layout, a novel document pre-training solution with layout knowledge enhancement in the whole workflow, to learn better representations that combine the features from text, layout, and image. Specifically, we first rearrange input sequences in the serialization stage, and then present a correlative pre-training task, reading order prediction, to learn the proper reading order of documents. To improve the layout awareness of the model, we integrate a spatial-aware disentangled attention into the multi-modal transformer and a replaced regions prediction task into the pre-training phase. Experimental results show that ERNIE-Layout achieves superior performance on various downstream tasks, setting new state-of-the-art on key information extraction, document image classification, and document question answering datasets. The code and models are publicly available at PaddleNLP.

---

## QGN

*Cao et al. Query-driven Generative Network for Document Information Extraction in the Wild. ACMMM, 2022.*


<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://dl.acm.org/doi/10.1145/3503161.3547877">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACM-brightgreen"></img>
  </a>
</p>

- **Highlights**: Capable of real-world OCR input
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: This paper focuses on solving Document Information Extraction (DIE) in the wild problem, which is rarely explored before. In contrast to existing studies mainly tailored for document cases in known templates with predefined layouts and keys under the ideal input without OCR errors involved, we aim to build up a more practical DIE paradigm for real-world scenarios where input document images may contain unknown layouts and keys in the scenes of the problematic OCR results. To achieve this goal, we propose a novel architecture, termed Query-driven Generative Network (QGN), which is equipped with two consecutive modules, i.e., Layout Context-aware Module (LCM) and Structured Generation Module (SGM). Given a document image with unseen layouts and fields, the former LCM yields the value prefix candidates serving as the query prompts for the SGM to generate the final key-value pairs even with OCR noise. To further investigate the potential of our method, we create a new large-scale dataset, named LArge-scale STructured Documents (LastDoc4000), containing 4,000 documents with 1,511 layouts and 3,500 different keys. In experiments, we demonstrate that our QGN consistently achieves the best F1-score on the new LastDoc4000 dataset by at most 30.32% absolute improvement. A more comprehensive experimental analysis and experiments on other public benchmarks also verify the effectiveness and robustness of our proposed method for the wild DIE task.

---

## GeoLayoutLM

*Luo et al. GeoLayoutLM: Geometric Pre-training for Visual Information Extraction. CVPR, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2304.10759">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/GeoLayoutLM">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Excellent Relation Extraction Performance
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Visual information extraction (VIE) plays an important role in Document Intelligence. Generally, it is divided into two tasks: semantic entity recognition (SER) and relation extraction (RE). Recently, pre-trained models for documents have achieved substantial progress in VIE, particularly in SER. However, most of the existing models learn the geometric representation in an implicit way, which has been found insufficient for the RE task since geometric information is especially crucial for RE. Moreover, we reveal another factor that limits the performance of RE lies in the objective gap between the pre-training phase and the fine-tuning phase for RE. To tackle these issues, we propose in this paper a multi-modal framework, named GeoLayoutLM, for VIE. GeoLayoutLM explicitly models the geometric relations in pre-training, which we call geometric pre-training. Geometric pre-training is achieved by three specially designed geometry-related pre-training tasks. Additionally, novel relation heads, which are pre-trained by the geometric pre-training tasks and fine-tuned for RE, are elaborately designed to enrich and enhance the feature representation. According to extensive experiments on standard VIE benchmarks, GeoLayoutLM achieves highly competitive scores in the SER task and significantly outperforms the previous state-of-the-arts for RE (\eg, the F1 score of RE on FUNSD is boosted from 80.35\% to 89.45\%).

---


## KVPFormer

*Hu et al. A Question-Answering Approach to Key Value Pair Extraction from Form-like Document Images. AAAI, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2304.07957">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Excellent Relation Extraction Performance
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: In this paper, we present a new question-answering (QA) based key-value pair extraction approach, called KVPFormer, to robustly extracting key-value relationships between entities from form-like document images. Specifically, KVPFormer first identifies key entities from all entities in an image with a Transformer encoder, then takes these key entities as questions and feeds them into a Transformer decoder to predict their corresponding answers (i.e., value entities) in parallel. To achieve higher answer prediction accuracy, we propose a coarse-to-fine answer prediction approach further, which first extracts multiple answer candidates for each identified question in the coarse stage and then selects the most likely one among these candidates in the fine stage. In this way, the learning difficulty of answer prediction can be effectively reduced so that the prediction accuracy can be improved. Moreover, we introduce a spatial compatibility attention bias into the self-attention/cross-attention mechanism for KVPFormer to better model the spatial interactions between entities. With these new techniques, our proposed KVPFormer achieves state-of-the-art results on FUNSD and XFUND datasets, outperforming the previous best-performing method by 7.2% and 13.2% in F1 score, respectively.

---

## TPP

*Zhang et al. Reading Order Matters: Information Extraction from Visually-rich DOcuments by Token Path Prediction. EMNLP, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2310.11016">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Text Reading Order Arrangement; Entity Linking
- **Modalities**: Semantic; Layout; Visual;
- **Abstract**: Recent advances in multimodal pre-trained models have significantly improved information extraction from visually-rich documents (VrDs), in which named entity recognition (NER) is treated as a sequence-labeling task of predicting the BIO entity tags for tokens, following the typical setting of NLP. However, BIO-tagging scheme relies on the correct order of model inputs, which is not guaranteed in real-world NER on scanned VrDs where text are recognized and arranged by OCR systems. Such reading order issue hinders the accurate marking of entities by BIO-tagging scheme, making it impossible for sequence-labeling methods to predict correct named entities. To address the reading order issue, we introduce Token Path Prediction (TPP), a simple prediction head to predict entity mentions as token sequences within documents. Alternative to token classification, TPP models the document layout as a complete directed graph of tokens, and predicts token paths within the graph as entities. For better evaluation of VrD-NER systems, we also propose two revised benchmark datasets of NER on scanned documents which can reflect real-world scenarios. Experiment results demonstrate the effectiveness of our method, and suggest its potential to be a universal solution to various information extraction tasks on documents.

---


## GraphLayoutLM

*Li et al. Enhancing Visually-Rich Document Understanding via Layout Structure Modeling. ACMMM, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2308.07777">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Excellent Entity Extraction Performance; LayoutLM with graph
- **Modalities**: Semantic; Layout; Visual;
- **Abstract**: In recent years, the use of multi-modal pre-trained Transformers has led to significant advancements in visually-rich document un- derstanding. However, existing models have mainly focused on features such as text and vision while neglecting the importance of layout relationship between text nodes. In this paper, we pro- pose GraphLayoutLM, a novel document understanding model that leverages the modeling of layout structure graph to inject document layout knowledge into the model. GraphLayoutLM utilizes a graph reordering algorithm to adjust the text sequence based on the graph structure. Additionally, our model uses a layout-aware multi-head self-attention layer to learn document layout knowledge. The pro- posed model enables the understanding of the spatial arrangement of text elements, improving document comprehension. We evaluate our model on various benchmarks, including FUNSD, XFUND and CORD, and achieve state-of-the-art results among these datasets. Our experimental results demonstrate that our proposed method provides a significant improvement over existing approaches and showcases the importance of incorporating layout information into document understanding models. We also conduct an ablation study to investigate the contribution of each component of our model. The results show that both the graph reordering algorithm and the layout-aware multi-head self-attention layer play a crucial role in achieving the best performance.

---


## HGALayoutLM

*Li et al. Hypergraph based Understanding for Document Semantic Entity Recognition. ACL, 2024.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2024-orange"></img>
  <a href="https://arxiv.org/abs/2407.06904">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Excellent Entity Extraction Performance
- **Modalities**: Semantic; Layout; Visual;
- **Abstract**: Semantic entity recognition is an important task in the field of visually-rich document understanding. It distinguishes the semantic types of text by analyzing the position relationship between text nodes and the relation between text content. The existing document understanding models mainly focus on entity categories while ignoring the extraction of entity boundaries. We build a novel hypergraph attention document semantic entity recognition framework, HGA, which uses hypergraph attention to focus on entity boundaries and entity categories at the same time. It can conduct a more detailed analysis of the document text representation analyzed by the upstream model and achieves a better performance of semantic information. We apply this method on the basis of GraphLayoutLM to construct a new semantic entity recognition model HGALayoutLM. Our experiment results on FUNSD, CORD, XFUND and SROIE show that our method can effectively improve the performance of semantic entity recognition tasks based on the original model. The results of HGALayoutLM on FUNSD and XFUND reach the new state-of-the-art results.

---

## DocFormerv2

*Appalaraju, Srikar, et al. DocFormerv2: Local Features for Document Understanding. AAAI, 2024.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2024-orange"></img>
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/27828">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-AAAI-brightgreen"></img>
  </a>
  <a href="https://github.com/uakarsh/docformerv2">
    <img alt="Code 1" src="https://img.shields.io/badge/DatasetRelease-Official-yellow"></img>
  </a>
</p>

- **Highlights**: Encdoer-decoder Structure
- **Modalities**: Semantic; Layout; Visual;
- **Abstract**: We propose DocFormerv2, a multi-modal transformer for Visual Document Understanding (VDU). The VDU do- main entails understanding documents (beyond mere OCR predictions) e.g., extracting information from a form, VQA for documents and other tasks. VDU is challenging as it needs a model to make sense of multiple modalities (visual, language and spatial) to make a prediction. Our approach, termed DocFormerv2 is an encoder-decoder transformer which takes as input - vision, language and spatial fea- tures. DocFormerv2 is pre-trained with unsupervised tasks employed asymmetrically i.e., two novel document tasks on encoder and one on the auto-regressive decoder. The un- supervised tasks have been carefully designed to ensure that the pre-training encourages local-feature alignment between multiple modalities. DocFormerv2 when evalu- ated on nine datasets shows state-of-the-art performance over strong baselines e.g. TabFact (4.3%), InfoVQA (1.4%), FUNSD (1%). Furthermore, to show generalization ca- pabilities, on three VQA tasks involving scene-text, Doc- Formerv2 outperforms previous comparably-sized models and even does better than much larger models (such as GIT2, PaLi and Flamingo) on some tasks. Extensive ab- lations show that due to its pre-training, DocFormerv2 un- derstands multiple modalities better than prior-art in VDU.

---

## DocTr

*Liao, Haofu, et al. DocTr: Document transformer for structured information extraction in documents. ICCV, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Liao_DocTr_Document_Transformer_for_Structured_Information_Extraction_in_Documents_ICCV_2023_paper.pdf">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-CVF-brightgreen"></img>
  </a>
</p>

- **Highlights**: Excellent Performance; New Paradigm
- **Modalities**: Semantic; Layout; Visual;
- **Abstract**: We present a new formulation for structured information extraction (SIE) from visually rich documents. We address the limitations of existing IOB tagging and graph-based for- mulations, which are either overly reliant on the correct ordering of input text or struggle with decoding a complex graph. Instead, motivated by anchor-based object detectors in computer vision, we represent an entity as an anchor word and a bounding box, and represent entity linking as the as- sociation between anchor words. This is more robust to text ordering, and maintains a compact graph for entity linking. The formulation motivates us to introduce 1) a Document Transformer (DocTr) that aims at detecting and associating entity bounding boxes in visually rich documents, and 2) a simple pre-training strategy that helps learn entity detection in the context of language. Evaluations on three SIE bench- marks show the effectiveness of the proposed formulation, and the overall approach outperforms existing solutions.

---

## LayoutMask

*Tu, Yi, et al. LayoutMask: Enhance Text-Layout Interaction in Multi-modal Pre-training for Document Understanding. ACL, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://aclanthology.org/2023.acl-long.847.pdf">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACL-brightgreen"></img>
  </a>
</p>

- **Highlights**: Capable for Variable Layout Granularity
- **Modalities**: Semantic; Layout; Visual;
- **Abstract**: Visually-rich Document Understanding (VrDU) has attracted much research attention over the past years. Pre-trained models on a large number of document images with transformer-based backbones have led to significant performance gains in this field. The major challenge is how to fusion the different modalities (text, layout, and image) of the documents in a unified model with different pre-training tasks. This paper focuses on improving text-layout interactions and proposes a novel multi-modal pre-training model, LayoutMask. LayoutMask uses local 1D position, instead of global 1D position, as layout input and has two pre-training objectives: (1) Masked Language Modeling: predicting masked tokens with two novel masking strategies; (2) Masked Position Modeling: predicting masked 2D positions to improve layout representation learning. LayoutMask can enhance the interactions between text and layout modalities in a unified model and produce adaptive and robust multi-modal representations for downstream tasks. Experimental results show that our proposed method can achieve state-of-the-art results on a wide variety of VrDU problems, including form understanding, receipt understanding, and document image classification.

<br>
<br>

# End-to-End Methods

## EATEN

*Guo et al. EATEN: Entity-Aware Attention for Single Shot Visual Text Extraction. ICDAR, 2019.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2019-orange"></img>
  <a href="https://arxiv.org/abs/1909.09380">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/beacandler/EATEN">
    <img alt="Code 1" src="https://img.shields.io/badge/DatasetRelease-Official-yellow"></img>
  </a>
</p>

- **Highlights**: Pure End-to-End
- **Modalities**: Visual
- **Abstract**: Extracting Text of Interest (ToI) from images is a crucial part of many OCR applications, such as entity recognition of cards, invoices, and receipts. Most of the existing works employ complicated engineering pipeline, which contains OCR and structure information extraction, to fulfill this task. This paper proposes an Entity-aware Attention Text Extraction Network called EATEN, which is an end-to-end trainable system to extract the ToIs without any post-processing. In the proposed framework, each entity is parsed by its corresponding entity-aware decoder, respectively. Moreover, we innovatively introduce a state transition mechanism which further improves the robustness of visual ToI extraction. In consideration of the absence of public benchmarks, we construct a dataset of almost 0.6 million images in three real-world scenarios (train ticket, passport and business card), which is publicly available at https://github.com/beacandler/EATEN. To the best of our knowledge, EATEN is the first single shot method to extract entities from images. Extensive experiments on these benchmarks demonstrate the state-of-the-art performance of EATEN.

---

## TRIE

*Zhang et al. TRIE: End-to-End Text Reading and Information Extraction for Document Understanding. ACMMM, 2020.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2020-orange"></img>
  <a href="https://arxiv.org/abs/2005.13118">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/hikopensource/DAVAR-Lab-OCR/tree/main/demo/text_ie/trie">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Seminal Work
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Since real-world ubiquitous documents (e.g., invoices, tickets, resumes and leaflets) contain rich information, automatic document image understanding has become a hot topic. Most existing works decouple the problem into two separate tasks, (1) text reading for detecting and recognizing texts in images and (2) information extraction for analyzing and extracting key elements from previously extracted plain text. However, they mainly focus on improving information extraction task, while neglecting the fact that text reading and information extraction are mutually correlated. In this paper, we propose a unified end-to-end text reading and information extraction network, where the two tasks can reinforce each other. Specifically, the multimodal visual and textual features of text reading are fused for information extraction and in turn, the semantics in information extraction contribute to the optimization of text reading. On three real-world datasets with diverse document images (from fixed layout to variable layout, from structured text to semi-structured text), our proposed method significantly outperforms the state-of-the-art methods in both efficiency and accuracy.

---

## VIES

*Wang et al. Towards Robust Visual Information Extraction in Real World: New Dataset and Novel Solution. AAAI, 2021.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/16378">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-AAAI-brightgreen"></img>
  </a>
  <a href="https://github.com/HCIILAB/EPHOIE">
    <img alt="Code 1" src="https://img.shields.io/badge/DatasetRelease-Official-yellow"></img>
  </a>
</p>

- **Highlights**: Segment/Char-level Encoding
- **Modalities**: Semantic; Layout; Visual
- **Abstract**: Visual information extraction (VIE) has attracted considerable attention recently owing to its various advanced applications such as document understanding, automatic marking and intelligent education. Most existing works decoupled this problem into several independent sub-tasks of text spotting (text detection and recognition) and information extraction, which completely ignored the high correlation among them during optimization. In this paper, we propose a robust visual information extraction system (VIES) towards real-world scenarios, which is a unified end-to-end trainable framework for simultaneous text detection, recognition and information extraction by taking a single document image as input and outputting the structured information. Specifically, the information extraction branch collects abundant visual and semantic representations from text spotting for multimodal feature fusion and conversely, provides higher-level semantic clues to contribute to the optimization of text spotting. Moreover, regarding the shortage of public benchmarks, we construct a fully-annotated dataset called EPHOIE (this https URL), which is the first Chinese benchmark for both text spotting and visual information extraction. EPHOIE consists of 1,494 images of examination paper head with complex layouts and background, including a total of 15,771 Chinese handwritten or printed text instances. Compared with the state-of-the-art methods, our VIES shows significant superior performance on the EPHOIE dataset and achieves a 9.01% F-score gain on the widely used SROIE dataset under the end-to-end scenario.

---

## Donut🍩

*Kim et al. OCR-free Document Understanding Transformer. ECCV, 2022.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://arxiv.org/abs/2111.15664">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/clovaai/donut">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Pure End-to-End; Structured IE; Data Synthesis
- **Modalities**: Visual
- **Abstract**: Understanding document images (e.g., invoices) is a core but challenging task since it requires complex functions such as reading text and a holistic understanding of the document. Current Visual Document Understanding (VDU) methods outsource the task of reading text to off-the-shelf Optical Character Recognition (OCR) engines and focus on the understanding task with the OCR outputs. Although such OCR-based approaches have shown promising performance, they suffer from 1) high computational costs for using OCR; 2) inflexibility of OCR models on languages or types of document; 3) OCR error propagation to the subsequent process. To address these issues, in this paper, we introduce a novel OCR-free VDU model named Donut, which stands for Document understanding transformer. As the first step in OCR-free VDU research, we propose a simple architecture (i.e., Transformer) with a pre-training objective (i.e., cross-entropy loss). Donut is conceptually simple yet effective. Through extensive experiments and analyses, we show a simple OCR-free VDU model, Donut, achieves state-of-the-art performances on various VDU tasks in terms of both speed and accuracy. In addition, we offer a synthetic data generator that helps the model pre-training to be flexible in various languages and domains.

---

## Dessurt

*Davis et al. End-to-end Document Recognition and Understanding with Dessurt. ECCV, 2022*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://arxiv.org/abs/2203.16618">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/herobd/dessurt">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Pure End-to-End
- **Modalities**: Visual
- **Abstract**: We introduce Dessurt, a relatively simple document understanding transformer capable of being fine-tuned on a greater variety of document tasks than prior methods. It receives a document image and task string as input and generates arbitrary text autoregressively as output. Because Dessurt is an end-to-end architecture that performs text recognition in addition to the document understanding, it does not require an external recognition model as prior methods do. Dessurt is a more flexible model than prior methods and is able to handle a variety of document domains and tasks. We show that this model is effective at 9 different dataset-task combinations.

---

## StrucTexTv2

*Yu et al. StrucTexTv2: Masked Visual-Textual Perdiction for Document Image Pre-training. ICLR, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2303.00289">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/PaddlePaddle/VIMER/tree/main/StrucTexT/v2">
    <img alt="DatasetRelease" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Low Inference Memory Consumption
- **Modalities**: Visual
- **Abstract**: In this paper, we present StrucTexTv2, an effective document image pre-training framework, by performing masked visual-textual prediction. It consists of two self-supervised pre-training tasks: masked image modeling and masked language modeling, based on text region-level image masking. The proposed method randomly masks some image regions according to the bounding box coordinates of text words. The objectives of our pre-training tasks are reconstructing the pixels of masked image regions and the corresponding masked tokens simultaneously. Hence the pre-trained encoder can capture more textual semantics in comparison to the masked image modeling that usually predicts the masked image patches. Compared to the masked multi-modal modeling methods for document image understanding that rely on both the image and text modalities, StrucTexTv2 models image-only input and potentially deals with more application scenarios free from OCR pre-processing. Extensive experiments on mainstream benchmarks of document image understanding demonstrate the effectiveness of StrucTexTv2. It achieves competitive or even new state-of-the-art performance in various downstream tasks such as image classification, layout analysis, table structure recognition, document OCR, and information extraction under the end-to-end scenario.

---

## ESP

*Yang et al. Modeling Entities as Semantic Points for Visual Information Extraction in the Wild. CVPR, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2303.13095">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://www.modelscope.cn/datasets/damo/SIBR/summary">
    <img alt="DatasetRelease" src="https://img.shields.io/badge/Dataset-Official-yellow"></img>
  </a>
</p>

- **Highlights**: Low Inference Memory Consumption; Entity Linking; Multi-line Entity Merging
- **Modalities**: Visual
- **Abstract**: Recently, Visual Information Extraction (VIE) has been becoming increasingly important in both the academia and industry, due to the wide range of real-world applications. Previously, numerous works have been proposed to tackle this problem. However, the benchmarks used to assess these methods are relatively plain, i.e., scenarios with real-world complexity are not fully represented in these benchmarks. As the first contribution of this work, we curate and release a new dataset for VIE, in which the document images are much more challenging in that they are taken from real applications, and difficulties such as blur, partial occlusion, and printing shift are quite common. All these factors may lead to failures in information extraction. Therefore, as the second contribution, we explore an alternative approach to precisely and robustly extract key information from document images under such tough conditions. Specifically, in contrast to previous methods, which usually either incorporate visual information into a multi-modal architecture or train text spotting and information extraction in an end-to-end fashion, we explicitly model entities as semantic points, i.e., center points of entities are enriched with semantic information describing the attributes and relationships of different entities, which could largely benefit entity labeling and linking. Extensive experiments on standard benchmarks in this field as well as the proposed dataset demonstrate that the proposed method can achieve significantly enhanced performance on entity labeling and linking, compared with previous state-of-the-art models.

---

## Kuang CFAM

*Kuang et al. Visual Information Extraction in the Wild: Practical Dataset and End-to-end Solution. ICDAR, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2305.07498">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/jfkuang/CFAM">
    <img alt="Code" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
  <a href="https://github.com/jfkuang/CFAM">
    <img alt="DatasetRelease" src="https://img.shields.io/badge/Dataset-Official-yellow"></img>
  </a>
</p>

- **Highlights**: Pure End-to-End; New dataset
- **Modalities**: Visual
- **Abstract**: Visual information extraction (VIE), which aims to simultaneously perform OCR and information extraction in a unified framework, has drawn increasing attention due to its essential role in various applications like understanding receipts, goods, and traffic signs. However, as existing benchmark datasets for VIE mainly consist of document images without the adequate diversity of layout structures, background disturbs, and entity categories, they cannot fully reveal the challenges of real-world applications. In this paper, we propose a large-scale dataset consisting of camera images for VIE, which contains not only the larger variance of layout, backgrounds, and fonts but also much more types of entities. Besides, we propose a novel framework for end-to-end VIE that combines the stages of OCR and information extraction in an end-to-end learning fashion. Different from the previous end-to-end approaches that directly adopt OCR features as the input of an information extraction module, we propose to use contrastive learning to narrow the semantic gap caused by the difference between the tasks of OCR and information extraction. We evaluate the existing end-to-end methods for VIE on the proposed dataset and observe that the performance of these methods has a distinguishable drop from SROIE (a widely used English dataset) to our proposed dataset due to the larger variance of layout and entities. These results demonstrate our dataset is more practical for promoting advanced VIE algorithms. In addition, experiments demonstrate that the proposed VIE method consistently achieves the obvious performance gains on the proposed and SROIE datasets.

---

## Pix2Struct

*Lee et al. Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding. ICML, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://proceedings.mlr.press/v202/lee23g.html">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-PMLR-brightgreen"></img>
  </a>
  <a href="https://github.com/google-research/pix2struct">
    <img alt="Code" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Pure End-to-End; New pretraining strategy
- **Modalities**: Visual
- **Abstract**: Visually-situated language is ubiquitous—sources range from textbooks with diagrams to web pages with images and tables, to mobile apps with buttons and forms. Perhaps due to this diversity, previous work has typically relied on domain-specific recipes with limited sharing of the underlying data, model architectures, and objectives. We present Pix2Struct, a pretrained image-to-text model for purely visual language understanding, which can be finetuned on tasks containing visually-situated language. Pix2Struct is pretrained by learning to parse masked screenshots of web pages into simplified HTML. The web, with its richness of visual elements cleanly reflected in the HTML structure, provides a large source of pretraining data well suited to the diversity of downstream tasks. Intuitively, this objective subsumes common pretraining signals such as OCR, language modeling, and image captioning. In addition to the novel pretraining strategy, we introduce a variable-resolution input representation and a more flexible integration of language and vision inputs, where language prompts such as questions are rendered directly on top of the input image. For the first time, we show that a single pretrained model can achieve state-of-the-art results in six out of nine tasks across four domains: documents, illustrations, user interfaces, and natural images.


---

## SeRum

*Cao, Haoyu, et al. Attention Where It Matters: Rethinking Visual Document Understanding with Selective Region Concentration. ICCV, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/pdf/2309.01131.pdf">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-PMLR-brightgreen"></img>
  </a>
</p>

- **Highlights**: Pure End-to-End; New pretraining strategy
- **Modalities**: Visual
- **Abstract**: We propose a novel end-to-end document understand- ing model called SeRum (SElective Region Understanding Model) for extracting meaningful information from docu-ment images, including document analysis, retrieval, and office automation. Unlike state-of-the-art approaches that rely on multi-stage technical schemes and are computa- tionally expensive, SeRum converts document image under- standing and recognition tasks into a local decoding pro- cess of the visual tokens of interest, using a content-aware token merge module. This mechanism enables the model to pay more attention to regions of interest generated by the query decoder, improving the model’s effectiveness and speeding up the decoding speed of the generative scheme. We also designed several pre-training tasks to enhance the understanding and local awareness of the model. Exper- imental results demonstrate that SeRum achieves state-of- the-art performance on document understanding tasks and competitive results on text spotting tasks. SeRum represents a substantial advancement towards enabling efficient and effective end-to-end document understanding.


---

## UDOP

*Tang et al. Unifying Vision, Text, and Layout for Universal Document Processing. CVPR, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://openaccess.thecvf.com/content/CVPR2023/html/Tang_Unifying_Vision_Text_and_Layout_for_Universal_Document_Processing_CVPR_2023_paper.html">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-CVF-brightgreen"></img>
  </a>
  <a href="https://github.com/microsoft/i-Code/tree/main/i-Code-Doc">
    <img alt="DatasetRelease" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Pure End-to-End
- **Modalities**: Visual
- **Abstract**: We propose Universal Document Processing (UDOP), a foundation Document AI model which unifies text, image, and layout modalities together with varied task formats, including document understanding and generation. UDOP leverages the spatial correlation between textual content and document image to model image, text, and layout modalities with one uniform representation. With a novel Vision-Text-Layout Transformer, UDOP unifies pretraining and multi-domain downstream tasks into a prompt-based sequence generation scheme. UDOP is pretrained on both large-scale unlabeled document corpora using innovative self-supervised objectives and diverse labeled data. UDOP also learns to generate document images from text and layout modalities via masked image reconstruction. To the best of our knowledge, this is the first time in the field of document AI that one model simultaneously achieves high-quality neural document editing and content customization. Our method sets the state-of-the-art on 8 Document AI tasks, e.g., document understanding and QA, across diverse data domains like finance reports, academic papers, and websites. UDOP ranks first on the leaderboard of the Document Understanding Benchmark.

---

## CREPE

*Yamoto Okamoto, et al. OCR-free Document Understanding Transformer. ICDAR, 2024.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2024-orange"></img>
  <a href="https://arxiv.org/pdf/2405.00260">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Pure End-to-End; Structured IE
- **Modalities**: Visual
- **Abstract**: In this study, we formulate an OCR-free sequence generation model for visual document understanding (VDU). Our model not only parses text from document images but also extracts the spatial coordinates of the text based on the multi-head architecture. Named as Coordinate-aware End-to-end Document Parser (CREPE), our method uniquely integrates these capabilities by introducing a special token for OCR text, and token-triggered coordinate decoding. We also proposed a weakly-supervised framework for cost-efficient training, requiring only parsing annotations without high-cost coordinate annotations. Our experimental evaluations demonstrate CREPE's state-of-the-art performances on document parsing tasks. Beyond that, CREPE's adaptability is further highlighted by its successful usage in other document understanding tasks such as layout analysis, document visual question answering, and so one. CREPE's abilities including OCR and semantic parsing not only mitigate error propagation issues in existing OCR-dependent methods, it also significantly enhance the functionality of sequence generation models, ushering in a new era for document understanding studies.


<br>
<br>

# Few-shot Methods

## LASER

*Wang Z, Shang J. Towards Few-shot Entity Recognition in Document Images: A Label-aware Sequence-to-Sequence Framework. ACL Findings, 2022.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://aclanthology.org/2022.findings-acl.329">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACL-brightgreen"></img>
  </a>
  <a href="https://github.com/zlwang-cs/LASER-release">
    <img alt="DatasetRelease" src="https://img.shields.io/badge/Dataset-Official-blue"></img>
  </a>
</p>

- **Highlights**: LayoutReader
- **Modalities**: Semantic; Layout; Visual
- **Abstract**:Entity recognition is a fundamental task in understanding document images. Traditional sequence labeling frameworks treat the entity types as class IDs and rely on extensive data and high-quality annotations to learn semantics which are typically expensive in practice. In this paper, we aim to build an entity recognition model requiring only a few shots of annotated document images. To overcome the data limitation, we propose to leverage the label surface names to better inform the model of the target entity type semantics and also embed the labels into the spatial embedding space to capture the spatial correspondence between regions and labels. Specifically, we go beyond sequence labeling and develop a novel labelaware seq2seq framework, LASER. The proposed model follows a new labeling scheme that generates the label surface names wordby word explicitly after generating the entities. During training, LASER refines the label semantics by updating the label surface name representations and also strengthens the label-region correlation. In this way, LASER recognizes the entities from document images through both semantic and layout correspondence. Extensive experiments on two benchmark datasets demonstrate the superiority of LASER under the few-shot setting.

---

## LF-Attn

*Cheng M, Qiu M, Shi X, et al. One-shot text field labeling using attention and belief propagation for structure information extraction. ACMM, 2022.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2022-orange"></img>
  <a href="https://dl.acm.org/doi/abs/10.1145/3394171.3413511">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACM-brightgreen"></img>
  </a>
  <a href="https://github.com/AlibabaPAI/one_shot_text_labeling">
    <img alt="DatasetRelease" src="https://img.shields.io/badge/Dataset-Official-blue"></img>
  </a>
</p>

- **Highlights**: Landmark Attention; Field Attention
- **Modalities**: Semantic; Layout
- **Abstract**: Structured information extraction from document images usually consists of three steps: text detection, text recognition, and text field labeling. While text detection and text recognition have been heavily studied and improved a lot in literature, text field labeling is less explored and still faces many challenges. Existing learning based methods for text labeling task usually require a large amount of labeled examples to train a specific model for each type of document. However, collecting large amounts of document images and labeling them is difficult and sometimes impossible due to privacy issues. Deploying separate models for each type of document also consumes a lot of resources. Facing these challenges, we explore one-shot learning for the text field labeling task. Existing one-shot learning methods for the task are mostly rule-based and have difficulty in labeling fields in crowded regions with few landmarks and fields consisting of multiple separate text regions. To alleviate these problems, we proposed a novel deep end-to-end trainable approach for one-shot text field labeling, which makes use of attention mechanism to transfer the layout information between document images. We further applied conditional random field on the transferred layout information for the refinement of field labeling. We collected and annotated a real-world one-shot field labeling dataset with a large variety of document types and conducted extensive experiments to examine the effectiveness of the proposed model. To stimulate research in this direction, the collected dataset and the one-shot model will be released.

---

## Deep Partial Graph Matching

*Yao M, Liu Z, Wang L, et al. One-shot Key Information Extraction from Document with Deep Partial Graph Matching. arXiv, 2021.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://arxiv.org/abs/2109.13967">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Highlights**: Graph Matching
- **Modalities**: Semantic; Layout
- **Abstract**: Automating the Key Information Extraction (KIE) from documents improves efficiency, productivity, and security in many industrial scenarios such as rapid indexing and archiving. Many existing supervised learning methods for the KIE task need to feed a large number of labeled samples and learn separate models for different types of documents. However, collecting and labeling a large dataset is time-consuming and is not a userfriendly requirement for many cloud platforms. To overcome these challenges, we propose a deep end-to-end trainable network for one-shot KIE using partial graph matching. Contrary to previous methods that the learning of similarity and solving are optimized separately, our method enables the learning of the two processes in an end-to-end framework. Existing one-shot KIE methods are either template or simple attention-based learning approach that struggle to handle texts that are shifted beyond their desired positions caused by printers, as illustrated in Fig. 1. To solve this problem, we add one-to-(at most)-one constraint such that we will find the globally optimized solution even if some texts are drifted. Further, we design a multimodal context ensemble block to boost the performance through fusing features of spatial, textual, and aspect representations. To promote research of KIE, we collected and annotated a one-shot document KIE dataset named DKIE with diverse types of images. The DKIE dataset consists of 2.5K document images captured by mobile phones in natural scenes, and it is the largest available one-shot KIE dataset up to now. The results of experiments on DKIE show that our method achieved state-of-the-art performance compared with recent one-shot and supervised learning approaches. The dataset and proposed one-shot KIE model will be released soon.


<br>
<br>

---

# LLM-based Methods

## ICL-D3IE

*He et al. ICL-D3IE: In-Context Learning with Diverse Demonstrations Updating for Document Information Extraction. arXiv preprint, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2303.05063">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/MAEHCM/ICL-D3IE">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-blue"></img>
  </a>
</p>

- **Highlights**: Seminal Work, In-Context Learning
- **Modalities**: Semantic Only
- **Abstract**: Large language models (LLMs), such as GPT-3 and ChatGPT, have demonstrated remarkable results in various natural language processing (NLP) tasks with in-context learning, which involves inference based on a few demonstration examples. Despite their successes in NLP tasks, no investigation has been conducted to assess the ability of LLMs to perform document information extraction (DIE) using in-context learning. Applying LLMs to DIE poses two challenges: the modality and task gap. To this end, we propose a simple but effective in-context learning framework called ICL-D3IE, which enables LLMs to perform DIE with different types of demonstration examples. Specifically, we extract the most difficult and distinct segments from hard training documents as hard demonstrations for benefiting all test instances. We design demonstrations describing relationships that enable LLMs to understand positional relationships. We introduce formatting demonstrations for easy answer extraction. Additionally, the framework improves diverse demonstrations by updating them iteratively. Our experiments on three widely used benchmark datasets demonstrate that the ICL-D3IE framework enables GPT-3/ChatGPT to achieve superior performance when compared to previous pre-trained methods fine-tuned with full training in both the in-distribution (ID) setting and in the out-of-distribution (OOD) setting.

---

## mPLUG-DocOwl

*Ye et al. mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding. arXiv preprint 2307.02499, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2307.02499">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/X-PLUG/mPLUG-DocOwl">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-blue"></img>
  </a>
</p>

- **Modalities**: Visual + Semantic Prompt
- **Abstract**: Document understanding refers to automatically extract, analyze and comprehend information from various types of digital documents, such as a web page. Existing Multi-model Large Language Models (MLLMs), including mPLUG-Owl, have demonstrated promising zero-shot capabilities in shallow OCR-free text recognition, indicating their potential for OCR-free document understanding. Nevertheless, without in-domain training, these models tend to ignore fine-grained OCR features, such as sophisticated tables or large blocks of text, which are essential for OCR-free document understanding. In this paper, we propose mPLUG-DocOwl based on mPLUG-Owl for OCR-free document understanding. Specifically, we first construct a instruction tuning dataset featuring a wide range of visual-text understanding tasks. Then, we strengthen the OCR-free document understanding ability by jointly train the model on language-only, general vision-and-language, and document instruction tuning dataset with our unified instruction tuning strategy. We also build an OCR-free document instruction understanding evaluation set LLMDoc to better compare models' capabilities on instruct compliance and document understanding. Experimental results show that our model outperforms existing multi-modal models, demonstrating its strong ability of document understanding. Besides, without specific fine-tuning, mPLUG-DocOwl generalizes well on various downstream tasks.

---


## mPLUG-DocOwl 1.5

*Hu et al. mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding. arXiv preprint 2403.12895, 2024.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2024-orange"></img>
  <a href="https://arxiv.org/abs/2403.12895">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/X-PLUG/mPLUG-DocOwl">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-blue"></img>
  </a>
</p>

- **Modalities**: Visual + Semantic Prompt
- **Abstract**: Structure information is critical for understanding the semantics of text-rich images, such as documents, tables, and charts. Existing Multimodal Large Language Models (MLLMs) for Visual Document Understanding are equipped with text recognition ability but lack general structure understanding abilities for text-rich document images. In this work, we emphasize the importance of structure information in Visual Document Understanding and propose the Unified Structure Learning to boost the performance of MLLMs. Our Unified Structure Learning comprises structure-aware parsing tasks and multi-grained text localization tasks across 5 domains: document, webpage, table, chart, and natural image. To better encode structure information, we design a simple and effective vision-to-text module H-Reducer, which can not only maintain the layout information but also reduce the length of visual features by merging horizontal adjacent patches through convolution, enabling the LLM to understand high-resolution images more efficiently. Furthermore, by constructing structure-aware text sequences and multi-grained pairs of texts and bounding boxes for publicly available text-rich images, we build a comprehensive training set DocStruct4M to support structure learning. Finally, we construct a small but high-quality reasoning tuning dataset DocReason25K to trigger the detailed explanation ability in the document domain. Our model DocOwl 1.5 achieves state-of-the-art performance on 10 visual document understanding benchmarks, improving the SOTA performance of MLLMs with a 7B LLM by more than 10 points in 5/10 benchmarks.

---

## UReader

*Ye et al. UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model. EMNLP Findings, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://aclanthology.org/2023.findings-emnlp.187/">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACL-brightgreen"></img>
  </a>
  <a href="https://github.com/LukeForeverYoung/UReader">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-blue"></img>
  </a>
</p>

- **Modalities**: Visual + Semantic Prompt
- **Abstract**: Text is ubiquitous in our visual world, conveying crucial information, such as in documents, websites, and everyday photographs. In this work, we propose UReader, a first exploration of universal OCR-free visually-situated language understanding based on the Multimodal Large Language Model (MLLM). By leveraging the shallow text recognition ability of the MLLM, we only finetuned 1.2% parameters and the training cost is much lower than previous work following domain-specific pretraining and finetuning paradigms. Concretely, UReader is jointly finetuned on a wide range of Visually-situated Language Understanding tasks via a unified instruction format. To enhance the visual text and semantic understanding, we further apply two auxiliary tasks with the same format, namely text reading and key points generation tasks. We design a shape-adaptive cropping module before the encoder-decoder architecture of MLLM to leverage the frozen low-resolution vision encoder for processing high-resolution images. Without downstream finetuning, our single model achieves state-of-the-art ocr-free performance in 8 out of 10 visually-situated language understanding tasks, across 5 domains: documents, tables, charts, natural images, and webpage screenshots. Codes and instruction-tuning datasets will be released.

---

## LLaVAR

*Zhang et al. LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding. arXiv preprint 2306.17107, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2306.17107">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/SALT-NLP/LLaVAR">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-blue"></img>
  </a>
</p>

- **Modalities**: Visual + Semantic Prompt
- **Abstract**: Document understanding refers to automatically extract, analyze and comprehend information from various types of digital documents, such as a web page. Existing Multi-model Large Language Models (MLLMs), including mPLUG-Owl, have demonstrated promising zero-shot capabilities in shallow OCR-free text recognition, indicating their potential for OCR-free document understanding. Nevertheless, without in-domain training, these models tend to ignore fine-grained OCR features, such as sophisticated tables or large blocks of text, which are essential for OCR-free document understanding. In this paper, we propose mPLUG-DocOwl based on mPLUG-Owl for OCR-free document understanding. Specifically, we first construct a instruction tuning dataset featuring a wide range of visual-text understanding tasks. Then, we strengthen the OCR-free document understanding ability by jointly train the model on language-only, general vision-and-language, and document instruction tuning dataset with our unified instruction tuning strategy. We also build an OCR-free document instruction understanding evaluation set LLMDoc to better compare models' capabilities on instruct compliance and document understanding. Experimental results show that our model outperforms existing multi-modal models, demonstrating its strong ability of document understanding. Besides, without specific fine-tuning, mPLUG-DocOwl generalizes well on various downstream tasks.

---

## UniDoc

*Feng et al. UniDoc: A Universal Large Multimodal Model for Simultaneous Text Detection, Recognition, Spotting and Understanding. arXiv preprint 2308.11592, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2308.11592">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Modalities**: Visual + Semantic Prompt
- **Abstract**: In the era of Large Language Models (LLMs), tremendous strides have been made in the field of multimodal understanding. However, existing advanced algorithms are limited to effectively utilizing the immense representation capabilities and rich world knowledge inherent to these large pre-trained models, and the beneficial connections among tasks within the context of text-rich scenarios have not been sufficiently explored. In this work, we introduce UniDoc, a novel multimodal model equipped with text detection and recognition capabilities, which are deficient in existing approaches. Moreover, UniDoc capitalizes on the beneficial interactions among tasks to enhance the performance of each individual task. To implement UniDoc, we perform unified multimodal instruct tuning on the contributed large-scale instruction following datasets. Quantitative and qualitative experimental results show that UniDoc sets state-of-the-art scores across multiple challenging benchmarks. To the best of our knowledge, this is the first large multimodal model capable of simultaneous text detection, recognition, spotting, and understanding.


---

## DocPedia

*Feng et al. DocPedia: Unleashing the Power of Large Multimodal Model in the Frequency Domain for Versatile Document Understanding. arXiv preprint 2311.11810, 2023.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2023-orange"></img>
  <a href="https://arxiv.org/abs/2311.11810">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Modalities**: Visual + Semantic Prompt
- **Abstract**: This work presents DocPedia, a novel large multimodal model (LMM) for versatile OCR-free document understanding, capable of parsing images up to 2,560×2,560 resolution. Unlike existing work either struggle with high-resolution documents or give up the large language model thus vision or language ability constrained, our DocPedia directly processes visual input in the frequency domain rather than the pixel space. The unique characteristic enables DocPedia to capture a greater amount of visual and textual information using a limited number of visual tokens. To consistently enhance both perception and comprehension abilities of our model, we develop a dual-stage training strategy and enrich instructions/annotations of all training tasks covering multiple document types. Extensive quantitative and qualitative experiments conducted on various publicly available benchmarks confirm the mutual benefits of jointly learning perception and comprehension tasks. The results provide further evidence of the effectiveness and superior performance of our DocPedia over other methods.


---

<!-- TODO replace cvpr links -->

## Monkey

*Li et al. Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models. CVPR, 2024.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2024-orange"></img>
  <a href="https://arxiv.org/abs/2311.06607">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/Yuliang-Liu/Monkey">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-blue"></img>
  </a>
</p>

- **Modalities**: Visual + Semantic Prompt
- **Abstract**: Large Multimodal Models (LMMs) have shown promise in vision-language tasks but struggle with high-resolution input and detailed scene understanding. Addressing these challenges, we introduce Monkey to enhance LMM capabilities. Firstly, Monkey processes input images by dividing them into uniform patches, each matching the size (e.g., 448×448) used in the original training of the welltrained vision encoder. Equipped with individual adapter for each patch, Monkey can handle higher resolutions up to 1344×896 pixels, enabling the detailed capture of complex visual information. Secondly, it employs a multi-level description generation method, enriching the context for scene-object associations. This two-part strategy ensures more effective learning from generated data: the higher resolution allows for a more detailed capture of visuals, which in turn enhances the effectiveness of comprehensive descriptions. Extensive ablative results validate the effectiveness of our designs. Additionally, experiments on 18 datasets further demonstrate that Monkey surpasses existing LMMs in many tasks like Image Captioning and various Visual Question Answering formats. Specially, in qualitative tests focused on dense text question answering, Monkey has exhibited encouraging results compared with GPT4V.


---

<!-- TODO replace cvpr links -->

## HRVDA

*Liu et al. HRVDA: High-Resolution Visual Document Assistant. CVPR, 2024.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2024-orange"></img>
  <a href="https://arxiv.org/abs/2404.06918">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Modalities**: Visual + Semantic Prompt
- **Abstract**: Leveraging vast training data, multimodal large language models (MLLMs) have demonstrated formidable general visual comprehension capabilities and achieved remarkable performance across various tasks. However, their performance in visual document understanding still leaves much room for improvement. This discrepancy is primarily attributed to the fact that visual document understanding is a fine-grained prediction task. In natural scenes, MLLMs typically use low-resolution images, leading to a substantial loss of visual information. Furthermore, general-purpose MLLMs do not excel in handling document-oriented instructions. In this paper, we propose a High-Resolution Visual Document Assistant (HRVDA), which bridges the gap between MLLMs and visual document understanding. This model employs a content filtering mechanism and an instruction filtering module to separately filter out the content-agnostic visual tokens and instruction-agnostic visual tokens, thereby achieving efficient model training and inference for high-resolution images. In addition, we construct a document-oriented visual instruction tuning dataset and apply a multi-stage training strategy to enhance the model's document modeling capabilities. Extensive experiments demonstrate that our model achieves state-of-the-art performance across multiple document understanding datasets, while maintaining training efficiency and inference speed comparable to low-resolution models.

---


<!-- TODO replace cvpr links -->

## DoCo

*Li et al. Enhancing Visual Document Understanding with Contrastive Learning in Large Visual-Language Models. CVPR, 2024.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2024-orange"></img>
  <a href="https://arxiv.org/abs/2402.19014">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
</p>

- **Modalities**: Visual + Semantic Prompt
- **Abstract**: Recently, the advent of Large Visual-Language Models (LVLMs) has received increasing attention across various domains, particularly in the field of visual document understanding (VDU). Different from conventional vision-language tasks, VDU is specifically concerned with text-rich scenarios containing abundant document elements. Nevertheless, the importance of fine-grained features remains largely unexplored within the community of LVLMs, leading to suboptimal performance in text-rich scenarios. In this paper, we abbreviate it as the fine-grained feature collapse issue. With the aim of filling this gap, we propose a contrastive learning framework, termed Document Object COntrastive learning (DoCo), specifically tailored for the downstream tasks of VDU. DoCo leverages an auxiliary multimodal encoder to obtain the features of document objects and align them to the visual features generated by the vision encoder of LVLM, which enhances visual representation in text-rich scenarios. It can represent that the contrastive learning between the visual holistic representations and the multimodal fine-grained features of document objects can assist the vision encoder in acquiring more effective visual cues, thereby enhancing the comprehension of text-rich documents in LVLMs. We also demonstrate that the proposed DoCo serves as a plug-and-play pre-training method, which can be employed in the pre-training of various LVLMs without inducing any increase in computational complexity during the inference process. Extensive experimental results on multiple benchmarks of VDU reveal that LVLMs equipped with our proposed DoCo can achieve superior performance and mitigate the gap between VDU and generic vision-language tasks.


---

<!-- TODO replace cvpr links -->

## OmniParser

*Wan et al. OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition. CVPR, 2024.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2024-orange"></img>
  <a href="https://arxiv.org/abs/2403.19128">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/AlibabaResearch/AdvancedLiterateMachinery">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-blue"></img>
  </a>
</p>

- **Modalities**: Visual
- **Abstract**: Recently, visually-situated text parsing (VsTP) has experienced notable advancements, driven by the increasing demand for automated document understanding and the emergence of Generative Large Language Models (LLMs) capable of processing document-based questions. Various methods have been proposed to address the challenging problem of VsTP. However, due to the diversified targets and heterogeneous schemas, previous works usually design task-specific architectures and objectives for individual tasks, which inadvertently leads to modal isolation and complex workflow. In this paper, we propose a unified paradigm for parsing visually-situated text across diverse scenarios. Specifically, we devise a universal model, called OmniParser, which can simultaneously handle three typical visually-situated text parsing tasks: text spotting, key information extraction, and table recognition. In OmniParser, all tasks share the unified encoder-decoder architecture, the unified objective: point-conditioned text generation, and the unified input & output representation: prompt & structured sequences. Extensive experiments demonstrate that the proposed OmniParser achieves state-of-the-art (SOTA) or highly competitive performances on 7 datasets for the three visually-situated text parsing tasks, despite its unified, concise design.


---

<!-- TODO replace cvpr links -->

## LayoutLLM

*Luo et al. LayoutLLM: Layout Instruction Tuning with Large Language Models for Document Understanding. CVPR, 2024.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2024-orange"></img>
  <a href="https://arxiv.org/abs/2404.05225">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-arXiv-brightgreen"></img>
  </a>
  <a href="https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/LayoutLLM">
    <img alt="Paper Link" src="https://img.shields.io/badge/Data-GitHub-yellow"></img>
  </a>

</p>

- **Modalities**: Visual, Semantic, Layout
- **Abstract**: Recently, leveraging large language models (LLMs) or multimodal large language models (MLLMs) for document understanding has been proven very promising. However, previous works that employ LLMs/MLLMs for document understanding have not fully explored and utilized the document layout information, which is vital for precise document understanding. In this paper, we propose LayoutLLM, an LLM/MLLM based method for document understanding. The core of LayoutLLM is a layout instruction tuning strategy, which is specially designed to enhance the comprehension and utilization of document layouts. The proposed layout instruction tuning strategy consists of two components: Layout-aware Pre-training and Layout-aware Supervised Fine-tuning. To capture the characteristics of document layout in Layout-aware Pre-training, three groups of pre-training tasks, corresponding to document-level, region-level and segment-level information, are introduced. Furthermore, a novel module called layout chain-of-thought (LayoutCoT) is devised to enable LayoutLLM to focus on regions relevant to the question and generate accurate answers. LayoutCoT is effective for boosting the performance of document understanding. Meanwhile, it brings a certain degree of interpretability, which could facilitate manual inspection and correction. Experiments on standard benchmarks show that the proposed LayoutLLM significantly outperforms existing methods that adopt open-source 7B LLMs/MLLMs for document understanding.


<br>
<br>

---



# Other Methods

## TCPN

*Wang et al. Tag, Copy or Predict: A Unified Weakly-Supervised Learning Framework for Visual Information Extraction using Sequences. IJCAI, 2021.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://www.ijcai.org/proceedings/2021/150">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-IJCAI-brightgreen"></img>
  </a>
</p>

- **Highlights**: OCR Error Correction
- **Modalities**: Semantic; Layout
- **Abstract**: Visual information extraction (VIE) has attracted increasing attention in recent years. The existing methods usually first organized optical character recognition (OCR) results in plain texts and then utilized token-level category annotations as supervision to train a sequence tagging model. However, it expends great annotation costs and may be exposed to label confusion, the OCR errors will also significantly affect the final performance. In this paper, we propose a unified weakly-supervised learning framework called TCPNet (Tag, Copy or Predict Network), which introduces 1) an efficient encoder to simultaneously model the semantic and layout information in 2D OCR results, 2) a weakly-supervised training method that utilizes only sequence-level supervision; and 3) a flexible and switchable decoder which contains two inference modes: one (Copy or Predict Mode) is to output key information sequences of different categories by copying a token from the input or predicting one in each time step, and the other (Tag Mode) is to directly tag the input sequence in a single forward pass. Our method shows new state-of-the-art performance on several public benchmarks, which fully proves its effectiveness.

---

## SPADE♠

*Hwang et al. Spatial Dependency Parsing for Semi-Structured Document Information Extraction. ACL Findings, 2021.*

<p>
  <img alt="year" src="https://img.shields.io/badge/Year-2021-orange"></img>
  <a href="https://aclanthology.org/2021.findings-acl.28/">
    <img alt="Paper Link" src="https://img.shields.io/badge/PaperLink-ACL-brightgreen"></img>
  </a>
  <a href="https://github.com/clovaai/spade">
    <img alt="Code 1" src="https://img.shields.io/badge/Code-Official-blue"></img>
  </a>
</p>

- **Highlights**: Structured IE; Entity Linking
- **Modalities**: Semantic; Layout
- **Abstract**: Information Extraction (IE) for semi-structured document images is often approached as a sequence tagging problem by classifying each recognized input token into one of the IOB (Inside, Outside, and Beginning) categories. However, such problem setup has two inherent limitations that (1) it cannot easily handle complex spatial relationships and (2) it is not suitable for highly structured information, which are nevertheless frequently observed in real-world document images. To tackle these issues, we first formulate the IE task as spatial dependency parsing problem that focuses on the relationship among text segment nodes in the documents. Under this setup, we then propose SPADE (SPAtial DEpendency parser) that models highly complex spatial relationships and an arbitrary number of information layers in the documents in an end-to-end manner. We evaluate it on various kinds of documents such as receipts, name cards, forms, and invoices, and show that it achieves a similar or better performance compared to strong baselines including BERT-based IOB taggger, with up to 37.7% improvement.
