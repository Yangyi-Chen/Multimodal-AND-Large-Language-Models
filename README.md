# Multi-modal Machine Learning

## Table of Contents 
- [Survey](#survey)
- [Structure](#structure)
- [Event Extraction](#event-extraction)
- [Situation Recognition](#situation-recognition)
- [Scene Graph](#scene-graph)
- [Compositionality](#compositionality)
- [Planning](#planning)
- [Reasoning](#reasoning)
- [Generation](#generation)
- [Representation Learning](#representation-learning)
- [Foundation Model](#foundation-model)
- [Pretraining](#pretraining)
- [Adaptation of Foundation Model](#adaptation-of-foundation-model)
- [Prompting](#prompting)
- [Efficiency](#efficiency)
- [Analysis](#analysis)
- [Grounding](#grounding)
- [VQA Task](#vqa-task)
- [Social Good](#social-good)
- [Application](#application)
- [Benchmark & Evaluation](#benchmark-&-evaluation)
- [Dataset](#dataset)
- [Robustness](#robustness)
- [Cognitive NeuronScience & Machine Learning](#cognitive-neuronscience-&-machine-learning)
- [Cognitive NeuronScience](#cognitive-neuronscience)
- [Backup](#backup)
- [Resource](#resource)

## Survey
- **Multimodal Learning with Transformers: A Survey;** Peng Xu, Xiatian Zhu, David A. Clifton
- **Multimodal Machine Learning: A Survey and Taxonomy;** Tadas Baltrusaitis, Chaitanya Ahuja, Louis-Philippe Morency; Introduce 4 challenges for multi-modal learning, including representation, translation, alignment, fusion, and co-learning. 
- **FOUNDATIONS & RECENT TRENDS IN MULTIMODAL MACHINE LEARNING: PRINCIPLES, CHALLENGES, & OPEN QUESTIONS;** Paul Pu Liang, Amir Zadeh, Louis-Philippe Morency
- **Multimodal research in vision and language: A review of current and emerging trends;** Shagun Uppal et al;
- **Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods;** Aditya Mogadala et al
- **Challenges and Prospects in Vision and Language Research;** Kushal Kafle et al
- **A Survey of Current Datasets for Vision and Language Research;** Francis Ferraro et al
- **VLP: A Survey on Vision-Language Pre-training;** Feilong Chen et al
- **A Survey on Multimodal Disinformation Detection;** Firoj Alam et al
- **Vision-Language Pre-training: Basics, Recent Advances, and Future Trends;** Zhe Gan et al
- **Deep Multimodal Representation Learning: A Survey;** Wenzhong Guo et al


## Structure
- **Finding Structural Knowledge in Multimodal-BERT;** Victor Milewski et al

## Event Extraction
- **Cross-media Structured Common Space for Multimedia Event Extraction;** Manling Li et al; Focus on image-text event extraction. A new benchmark and baseline are proposed. 
- **Visual Semantic Role Labeling for Video Understanding;** Arka Sadhu et al; A new benchmark is proposed. 
- **GAIA: A Fine-grained Multimedia Knowledge Extraction System;** Manling Li et al; Demo paper. Extract knowledge (relation, event) from multimedia data. 

## Situation Recognition
- **Situation Recognition: Visual Semantic Role Labeling for Image Understanding;** Mark Yatskar et al; Focus on image understanding. Given images, do the semantic role labeling task. No text available. A new benchmark and baseline are proposed. 
- **Commonly Uncommon: Semantic Sparsity in Situation Recognition;** Mark Yatskar et al; Address the long-tail problem.


## Scene Graph
- **Visual Distant Supervision for Scene Graph Generation;** Yuan Yao et al
- **Learning to Generate Scene Graph from Natural Language Supervision;** Yiwu Zhong et al
- **Weakly Supervised Visual Semantic Parsing;** Alireza Zareian, Svebor Karaman, Shih-Fu Chang
- **Scene Graph Prediction with Limited Labels;** Vincent S. Chen, Paroma Varma, Ranjay Krishna, Michael Bernstein, Christopher Re, Li Fei-Fei
- **Neural Motifs: Scene Graph Parsing with Global Context;** Rowan Zellers et al
- **Fine-Grained Scene Graph Generation with Data Transfer;** Ao Zhang et al
- **Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning;** Tao He et al
- **Action Genome: Actions as Composition of Spatio-temporal Scene Graphs;** Jingwei Ji et al; Spatio-temporal scene graphs. 




## Compositionality
- **CREPE: Can Vision-Language Foundation Models Reason Compositionally?;** Zixian Ma et al
- **Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality;** Tristan Thrush et al
- **WHEN AND WHY VISION-LANGUAGE MODELS BE- HAVE LIKE BAGS-OF-WORDS, AND WHAT TO DO ABOUT IT?;** Mert Yuksekgonul et al
- **GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering;** Drew A. Hudson et al
- **COVR: A Test-Bed for Visually Grounded Compositional Generalization with Real Images;** Ben Bogin et al
- **Cops-Ref: A new Dataset and Task on Compositional Referring Expression Comprehension;** Zhenfang Chen et al







## Planning
- **Multimedia Generative Script Learning for Task Planning;** Qingyun Wang et al; Next step prediction.
- **PlaTe: Visually-Grounded Planning with Transformers in Procedural Tasks;** Jiankai Sun et al; Procedure planning. 
- **P3IV: Probabilistic Procedure Planning from Instructional Videos with Weak Supervision;** He Zhao et al; Procedure planning. Using text as weak supervision to replace video clips. 
- **Procedure Planning in Instructional Videos;** Chien-Yi Chang et al; Procedure planning.


## Reasoning
- **VisualCOMET: Reasoning about the Dynamic Context of a Still Image;** Jae Sung Park et al; Benchmark dataset, requiring models to reason about a still iamge (what happen past & next). 
- **From Recognition to Cognition: Visual Commonsense Reasoning;** Rowan Zellers et al; Benchmark dataset, requiring models to go beyond the recognition level to cognition. Need to reason about a still image and give rationales. 
- **Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering;** Pan Lu et al
- **See, Think, Confirm: Interactive Prompting Between Vision and Language Models for Knowledge-based Visual Reasoning;** Zhenfang Chen et al
- **An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA;** Zhengyuan Yang et al

## Generation
- **ClipCap: CLIP Prefix for Image Captioning;** Ron Mokady et al; Train an light-weight encoder to convert CLIP embeddings to prefix token embeddings of GPT-2. 
- **Multimodal Knowledge Alignment with Reinforcement Learning;** Youngjae Yu et al; Use RL to train an encoder that projects multimodal inputs into the word embedding space of GPT-2. 

## Representation Learning
- **Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering;** Peter Anderson et al
- **Fusion of Detected Objects in Text for Visual Question Answering;** Chris Alberti et al
- **VLMixer: Unpaired Vision-Language Pre-training via Cross-Modal CutMix;** Teng Wang et al
- **Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision;** Hao Tan et al; Use visual supervision to pretrain language models. 



## Foundation Model 
### First Generation: Using region-based features; can be classified as one- and two- streams model architectures; Before 2020.6;
- **Multimodal Pretraining Unmasked: A Meta-Analysis and a Unified Framework of Vision-and-Language BERTs;** Emanuele Bugliarello et al; A meta-analysis of the first generation VL models and a unified framework. 
- **ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks;** Jiasen Lu et al
- **LXMERT: Learning Cross-Modality Encoder Representations from Transformers;** Hao Tan et al
- **VISUALBERT: A SIMPLE AND PERFORMANT BASELINE FOR VISION AND LANGUAGE;** Liunian Harold Li et al
- **UNITER: UNiversal Image-TExt Representation Learning;** Yen-Chun Chen et al
- **VL-BERT: PRE-TRAINING OF GENERIC VISUAL-LINGUISTIC REPRESENTATIONS;** Weijie Su et al
- **IMAGEBERT: CROSS-MODAL PRE-TRAINING WITH LARGE-SCALE WEAK-SUPERVISED IMAGE-TEXT DATA;** Di Qi et al
- **Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training;** Gen Li et al
- **UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning;** Wei Li et al; Motivate to use unimodal data to improve the performance of VL tasks. 

**Introduce image tags to learn image-text alignments.**
- **Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks;** Xiujun Li et al
- **VinVL: Revisiting Visual Representations in Vision-Language Models;** Pengchuan Zhang et al
- **Unsupervised Vision-and-Language Pre-training Without Parallel Images and Captions;** Liunian Harold Li et al; Consider the unsupervised setting. 


### Second Generation: Get rid of ROI and object detectors for acceleration; Moving to large pretraining datasets; Moving to unified architectures for understanding and generation tasks; Mostly before 2022.6.
- **An Empirical Study of Training End-to-End Vision-and-Language Transformers;** Zi-Yi Dou et al; Meta-analysis. Investigate how to design and pre-train a fully transformer-based VL model in an end-to-end manner.
- **Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers;** Zhicheng Huang et al; Throw away region-based features, bounding boxes, and object detectors. Directly input the raw pixels and use CNN to extract features. 
- **ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision;** Wonjae Kim et al; Get rid of heavy computation of ROI and CNN through utilizing ViT. 
- **Seeing Out of tHe bOx: End-to-End Pre-training for Vision-Language Representation Learning;** Zhicheng Huang et al
- **E2E-VLP: End-to-End Vision-Language Pre-training Enhanced by Visual Learning;** Haiyang Xu et al; Get rid of bounding boxes; Introduce object detection and image captioning as pretraining tasks with a encoder-decoder structure. 
- **Align before Fuse: Vision and Language Representation Learning with Momentum Distillation;** Junnan Li et al; Propose ALBEF.
- **simvlm: simple visual language model pre-training with weak supervision;** Zirui Wang et al; Get rid of bounding boxes; Further argue that the pretraining objectives are complicated and not scalable; Consider the zero-shot behaviors, emergent by pretraining on large datasets.  
- **UFO: A UniFied TransfOrmer for Vision-Language Representation Learning;** Jianfeng Wang et al
- **VLMO: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts;** Hangbo Bao et al; Introduce the mixture-of-experts method to model text and image separately and use a specific expert to learn the cross-modal fusion (Multiway Transformer), which is later adopted by BEiT-3; Ensure better image-text retrieval (performance & speed) and VL tasks; 
- **Learning Transferable Visual Models From Natural Language Supervision;** Alec Radford et al; Using large noisy pretraining datasets.
- **Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision;** Chao Jia et al; Using large noisy pretraining datasets.
- **FILIP: FINE-GRAINED INTERACTIVE LANGUAGE-IMAGE PRE-TRAINING;** Lewei Yao et al; Further improve CLIP & ALIGN by introducing fine-grained alignments. 


**Special designs tailored to enhance the position encoding & grounding.**
- **UniTAB: Unifying Text and Box Outputs for Grounded Vision-Language Modeling;** Zhengyuan Yang et al
- **PEVL: Position-enhanced Pre-training and Prompt Tuning for Vision-language Models;** Yuan Yao et al; Introduce explicit object position modeling.A woman < 310 mask 406 475 > is watching the mask < 175 86 254 460 >;
- **GLIPv2: Unifying Localization and VL Understanding;** Haotian Zhang et al; Further show that GLIP's pretraining method can benefit the VL task (Unifying localization and understanding). 

**Motivate to use unparalleled image & text data to build a unified model for VL, vision, and language tasks and potentially bring better performance.**
- **Uni-Perceiver: Pre-training Unified Architecture for Generic Perception for Zero-shot and Few-shot Tasks;** Siamese network to encode various modalities.
- **FLAVA: A Foundational Language And Vision Alignment Model;** Amanpreet Singh et al; A unified backbone model (need task-specific heads) for NLP, CV, and VL tasks.
- **UNIMO-2: End-to-End Unified Vision-Language Grounded Learning;** Wei Li et al; Design a new method "Grounded Dictionary Learning", similar to the sense of "continuous" image tags to align two modalities. 



### Third Generation: 2022->Now. 
- **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation;** Junnan Li et al; New unified architecture and new method to generate and then filter captions.  
- **OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework;** Peng Wang et al; A unified model (framework) to handle text, image, and image-text tasks. 
- **CoCa: Contrastive Captioners are Image-Text Foundation Models;** Jiahui Yu et al
- **Flamingo: a Visual Language Model for Few-Shot Learning;** Jean-Baptiste Alayrac et al; Designed for few-shot learning. 
- **UNIFIED-IO: A UNIFIED MODEL FOR VISION, LANGUAGE, AND MULTI-MODAL TASKS;** Jiasen Lu et al; Examine whether a single unified model can solve a variety of tasks (NLP, CV, VL) simultaneously; Construct a massive multi-tasking dataset by ensembling 95 datasets from 62 publicly available data sources, including Image Synthesis, Keypoint Estimation, Depth Estimation, Object Segmentation, et al; Focusing on multi-task fine-tuning.
- **Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks;** Wenhui Wang et al; BEIT-3. 
- **OmniVL: One Foundation Model for Image-Language and Video-Language Tasks;** Junke Wang et al; Support both image-language and video-language tasks and show the positive transfer in three modalities.
- **Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks;** Hao Li et al; Propose a generalist model that can also handle object detection and instance segmentation tasks.
- **X2-VLM: All-In-One Pre-trained Model For Vision-Language Tasks;** Yan Zeng et al; Propose a unified model for image-language and video-text-language tasks; Modeling the fine-grained alignments between image regions and descriptions. 
- **Generalized Decoding for Pixel, Image, and Language;** Xueyan Zou et al
- **Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks;** Xinsong Zhang et al
- **Foundation Transformers;** Hongyu Wang et al; Propose a new unified architecture. 








### Others
- **Unified Vision-Language Pre-Training for Image Captioning and VQA;** Luowei Zhou et al
- **Unifying Vision-and-Language Tasks via Text Generation;** Jaemin Cho et al
- **MERLOT: Multimodal Neural Script Knowledge Models;** Rowan Zellers et al
- **MERLOT Reserve: Neural Script Knowledge through Vision and Language and Sound;** Rowan Zellers et al
- **CLIP-Event: Connecting Text and Images with Event Structures;** Manling Li et al; The new model CLIP-Event, specifically designed for multi-modal event extraction. Introducing new pretraining tasks to enable strong zero-shot performances. From object-centric representations to event-centric representations. 




## Pretraining
- **SGEITL: Scene Graph Enhanced Image-Text Learning for Visual Commonsense Reasoning;** Zhecan Wang et al; Incorporating scene graphs in pretraining and fine-tuning improves performance of VCR tasks. 
- **ERNIE-ViL: Knowledge Enhanced Vision-Language Representations through Scene Graphs;** Fei Yu et al
- **Large-Scale Adversarial Training for Vision-and-Language Representation Learning;** Zhe Gan et al
- **Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts;** Yan Zeng et al
- **BEIT: BERT Pre-Training of Image Transformers;** Hangbo Bao et al; Pre-trained CV model. 
- **BEIT V2: Masked Image Modeling with Vector-Quantized Visual Tokenizers;** Zhiliang Peng et al; Pre-trained CV model.
- **VirTex: Learning Visual Representations from Textual Annotations;** Karan Desai et al; Pretraining CV models through the dense image captioning task.
- **Florence: A New Foundation Model for Computer Vision;** Lu Yuan et al; Pre-trained CV model.  
- **Grounded Language-Image Pre-training;** Liunian Harold Li et al; Learning object-level, language-aware, and semantic-rich visual representations. Introducing phrase grounding to the pretraining task and focusing on object detection as the downstream task; Propose GLIP.

**Novel techniques.**
- **CM3: A CAUSAL MASKED MULTIMODAL MODEL OF THE INTERNET;** Armen Aghajanyan et al
- **PaLI: A Jointly-Scaled Multilingual Language-Image Model;** Xi Chen et al
- **KB-VLP: Knowledge Based Vision and Language Pretraining;** Kezhen Chen et al
- **PIX2STRUCT: SCREENSHOT PARSING AS PRETRAINING FOR VISUAL LANGUAGE UNDERSTANDING;** Kenton Lee et al
- **Retrievetrrieval-Augmented Multimodal Language Modeling;** Michihiro Yasunaga et al
- **Teaching Structured Vision & Language Concepts to Vision & Language Models;** Sivan Doveh et al
- **Unifying Vision, Text, and Layout for Universal Document Processing;** Zineng Tang et al
- **MATCHA : Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering;** Fangyu Liu et al’
- **Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training;** Filip Radenovic et al



## Adaptation of Foundation Model
- **Language Models with Image Descriptors are Strong Few-Shot Video-Language Learners;** Zhenhailong Wang et al
- **Multimodal Few-Shot Learning with Frozen Language Models;** Maria Tsimpoukelli et al; Use prefix-like image-embedding to stear the text generation process to achieve few-shot learning. 






## Prompting
- **Learning to Prompt for Vision-Language Models;** Kaiyang Zhou et al; Soft prompt tuning. Useing few-shot learning to improve performance on both in-distribution and out-of-distribution data. Few-shot setting. 
- **Unsupervised Prompt Learning for Vision-Language Models;** Tony Huang et al; Soft prompt tuning. Unsupervised setting. 
- **Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling;** Renrui Zhang et al; Few-shot setting.
- **CLIP-Adapter: Better Vision-Language Models with Feature Adapters;** Peng Gao et al; Few-shot setting.
- **Neural Prompt Search;** Yuanhan Zhang et al; Explore the combination of LoRA, Adapter, Soft prompt tuning. In full-data, few-shot, and domain shift settings. 
- **Visual Prompt Tuning;** Menglin Jia et al; Soft prompt tuning + head tuning. Show better performance in few-shot and full-data settings than full-parameters tuning. Quite different from the NLP field. 
- **Prompt Distribution Learning;** Yuning Lu et al; Soft prompt tuning. Few-shot setting.
- **Conditional Prompt Learning for Vision-Language Models;** identify a critical problem of CoOp: the learned con- text is not generalizable to wider unseen classes within the same dataset; Propose to learn a DNN that can generate for each image an input-conditional token (vector).
- **Learning to Prompt for Continual Learning;** Zifeng Wang et al; Continual learning setting. Maintain a prompt pool. 
- **Exploring Visual Prompts for Adapting Large-Scale Models;** Hyojin Bahng et al; Employ adversarial reprogramming as visual prompts. Full-data setting.
- **Learning multiple visual domains with residual adapters;** Sylvestre-Alvise Rebuff et al; Use adapter to transfer pretrained knowledge to multiple domains while freeze the base model parameters. Work in the CV filed & full-data transfer learning. 
- **Efficient parametrization of multi-domain deep neural networks;** Sylvestre-Alvise Rebuff et al; Still use adapter for transfer learning, with more comprehensive empirical study for an ideal choice. 
- **Prompting Visual-Language Models for Efficient Video Understanding;** Chen Ju et al; Video tasks. Few-shots & zero-shots. Soft prompt tuning. 
- **Visual Prompting via Image Inpainting;** Amir Bar et al; In-context learning in CV. Use pretrained masked auto-encoder. 
- **CLIP Models are Few-shot Learners: Empirical Studies on VQA and Visual Entailment;** Haoyu Song et al; Propose a parameter-efficient tuning method (bias tuning), function well in few-shot setting.  
- **LEARNING TO COMPOSE SOFT PROMPTS FOR COMPOSITIONAL ZERO-SHOT LEARNING;** Nihal V. Nayak et al; zero-shot setting, inject some knowledge in the learning process.
- **Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models;** Manli Shu et al; Learn soft-prompt in the test-time. 
- **Multitask Vision-Language Prompt Tuning;** Sheng Shen et al; Few-shot. 
- **A Good Prompt Is Worth Millions of Parameters: Low-resource Prompt-based Learning for Vision-Language Models;** Woojeong Jin et al
- **CPT: COLORFUL PROMPT TUNING FOR PRE-TRAINED VISION-LANGUAGE MODELS;** Yuan Yao et al; Good few-shot & zero-shot performance on RefCOCO datasets. 



## Efficiency
- **M3SAT: A SPARSELY ACTIVATED TRANSFORMER FOR EFFICIENT MULTI-TASK LEARNING FROM MULTIPLE MODALITIES;** Anonymous
- **Prompt Tuning for Generative Multimodal Pretrained Models;** Hao Yang et al; Implement prefix-tuning in OFA. Try full-data setting and demonstrate comparable performance. 
- **Fine-tuning Image Transformers using Learnable Memory;** Mark Sandler et al; Add soft prompts in each layer. full-data.
- **Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks;** Jeffrey O. Zhang et al; Transfer learning. 
- **Polyhistor: Parameter-Efficient Multi-Task Adaptation for Dense Vision Tasks;** Yen-Cheng Liu et al
- **Task Residual for Tuning Vision-Language Models;** Tao Yu et al





## Analysis
- **What Does BERT with Vision Look At?** Liunian Harold Li et al
- **Visual Referring Expression Recognition: What Do Systems Actually Learn?;** Volkan Cirik et al
- **Characterizing and Overcoming the Greedy Nature of Learning in Multi-modal Deep Neural Networks;** Nan Wu et al; Study the problem of only relying on one certain modality in training when using multi-modal models. 
- **Behind the Scene: Revealing the Secrets of Pre-trained Vision-and-Language Models;** Jize Cao et al
- **Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning** Weixin Liang et al

## Grounding
- **Flickr30k Entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models;** Bryan A. Plummer et al; A new benchmark dataset, annotating phrase-region correspondences. 
- **Visual Grounding Strategies for Text-Only Natural Language Processing;** Propose to improve the NLP tasks performance by grounding to images. Two methods are proposed.  
- **Visually Grounded Neural Syntax Acquisition;** Haoyue Shi et al
- **PIGLeT: Language Grounding Through Neuro-Symbolic Interaction in a 3D World;** Rowan Zellers et al

## VQA Task
- **WeaQA: Weak Supervision via Captions for Visual Question Answering;** Pratyay Banerjee et al
- **Don’t Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering;** Aishwarya Agrawal et al
- **Language Prior Is Not the Only Shortcut: A Benchmark for Shortcut Learning in VQA;** Qingyi Si et al
- **Towards Robust Visual Question Answering: Making the Most of Biased Samples via Contrastive Learning;** Qingyi Si et al
- **Plug-and-Play VQA: Zero-shot VQA by Conjoining Large Pretrained Models with Zero Training;** Anthony Meng Huat Tiong et al
- **FROM IMAGES TO TEXTUAL PROMPTS: ZERO-SHOT VQA WITH FROZEN LARGE LANGUAGE MODELS;** Jiaxian Guo et al
- **SQuINTing at VQA Models: Introspecting VQA Models with Sub-Questions;** Ramprasaath R. Selvaraju et al




## Social Good
- **The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes;** Douwe Kiela et al; Multi-modal hate-speech detection. 
- **Detecting Cross-Modal Inconsistency to Defend Against Neural Fake News;** Reuben Tan et al; Multi-modal fake news dedetection.
- **InfoSurgeon: Cross-Media Fine-grained Information Consistency Checking for Fake News Detection;** Yi R. Fung et al; Cross-modal fake news detection.
- **EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection;** Yaqing Wang et al
- **End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models;** Barry Menglong Yao et al
- **SAFE: Similarity-Aware Multi-Modal Fake News Detection;** Xinyi Zhou et al
- **r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection;** Kai Nakamura et al; Fake news detection dataset.
- **Fact-Checking Meets Fauxtography: Verifying Claims About Images;** Dimitrina Zlatkova et al; Claim-Images pairs. 

## Application
- **Evaluating and Improving Factuality in Multimodal Abstractive Summarization;** David Wan et al


## Benchmark & Evaluation
- **How Much Can CLIP Benefit Vision-and-Language Tasks?;** Sheng Shen et al; Explore two scenarios: 1) plugging CLIP into task-specific fine-tuning; 2) combining CLIP with V&L pre-training and transferring to downstream tasks. Show the boost in performance when using CLIP as the image encoder. 
- **Multimodal datasets: misogyny, pornography, and malignant stereotypes;** Abeba Birhane et al
- **WHEN AND WHY VISION-LANGUAGE MODELS BE- HAVE LIKE BAGS-OF-WORDS, AND WHAT TO DO ABOUT IT?;** Mert Yuksekgonul et al
- **Understanding ME? Multimodal Evaluation for Fine-grained Visual Commonsense;** Zhecan Wang et al


## Dataset
- **Visual Entailment: A Novel Task for Fine-Grained Image Understanding;** Ning Xie et al; Visual entailment task. SNLI-VE.  
- **Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering;** Yash Goyal et al; VQA-V2.
- **From Recognition to Cognition: Visual Commonsense Reasoning;** Rowan Zellers et al; VCR.
- **A Corpus for Reasoning About Natural Language Grounded in Photographs;** Alane Suhr et al; NLVR2.
- **VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models;** Wangchunshu Zhou et al; VLUE.
- **Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning;** Piyush Sharma et al
- **Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts;** Soravit Changpinyo et al 
- **LAION-5B: An open large-scale dataset for training next generation image-text models;** Christoph Schuhmann et al
- **Bloom Library: Multimodal Datasets in 300+ Languages for a Variety of Downstream Tasks;** Colin Leong et al
- **Find Someone Who: Visual Commonsense Understanding in Human-Centric Grounding;** Haoxuan You et al
- **MULTIINSTRUCT: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning;** Zhiyang Xu et al

## Robustness
- **Domino: Discovering Systematic Errors with Cross-Modal Embeddings;** Sabri Eyuboglu et al
- **Learning Visually-Grounded Semantics from Contrastive Adversarial Samples;** Haoyue Shi et al
- **Visually Grounded Reasoning across Languages and Cultures;** Fangyu Liu et al
- **A Closer Look at the Robustness of Vision-and-Language Pre-trained Models;** Linjie Li et al; Compile a list of robustness-VQA datasets.
- **Plausible May Not Be Faithful: Probing Object Hallucination in Vision-Language Pre-training;** Wenliang Dai et al
- **ROBUSTNESS ANALYSIS OF VIDEO-LANGUAGE MODELS AGAINST VISUAL AND LANGUAGE PERTURBATIONS;** Madeline C. Schiappa et al
- **Context-Aware Robust Fine-Tuning;** Xiaofeng Mao et al
- **Task Bias in Vision-Language Models;** Sachit Menon et al
- **Are Multimodal Models Robust to Image and Text Perturbations?;** Jielin Qiu et al
- **CPL: Counterfactual Prompt Learning for Vision and Language Models;** Xuehai He et al

## Cognitive NeuronScience & Machine Learning
- **Mind Reader: Reconstructing complex images from brain activities;** Sikun Lin et al
- **Joint processing of linguistic properties in brains and language models;** Subba Reddy Oota et al
- **Do Large Language Models know what humans know?;** Sean Trott et al
- **Few-shot Language Coordination by Modeling Theory of Mind;** Hao Zhu et al
- **Few-Shot Character Understanding in Movies as an Assessment to Meta-Learning of Theory-of-Mind;** Mo Yu et al 
- **Neural Theory-of-Mind? On the Limits of Social Intelligence in Large LMs;** Maarten Sap et al
- **A Cognitive Evaluation of Instruction Generation Agents tl;dr They Need Better Theory-of-Mind Capabilities;** Lingjun Zhao et al
- **MINDCRAFT: Theory of Mind Modeling for Situated Dialogue in Collaborative Tasks;** Cristian-Paul Bara et al
- **TVSHOWGUESS: Character Comprehension in Stories as Speaker Guessing;** Yisi Sang et al
- **Is the Brain Mechanism for Hierarchical Structure Building Universal Across Languages? An fMRI Study of Chinese and English;** Xiaohan Zhang et al
- **TRAINING LANGUAGE MODELS FOR DEEPER UNDERSTANDING IMPROVES BRAIN ALIGNMENT;** Khai Loong Aw et al
- **Abstract Visual Reasoning with Tangram Shapes;** Anya Ji et al


## Cognitive NeuronScience
- **Functional specificity in the human brain: A window into the functional architecture of the mind;** Nancy Kanwisher et al
- **Visual motion aftereffect in human cortical area MT revealed by functional magnetic resonance imaging;** Roger B. H. Tootell et al
- **Speed of processing in the human visual system;** Simon Thorpe et al
- **A Cortical Area Selective for Visual Processing of the Human Body;** Paul E. Downing et al
- **Triple Dissociation of Faces, Bodies, and Objects in Extrastriate Cortex;** David Pitcher et al
- **Distributed and Overlapping Representations of Faces and Objects in Ventral Temporal Cortex;** James V. Haxby et al
- **Rectilinear Edge Selectivity Is Insufficient to Explain the Category Selectivity of the Parahippocampal Place Area;** Peter B. Bryan et al
- **Selective scene perception deficits in a case of topographical disorientation;** Jessica Robin et al
- **The cognitive map in humans: spatial navigation and beyond;** Russell A Epstein et al
- **From simple innate biases to complex visual concepts;** Shimon Ullman et al
- **Face perception in monkeys reared with no exposure to faces;** Yoichi Sugita et al
- **Functional neuroanatomy of intuitive physical inference;** Jason Fischer et al
- **Recruitment of an Area Involved in Eye Movements During Mental Arithmetic;** André Knops et al
- **Intonational speech prosody encoding in the human auditory cortex;** C. Tang et al


## Backup
- **On the Principles of Parsimony and Self-Consistency for the Emergence of Intelligence;** Yi Ma et al


## Resource
- **LAVIS-A One-stop Library for Language-Vision Intelligence;** https://github.com/salesforce/LAVIS

