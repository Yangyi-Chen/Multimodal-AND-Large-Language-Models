# Multimodal & Large Language Models

**Note:** This paper list is only used to record papers I read in the daily arxiv for personal needs. If you find I missed some important and exciting work, it would be super helpful to let me know. Thanks!

## Table of Contents

- [Survey](#survey)
- [Position Paper](#position-paper)
- [Structure](#structure)
- [Planning](#planning)
- [Reasoning](#reasoning)
- [Generation](#generation)
- [Representation Learning](#representation-learning)
- [LLM Analysis](#llm-analysis)
- [LLM Safety](#llm-safety)
- [LLM Evaluation](#llm-evaluation)
- [LLM Reasoning](#llm-reasoning)
- [LLM Application](#llm-application)
- [LLM with Memory](#llm-with-memory)
- [LLM with Human](#llm-with-human)
- [LLM Foundation](#llm-foundation)
- [Scaling Law](#scaling-law)
- [LLM Data Engineering](#llm-data-engineering)
- [VLM Data Engineering](#vlm-data-engineering)
- [Alignment](#alignment)
- [Scalable Oversight&amp;SuperAlignment](#scalable-oversight-&-superalignment)
- [RL Foundation](#rl-foundation)
- [Beyond Bandit](#beyond-bandit)
- [Agent](#agent)
- [Interaction](#interaction)
- [Critique Modeling](#critic-modeling)
- [MoE/Specialized](#moe/specialized)
- [Vision-Language Foundation Model](#vision-language-foundation-model)
- [Vision-Language Model Analysis &amp; Evaluation](#vision-language-model-analysis&evaluation)
- [Vision-Language Model Application](#vision-language-model-application)
- [Multimodal Foundation Model](#multimodal-foundation-model)
- [Image Generation](#image-generation)
- [Document Understanding](#document-understanding)
- [Tool Learning](#tool-learning)
- [Instruction Tuning](#instruction-tuning)
- [Incontext Learning](#incontext-learning)
- [Learning from Feedback](#learning-from-feedback)
- [Video Foundation Model](#video-foundation-model)
- [Key Frame Detection](#key-frame-detection)
- [Pretraining](#pretraining)
- [Vision Model](#vision-model)
- [Adaptation of Foundation Model](#adaptation-of-foundation-model)
- [Prompting](#prompting)
- [Efficiency](#efficiency)
- [Analysis](#analysis)
- [Grounding](#grounding)
- [VQA Task](#vqa-task)
- [VQA Dataset](#vqa-dataset)
- [Social Good](#social-good)
- [Application](#application)
- [Benchmark &amp; Evaluation](#benchmark-&-evaluation)
- [Dataset](#dataset)
- [Robustness](#robustness)
- [Hallucination&amp;Factuality](#hallucination&factuality)
- [Cognitive NeuronScience &amp; Machine Learning](#cognitive-neuronscience-&-machine-learning)
- [Theory of Mind](#theory-of-mind)
- [Cognitive NeuronScience](#cognitive-neuronscience)
- [World Model](#world-model)
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
- **The Contribution of Knowledge in Visiolinguistic Learning: A Survey on Tasks and Challenges;** Maria Lymperaiou et al
- **Augmented Language Models: a Survey;** Grégoire Mialon et al
- **Multimodal Deep Learning;** Matthias Aßenmacher et al
- **Sparks of Artificial General Intelligence: Early experiments with GPT-4;** Sebastien Bubeck et al
- **Retrieving Multimodal Information for Augmented Generation: A Survey;** Ruochen Zhao et al
- **Is Prompt All You Need? No. A Comprehensive and Broader View of Instruction Learning;** Renze Lou et al
- **A Survey of Large Language Models;** Wayne Xin Zhao et al
- **Tool Learning with Foundation Models;** Yujia Qin et al
- **A Cookbook of Self-Supervised Learning;** Randall Balestriero et al
- **Foundation Models for Decision Making: Problems, Methods, and Opportunities;** Sherry Yang et al
- **Bridging the Gap: A Survey on Integrating (Human) Feedback for Natural Language Generation;** Patrick Fernandes et al
- **Reasoning with Language Model Prompting: A Survey;** Shuofei Qiao et al
- **Towards Reasoning in Large Language Models: A Survey;** Jie Huang et al
- **Beyond One-Model-Fits-All: A Survey of Domain Specialization for Large Language Models;** Chen Ling et al
- **Unifying Large Language Models and Knowledge Graphs: A Roadmap;** Shirui Pan et al
- **Interactive Natural Language Processing;** Zekun Wang et al
- **A Survey on Multimodal Large Language Models;** Shukang Yin et al
- **TRUSTWORTHY LLMS: A SURVEY AND GUIDELINE FOR EVALUATING LARGE LANGUAGE MODELS’ ALIGNMENT;** Yang Liu et al
- **Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback;** Stephen Casper et al
- **Automatically Correcting Large Language Models: Surveying the landscape of diverse self-correction strategies;** Liangming Pan et al
- **Challenges and Applications of Large Language Models;** Jean Kaddour et al
- **Aligning Large Language Models with Human: A Survey;** Yufei Wang et al
- **Instruction Tuning for Large Language Models: A Survey;** Shengyu Zhang et al
- **From Instructions to Intrinsic Human Values —— A Survey of Alignment Goals for Big Models;** Jing Yao et al
- **A Survey of Safety and Trustworthiness of Large Language Models through the Lens of Verification and Validation;** Xiaowei Huang et al
- **Explainability for Large Language Models: A Survey;** Haiyan Zhao et al
- **Siren’s Song in the AI Ocean: A Survey on Hallucination in Large Language Models;** Yue Zhang et al
- **Survey on Factuality in Large Language Models: Knowledge, Retrieval and Domain-Specificity;** Cunxiang Wang et al
- **ChatGPT’s One-year Anniversary: Are Open-Source Large Language Models Catching up?;** Hailin Chen et al
- **Vision-Language Instruction Tuning: A Review and Analysis;** Chen Li et al
- **The Mystery and Fascination of LLMs: A Comprehensive Survey on the Interpretation and Analysis of Emergent Abilities;** Yuxiang Zhou et al
- **Efficient Large Language Models: A Survey;** Zhongwei Wan et al
- **The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision);** Zhengyuan Yang et al
- **Igniting Language Intelligence: The Hitchhiker’s Guide From Chain-of-Thought Reasoning to Language Agents;** Zhuosheng Zhang et al
- **Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis;** Yafei Hu et al
- **Multimodal Foundation Models: From Specialists to General-Purpose Assistants;** Chunyuan Li et al
- **A Survey on Large Language Model based Autonomous Agents;** Lei Wang et al
- **Video Understanding with Large Language Models: A Survey;** Yunlong Tang et al
- **A Survey of Preference-Based Reinforcement Learning Methods;** Christian Wirth et al
- **AI Alignment: A Comprehensive Survey;** Jiaming Ji et al
- **A SURVEY OF REINFORCEMENT LEARNING FROM HUMAN FEEDBACK;** Timo Kaufmann et al
- **TRUSTLLM: TRUSTWORTHINESS IN LARGE LANGUAGE MODELS;** Lichao Sun et al
- **AGENT AI: SURVEYING THE HORIZONS OF MULTIMODAL INTERACTION;** Zane Durante et al
- **Autotelic Agents with Intrinsically Motivated Goal-Conditioned Reinforcement Learning: A Short Survey;** Cedric Colas et al
- **Safety of Multimodal Large Language Models on Images and Text;** Xin Liu et al
- **MM-LLMs: Recent Advances in MultiModal Large Language Models;** Duzhen Zhang et al
- **Rethinking Interpretability in the Era of Large Language Models;** Chandan Singh et al
- **Large Multimodal Agents: A Survey;** Junlin Xie et al
- **A Survey on Data Selection for Language Models;** Alon Albalak et al
- **What Are Tools Anyway? A Survey from the Language Model Perspective;** Zora Zhiruo Wang et al

## Position Paper

- **Eight Things to Know about Large Language Models;** Samuel R. Bowman et al
- **A PhD Student’s Perspective on Research in NLP in the Era of Very Large Language Models;** Oana Ignat et al
- **Brain in a Vat: On Missing Pieces Towards Artificial General Intelligence in Large Language Models;** Yuxi Ma et al
- **Towards AGI in Computer Vision: Lessons Learned from GPT and Large Language Models;** Lingxi Xie et al
- **A Path Towards Autonomous Machine Intelligence;** Yann LeCun et al
- **GPT-4 Can’t Reason;** Konstantine Arkoudas et al
- **Cognitive Architectures for Language Agents;** Theodore Sumers et al
- **Large Search Model: Redefining Search Stack in the Era of LLMs;** Liang Wang et al
- **PROAGENT: FROM ROBOTIC PROCESS AUTOMATION TO AGENTIC PROCESS AUTOMATION;** Yining Ye et al
- **Language Models, Agent Models, and World Models: The LAW for Machine Reasoning and Planning;** Zhiting Hu et al
- **A Roadmap to Pluralistic Alignment;** Taylor Sorensen et al
- **Towards Unified Alignment Between Agents, Humans, and Environment;** Zonghan Yang et al
- **Video as the New Language for Real-World Decision Making;** Sherry Yang et al

## Structure

- **Finding Structural Knowledge in Multimodal-BERT;** Victor Milewski et al
- **Going Beyond Nouns With Vision & Language Models Using Synthetic Data;** Paola Cascante-Bonilla et al
- **Measuring Progress in Fine-grained Vision-and-Language Understanding;** Emanuele Bugliarello et al
- **PV2TEA: Patching Visual Modality to Textual-Established Information Extraction;** Hejie Cui et al

**Event Extraction**

- **Cross-media Structured Common Space for Multimedia Event Extraction;** Manling Li et al; Focus on image-text event extraction. A new benchmark and baseline are proposed.
- **Visual Semantic Role Labeling for Video Understanding;** Arka Sadhu et al; A new benchmark is proposed.
- **GAIA: A Fine-grained Multimedia Knowledge Extraction System;** Manling Li et al; Demo paper. Extract knowledge (relation, event) from multimedia data.
- **MMEKG: Multi-modal Event Knowledge Graph towards Universal Representation across Modalities;** Yubo Ma et al

**Situation Recognition**

- **Situation Recognition: Visual Semantic Role Labeling for Image Understanding;** Mark Yatskar et al; Focus on image understanding. Given images, do the semantic role labeling task. No text available. A new benchmark and baseline are proposed.
- **Commonly Uncommon: Semantic Sparsity in Situation Recognition;** Mark Yatskar et al; Address the long-tail problem.
- **Grounded Situation Recognition;** Sarah Pratt et al
- **Rethinking the Two-Stage Framework for Grounded Situation Recognition;** Meng Wei et al
- **Collaborative Transformers for Grounded Situation Recognition;** Junhyeong Cho et al

**Scene Graph**

- **Action Genome: Actions as Composition of Spatio-temporal Scene Graphs;** Jingwei Ji et al; Spatio-temporal scene graphs (video).
- **Unbiased Scene Graph Generation from Biased Training;** Kaihua Tang et al
- **Visual Distant Supervision for Scene Graph Generation;** Yuan Yao et al
- **Learning to Generate Scene Graph from Natural Language Supervision;** Yiwu Zhong et al
- **Weakly Supervised Visual Semantic Parsing;** Alireza Zareian, Svebor Karaman, Shih-Fu Chang
- **Scene Graph Prediction with Limited Labels;** Vincent S. Chen, Paroma Varma, Ranjay Krishna, Michael Bernstein, Christopher Re, Li Fei-Fei
- **Neural Motifs: Scene Graph Parsing with Global Context;** Rowan Zellers et al
- **Fine-Grained Scene Graph Generation with Data Transfer;** Ao Zhang et al
- **Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning;** Tao He et al
- **COMPOSITIONAL PROMPT TUNING WITH MOTION CUES FOR OPEN-VOCABULARY VIDEO RELATION DETECTION;** Kaifeng Gao et al; Video.
- **LANDMARK: Language-guided Representation Enhancement Framework for Scene Graph Generation;** Xiaoguang Chang et al
- **TRANSFORMER-BASED IMAGE GENERATION FROM SCENE GRAPHS;** Renato Sortino et al
- **The Devil is in the Labels: Noisy Label Correction for Robust Scene Graph Generation;** Lin Li et al
- **Knowledge-augmented Few-shot Visual Relation Detection;** Tianyu Yu et al
- **Prototype-based Embedding Network for Scene Graph Generation;** Chaofan Zhen et al
- **Unified Visual Relationship Detection with Vision and Language Models;** Long Zhao et al
- **Structure-CLIP: Enhance Multi-modal Language Representations with Structure Knowledge;** Yufeng Huang et al

**Attribute**

- **COCO Attributes: Attributes for People, Animals, and Objects;** Genevieve Patterson et al
- **Human Attribute Recognition by Deep Hierarchical Contexts;** Yining Li et al; Attribute prediction in specific domains.
- **Emotion Recognition in Context;** Ronak Kosti et al; Attribute prediction in specific domains.
- **The iMaterialist Fashion Attribute Dataset;** Sheng Guo et al; Attribute prediction in specific domains.
- **Learning to Predict Visual Attributes in the Wild;** Khoi Pham et al
- **Open-vocabulary Attribute Detection;** Marıa A. Bravo et al
- **OvarNet: Towards Open-vocabulary Object Attribute Recognition;** Keyan Chen et al

**Compositionality**

- **CREPE: Can Vision-Language Foundation Models Reason Compositionally?;** Zixian Ma et al
- **Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality;** Tristan Thrush et al
- **WHEN AND WHY VISION-LANGUAGE MODELS BEHAVE LIKE BAGS-OF-WORDS, AND WHAT TO DO ABOUT IT?;** Mert Yuksekgonul et al
- **GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering;** Drew A. Hudson et al
- **COVR: A Test-Bed for Visually Grounded Compositional Generalization with Real Images;** Ben Bogin et al
- **Cops-Ref: A new Dataset and Task on Compositional Referring Expression Comprehension;** Zhenfang Chen et al
- **Do Vision-Language Pretrained Models Learn Composable Primitive Concepts?;** Tian Yun et al
- **SUGARCREPE: Fixing Hackable Benchmarks for Vision-Language Compositionality;** Cheng-Yu Hsieh et al
- **An Examination of the Compositionality of Large Generative Vision-Language Models;** Teli Ma et al

**Concept**

- **Cross-Modal Concept Learning and Inference for Vision-Language Models;** Yi Zhang et al
- **Hierarchical Visual Primitive Experts for Compositional Zero-Shot Learning;** Hanjae Kim et al

## Planning

- **Multimedia Generative Script Learning for Task Planning;** Qingyun Wang et al; Next step prediction.
- **PlaTe: Visually-Grounded Planning with Transformers in Procedural Tasks;** Jiankai Sun et al; Procedure planning.
- **P3IV: Probabilistic Procedure Planning from Instructional Videos with Weak Supervision;** He Zhao et al; Procedure planning. Using text as weak supervision to replace video clips.
- **Procedure Planning in Instructional Videos;** Chien-Yi Chang et al; Procedure planning.
- **ViLPAct: A Benchmark for Compositional Generalization on Multimodal Human Activities;** Terry Yue Zhuo et al
- **Actional Atomic-Concept Learning for Demystifying Vision-Language Navigation;** Bingqian Lin et al

## Reasoning

- **VisualCOMET: Reasoning about the Dynamic Context of a Still Image;** Jae Sung Park et al; Benchmark dataset, requiring models to reason about a still image (what happen past & next).
- **Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering;** Pan Lu et al
- **See, Think, Confirm: Interactive Prompting Between Vision and Language Models for Knowledge-based Visual Reasoning;** Zhenfang Chen et al
- **An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA;** Zhengyuan Yang et al
- **Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering;** Pan Lu et al
- **Multimodal Chain-of-Thought Reasoning in Language Models;** Zhuosheng Zhang et al
- **LAMPP: Language Models as Probabilistic Priors for Perception and Action;** Belinda Z. Li et al
- **Visual Chain of Thought: Bridging Logical Gaps with Multimodal Infillings;** Daniel Rose et al
- **Symbol-LLM: Leverage Language Models for Symbolic System in Visual Human Activity Reasoning;** Xiaoqian Wu et al

**Common sense.**

- **Improving Commonsense in Vision-Language Models via Knowledge Graph Riddles;** Shuquan Ye et al
- **VIPHY: Probing “Visible” Physical Commonsense Knowledge;** Shikhar Singh et al
- **Visual Commonsense in Pretrained Unimodal and Multimodal Models;** Chenyu Zhang et al

## Generation

- **ClipCap: CLIP Prefix for Image Captioning;** Ron Mokady et al; Train an light-weight encoder to convert CLIP embeddings to prefix token embeddings of GPT-2.
- **Multimodal Knowledge Alignment with Reinforcement Learning;** Youngjae Yu et al; Use RL to train an encoder that projects multimodal inputs into the word embedding space of GPT-2.

## Representation Learning

- **Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering;** Peter Anderson et al
- **Fusion of Detected Objects in Text for Visual Question Answering;** Chris Alberti et al
- **VLMixer: Unpaired Vision-Language Pre-training via Cross-Modal CutMix;** Teng Wang et al
- **Vision-Language Pre-Training with Triple Contrastive Learning;** Jinyu Yang et al
- **Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision;** Hao Tan et al; Use visual supervision to pretrain language models.
- **HighMMT: Quantifying Modality & Interaction Heterogeneity for High-Modality Representation Learning;** Paul Pu Liang et al
- **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture;** Mahmoud Assran et al
- **PIGLeT: Language Grounding Through Neuro-Symbolic Interaction in a 3D World;** Rowan Zellers et al
- **Learning the Effects of Physical Actions in a Multi-modal Environment;** Gautier Dagan et al
- **Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models;** Zhiqiu Lin et al
- **Learning Visual Representations via Language-Guided Sampling;** Mohamed El Banani et al
- **Image as Set of Points;** Xu Ma et al
- **ARCL: ENHANCING CONTRASTIVE LEARNING WITH AUGMENTATION-ROBUST REPRESENTATIONS;** Xuyang Zhao et al
- **BRIDGING THE GAP TO REAL-WORLD OBJECT-CENTRIC LEARNING;** Maximilian Seitzer et al
- **Learning Transferable Spatiotemporal Representations from Natural Script Knowledge;** Ziyun Zeng et al
- **Understanding and Constructing Latent Modality Structures in Multi-Modal Representation Learning;** Qian Jiang et al

## LLM Analysis

- **GROKKING: GENERALIZATION BEYOND OVERFITTING ON SMALL ALGORITHMIC DATASETS;** Alethea Power et al
- **Unified View of Grokking, Double Descent and Emergent Abilities: A Perspective from Circuits Competition;** Yufei Huang et al
- **A Categorical Archive of ChatGPT Failures;** Ali Borji et al
- **Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling;** Stella Biderman  et al
- **Are Emergent Abilities of Large Language Models a Mirage?;** Rylan Schaeffer et al
- **A Drop of Ink may Make a Million Think: The Spread of False Information in Large Language Models;** Ning Bian et al
- **Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting;** Miles Turpin et al
- **SYMBOL TUNING IMPROVES IN-CONTEXT LEARNING IN LANGUAGE MODELS;** Jerry Wei et al
- **What In-Context Learning “Learns” In-Context: Disentangling Task Recognition and Task Learning;** Jane Pan et al
- **Measuring the Knowledge Acquisition-Utilization Gap in Pretrained Language Models;** Amirhossein Kazemnejad et al
- **Scaling Data-Constrained Language Models;** Niklas Muennighoff et al
- **The False Promise of Imitating Proprietary LLMs;** Arnav Gudibande et al
- **Counterfactual reasoning: Testing language models’ understanding of hypothetical scenarios;** Jiaxuan Li et al
- **Inverse Scaling: When Bigger Isn’t Better;** Ian R. McKenzie et al
- **DECODINGTRUST: A Comprehensive Assessment of Trustworthiness in GPT Models;** Boxin Wang et al
- **Lost in the Middle: How Language Models Use Long Contexts;** Nelson F. Liu et al
- **Won’t Get Fooled Again: Answering Questions with False Premises;** Shengding Hu et al
- **Generating Benchmarks for Factuality Evaluation of Language Models;** Dor Muhlgay et al
- **Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations;** Yanda Chen et al
- **Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation;** Ruiyang Ren et al
- **Large Language Models Struggle to Learn Long-Tail Knowledge;** Nikhil Kandpal et al
- **SCALING RELATIONSHIP ON LEARNING MATHEMATICAL REASONING WITH LARGE LANGUAGE MODELS;** Zheng Yuan et al
- **Multimodal Neurons in Pretrained Text-Only Transformers;** Sarah Schwettmann et al
- **SIMPLE SYNTHETIC DATA REDUCES SYCOPHANCY IN LARGE LANGUAGE MODELS;** Jerry Wei et al
- **Studying Large Language Model Generalization with Influence Functions;** Roger Grosse et al
- **Taken out of context: On measuring situational awareness in LLMs;** Lukas Berglund et al
- **OpinionGPT: Modelling Explicit Biases in Instruction-Tuned LLMs;** Patrick Haller et al
- **Neurons in Large Language Models: Dead, N-gram, Positional;** Elena Voita et al
- **Are Emergent Abilities in Large Language Models just In-Context Learning?;** Sheng Lu et al
- **The Reversal Curse: LLMs trained on “A is B” fail to learn “B is A”;** Lukas Berglund et al
- **Language Modeling Is Compression;** Grégoire Delétang et al
- **FROM LANGUAGE MODELING TO INSTRUCTION FOLLOWING: UNDERSTANDING THE BEHAVIOR SHIFT IN LLMS AFTER INSTRUCTION TUNING;** Xuansheng Wu et al
- **RESOLVING KNOWLEDGE CONFLICTS IN LARGE LANGUAGE MODELS;** Yike Wang et al
- **LARGE LANGUAGE MODELS CANNOT SELF-CORRECT REASONING YET;** Jie Huang et al
- **ASK AGAIN, THEN FAIL: LARGE LANGUAGE MODELS’ VACILLATIONS IN JUDGEMENT;** Qiming Xie et al
- **FRESHLLMS: REFRESHING LARGE LANGUAGE MODELS WITH SEARCH ENGINE AUGMENTATION;** Tu Vu et al
- **Demystifying Embedding Spaces using Large Language Models;** Guy Tennenholtz et al
- **An Emulator for Fine-Tuning Large Language Models using Small Language Models;** Eric Mitchell et al
- **UNVEILING A CORE LINGUISTIC REGION IN LARGE LANGUAGE MODELS;** Jun Zhao et al
- **DETECTING PRETRAINING DATA FROM LARGE LANGUAGE MODELS;** Weijia Shi et al
- **BENCHMARKING AND IMPROVING GENERATOR-VALIDATOR CONSISTENCY OF LMS;** Xiang Lisa Li et al
- **UNLOCK PREDICTABLE SCALING FROM EMERGENT ABILITIES;** Shengding Hu et al
- **Trusted Source Alignment in Large Language Models;** Vasilisa Bashlovkina et al
- **THE UNLOCKING SPELL ON BASE LLMS: RETHINKING ALIGNMENT VIA IN-CONTEXT LEARNING;** Bill Yuchen Lin et al
- **Can Large Language Models Really Improve by Self-critiquing Their Own Plans?;** Karthik Valmeekam et al
- **TELL, DON’T SHOW: DECLARATIVE FACTS INFLUENCE HOW LLMS GENERALIZE;** Alexander Meinke et al
- **A Closer Look at the Limitations of Instruction Tuning;** Sreyan Ghosh et al
- **PERSONAS AS A WAY TO MODEL TRUTHFULNESS IN LANGUAGE MODELS;** Nitish Joshi et al
- **Beyond Probabilities: Unveiling the Misalignment in Evaluating Large Language Models;** Chenyang Lyu et al
- **Dated Data: Tracing Knowledge Cutoffs in Large Language Models;** Jeffrey Cheng et al


**Calibration & Uncertainty**

- **Knowledge of Knowledge: Exploring Known-Unknowns Uncertainty with Large Language Models;** Alfonso Amayuelas et al
- **Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs;** Miao Xiong et al
- **LLAMAS KNOW WHAT GPTS DON’T SHOW: SURROGATE MODELS FOR CONFIDENCE ESTIMATION;** Vaishnavi Shrivastava et al
- **Navigating the Grey Area: How Expressions of Uncertainty and Overconfidence Affect Language Models;** Kaitlyn Zhou et al
- **R-Tuning: Teaching Large Language Models to Refuse Unknown Questions;** Hanning Zhang et al
- **Relying on the Unreliable: The Impact of Language Models’ Reluctance to Express Uncertainty;** Kaitlyn Zhou et al
- **Prudent Silence or Foolish Babble? Examining Large Language Models’ Responses to the Unknown;** Genglin Liu et al
- **Benchmarking LLMs via Uncertainty Quantification;** Fanghua Ye et al
- **Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback;** Katherine Tian et al
- **Deal, or no deal (or who knows)? Forecasting Uncertainty in Conversations using Large Language Models;** Anthony Sicilia et al
- **Calibrating Long-form Generations from Large Language Models;** Yukun Huang et al
- **Distinguishing the Knowable from the Unknowable with Language Models;** Gustaf Ahdritz et al
- **Introspective Planning: Guiding Language-Enabled Agents to Refine Their Own Uncertainty;** Kaiqu Liang et al
- **Asking the Right Question at the Right Time: Human and Model Uncertainty Guidance to Ask Clarification Questions;** Alberto Testoni et al
- **Into the Unknown: Self-Learning Large Language Models;** Teddy Ferdinan et al
- **The Internal State of an LLM Knows When It’s Lying;** Amos Azaria et al
- **SELFCHECKGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models;** Potsawee Manakul et al
- **Calibrating Large Language Models with Sample Consistency;** Qing Lyu et al
- **Gotcha! Don’t trick me with unanswerable questions! Self-aligning Large Language Models for Responding to Unknown Questions;** Yang Deng et al
- **Unfamiliar Finetuning Examples Control How Language Models Hallucinate;** Katie Kang et al



## LLM Safety

- **Learning Human Objectives by Evaluating Hypothetical Behavior;** Siddharth Reddy et al
- **Universal and Transferable Adversarial Attacks on Aligned Language Models;** Andy Zou et al
- **XSTEST: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models;** Paul Röttger et al
- **Jailbroken: How Does LLM Safety Training Fail? Content Warning: This paper contains examples of harmful language;** Alexander Wei et al
- **FUNDAMENTAL LIMITATIONS OF ALIGNMENT IN LARGE LANGUAGE MODELS;** Yotam Wolf et al
- **BEAVERTAILS: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset;** Jiaming Ji et al
- **GPT-4 IS TOO SMART TO BE SAFE: STEALTHY CHAT WITH LLMS VIA CIPHER;** Youliang Yuan et al
- **Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment;** Rishabh Bhardwaj et al
- **Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs;** Yuxia Wang et al
- **SafetyBench: Evaluating the Safety of Large Language Models with Multiple Choice Questions;** Zhexin Zhang et al
- **Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions;** Federico Bianchi et al
- **Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations;** Hakan Inan et al
- **EMULATED DISALIGNMENT: SAFETY ALIGNMENT FOR LARGE LANGUAGE MODELS MAY BACKFIRE!;** Zhanhui Zhou et al
- **Logits of API-Protected LLMs Leak Proprietary Information;** Matthew Finlayson et al

## LLM Evaluation

- **IS CHATGPT A GENERAL-PURPOSE NATURAL LANGUAGE PROCESSING TASK SOLVER?;** Chengwei Qin et al
- **AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models;** Wanjun Zhong et al
- **A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity;** Yejin Bang et al
- **On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective;** Jindong Wang et al
- **A Comprehensive Capability Analysis of GPT-3 and GPT-3.5 Series Models;** Junjie Ye et al
- **KoLA: Carefully Benchmarking World Knowledge of Large Language Models;** Jifan Yu et al
- **SCIBENCH: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models;** Xiaoxuan Wang et al
- **FLASK: FINE-GRAINED LANGUAGE MODEL EVALUATION BASED ON ALIGNMENT SKILL SETS;** Seonghyeon Ye et al
- **Efficient Benchmarking (of Language Models);** Yotam Perlitz et al
- **Can Large Language Models Understand Real-World Complex Instructions?;** Qianyu He et al
- **NLPBENCH: EVALUATING LARGE LANGUAGE MODELS ON SOLVING NLP PROBLEMS;** Linxin Song et al
- **CALIBRATING LLM-BASED EVALUATOR;** Yuxuan Liu et al
- **GPT-FATHOM: BENCHMARKING LARGE LANGUAGE MODELS TO DECIPHER THE EVOLUTIONARY PATH TOWARDS GPT-4 AND BEYOND;** Shen Zheng et al
- **L2CEval: Evaluating Language-to-Code Generation Capabilities of Large Language Models;** Ansong Ni et al
- **Revisiting Out-of-distribution Robustness in NLP: Benchmark, Analysis, and LLMs Evaluations;** Lifan Yuan et al
- **TIGERSCORE: TOWARDS BUILDING EXPLAINABLE METRIC FOR ALL TEXT GENERATION TASKS;** Dongfu Jiang et al
- **DO LARGE LANGUAGE MODELS KNOW ABOUT FACTS?;** Xuming Hu et al
- **PROMETHEUS: INDUCING FINE-GRAINED EVALUATION CAPABILITY IN LANGUAGE MODELS;** Seungone Kim et al
- **CRITIQUE ABILITY OF LARGE LANGUAGE MODELS;** Liangchen Luo et al
- **BotChat: Evaluating LLMs’ Capabilities of Having Multi-Turn Dialogues;** Haodong Duan et al
- **Instruction-Following Evaluation for Large Language Models;** Jeffrey Zhou et al
- **GAIA: A Benchmark for General AI Assistants;** Gregoire Mialon et al
- **ML-BENCH: LARGE LANGUAGE MODELS LEVERAGE OPEN-SOURCE LIBRARIES FOR MACHINE LEARNING TASKS;** Yuliang Liu et al
- **TASKBENCH: BENCHMARKING LARGE LANGUAGE MODELS FOR TASK AUTOMATION;** Yongliang Shen et al
- **GENERATIVE JUDGE FOR EVALUATING ALIGNMENT;** Junlong Li et al
- **InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks;** Xueyu Hu et al
- **AGENTBOARD: AN ANALYTICAL EVALUATION BOARD OF MULTI-TURN LLM AGENTS;** Chang Ma et al
- **WEBLINX: Real-World Website Navigation with Multi-Turn Dialogue;** Xing Han Lu et al
- **MatPlotAgent: Method and Evaluation for LLM-Based Agentic Scientific Data Visualization;** Zhiyu Yang et al
- **Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference;** Wei-Lin Chiang et al
- **DevBench: A Comprehensive Benchmark for Software Development;** Bowen Li et al
- **REWARDBENCH: Evaluating Reward Models for Language Modeling;** Nathan Lambert et al



## LLM Reasoning

- **STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning;** Eric Zelikman et al
- **Generated Knowledge Prompting for Commonsense Reasoning;** Jiacheng Liu et al
- **SELF-CONSISTENCY IMPROVES CHAIN OF THOUGHT REASONING IN LANGUAGE MODELS;** Xuezhi Wang et al
- **LEAST-TO-MOST PROMPTING ENABLES COMPLEX REASONING IN LARGE LANGUAGE MODELS;** Denny Zhou et al
- **REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS;** Shunyu Yao et al
- **The Capacity for Moral Self-Correction in Large Language Models;** Deep Ganguli et al
- **Learning to Reason and Memorize with Self-Notes;** Jack lanchantin et al
- **Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models;** Lei Wang et al
- **T-SciQ: Teaching Multimodal Chain-of-Thought Reasoning via Large Language Model Signals for Science Question Answering;** Lei Wang et al
- **Tree of Thoughts: Deliberate Problem Solving with Large Language Models;** Shunyu Yao et al
- **Introspective Tips: Large Language Model for In-Context Decision Making;** Liting Chen et al
- **Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples;** Abulhair Saparov et al
- **Reasoning with Language Model is Planning with World Model;** Shibo Hao et al
- **Interpretable Math Word Problem Solution Generation Via Step-by-step Planning;** Mengxue Zhang et al
- **Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters;** Boshi Wang et al
- **Recursion of Thought: A Divide-and-Conquer Approach to Multi-Context Reasoning with Language Models;** Soochan Lee et al
- **Large Language Models Are Reasoning Teachers;** Namgyu Ho et al
- **Meta-Reasoning: Semantics-Symbol Deconstruction For Large Language Models;** Yiming Wang et al
- **BeamSearchQA: Large Language Models are Strong Zero-Shot QA Solver;** Hao Sun et al
- **AdaPlanner: Adaptive Planning from Feedback with Language Models;** Haotian Sun et al
- **ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models;** Binfeng Xu et al
- **SKILLS-IN-CONTEXT PROMPTING: UNLOCKING COMPOSITIONALITY IN LARGE LANGUAGE MODELS;** Jiaao Chen et al
- **SOLVING CHALLENGING MATH WORD PROBLEMS USING GPT-4 CODE INTERPRETER WITH CODE-BASED SELF-VERIFICATION;** Aojun Zhou et al
- **MAMMOTH: BUILDING MATH GENERALIST MODELS THROUGH HYBRID INSTRUCTION TUNING;** Xiang Yue et al
- **DESIGN OF CHAIN-OF-THOUGHT IN MATH PROBLEM SOLVING;** Zhanming Jie et al
- **NATURAL LANGUAGE EMBEDDED PROGRAMS FOR HYBRID LANGUAGE SYMBOLIC REASONING;** Tianhua Zhang et al
- **MATHCODER: SEAMLESS CODE INTEGRATION IN LLMS FOR ENHANCED MATHEMATICAL REASONING;** Ke Wang et al
- **META-COT: GENERALIZABLE CHAIN-OF-THOUGHT PROMPTING IN MIXED-TASK SCENARIOS WITH LARGE LANGUAGE MODELS;** Anni Zou et al
- **TOOLCHAIN: EFFICIENT ACTION SPACE NAVIGATION IN LARGE LANGUAGE MODELS WITH A SEARCH;** Yuchen Zhuang et al
- **Learning From Mistakes Makes LLM Better Reasoner;** Shengnan An et al
- **Chain of Code: Reasoning with a Language Model-Augmented Code Emulator;** Chengshu Li et al
- **Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives;** Wenqi Zhang et al
- **Divide and Conquer for Large Language Models Reasoning;** Zijie Meng et al
- **The Impact of Reasoning Step Length on Large Language Models;** Mingyu Jin et al
- **REFT: Reasoning with REinforced Fine-Tuning;** Trung Quoc Luong et al
- **Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding;** Mirac Suzgun et al
- **SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures;** Pei Zhou et al
- **Guiding Large Language Models with Divide-and-Conquer Program for Discerning Problem Solving;** Yizhou Zhang et al
- **Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning;** Zhiheng Xi et al
- **V-STaR: Training Verifiers for Self-Taught Reasoners;** Arian Hosseini et al
- **Verified Multi-Step Synthesis using Large Language Models and Monte Carlo Tree Search;** David Brandfonbrener et al
- **BOOSTING OF THOUGHTS: TRIAL-AND-ERROR PROBLEM SOLVING WITH LARGE LANGUAGE MODELS;** Sijia Chen et al
- **Language Agents as Optimizable Graphs;** Mingchen Zhuge et al
- **MathScale: Scaling Instruction Tuning for Mathematical Reasoning;** Zhengyang Tang et al
- **Teaching Large Language Models to Reason with Reinforcement Learning;** Alex Havrilla et al
- **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking;** Eric Zelikman et al

**Self-consistency**

- **Enhancing Self-Consistency and Performance of Pre-Trained Language Models through Natural Language Inference;** Eric Mitchell et al
- **Two Failures of Self-Consistency in the Multi-Step Reasoning of LLMs;** Angelica Chen et al
- **Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation;** Niels Mündler et al
- **Measuring and Narrowing the Compositionality Gap in Language Models;** Ofir Press et al
- **Self-consistency for open-ended generations;** Siddhartha Jain et al
- **Question Decomposition Improves the Faithfulness of Model-Generated Reasoning;** Ansh Radhakrishnan et al
- **Measuring Faithfulness in Chain-of-Thought Reasoning;** Tamera Lanham et al
- **SELFCHECK: USING LLMS TO ZERO-SHOT CHECK THEIR OWN STEP-BY-STEP REASONING;** Ning Miao et al
- **On Measuring Faithfulness or Self-consistency of Natural Language Explanations;** Letitia Parcalabescu et al
- **Chain-of-Thought Unfaithfulness as Disguised Accuracy;** Oliver Bentham et al

(with images)

- **Sunny and Dark Outside?! Improving Answer Consistency in VQA through Entailed Question Generation;** Arijit Ray et al
- **Maintaining Reasoning Consistency in Compositional Visual Question Answering;** Chenchen Jing et al
- **SQuINTing at VQA Models: Introspecting VQA Models with Sub-Questions;** Ramprasaath R. Selvaraju et al
- **Logical Implications for Visual Question Answering Consistency;** Sergio Tascon-Morales et al
- **Exposing and Addressing Cross-Task Inconsistency in Unified Vision-Language Models;** Adyasha Maharana et al
- **Co-VQA: Answering by Interactive Sub Question Sequence;** Ruonan Wang et al
- **IdealGPT: Iteratively Decomposing Vision and Language Reasoning via Large Language Models;** Haoxuan You et al
- **Understanding ME? Multimodal Evaluation for Fine-grained Visual Commonsense;** Zhecan Wang et al

## LLM Application

- **ArK: Augmented Reality with Knowledge Interactive Emergent Ability;** Qiuyuan Huang et al
- **Can Large Language Models Be an Alternative to Human Evaluation?;** Cheng-Han Chiang et al
- **Few-shot In-context Learning for Knowledge Base Question Answering;** Tianle Li et al
- **AutoML-GPT: Automatic Machine Learning with GPT;** Shujian Zhang et al
- **Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs;** Jinyang Li et al
- **Language models can explain neurons in language models;** Steven Bills et al
- **Large Language Model Programs;** Imanol Schlag et al
- **Evaluating Factual Consistency of Summaries with Large Language Models;** Shiqi Chen et al
- **WikiChat: A Few-Shot LLM-Based Chatbot Grounded with Wikipedia;** Sina J. Semnani et al
- **Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning;** Xiaoming Shi et al
- **Images in Language Space: Exploring the Suitability of Large Language Models for Vision & Language Tasks;** Sherzod Hakimov et al
- **PEARL: Prompting Large Language Models to Plan and Execute Actions Over Long Documents;** Simeng Sun et al
- **LayoutGPT: Compositional Visual Planning and Generation with Large Language Models;** Weixi Feng et al
- **Judging LLM-as-a-judge with MT-Bench and Chatbot Arena;** Lianmin Zheng et al
- **LLM-BLENDER: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion;** Dongfu Jiang et al
- **Benchmarking Foundation Models with Language-Model-as-an-Examiner;** Yushi Bai et al
- **AudioPaLM: A Large Language Model That Can Speak and Listen;** Paul K. Rubenstein et al
- **Human-in-the-Loop through Chain-of-Thought;** Zefan Cai et al
- **LARGE LANGUAGE MODELS ARE EFFECTIVE TEXT RANKERS WITH PAIRWISE RANKING PROMPTING;** Zhen Qin et al
- **Language to Rewards for Robotic Skill Synthesis;** Wenhao Yu et al
- **Visual Programming for Text-to-Image Generation and Evaluation;** Jaemin Cho et al
- **Mindstorms in Natural Language-Based Societies of Mind;** Mingchen Zhuge et al
- **Responsible Task Automation: Empowering Large Language Models as Responsible Task Automators;** Zhizheng Zhang et al
- **Large Language Models as General Pattern Machines;** Suvir Mirchandani et al
- **A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generation;** Neeraj Varshney et al
- **VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models;** Wenlong Huang et al
- **External Reasoning: Towards Multi-Large-Language-Models Interchangeable Assistance with Human Feedback;** Akide Liu et al
- **OCTOPACK: INSTRUCTION TUNING CODE LARGE LANGUAGE MODELS;** Niklas Muennighoff et al
- **Tackling Vision Language Tasks Through Learning Inner Monologues;** Diji Yang et al
- **Can Language Models Learn to Listen?;** Evonne Ng et al
- **PROMPT2MODEL: Generating Deployable Models from Natural Language Instructions;** Vijay Viswanathan et al
- **AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models;** Zhaopeng Gu et al
- **LARGE LANGUAGE MODELS AS OPTIMIZERS;** Chengrun Yang et al
- **Large Language Model for Science: A Study on P vs. NP;** Qingxiu Dong et al
- **Physically Grounded Vision-Language Models for Robotic Manipulation;** Jensen Gao et al
- **Compositional Foundation Models for Hierarchical Planning;** Anurag Ajay et al
- **STRUC-BENCH: Are Large Language Models Really Good at Generating Complex Structured Data?;** Xiangru Tang et al
- **XATU: A Fine-grained Instruction-based Benchmark for Explainable Text Updates;** Haopeng Zhang et al
- **TEXT2REWARD: AUTOMATED DENSE REWARD FUNCTION GENERATION FOR REINFORCEMENT LEARNING;** Tianbao Xie et al
- **EUREKA: HUMAN-LEVEL REWARD DESIGN VIA CODING LARGE LANGUAGE MODELS;** Yecheng Jason Ma et al
- **CREATIVE ROBOT TOOL USE WITH LARGE LANGUAGE MODELS;** Mengdi Xu et al
- **Goal Driven Discovery of Distributional Differences via Language Descriptions;** Ruiqi Zhong et al
- **Can large language models provide useful feedback on research papers? A large-scale empirical analysis.;** Weixin Liang et al
- **DRIVEGPT4: INTERPRETABLE END-TO-END AUTONOMOUS DRIVING VIA LARGE LANGUAGE MODEL;** Zhenhua Xu et al
- **QUALEVAL: QUALITATIVE EVALUATION FOR MODEL IMPROVEMENT;** Vishvak Murahari et al
- **LLM AUGMENTED LLMS: EXPANDING CAPABILITIES THROUGH COMPOSITION;** Rachit Bansal et al
- **SpeechAgents: Human-Communication Simulation with Multi-Modal Multi-Agent Systems;** Dong Zhang et al
- **DEMOCRATIZING FINE-GRAINED VISUAL RECOGNITION WITH LARGE LANGUAGE MODELS;** Mingxuan Liu et al
- **Solving olympiad geometry without human demonstrations;** Trieu H. Trinh et al
- **AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Accuracy;** Philipp Schoenegger et al
- **What Evidence Do Language Models Find Convincing?;** Alexander Wan et al

## LLM with Memory

- **Neural Turing Machines;** Alex Graves et al
- **Narrative Question Answering with Cutting-Edge Open-Domain QA Techniques: A Comprehensive Study;** Xiangyang Mou et al
- **Memory and Knowledge Augmented Language Models for Inferring Salience in Long-Form Stories;** David Wilmot et al
- **MemPrompt: Memory-assisted Prompt Editing with User Feedback;** Aman Madaan et al
- **LANGUAGE MODEL WITH PLUG-IN KNOWLEDGE MEMORY;** Xin Cheng et al
- **Assessing Working Memory Capacity of ChatGPT;** Dongyu Gong et al
- **Prompted LLMs as Chatbot Modules for Long Open-domain Conversation;** Gibbeum Lee et al
- **Beyond Goldfish Memory: Long-Term Open-Domain Conversation;** Jing Xu et al
- **Memory Augmented Large Language Models are Computationally Universal;** Dale Schuurmans et al
- **MemoryBank: Enhancing Large Language Models with Long-Term Memory;** Wanjun Zhong et al
- **Adaptive Chameleon or Stubborn Sloth: Unraveling the Behavior of Large Language Models in Knowledge Clashes;** Jian Xie et al
- **RET-LLM: Towards a General Read-Write Memory for Large Language Models;** Ali Modarressi et al
- **RECURRENTGPT: Interactive Generation of (Arbitrarily) Long Text;** Wangchunshu Zhou et al
- **MEMORIZING TRANSFORMERS;** Yuhuai Wu et al
- **Augmenting Language Models with Long-Term Memory;** Weizhi Wang et al
- **Statler: State-Maintaining Language Models for Embodied Reasoning;** Takuma Yoneda et al
- **LONGNET: Scaling Transformers to 1,000,000,000 Tokens;** Jiayu Ding et al
- **In-context Autoencoder for Context Compression in a Large Language Model;** Tao Ge et al
- **MemoChat: Tuning LLMs to Use Memos for Consistent Long-Range Open-Domain Conversation;** Junru Lu et al
- **KnowledGPT: Enhancing Large Language Models with Retrieval and Storage Access on Knowledge Bases;** Xintao Wang et al
- **LONGBENCH: A BILINGUAL, MULTITASK BENCHMARK FOR LONG CONTEXT UNDERSTANDING;** Yushi Bai et al
- **ChipNeMo: Domain-Adapted LLMs for Chip Design;** Mingjie Liu et al
- **LongAlign: A Recipe for Long Context Alignment of Large Language Models;** Yushi Bai et al
- **RAPTOR: RECURSIVE ABSTRACTIVE PROCESSING FOR TREE-ORGANIZED RETRIEVAL;** Parth Sarthi et al
- **UniMem: Towards a Unified View of Long-Context Large Language Models;** Junjie Fang et al
- **A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts;** Kuang-Huei Lee et al
- **Transformers Can Achieve Length Generalization But Not Robustly;** Yongchao Zhou et al
- **Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context;** Gemini Team, Google


**Retrieval-augmented LLM**

- **Training Language Models with Memory Augmentation;** Zexuan Zhong et al
- **Enabling Large Language Models to Generate Text with Citations;** Tianyu Gao et al
- **Multiview Identifiers Enhanced Generative Retrieval;** Yongqi Li et al
- **Meta-training with Demonstration Retrieval for Efficient Few-shot Learning;** Aaron Mueller et al
- **SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION;** Akari Asai et ak
- **RAP: Retrieval-Augmented Planning with Contextual Memory for Multimodal LLM Agents;** Tomoyuki Kagaya et al



## LLM with Human

- **CoAuthor: Designing a Human-AI Collaborative Writing Dataset for Exploring Language Model Capabilities;** Mina Lee et al
- **RewriteLM: An Instruction-Tuned Large Language Model for Text Rewriting;** Lei Shu et al
- **LeanDojo: Theorem Proving with Retrieval-Augmented Language Models;** Kaiyu Yang et al
- **Evaluating Human-Language Model Interaction;** Mina Lee et al

## LLM Foundation

- **Retentive Network: A Successor to Transformer for Large Language Models;** Yutao Sun et al
- **Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models;** Mayee F. Chen et al
- **Secrets of RLHF in Large Language Models Part I: PPO;** Rui Zheng et al
- **EduChat: A Large-Scale Language Model-based Chatbot System for Intelligent Education;** Yuhao Dan et al
- **WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct;** Haipeng Luo et al
- **SlimPajama-DC: Understanding Data Combinations for LLM Training;** Zhiqiang Shen et al
- **LMSYS-CHAT-1M: A LARGE-SCALE REAL-WORLD LLM CONVERSATION DATASET;** Lianmin Zheng et al
- **Mistral 7B;** Albert Q. Jiang et al
- **Tokenizer Choice For LLM Training: Negligible or Crucial?;** Mehdi Ali et al
- **ZEPHYR: DIRECT DISTILLATION OF LM ALIGNMENT;** Lewis Tunstall et al
- **LEMUR: HARMONIZING NATURAL LANGUAGE AND CODE FOR LANGUAGE AGENTS;** Yiheng Xu et al
- **System 2 Attention (is something you might need too);** Jason Weston et al
- **Camels in a Changing Climate: Enhancing LM Adaptation with TÜLU 2;** Hamish Ivison et al
- **The Falcon Series of Open Language Models;** Ebtesam Almazrouei et al
- **LLM360: Towards Fully Transparent Open-Source LLMs;** Zhengzhong Liu et al
- **OLMO: Accelerating the Science of Language Models;** Dirk Groeneveld et al
- **Executable Code Actions Elicit Better LLM Agents;** Xingyao Wang et al
- **InternLM-Math: Open Math Large Language Models Toward Verifiable Reasoning;** Huaiyuan Ying et al
- **Gemma: Open Models Based on Gemini Research and Technology;** Gemma Team
- **StarCoder 2 and The Stack v2: The Next Generation;** Anton Lozhkov et al
- **Yi: Open Foundation Models by 01.AI;** 01.AI 

## Scaling Law
- **SCALING LAWS FOR SPARSELY-CONNECTED FOUNDATION MODELS;** Elias Frantar et al
- **Language models scale reliably with over-training and on downstream tasks;** Samir Yitzhak Gadre et al
- **Unraveling the Mystery of Scaling Laws: Part I;** Hui Su et al
- **Understanding Emergent Abilities of Language Models from the Loss Perspective;** Zhengxiao Du et al


## LLM Data Engineering

- **Textbooks Are All You Need II: phi-1.5 technical report;** Yuanzhi Li et al
- **Orca: Progressive Learning from Complex Explanation Traces of GPT-4;** Subhabrata Mukherjee et al
- **Symbol-LLM: Towards Foundational Symbol-centric Interface For Large Language Models;** Fangzhi Xu et al
- **Orca 2: Teaching Small Language Models How to Reason;** Arindam Mitra et al
- **REST MEETS REACT: SELF-IMPROVEMENT FOR MULTI-STEP REASONING LLM AGENT;** Renat Aksitov et al
- **WHAT MAKES GOOD DATA FOR ALIGNMENT? A COMPREHENSIVE STUDY OF AUTOMATIC DATA SELECTION IN INSTRUCTION TUNING;** Wei Liu et al
- **ChatQA: Building GPT-4 Level Conversational QA Models;** Zihan Liu et al
- **AGENTOHANA: DESIGN UNIFIED DATA AND TRAINING PIPELINE FOR EFFECTIVE AGENT LEARNING;** Jianguo Zhang et al


## VLM Data Engineering
- **Multimodal ArXiv: A Dataset for Improving Scientific Comprehension of Large Vision-Language Models;** Lei Li et al



## Alignment

- **AI Alignment Research Overview;** Jacob Steinhardt 
- **Language Model Alignment with Elastic Reset;** Michael Noukhovitch et al
- **Alignment for Honesty;** Yuqing Yang et al
- **Align on the Fly: Adapting Chatbot Behavior to Established Norms;** Chunpu Xu et al
- **Combining weak-to-strong generalization with scalable oversight A high-level view on how this new approach fits into our alignment plans;** JAN LEIKE
- **SLEEPER AGENTS: TRAINING DECEPTIVE LLMS THAT PERSIST THROUGH SAFETY TRAINING;** Evan Hubinger et al
- **Towards Efficient and Exact Optimization of Language Model Alignment;** Haozhe Ji et al
- **Aligner: Achieving Efficient Alignment through Weak-to-Strong Correction;** Jiaming Ji et al
- **DeAL: Decoding-time Alignment for Large Language Models;** James Y. Huang et al
- **Step-On-Feet Tuning: Scaling Self-Alignment of LLMs via Bootstrapping;** Haoyu Wang et al
- **Dissecting Human and LLM Preferences;** Junlong Li1 et al
- **Reformatted Alignment;** Run-Ze Fan et al
- **Capability or Alignment? Respect the LLM Base Model’s Capability During Alignment;** Jingfeng Yang
- **Learning or Self-aligning? Rethinking Instruction Fine-tuning;** Mengjie Ren et al
- **Ensuring Safe and High-Quality Outputs: A Guideline Library Approach for Language Models;** Yi Luo et al


## Scalable Oversight & SuperAlignment

- **Supervising strong learners by amplifying weak experts;** Paul Christiano et al
- **Deep Reinforcement Learning from Human Preferences;** Paul F Christiano et al
- **AI safety via debate;** Geoffrey Irving et al
- **Scalable agent alignment via reward modeling: a research direction;** Jan Leike et al
- **Recursively Summarizing Books with Human Feedback;** Jeff Wu et al
- **Self-critiquing models for assisting human evaluators;** William Saunders et al
- **Measuring Progress on Scalable Oversight for Large Language Models;** Samuel R. Bowman et al
- **Debate Helps Supervise Unreliable Experts;** Julian Michael et al
- **WEAK-TO-STRONG GENERALIZATION: ELICITING STRONG CAPABILITIES WITH WEAK SUPERVISION;** Collin Burns et al
- **Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models;** Zixiang Chen et al
- **Discovering Language Model Behaviors with Model-Written Evaluations;** Ethan Perez et al
- **Towards Explainable Harmful Meme Detection through Multimodal Debate between Large Language Models;** Hongzhan Lin et al
- **Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate;** Tian Liang et al
- **Improving Weak-to-Strong Generalization with Scalable Oversight and Ensemble Learning;** Jitao Sang et al
- **PRD: Peer Rank and Discussion Improve Large Language Model based Evaluations;** Ruosen Li et al
- **Vision Superalignment: Weak-to-Strong Generalization for Vision Foundation Models;** Jianyuan Guo et al
- **Improving Factuality and Reasoning in Language Models through Multiagent Debate;** Yilun Du et al
- **CHATEVAL: TOWARDS BETTER LLM-BASED EVALUATORS THROUGH MULTI-AGENT DEBATE;** Chi-Min Chan et al
- **Debating with More Persuasive LLMs Leads to More Truthful Answers;** Akbir Khan et al
- **Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision;** Zhiqing Sun et al


## RL Foundation

- **PREFERENCES IMPLICIT IN THE STATE OF THE WORLD;** Rohin Shah et al
- **Hindsight Experience Replay;** Marcin Andrychowicz et al
- **Learning to Reach Goals via Iterated Supervised Learning;** Dibya Ghosh et al
- **The Wisdom of Hindsight Makes Language Models Better Instruction Followers;** Tianjun Zhang et al
- **REWARD UNCERTAINTY FOR EXPLORATION IN PREFERENCE-BASED REINFORCEMENT LEARNING;** Xinran Liang et al


## Beyond Bandit

- **ZERO-SHOT GOAL-DIRECTED DIALOGUE VIA RL ON IMAGINED CONVERSATIONS;** Joey Hong et al
- **LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language Models;** Marwa Abdulhai et al
- **ELICITING HUMAN PREFERENCES WITH LANGUAGE MODELS;** Belinda Z. Li et al
- **ITERATED DECOMPOSITION: IMPROVING SCIENCE Q&A BY SUPERVISING REASONING PROCESSES;** Justin Reppert et al
- **Let’s Verify Step by Step;** Hunter Lightman et al
- **Solving math word problems with process and outcome-based feedback;** Jonathan Uesato et al
- **EMPOWERING LANGUAGE MODELS WITH ACTIVE INQUIRY FOR DEEPER UNDERSTANDING A PREPRINT;** Jing-Cheng Pang et al
- **Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models;** Zhiyuan Hu et al
- **Tell Me More! Towards Implicit User Intention Understanding of Language Model-Driven Agents;** Cheng Qian et al
- **MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues;** Ge Bai et al
- **ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL;** Yifei Zhou et al
- **Probing the Multi-turn Planning Capabilities of LLMs via 20 Question Games;** Yizhe Zhang et al

## Agent

- **Generative Agents: Interactive Simulacra of Human Behavior;** Joon Sung Park et al
- **SWIFTSAGE: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks;** Bill Yuchen Lin et al
- **Large Language Model Is Semi-Parametric Reinforcement Learning Agent;** Danyang Zhang et al
- **The Role of Summarization in Generative Agents: A Preliminary Perspective;** Xiachong Feng et al
- **CAMEL: Communicative Agents for “Mind” Exploration of Large Scale Language Model Society;** Guohao Li et al
- **Plan, Eliminate, and Track-Language Models are Good Teachers for Embodied Agents;** Yue Wu et al
- **Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents;** Zihao Wang et al
- **Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory;** Xizhou Zhu et al
- **TOWARDS A UNIFIED AGENT WITH FOUNDATION MODELS;** Norman Di Palo et al
- **MotionLM: Multi-Agent Motion Forecasting as Language Modeling;** Ari Seff et al
- **A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis;** Izzeddin Gur et al
- **Guide Your Agent with Adaptive Multimodal Rewards;** Changyeon Kim et al
- **Generative Agents: Interactive Simulacra of Human Behavior;** Joon Sung Park et al
- **AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors in Agents;** Weize Chen et al
- **METAGPT: META PROGRAMMING FOR MULTI-AGENT COLLABORATIVE FRAMEWORK;** Sirui Hong et al
- **YOU ONLY LOOK AT SCREENS: MULTIMODAL CHAIN-OF-ACTION AGENTS;** Zhuosheng Zhang et al
- **SELF: LANGUAGE-DRIVEN SELF-EVOLUTION FOR LARGE LANGUAGE MODEL;** Jianqiao Lu et al
- **Towards End-to-End Embodied Decision Making via Multi-modal Large Language Model: Explorations with GPT4-Vision and Beyond;** Liang Chen et al
- **A Zero-Shot Language Agent for Computer Control with Structured Reflection;** Tao Li et al
- **Character-LLM: A Trainable Agent for Role-Playing;** Yunfan Shao et al
- **CLIN: A CONTINUALLY LEARNING LANGUAGE AGENT FOR RAPID TASK ADAPTATION AND GENERALIZATION;** Bodhisattwa Prasad Majumder et al
- **FIREACT: TOWARD LANGUAGE AGENT FINE-TUNING;** Baian Chen et al
- **TrainerAgent: Customizable and Efficient Model Training through LLM-Powered Multi-Agent System;** Haoyuan Li et al
- **LUMOS: LEARNING AGENTS WITH UNIFIED DATA, MODULAR DESIGN, AND OPEN-SOURCE LLMS;** Da Yin et al
- **TaskWeaver: A Code-First Agent Framework;** Bo Qiao et al
- **Pangu-Agent: A Fine-Tunable Generalist Agent with Structured Reasoning;** Filippos Christianos et al
- **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation;** Qingyun Wu et al
- **TRUE KNOWLEDGE COMES FROM PRACTICE: ALIGNING LLMS WITH EMBODIED ENVIRONMENTS VIA REINFORCEMENT LEARNING;** Weihao Tan et al
- **Investigate-Consolidate-Exploit: A General Strategy for Inter-Task Agent Self-Evolution;** Cheng Qian et al
- **OS-COPILOT: TOWARDS GENERALIST COMPUTER AGENTS WITH SELF-IMPROVEMENT;** Zhiyong Wu et al
- **LONGAGENT: Scaling Language Models to 128k Context through Multi-Agent Collaboration;** Jun Zhao et al
- **When is Tree Search Useful for LLM Planning? It Depends on the Discriminator;** Ziru Chen et al
- **DATA INTERPRETER: AN LLM AGENT FOR DATA SCIENCE;** Sirui Hong et al
- **Towards General Computer Control: A Multimodal Agent for Red Dead Redemption II as a Case Study;** Weihao Tan et al
- **SOTOPIA-π: Interactive Learning of Socially Intelligent Language Agents;** Ruiyi Wang et al
- **Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models;** Zehui Chen et al
- **LLM Agent Operating System;** Kai Mei et al


**AutoTelic Agent**

- **AUGMENTING AUTOTELIC AGENTS WITH LARGE LANGUAGE MODELS;** Cedric Colas et al
- **Visual Reinforcement Learning with Imagined Goals;** Ashvin Nair et al


**Evaluation**

- **AgentBench: Evaluating LLMs as Agents;** Xiao Liu et al
- **EVALUATING MULTI-AGENT COORDINATION ABILITIES IN LARGE LANGUAGE MODELS;** Saaket Agashe et al
- **OpenAgents: AN OPEN PLATFORM FOR LANGUAGE AGENTS IN THE WILD;** Tianbao Xie et al
- **SMARTPLAY : A BENCHMARK FOR LLMS AS INTELLIGENT AGENTS;** Yue Wu et al
- **WorkArena: How Capable are Web Agents at Solving Common Knowledge Work Tasks?;** Alexandre Drouin et al


**VL Related Task**

- **LANGNAV: LANGUAGE AS A PERCEPTUAL REPRESENTATION FOR NAVIGATION;** Bowen Pan et al
- **VIDEO LANGUAGE PLANNING;** Yilun Du et al
- **Fuyu-8B: A Multimodal Architecture for AI Agents;** Rohan Bavishi et al
- **GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation;** An Yan et al
- **Embodied Multi-Modal Agent trained by an LLM from a Parallel TextWorld;** Yijun Yang et al
- **STEVE: See and Think: Embodied Agent in Virtual Environment;** Zhonghan Zhao et al
- **JARVIS-1: Open-world Multi-task Agents with Memory-Augmented Multimodal Language Models;** Zihao Wang et al
- **STEVE-EYE: EQUIPPING LLM-BASED EMBODIED AGENTS WITH VISUAL PERCEPTION IN OPEN WORLDS;** Sipeng Zheng et al
- **OCTOPUS: EMBODIED VISION-LANGUAGE PROGRAMMER FROM ENVIRONMENTAL FEEDBACK;** Jingkang Yang et al
- **CogAgent: A Visual Language Model for GUI Agents;** Wenyi Hong et al
- **GPT-4V(ision) is a Generalist Web Agent, if Grounded;** Boyuan Zheng et al
- **WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models;** Hongliang He et al
- **MOBILE-AGENT: AUTONOMOUS MULTI-MODAL MOBILE DEVICE AGENT WITH VISUAL PERCEPTION;** Junyang Wang et al
- **V-IRL: Grounding Virtual Intelligence in Real Life;** Jihan Yang et al
- **An Interactive Agent Foundation Model;** Zane Durante et al
- **RoboCodeX: Multimodal Code Generation for Robotic Behavior Synthesis;** Yao Mu et al
- **Scaling Instructable Agents Across Many Simulated Worlds;** SIMA Team 

## Interaction

- **Imitating Interactive Intelligence;** Interactive Agents Group
- **Creating Multimodal Interactive Agents with Imitation and Self-Supervised Learning;** Interactive Agents Team
- **Evaluating Multimodal Interactive Agents;** Interactive Agents Team
- **Improving Multimodal Interactive Agents with Reinforcement Learning from Human Feedback;** Interactive Agents Team
- **LatEval: An Interactive LLMs Evaluation Benchmark with Incomplete Information from Lateral Thinking Puzzles;** Shulin Huang et al
- **BENCHMARKING LARGE LANGUAGE MODELS AS AI RESEARCH AGENTS;** Qian Huang et al
- **ADAPTING LLM AGENTS THROUGH COMMUNICATION;** Kuan Wang et al
- **PARROT: ENHANCING MULTI-TURN CHAT MODELS BY LEARNING TO ASK QUESTIONS;** Yuchong Sun et al
- **LLAMA RIDER: SPURRING LARGE LANGUAGE MODELS TO EXPLORE THE OPEN WORLD;** Yicheng Feng et al
- **AGENTTUNING: ENABLING GENERALIZED AGENT ABILITIES FOR LLMS;** Aohan Zeng et al
- **MINT: Evaluating LLMs in Multi-Turn Interaction with Tools and Language Feedback;** Xingyao Wang et al
- **LLF-Bench: Benchmark for Interactive Learning from Language Feedback;** Ching-An Cheng et al
- **MT-Eval: A Multi-Turn Capabilities Evaluation Benchmark for Large Language Models;** Wai-Chung Kwan et al
- **Can large language models explore in-context?;** Akshay Krishnamurthy et al

## Critique Modeling

- **Learning Evaluation Models from Large Language Models for Sequence Generation;** Chenglong Wang et al
- **RETROFORMER: RETROSPECTIVE LARGE LANGUAGE AGENTS WITH POLICY GRADIENT OPTIMIZATION;** Weiran Yao et al
- **Shepherd: A Critic for Language Model Generation;** Tianlu Wang et al
- **GENERATING SEQUENCES BY LEARNING TO [SELF-]CORRECT;** Sean Welleck et al
- **ZYN: Zero-Shot Reward Models with Yes-No Questions;** Victor Gallego et al
- **LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked;** Alec Helbling et al
- **RAIN: Your Language Models Can Align Themselves Yuhui Liwithout Finetuning;** Yuhui Li et al
- **SYNDICOM: Improving Conversational Commonsense with Error-Injection and Natural Language Feedback;** Christopher Richardson et al
- **LET’S REWARD STEP BY STEP: STEP-LEVEL REWARD MODEL AS THE NAVIGATORS FOR REASONING;** Qianli Ma et al
- **MAF: Multi-Aspect Feedback for Improving Reasoning in Large Language Models;** Deepak Nathani et al
- **DON’T THROW AWAY YOUR VALUE MODEL! MAKING PPO EVEN BETTER VIA VALUE-GUIDED MONTE-CARLO TREE SEARCH DECODING;** Jiacheng Liu et al
- **COFFEE: Boost Your Code LLMs by Fixing Bugs with Feedback;** Seungjun Moon et al
- **Cappy: Outperforming and Boosting Large Multi-Task LMs with a Small Scorer;** Bowen Tan et al
- **Pinpoint, Not Criticize: Refining Large Language Models via Fine-Grained Actionable Feedback;** Wenda Xu et al
- **Digital Socrates: Evaluating LLMs through explanation critiques;** Yuling Gu et al
- **Outcome-supervised Verifiers for Planning in Mathematical Reasoning;** Fei Yu et al
- **Data-Efficient Alignment of Large Language Models with Human Feedback Through Natural Language;** Di Jin et al
- **CRITIQUELLM: Scaling LLM-as-Critic for Effective and Explainable Evaluation of Large Language Model Generation;** Pei Ke et al
- **Mismatch Quest: Visual and Textual Feedback for Image-Text Misalignment;** Brian Gordon et al
- **MATH-SHEPHERD: A LABEL-FREE STEP-BY-STEP VERIFIER FOR LLMS IN MATHEMATICAL REASONING;** Peiyi Wang et al
- **The Critique of Critique;** Shichao Sun et al
- **LLMCRIT: Teaching Large Language Models to Use Criteria;** Weizhe Yuan et al



## MoE/Specialized

- **OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER;** Noam Shazeer et al
- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity;** William Fedus et al
- **DEMIX Layers: Disentangling Domains for Modular Language Modeling;** Suchin Gururangan et al
- **ModuleFormer: Learning Modular Large Language Models From Uncurated Data;** Yikang Shen et al
- **Mixture-of-Experts Meets Instruction Tuning: A Winning Combination for Large Language Models;** Sheng Shen et al
- **From Sparse to Soft Mixtures of Experts;** Joan Puigcerver et al
- **SELF-SPECIALIZATION: UNCOVERING LATENT EXPERTISE WITHIN LARGE LANGUAGE MODELS;** Junmo Kang et al
- **HOW ABILITIES IN LARGE LANGUAGE MODELS ARE AFFECTED BY SUPERVISED FINE-TUNING DATA COMPOSITION;** Guanting Dong et al
- **OPENWEBMATH: AN OPEN DATASET OF HIGH-QUALITY MATHEMATICAL WEB TEXT;** Keiran Paster et al
- **LLEMMA: AN OPEN LANGUAGE MODEL FOR MATHEMATICS;** Zhangir Azerbayev et al
- **Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models;** Keming Lu et al
- **Mixtral of Experts;** Albert Q. Jiang et al
- **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models;** Damai Dai et al
- **MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts;** Maciej Pioro et al
- **OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models;** Fuzhao Xue et al
- **Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM;** Sainbayar Sukhbaatar et al

## Vision-Language Foundation Model

### First Generation: Using region-based features; can be classified as one- and two- streams model architectures; Before 2020.6;

- **Multimodal Pretraining Unmasked: A Meta-Analysis and a Unified Framework of Vision-and-Language BERTs;** Emanuele Bugliarello et al; A meta-analysis of the first generation VL models and a unified framework.
- **Decoupling the Role of Data, Attention, and Losses in Multimodal Transformers;** Lisa Anne Hendricks et al
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
- **Tag2Text: Guiding Vision-Language Model via Image Tagging;** Xinyu Huang et al

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
- **PERCEIVER IO: A GENERAL ARCHITECTURE FOR STRUCTURED INPUTS & OUTPUTS;** Andrew Jaegle et al
- **X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages;** Feilong Chen et al

**Special designs tailored to enhance the position encoding & grounding.**

- **UniTAB: Unifying Text and Box Outputs for Grounded Vision-Language Modeling;** Zhengyuan Yang et al
- **PEVL: Position-enhanced Pre-training and Prompt Tuning for Vision-language Models;** Yuan Yao et al; Introduce explicit object position modeling.A woman < 310 mask 406 475 > is watching the mask < 175 86 254 460 >;
- **GLIPv2: Unifying Localization and VL Understanding;** Haotian Zhang et al; Further show that GLIP's pretraining method can benefit the VL task (Unifying localization and understanding).
- **DesCo: Learning Object Recognition with Rich Language Descriptions;** Liunian Harold Li et al

**Motivate to use unparalleled image & text data to build a unified model for VL, vision, and language tasks and potentially bring better performance.**

- **Uni-Perceiver: Pre-training Unified Architecture for Generic Perception for Zero-shot and Few-shot Tasks;** Xizhou Zhu et al; Siamese network to encode various modalities.
- **FLAVA: A Foundational Language And Vision Alignment Model;** Amanpreet Singh et al; A unified backbone model (need task-specific heads) for NLP, CV, and VL tasks.
- **UNIMO-2: End-to-End Unified Vision-Language Grounded Learning;** Wei Li et al; Design a new method "Grounded Dictionary Learning", similar to the sense of "continuous" image tags to align two modalities.

### Third Generation: Chasing for one unified/general/generalist model to include more VL/NLP/CV tasks; Becoming larger & Stronger; 2022->Now.

- **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation;** Junnan Li et al; New unified architecture and new method to generate and then filter captions.
- **OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework;** Peng Wang et al; A unified model (framework) to handle text, image, and image-text tasks.
- **Webly Supervised Concept Expansion for General Purpose Vision Models;** Amita Kamath et al
- **Language Models are General-Purpose Interfaces;** Yaru Hao et al
- **GIT: A Generative Image-to-text Transformer for Vision and Language;** Jianfeng Wang et al
- **CoCa: Contrastive Captioners are Image-Text Foundation Models;** Jiahui Yu et al
- **Flamingo: a Visual Language Model for Few-Shot Learning;** Jean-Baptiste Alayrac et al; Designed for few-shot learning.
- **Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks;** Wenhui Wang et al; BEIT-3.
- **OmniVL: One Foundation Model for Image-Language and Video-Language Tasks;** Junke Wang et al; Support both image-language and video-language tasks and show the positive transfer in three modalities.
- **Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks;** Hao Li et al; Propose a generalist model that can also handle object detection and instance segmentation tasks.
- **X2-VLM: All-In-One Pre-trained Model For Vision-Language Tasks;** Yan Zeng et al; Propose a unified model for image-language and video-text-language tasks; Modeling the fine-grained alignments between image regions and descriptions.
- **Toward Building General Foundation Models for Language, Vision, and Vision-Language Understanding Tasks;** Xinsong Zhang et al
- **mPLUG-2: A Modularized Multi-modal Foundation Model Across Text, Image and Video;** Haiyang Xu et al
- **KOSMOS-2: Grounding Multimodal Large Language Models to the World;** Zhiliang Peng et al
- **PaLI-X: On Scaling up a Multilingual Vision and Language Model;** Xi Chen et al
- **UNIFIED LANGUAGE-VISION PRETRAINING WITH DYNAMIC DISCRETE VISUAL TOKENIZATION;** Yang Jin et al
- **PALI-3 VISION LANGUAGE MODELS: SMALLER, FASTER, STRONGER;** Xi Chen et al

**Generalist models**

- **UNIFIED-IO: A UNIFIED MODEL FOR VISION, LANGUAGE, AND MULTI-MODAL TASKS;** Jiasen Lu et al; Examine whether a single unified model can solve a variety of tasks (NLP, CV, VL) simultaneously; Construct a massive multi-tasking dataset by ensembling 95 datasets from 62 publicly available data sources, including Image Synthesis, Keypoint Estimation, Depth Estimation, Object Segmentation, et al; Focusing on multi-task fine-tuning.
- **Generalized Decoding for Pixel, Image, and Language;** Xueyan Zou et al
- **Foundation Transformers;** Hongyu Wang et al; Propose a new unified architecture.
- **A Generalist Agent;** Scott Reed et al
- **PaLM-E: An Embodied Multimodal Language Model;** Danny Driess et al
- **IMAGEBIND: One Embedding Space To Bind Them All;** Rohit Girdhar et al

### Fourth Generation: Relying on LLMs and instruction tuning

- **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models;** Junnan Li et al
- **Grounding Language Models to Images for Multimodal Inputs and Outputs;** Jing Yu Koh et al
- **Language Is Not All You Need: Aligning Perception with Language Models;** Shaohan Huang et al
- **Otter: A Multi-Modal Model with In-Context Instruction Tuning;** Bo Li et al
- **Visual Instruction Tuning;** Haotian Liu et al
- **MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models;** Deyao Zhu et al
- **InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning;** Wenliang Dai et al
- **LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model;** Peng Gao et al
- **LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding;** Yanzhe Zhang et al
- **MultiModal-GPT: A Vision and Language Model for Dialogue with Humans;** Tao Gong et al
- **GPT-4 Technical Report;** OpenAI
- **mPLUG-Owl : Modularization Empowers Large Language Models with Multimodality;** Qinghao Ye et al
- **VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks;** Wenhai Wang et al
- **PandaGPT: One Model To Instruction-Follow Them All;** Yixuan Su et al
- **Generating Images with Multimodal Language Models;** Jing Yu Koh et al
- **What Matters in Training a GPT4-Style Language Model with Multimodal Inputs?;** Yan Zeng et al
- **GPT4RoI: Instruction Tuning Large Language Model on Region-of-Interest;** Shilong Zhang et al
- **Generative Pretraining in Multimodality;** Quan Sun et al
- **Planting a SEED of Vision in Large Language Model;** Yuying Ge et al
- **ChatSpot: Bootstrapping Multimodal LLMs via Precise Referring Instruction Tuning;** Liang Zhao et al
- **Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning;** Lili Yu et al
- **The All-Seeing Project: Towards Panoptic Visual Recognition and Understanding of the Open World;** Weiyun Wang et al
- **EMPOWERING VISION-LANGUAGE MODELS TO FOLLOW INTERLEAVED VISION-LANGUAGE INSTRUCTIONS;** Juncheng Li et al
- **RegionBLIP: A Unified Multi-modal Pre-training Framework for Holistic and Regional Comprehension;** Qiang Zhou et al
- **LISA: REASONING SEGMENTATION VIA LARGE LANGUAGE MODEL;** Xin Lai et al
- **Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities;** Jinze Bai et al
- **InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4;** Lai Wei et al
- **StableLLaVA: Enhanced Visual Instruction Tuning with Synthesized Image-Dialogue Data;** Yanda Li et al
- **Large Multilingual Models Pivot Zero-Shot Multimodal Learning across Languages;** Jinyi Hu et al
- **MMICL: EMPOWERING VISION-LANGUAGE MODEL WITH MULTI-MODAL IN-CONTEXT LEARNING;** Haozhe Zhao et al
- **An Empirical Study of Scaling Instruction-Tuned Large Multimodal Models;** Yadong Lu et al
- **ALIGNING LARGE MULTIMODAL MODELS WITH FACTUALLY AUGMENTED RLHF;** Zhiqing Sun et al
- **Reformulating Vision-Language Foundation Models and Datasets Towards Universal Multimodal Assistants;** Tianyu Yu et al
- **AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model;** Seungwhan Moon et al
- **InternLM-XComposer: A Vision-Language Large Model for Advanced Text-image Comprehension and Composition;** Pan Zhang et al
- **DREAMLLM: SYNERGISTIC MULTIMODAL COMPREHENSION AND CREATION;** Runpei Dong et al
- **HALLE-SWITCH: RETHINKING AND CONTROLLING OBJECT EXISTENCE HALLUCINATIONS IN LARGE VISION LANGUAGE MODELS FOR DETAILED CAPTION;** Bohan Zhai et al
- **Improved Baselines with Visual Instruction Tuning;** Haotian Liu et al
- **Fuyu-8B: A Multimodal Architecture for AI Agents;** Rohan Bavishi et al
- **MINIGPT-5: INTERLEAVED VISION-AND-LANGUAGE GENERATION VIA GENERATIVE VOKENS;** Kaizhi Zheng et al
- **Making LLaMA SEE and Draw with SEED Tokenizer;** Yuying Ge et al
- **To See is to Believe: Prompting GPT-4V for Better Visual Instruction Tuning;** Junke Wang et al
- **TEAL: TOKENIZE AND EMBED ALL FOR MULTI-MODAL LARGE LANGUAGE MODELS;** Zhen Yang et al
- **mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration;** Qinghao Ye et al
- **LION: Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge;** Gongwei Chen et al
- **OtterHD: A High-Resolution Multi-modality Model;** Bo Li et al
- **ShareGPT4V: Improving Large Multi-Modal Models with Better Captions;** Lin Chen et al
- **PerceptionGPT: Effectively Fusing Visual Perception into LLM;** Renjie Pi et al
- **OCTAVIUS: MITIGATING TASK INTERFERENCE IN MLLMS VIA MOE;** Zeren Chen et al
- **COGVLM: VISUAL EXPERT FOR LARGE LANGUAGE MODELS;** Weihan Wang et al
- **Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models;** Zhang Li et al
- **Omni-SMoLA: Boosting Generalist Multimodal Models with Soft Mixture of Low-rank Experts;** Jialin Wu et al
- **SILKIE: PREFERENCE DISTILLATION FOR LARGE VISUAL LANGUAGE MODELS;** Lei Li et al
- **GLaMM: Pixel Grounding Large Multimodal Model;** Hanoona Rasheed et al
- **TEXTBIND: MULTI-TURN INTERLEAVED MULTIMODAL INSTRUCTION-FOLLOWING IN THE WILD;** Huayang Li et al
- **DRESS: Instructing Large Vision-Language Models to Align and Interact with Humans via Natural Language Feedback;** Yangyi Chen et al
- **Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding;** Peng Jin et al
- **LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models;** Hao Zhang et al
- **mPLUG-PaperOwl: Scientific Diagram Analysis with the Multimodal Large Language Model;** Anwen Hu et al
- **Q-Instruct: Improving Low-level Visual Abilities for Multi-modality Foundation Models;** Haoning Wu et al
- **SPHINX: THE JOINT MIXING OF WEIGHTS, TASKS, AND VISUAL EMBEDDINGS FOR MULTI-MODAL LARGE LANGUAGE MODELS;** Ziyi Lin et al
- **DeepSpeed-VisualChat: Multi Round Multi Images Interleave Chat via Multi-Modal Casual Attention;** Zhewei Yao et al
- **Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models;** Haoran Wei et al
- **Osprey: Pixel Understanding with Visual Instruction Tuning;** Yuqian Yuan et al
- **Generative Multimodal Models are In-Context Learners;** Quan Sun et al
- **Gemini: A Family of Highly Capable Multimodal Models;** Gemini Team
- **CaMML: Context-Aware Multimodal Learner for Large Models;** Yixin Chen et al
- **MM-Interleaved: Interleaved Image-Text Generative Modeling via Multi-modal Feature Synchronizer;** Changyao Tian et al
- **InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Models;** Xiaoyi Dong et al
- **MoE-LLaVA: Mixture of Experts for Large Vision-Language Models;** Bin Lin et al
- **MouSi: Poly-Visual-Expert Vision-Language Models;** Xiaoran Fan et al
- **SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models;** Peng Gao et al
- **Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models;** Gen Luo et al
- **DeepSeek-VL: Towards Real-World Vision-Language Understanding;** Haoyu Lu et al
- **UniCode: Learning a Unified Codebook for Multimodal Large Language Models;** Sipeng Zheng et al
- **MoAI: Mixture of All Intelligence for Large Language and Vision Models;** Byung-Kwan Lee et al
- **LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution Images;** Ruyi Xu et al


### Others

- **Unified Vision-Language Pre-Training for Image Captioning and VQA;** Luowei Zhou et al
- **Unifying Vision-and-Language Tasks via Text Generation;** Jaemin Cho et al
- **MERLOT Reserve: Neural Script Knowledge through Vision and Language and Sound;** Rowan Zellers et al
- **CLIP-Event: Connecting Text and Images with Event Structures;** Manling Li et al; The new model CLIP-Event, specifically designed for multi-modal event extraction. Introducing new pretraining tasks to enable strong zero-shot performances. From object-centric representations to event-centric representations.
- **Scaling Vision-Language Models with Sparse Mixture of Experts;** Sheng Shen et al
- **MaMMUT: A Simple Architecture for Joint Learning for MultiModal Tasks;** Weicheng Kuo et al
- **Vision-Flan: Scaling Human-Labeled Tasks in Visual Instruction Tuning;** Zhiyang Xu et al

## Vision-Language Model Application

- **VISION-LANGUAGE FOUNDATION MODELS AS EFFECTIVE ROBOT IMITATORS;** Xinghang Li et al
- **LLaVA-Interactive: An All-in-One Demo for Image Chat, Segmentation, Generation and Editing;** Wei-Ge Chen et al
- **Vision-Language Models as a Source of Rewards;** Kate Baumli et al
- **SELF-IMAGINE: EFFECTIVE UNIMODAL REASONING WITH MULTIMODAL MODELS USING SELF-IMAGINATION;** Syeda Nahida Akter et al
- **Code as Reward: Empowering Reinforcement Learning with VLMs;** David Venuto et al
- **MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark;** Dongping Chen et al
- **PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs;** Soroush Nasiriany et al

## Vision-Language Model Analysis & Evaluation

- **What Makes for Good Visual Tokenizers for Large Language Models?;** Guangzhi Wang et al
- **LVLM-eHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models;** Peng Xu et al
- **MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models;** Chaoyou Fu et al
- **JourneyDB: A Benchmark for Generative Image Understanding;** Junting Pan et al
- **MMBench: Is Your Multi-modal Model an All-around Player?;** Yuan Liu et al
- **SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension;** Bohao Li et al
- **Tiny LVLM-eHub: Early Multimodal Experiments with Bard;** Wenqi Shao et al
- **MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities;** Weihao Yu et al
- **VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use;** Yonatan Bitton et al
- **TouchStone: Evaluating Vision-Language Models by Language Models;** Shuai Bai et al
- **Investigating the Catastrophic Forgetting in Multimodal Large Language Models;** Yuexiang Zhai et al
- **DEMYSTIFYING CLIP DATA;** Hu Xu et al
- **Measuring and Improving Chain-of-Thought Reasoning in Vision-Language Models;** Yangyi Chen et al
- **REFORM-EVAL: EVALUATING LARGE VISION LANGUAGE MODELS VIA UNIFIED RE-FORMULATION OF TASK-ORIENTED BENCHMARKS;** Zejun Li1 et al
- **REVO-LION: EVALUATING AND REFINING VISION LANGUAGE INSTRUCTION TUNING DATASETS;** Ning Liao et al
- **BEYOND TASK PERFORMANCE: EVALUATING AND REDUCING THE FLAWS OF LARGE MULTIMODAL MODELS WITH IN-CONTEXT LEARNING;** Mustafa Shukor et al
- **Grounded Intuition of GPT-Vision’s Abilities with Scientific Images;** Alyssa Hwang et al
- **Holistic Evaluation of Text-to-Image Models;** Tony Lee et al
- **CORE-MM: COMPLEX OPEN-ENDED REASONING EVALUATION FOR MULTI-MODAL LARGE LANGUAGE MODELS;** Xiaotian Han et al
- **HALLUSIONBENCH: An Advanced Diagnostic Suite for Entangled Language Hallucination & Visual Illusion in Large Vision-Language Models;** Tianrui Guan et al
- **SEED-Bench-2: Benchmarking Multimodal Large Language Models;** Bohao Li et al
- **MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI;** Xiang Yue et al
- **MATHVISTA: EVALUATING MATH REASONING IN VISUAL CONTEXTS WITH GPT-4V, BARD, AND OTHER LARGE MULTIMODAL MODELS;** Pan Lu et al
- **VILA: On Pre-training for Visual Language Models;** Ji Lin et al
- **TUNING LAYERNORM IN ATTENTION: TOWARDS EFFICIENT MULTI-MODAL LLM FINETUNING;** Bingchen Zhao et al
- **Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs;** Shengbang Tong et al
- **FROZEN TRANSFORMERS IN LANGUAGE MODELS ARE EFFECTIVE VISUAL ENCODER LAYERS;** Ziqi Pang et al
- **Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models;** Siddharth Karamcheti et al
- **Design2Code: How Far Are We From Automating Front-End Engineering?;** Chenglei Si et al
- **MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training;** Brandon McKinzie et al
- **MATHVERSE: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?;** Renrui Zhang et al


## Multimodal Foundation Model

- **MotionGPT: Human Motion as a Foreign Language;** Biao Jiang et al
- **Meta-Transformer: A Unified Framework for Multimodal Learning;** Yiyuan Zhang et al
- **3D-LLM: Injecting the 3D World into Large Language Models;** Yining Hong et al
- **BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs;** Yang Zhao et al
- **VIT-LENS: Towards Omni-modal Representations;** Weixian Lei et al
- **LLASM: LARGE LANGUAGE AND SPEECH MODEL;** Yu Shu et al
- **Point-Bind & Point-LLM: Aligning Point Cloud with Multi-modality for 3D Understanding, Generation, and Instruction Following;** Ziyu Guo et al
- **NExT-GPT: Any-to-Any Multimodal LLM;** Shengqiong Wu et al
- **ImageBind-LLM: Multi-modality Instruction Tuning;** Jiaming Han et al
- **LAURAGPT: LISTEN, ATTEND, UNDERSTAND, AND REGENERATE AUDIO WITH GPT;** Jiaming Wang et al
- **AN EMBODIED GENERALIST AGENT IN 3D WORLD;** Jiangyong Huang et al
- **VIT-LENS-2: Gateway to Omni-modal Intelligence;** Weixian Lei et al
- **CoDi-2: In-Context, Interleaved, and Interactive Any-to-Any Generation;** Zineng Tang et al
- **X-InstructBLIP: A Framework for aligning X-Modal instruction-aware representations to LLMs and Emergent Cross-modal Reasoning;** Artemis Panagopoulou et al
- **Merlin: Empowering Multimodal LLMs with Foresight Minds;** En Yu et al
- **OneLLM: One Framework to Align All Modalities with Language;** Jiaming Han et al
- **Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action;** Jiasen Lu et al
- **WORLD MODEL ON MILLION-LENGTH VIDEO AND LANGUAGE WITH RINGATTENTION;** Hao Liu et al
- **LLMBind: A Unified Modality-Task Integration Framework;** Bin Zhu et al

## Image Generation

- **Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors;** Oran Gafni et al
- **Modeling Image Composition for Complex Scene Generation;** Zuopeng Yang et al
- **Frido: Feature Pyramid Diffusion for Complex Scene Image Synthesis;** Wan-Cyuan Fan et al
- **ReCo: Region-Controlled Text-to-Image Generation;** Zhengyuan Yang et al
- **UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild;** Can Qin et al
- **Going Beyond Nouns With Vision & Language Models Using Synthetic Data;** Paola Cascante-Bonilla et al
- **GUIDING INSTRUCTION-BASED IMAGE EDITING VIA MULTIMODAL LARGE LANGUAGE MODELS;** Tsu-Jui Fu et al
- **KOSMOS-G: Generating Images in Context with Multimodal Large Language Models;** Xichen Pan et al
- **DiagrammerGPT: Generating Open-Domain, Open-Platform Diagrams via LLM Planning;** Abhay Zala et al
- **LLMGA: Multimodal Large Language Model based Generation Assistant;** Bin Xia et al
- **ChatIllusion: Efficient-Aligning Interleaved Generation ability with Visual Instruction Model;** Xiaowei Chi et al
- **Gen4Gen: Generative Data Pipeline for Generative Multi-Concept Composition;** Chun-Hsiao Yeh et al





## Document Understanding

- **LayoutLM: Pre-training of Text and Layout for Document Image Understanding;** Yiheng Xu et al
- **LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding;** Yang Xu et al
- **LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking;** Yupan Huang et al
- **StrucTexT: Structured Text Understanding with Multi-Modal Transformers;** Yulin Li et al
- **LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding;** Jiapeng Wang et al
- **PIX2STRUCT: SCREENSHOT PARSING AS PRETRAINING FOR VISUAL LANGUAGE UNDERSTANDING;** Kenton Lee et al
- **Unifying Vision, Text, and Layout for Universal Document Processing;** Zineng Tang et al
- **STRUCTEXTV2: MASKED VISUAL-TEXTUAL PREDIC- TION FOR DOCUMENT IMAGE PRE-TRAINING;** Yuechen Yu et al
- **UniChart: A Universal Vision-language Pretrained Model for Chart Comprehension and Reasoning;** Ahmed Masry et al
- **Cream: Visually-Situated Natural Language Understanding with Contrastive Reading Model and Frozen Large Language Models;** Geewook Kim et al
- **LayoutMask: Enhance Text-Layout Interaction in Multi-modal Pre-training for Document Understanding;** Yi Tu et al
- **mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding;** Jiabo Ye et al
- **KOSMOS-2.5: A Multimodal Literate Model;** Tengchao Lv et al
- **STRUCTCHART: PERCEPTION, STRUCTURING, REASONING FOR VISUAL CHART UNDERSTANDING;** Renqiu Xia et al
- **UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model;** Jiabo Ye et al
- **MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning;** Fuxiao Liu et al
- **ChartLlama: A Multimodal LLM for Chart Understanding and Generation;** Yucheng Han et al
- **G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model;** Jiahui Gao et al
- **ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning;** Fanqing Meng et al
- **ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning;** Renqiu Xia et al
- **Enhancing Vision-Language Pre-training with Rich Supervisions;** Yuan Gao et al
- **TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document;** Yuliang Liu et al
- **ChartInstruct: Instruction Tuning for Chart Comprehension and Reasoning;** Ahmed Masry et al
- **Towards Improving Document Understanding: An Exploration on Text-Grounding via MLLMs;** Yonghui Wang et al
- **Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs;** Victor Carbune et al

**Dataset**

- **A Diagram Is Worth A Dozen Images;** Aniruddha Kembhavi et al
- **ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning;** Ahmed Masry et al
- **PDF-VQA: A New Dataset for Real-World VQA on PDF Documents;** Yihao Ding et al
- **DocumentNet: Bridging the Data Gap in Document Pre-Training;** Lijun Yu et al
- **Do LVLMs Understand Charts? Analyzing and Correcting Factual Errors in Chart Captioning;** Kung-Hsiang Huang et al


***Table***

- **Visual Understanding of Complex Table Structures from Document Images;** Sachin Raja et al
- **Improving Table Structure Recognition with Visual-Alignment Sequential Coordinate Modeling;** Yongshuai Huang et al
- **Table-GPT: Table-tuned GPT for Diverse Table Tasks;** Peng Li et al

## Tool Learning

**NLP**

- **TALM: Tool Augmented Language Models;** Aaron Paris et al
- **WebGPT: Browser-assisted question-answering with human feedback;** Reiichiro Nakano et al
- **LaMDA: Language Models for Dialog Applications;** Romal Thoppilan et al
- **BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage;** Kurt Shuster et al
- **PAL: program-aided language models;** Luyu Gao et al
- **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks;** Wenhu Chen et al
- **A neural network solves, explains, and generates university math problems by program synthesis and few-shot learning at human level;** Iddo Droria et al
- **React: synergizing reasoning and acting in language models;** Shunyu Yao et al
- **MIND’S EYE: GROUNDED LANGUAGE MODEL REASONING THROUGH SIMULATION;** Ruibo Liu et al
- **Toolformer: Language Models Can Teach Themselves to Use Tools;** Timo Schick et al
- **Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback;** Baolin Peng et al
- **ART: Automatic multi-step reasoning and tool-use for large language models;** Bhargavi Paranjape et al
- **Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models;** Pan Lu et al
- **AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head;** Rongjie Huang et al
- **Augmented Large Language Models with Parametric Knowledge Guiding;** Ziyang Luo et al
- **COOK: Empowering General-Purpose Language Models with Modular and Collaborative Knowledge;** Shangbin Feng et al
- **StructGPT: A General Framework for Large Language Model to Reason over Structured Data;** Jinhao Jiang et al
- **Chain of Knowledge: A Framework for Grounding Large Language Models with Structured Knowledge Bases;** Xingxuan Li et al
- **CREATOR: Disentangling Abstract and Concrete Reasonings of Large Language Models through Tool Creation;** Cheng Qian et al
- **ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases;** Qiaoyu Tang et al
- **WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences;** Xiao Liu et al
- **RestGPT: Connecting Large Language Models with Real-World Applications via RESTful APIs;** Yifan Song et al
- **MIND2WEB: Towards a Generalist Agent for the Web;** Xiang Deng et al
- **Certified Reasoning with Language Models;** Gabriel Poesia et al
- **ToolQA: A Dataset for LLM Question Answering with External Tools;** Yuchen Zhuang et al
- **On the Tool Manipulation Capability of Open-source Large Language Models;** Qiantong Xu, Fenglu Hong, Bo Li, Changran Hu, Zhengyu Chen, Jian Zhang et al
- **CHATDB: AUGMENTING LLMS WITH DATABASES AS THEIR SYMBOLIC MEMORY;** Chenxu Hu et al
- **MultiTool-CoT: GPT-3 Can Use Multiple External Tools with Chain of Thought Prompting;** Tatsuro Inaba et al
- **Making Language Models Better Tool Learners with Execution Feedback;** Shuofei Qiao et al
- **CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing;** Zhibin Gou et al
- **ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models;** Zhipeng Chen et al
- **Fact-Checking Complex Claims with Program-Guided Reasoning;** Liangming Pan et al
- **Gorilla: Large Language Model Connected with Massive APIs;** Shishir G. Patil et al
- **ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings;** Shibo Hao et al
- **Large Language Models as Tool Makers;** Tianle Cai et al
- **VOYAGER: An Open-Ended Embodied Agent with Large Language Models;** Guanzhi Wang et al
- **FACTOOL: Factuality Detection in Generative AI A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios;** I-Chun Chern et al
- **WebArena: A Realistic Web Environment for Building Autonomous Agents;** Shuyan Zhou et al
- **TOOLLLM: FACILITATING LARGE LANGUAGE MODELS TO MASTER 16000+ REAL-WORLD APIS;** Yujia Qin et al
- **Tool Documentation Enables Zero-Shot Tool-Usage with Large Language Models;** Cheng-Yu Hsieh et al
- **ExpeL: LLM Agents Are Experiential Learners;** Andrew Zhao et al
- **Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum;** Shen Gao et al
- **Self-driven Grounding: Large Language Model Agents with Automatical Language-aligned Skill Learning;** Shaohui Peng et al
- **Identifying the Risks of LM Agents with an LM-Emulated Sandbox;** Yangjun Ruan et al
- **TORA: A TOOL-INTEGRATED REASONING AGENT FOR MATHEMATICAL PROBLEM SOLVING;** Zhibin Gou et al
- **CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets;** Lifan Yuan et al
- **METATOOL BENCHMARK: DECIDING WHETHER TO USE TOOLS AND WHICH TO USE;** Yue Huang et al
- **A Comprehensive Evaluation of Tool-Assisted Generation Strategies;** Alon Jacovi et al
- **TPTU-v2: Boosting Task Planning and Tool Usage of Large Language Model-based Agents in Real-world Systems;** Yilun Kong et al
- **GITAGENT: FACILITATING AUTONOMOUS AGENT WITH GITHUB BY TOOL EXTENSION;** Bohan Lyu et al
- **TROVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks;** Zhiruo Wang et al
- **Towards Uncertainty-Aware Language Agent;** Jiuzhou Han et al
- **Tool-LMM: A Large Multi-Modal Model for Tool Agent Learning;** Chenyu Wang et al
- **Skill Set Optimization: Reinforcing Language Model Behavior via Transferable Skills;** Kolby Nottingham et al
- **AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API Calls;** Yu Du et al
- **SCIAGENT: Tool-augmented Language Models for Scientific Reasoning;** Yubo Ma et al
- **API-BLEND: A Comprehensive Corpora for Training and Benchmarking API LLMs;** Kinjal Basu et al
- **Empowering Large Language Model Agents through Action Learning;** Haiteng Zhao et al
- **LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error;** Boshi Wang et al
- **StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models;** Zhicheng Guo et al


**With Visual Tools**

- **Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models;** Chenfei Wu et al
- **ChatGPT Asks, BLIP-2 Answers: Automatic Questioning Towards Enriched Visual Descriptions;** Deyao Zhu et al
- **Video ChatCaptioner: Towards Enriched Spatiotemporal Descriptions;** Jun Chen et al
- **Visual Programming: Compositional visual reasoning without training;** Tanmay Gupta et al
- **ViperGPT: Visual Inference via Python Execution for Reasoning;** Dídac Surís et al
- **Chat with the Environment: Interactive Multimodal Perception using Large Language Models;** Xufeng Zhao et al
- **MM-REACT : Prompting ChatGPT for Multimodal Reasoning and Action;** Zhengyuan Yang et al
- **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace;** Yongliang Shen et al
- **TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs;** Yaobo Liang et al
- **OpenAGI: When LLM Meets Domain Experts;** Yingqiang Ge et al; Benchmark.
- **Inner Monologue: Embodied Reasoning through Planning with Language Models;** Wenlong Huang et al
- **Caption Anything: Interactive Image Description with Diverse Multimodal Controls;** Teng Wang et al
- **InternChat: Solving Vision-Centric Tasks by Interacting with Chatbots Beyond Language;** Zhaoyang Liu et al
- **Modular Visual Question Answering via Code Generation;** Sanjay Subramanian et al
- **Towards Language Models That Can See: Computer Vision Through the LENS of Natural Language;** William Berrios et al
- **AVIS: Autonomous Visual Information Seeking with Large Language Models;** Ziniu Hu et al
- **AssistGPT: A General Multi-modal Assistant that can Plan, Execute, Inspect, and Learn;** Difei Gao et al
- **GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction;** Rui Yang et al
- **LLM-Grounder: Open-Vocabulary 3D Visual Grounding with Large Language Model as an Agent;** Jianing Yang et al
- **Idea2Img: Iterative Self-Refinement with GPT-4V(ision) for Automatic Image Design and Generation;** Zhengyuan Yang et al
- **ControlLLM: Augment Language Models with Tools by Searching on Graphs;** Zhaoyang Liu et al
- **MM-VID: Advancing Video Understanding with GPT-4V(ision);** Kevin Lin et al
- **Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models;** Yushi Hu et al
- **CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations;** Ji Qi et al
- **CLOVA: a closed-loop visual assistant with tool usage and update;** Zhi Gao et al
- **m&m’s: A Benchmark to Evaluate Tool-Use for multi-step multi-modal Tasks;** Zixian Ma et al


## Instruction Tuning

- **Cross-Task Generalization via Natural Language Crowdsourcing Instructions;** Swaroop Mishra et al
- **FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS;** Jason Wei et al
- **MULTITASK PROMPTED TRAINING ENABLES ZERO-SHOT TASK GENERALIZATION;** Victor Sanh et al
- **Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks;** Yizhong Wang et al
- **Learning Instructions with Unlabeled Data for Zero-Shot Cross-Task Generalization;** Yuxian Gu et al
- **Scaling Instruction-Finetuned Language Models;** Hyung Won Chung et al
- **Task-aware Retrieval with Instructions;** Akari Asai et al
- **One Embedder, Any Task: Instruction-Finetuned Text Embeddings;** Hongjin Su et al
- **Boosting Natural Language Generation from Instructions with Meta-Learning;** Budhaditya Deb et al
- **Exploring the Benefits of Training Expert Language Models over Instruction Tuning;** Joel Jang et al
- **OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization;** Srinivasan Iyer et al
- **Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor;** Or Honovich et al
- **WeaQA: Weak Supervision via Captions for Visual Question Answering;** Pratyay Banerjee et al
- **MULTIINSTRUCT: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning;** Zhiyang Xu et al
- **SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions;** Yizhong Wang et al
- **Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases;** Yunjie Ji et al
- **INSTRUCTION TUNING WITH GPT-4;** Baolin Peng et al
- **The Flan Collection: Designing Data and Methods for Effective Instruction Tuning;** Shayne Longpre et al
- **LongForm: Optimizing Instruction Tuning for Long Text Generation with Corpus Extraction;** Abdullatif Köksal et al
- **GUESS THE INSTRUCTION! FLIPPED LEARNING MAKES LANGUAGE MODELS STRONGER ZERO-SHOT LEARNERS;** Seonghyeon Ye et al
- **In-Context Instruction Learning;** Seonghyeon Ye et al
- **WizardLM: Empowering Large Language Models to Follow Complex Instructions;** Can Xu et al
- **Controlled Text Generation with Natural Language Instructions;** Wangchunshu Zhou et al
- **Poisoning Language Models During Instruction Tuning;** Alexander Wan et al
- **Improving Cross-Task Generalization with Step-by-Step Instructions;** Yang Wu et al
- **VideoChat: Chat-Centric Video Understanding;** KunChang Li et al
- **SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities;** Dong Zhang et al
- **Prompting with Pseudo-Code Instructions;** Mayank Mishra et al
- **LIMA: Less Is More for Alignment;** Chunting Zhou et al
- **ExpertPrompting: Instructing Large Language Models to be Distinguished Experts;** Benfeng Xu et al
- **HINT: Hypernetwork Instruction Tuning for Efficient Zero- & Few-Shot Generalisation;** Hamish Ivison et al
- **Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models;** Gen Luo et al
- **SAIL: Search-Augmented Instruction Learning;** Hongyin Luo et al
- **Did You Read the Instructions? Rethinking the Effectiveness of Task Definitions in Instruction Learning;** Fan Yin et al
- **DYNOSAUR: A Dynamic Growth Paradigm for Instruction-Tuning Data Curation;** Da Yin et al
- **MACAW-LLM: MULTI-MODAL LANGUAGE MODELING WITH IMAGE, AUDIO, VIDEO, AND TEXT INTEGRATION;** Chenyang Lyu et al
- **How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources;** Yizhong Wang et al
- **INSTRUCTEVAL: Towards Holistic Evaluation of Instruction-Tuned Large Language Models;** Yew Ken Chia et al
- **MIMIC-IT: Multi-Modal In-Context Instruction Tuning;** Bo Li et al
- **Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning;** Fuxiao Liu et al
- **M3IT: A Large-Scale Dataset towards Multi-Modal Multilingual Instruction Tuning;** Lei Li et al
- **InstructEval: Systematic Evaluation of Instruction Selection Methods;** Anirudh Ajith et al
- **LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark;** Zhenfei Yin et al
- **Instruction Mining: High-Quality Instruction Data Selection for Large Language Models;** Yihan Cao et al
- **ALPAGASUS: TRAINING A BETTER ALPACA WITH FEWER DATA;** Lichang Chen et al
- **Exploring Format Consistency for Instruction Tuning;** Shihao Liang et al
- **Self-Alignment with Instruction Backtranslation;** Xian Li et al
- **#INSTAG: INSTRUCTION TAGGING FOR DIVERSITY AND COMPLEXITY ANALYSIS;** Keming Lu et al
- **CITING: LARGE LANGUAGE MODELS CREATE CURRICULUM FOR INSTRUCTION TUNING;** Tao Feng et al
- **Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models;** Haoran Li et al

## Incontext Learning

- **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?;** Sewon Min et al
- **Extrapolating to Unnatural Language Processing with GPT-3's In-context Learning: The Good, the Bad, and the Mysterious;** Frieda Rong et al
- **Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning;** Haokun Liu et al
- **Learning To Retrieve Prompts for In-Context Learning;** Ohad Rubin et al
- **An Explanation of In-context Learning as Implicit Bayesian Inference;** Sang Michael Xie, Aditi Raghunathan, Percy Liang, Tengyu Ma
- **MetaICL: Learning to Learn In Context;** Sewon Min et al
- **PROMPTING GPT-3 TO BE RELIABLE;** Chenglei Si et al
- **Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm;** Laria Reynolds et al
- **Do Prompt-Based Models Really Understand the Meaning of their Prompts?;** Albert Webson et al
- **On the Relation between Sensitivity and Accuracy in In-context Learning;** Yanda Chen et al
- **Meta-learning via Language Model In-context Tuning;** Yanda Chen et al
- **Extrapolating to Unnatural Language Processing with GPT-3's In-context Learning: The Good, the Bad, and the Mysterious;** Frieda Rong
- **SELECTIVE ANNOTATION MAKES LANGUAGE MODELS BETTER FEW-SHOT LEARNERS;** Hongjin Su et al
- **Robustness of Demonstration-based Learning Under Limited Data Scenario;** Hongxin Zhang et al; Demonstration-based learning, tuning the parameters.
- **Active Example Selection for In-Context Learning;** Yiming Zhang et al
- **Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity;** Yao Lu et al
- **Calibrate Before Use: Improving Few-Shot Performance of Language Models;** Tony Z. Zhao et al
- **DIALOGIC: Controllable Dialogue Simulation with In-Context Learning;** Zekun Li et al
- **PRESERVING IN-CONTEXT LEARNING ABILITY IN LARGE LANGUAGE MODEL FINE-TUNING;** Yihan Wang et al
- **Teaching Algorithmic Reasoning via In-context Learning;** Hattie Zhou et al
- **On the Compositional Generalization Gap of In-Context Learning** Arian Hosseini et al
- **Transformers generalize differently from information stored in context vs weights;** Stephanie C.Y. Chan et al
- **OVERTHINKING THE TRUTH: UNDERSTANDING HOW LANGUAGE MODELS PROCESS FALSE DEMONSTRATIONS;** Anonymous
- **In-context Learning and Induction Heads;** Catherine Olsson et al
- **Complementary Explanations for Effective In-Context Learning;** Xi Ye et al
- **What is Not in the Context? Evaluation of Few-shot Learners with Informative Demonstrations;** Michal Štefánik et al
- **Robustness of Learning from Task Instructions;** Jiasheng Gu et al
- **Structured Prompting: Scaling In-Context Learning to 1,000 Examples;** Yaru Hao et al
- **Transformers learn in-context by gradient descent;** Johannes von Oswald et al
- **Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale;** Hritik Bansal et al
- **Z-ICL: Zero-Shot In-Context Learning with Pseudo-Demonstrations;** Xinxi Lyu et al
- **Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters;** Boshi Wang et al
- **Careful Data Curation Stabilizes In-context Learning;** Ting-Yun Chang et al
- **Parallel Context Windows Improve In-Context Learning of Large Language Models;** Nir Ratner et al
- **Investigating Fusion Methods for In-Context Learning;** Qinyuan Ye et al
- **Batch Prompting: Efficient Inference with Large Language Model APIs;** Zhoujun Cheng et al
- **Explanation Selection Using Unlabeled Data for In-Context Learning;** Xi Ye et al
- **Compositional Exemplars for In-context Learning;** Jiacheng Ye et al
- **Distinguishability Calibration to In-Context Learning;** Hongjing Li et al
- **How Does In-Context Learning Help Prompt Tuning?;** Simeng Sun et al
- **Guiding Large Language Models via Directional Stimulus Prompting;** Zekun Li et al
- **In-Context Instruction Learning;** Seonghyeon Ye et al
- **LARGER LANGUAGE MODELS DO IN-CONTEXT LEARNING DIFFERENTLY;** Jerry Wei et al
- **kNN PROMPTING: BEYOND-CONTEXT LEARNING WITH CALIBRATION-FREE NEAREST NEIGHBOR INFERENCE;** Benfeng Xu et al
- **Learning In-context Learning for Named Entity Recognition;** Jiawei Chen et al
- **SELF-ICL: Zero-Shot In-Context Learning with Self-Generated Demonstrations;** Wei-Lin Chen et al
- **Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation;** Marius Mosbach et al
- **Large Language Models Can be Lazy Learners: Analyze Shortcuts in In-Context Learning;** Ruixiang Tang et al
- **IN-CONTEXT REINFORCEMENT LEARNING WITH ALGORITHM DISTILLATION;** Michael Laskin et al
- **Supervised Pretraining Can Learn In-Context Reinforcement Learning;** Jonathan N. Lee et al
- **Learning to Retrieve In-Context Examples for Large Language Models;** Liang Wang et al
- **IN-CONTEXT LEARNING IN LARGE LANGUAGE MODELS LEARNS LABEL RELATIONSHIPS BUT IS NOT CONVENTIONAL LEARNING;** Jannik Kossen et al
- **In-Context Alignment: Chat with Vanilla Language Models Before Fine-Tuning;** Xiaochuang Han et al

## Learning from Feedback

- **Decision Transformer: Reinforcement Learning via Sequence Modeling;** Lili Chen et al
- **Quark: Controllable Text Generation with Reinforced (Un)learning;** Ximing Lu et al
- **Learning to Repair: Repairing model output errors after deployment using a dynamic memory of feedback;** Niket Tandon et al
- **MemPrompt: Memory-assisted Prompt Editing with User Feedback;** Aman Madaan et al
- **Training language models to follow instructions with human feedback;** Long Ouyang et al
- **Pretraining Language Models with Human Preferences;** Tomasz Korbak et al
- **Training Language Models with Language Feedback;** Jérémy Scheurer et al
- **Training Language Models with Language Feedback at Scale;**  Jérémy Scheurer et al
- **Improving Code Generation by Training with Natural Language Feedback;** Angelica Chen et al
- **REFINER: Reasoning Feedback on Intermediate Representations;** Debjit Paul et al
- **RRHF: Rank Responses to Align Language Models with Human Feedback without tears;** Zheng Yuan et al
- **Constitutional AI: Harmlessness from AI Feedback;** Yuntao Bai et al
- **Chain of Hindsight Aligns Language Models with Feedback;** Hao Liu et al
- **Self-Edit: Fault-Aware Code Editor for Code Generation;** Kechi Zhang et al
- **RL4F: Generating Natural Language Feedback with Reinforcement Learning for Repairing Model Outputs;** Afra Feyza Akyürek et al
- **Learning to Simulate Natural Language Feedback for Interactive Semantic Parsing;** Hao Yan et al
- **Improving Language Model Negotiation with Self-Play and In-Context Learning from AI Feedback;** Yao Fu et al
- **Fine-Grained Human Feedback Gives Better Rewards for Language Model Training;** Zeqiu Wu et al
- **Aligning Large Language Models through Synthetic Feedback;** Sungdong Kim1 et al
- **Improving Language Models via Plug-and-Play Retrieval Feedback;** Wenhao Yu et al
- **Improving Open Language Models by Learning from Organic Interactions;** Jing Xu et al
- **Demystifying GPT Self-Repair for Code Generation;** Theo X. Olausson et al
- **Reflexion: Language Agents with Verbal Reinforcement Learning;** Noah Shinn et al
- **Evaluating Language Models for Mathematics through Interactions;** Katherine M. Collins et al
- **InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback;** John Yang et al
- **System-Level Natural Language Feedback;** Weizhe Yuan et al
- **Preference Ranking Optimization for Human Alignment;** Feifan Song et al
- **Let Me Teach You: Pedagogical Foundations of Feedback for Language Models;** Beatriz Borges et al
- **AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback;** Yann Dubois et al
- **Training Socially Aligned Language Models in Simulated Human Society;** Ruibo Liu et al
- **RLTF: Reinforcement Learning from Unit Test Feedback;** Jiate Liu et al
- **Chain of Hindsight Aligns Language Models with Feedback;** Hao Liu et al
- **LETI: Learning to Generate from Textual Interactions;** Xingyao Wang et al
- **Direct Preference Optimization: Your Language Model is Secretly a Reward Model;** Rafael Rafailov et al
- **FigCaps-HF: A Figure-to-Caption Generative Framework and Benchmark with Human Feedback;** Ashish Singh et al
- **Leveraging Implicit Feedback from Deployment Data in Dialogue;** Richard Yuanzhe Pang et al
- **RLCD: REINFORCEMENT LEARNING FROM CONTRAST DISTILLATION FOR LANGUAGE MODEL ALIGNMENT;** Kevin Yang et al
- **Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback;** Viet Dac Lai et al
- **Reinforced Self-Training (ReST) for Language Modeling;** Caglar Gulcehre et al
- **EVERYONE DESERVES A REWARD: LEARNING CUSTOMIZED HUMAN PREFERENCES;** Pengyu Cheng et al
- **RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback;** Harrison Lee et al
- **STABILIZING RLHF THROUGH ADVANTAGE MODEL AND SELECTIVE REHEARSAL;** Baolin Peng et al
- **OPENCHAT: ADVANCING OPEN-SOURCE LANGUAGE MODELS WITH MIXED-QUALITY DATA;** Guan Wang et al
- **HUMAN FEEDBACK IS NOT GOLD STANDARD;** Tom Hosking et al
- **A LONG WAY TO GO: INVESTIGATING LENGTH CORRELATIONS IN RLHF;** Prasann Singhal et al
- **CHAT VECTOR: A SIMPLE APPROACH TO EQUIP LLMS WITH NEW LANGUAGE CHAT CAPABILITIES;** Shih-Cheng Huang et al
- **SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF;** Yi Dong et al
- **UNDERSTANDING THE EFFECTS OF RLHF ON LLM GENERALISATION AND DIVERSITY;** Robert Kirk et al
- **GAINING WISDOM FROM SETBACKS: ALIGNING LARGE LANGUAGE MODELS VIA MISTAKE ANALYSIS;** Kai Chen et al
- **Tuna: Instruction Tuning using Feedback from Large Language Models;** Haoran Li et al
- **Teaching Language Models to Self-Improve through Interactive Demonstrations;** Xiao Yu et al
- **Democratizing Reasoning Ability: Tailored Learning from Large Language Model;** Zhaoyang Wang et al
- **ENABLE LANGUAGE MODELS TO IMPLICITLY LEARN SELF-IMPROVEMENT FROM DATA;** Ziqi Wang et al
- **ULTRAFEEDBACK: BOOSTING LANGUAGE MODELS WITH HIGH-QUALITY FEEDBACK;** Ganqu Cui et al
- **HELPSTEER: Multi-attribute Helpfulness Dataset for STEERLM;** Zhilin Wang et al
- **Knowledgeable Preference Alignment for LLMs in Domain-specific Question Answering;** Yichi Zhang et al
- **TOOL-AUGMENTED REWARD MODELING;** Lei Li et al
- **Nash Learning from Human Feedback;** Rémi Munos et al
- **Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models;** Avi Singh et al
- **When Life Gives You Lemons, Make Cherryade: Converting Feedback from Bad Responses into Good Labels;** Weiyan Shi et al
- **ConstitutionMaker: Interactively Critiquing Large Language Models by Converting Feedback into Principles;** Savvas Petridis et al
- **REASONS TO REJECT? ALIGNING LANGUAGE MODELS WITH JUDGMENTS;** Weiwen Xu et al
- **Some things are more CRINGE than others: Preference Optimization with the Pairwise Cringe Loss;** Jing Xu et al
- **Mitigating Unhelpfulness in Emotional Support Conversations with Multifaceted AI Feedback;** Jiashuo Wang et al
- **Secrets of RLHF in Large Language Models Part II: Reward Modeling;** Binghai Wang et al
- **Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation;** Haoran Xu et al
- **WARM: On the Benefits of Weight Averaged Reward Models;** Alexandre Ramé et al
- **Self-Rewarding Language Models;** Weizhe Yuan et al
- **Dense Reward for Free in Reinforcement Learning from Human Feedback;** Alex J. Chan et al
- **Efficient Exploration for LLMs;** Vikranth Dwaracherla et al
- **KTO: Model Alignment as Prospect Theoretic Optimization;** Kawin Ethayarajh et al
- **LiPO: Listwise Preference Optimization through Learning-to-Rank;** Tianqi Liu et al
- **Direct Language Model Alignment from Online AI Feedback;** Shangmin Guo et al
- **Noise Contrastive Alignment of Language Models with Explicit Rewards;** Huayu Chen et al
- **RLVF: Learning from Verbal Feedback without Overgeneralization;** Moritz Stephan et al
- **OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement;** Tianyu Zheng et al
- **A Critical Evaluation of AI Feedback for Aligning Large Language Models;** Archit Sharma et al




## Video Foundation Model

- **VideoBERT: A Joint Model for Video and Language Representation Learning;** Chen Sun et al
- **LEARNING VIDEO REPRESENTATIONS USING CONTRASTIVE BIDIRECTIONAL TRANSFORMER;** Chen Sun et al
- **End-to-End Learning of Visual Representations from Uncurated Instructional Videos;** Antoine Miech et al
- **HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training;** Linjie Li et al
- **Multi-modal Transformer for Video Retrieval;** Valentin Gabeur et al
- **ActBERT: Learning Global-Local Video-Text Representations;** Linchao Zhu et al
- **Spatiotemporal Contrastive Video Representation Learning;** Rui Qian et al
- **DECEMBERT: Learning from Noisy Instructional Videos via Dense Captions and Entropy Minimization;** Zineng Tang et al
- **HiT: Hierarchical Transformer with Momentum Contrast for Video-Text Retrieval;** Song Liu et al
- **Self-Supervised MultiModal Versatile Networks;** Jean-Baptiste Alayrac et al
- **COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning;** Simon Ging et al
- **VIMPAC: Video Pre-Training via Masked Token Prediction and Contrastive Learning;** Hao Tan et al
- **Less is More: CLIPBERT for Video-and-Language Learning via Sparse Sampling;** Jie Lei et al
- **Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval;** Max Bain et al
- **CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval;** Huaishao Luo et al
- **MERLOT: Multimodal Neural Script Knowledge Models;** Rowan Zellers et al
- **VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text;** Hassan Akbari et al
- **VIOLET: End-to-End Video-Language Transformers with Masked Visual-token Modeling;** Tsu-Jui Fu et al
- **CoCo-BERT: Improving Video-Language Pre-training with Contrastive Cross-modal Matching and Denoising;** Jianjie Luo et al
- **LAVENDER: Unifying Video-Language Understanding as Masked Language Modeling;** Linjie Li et al
- **CLIP-VIP: ADAPTING PRE-TRAINED IMAGE-TEXT MODEL TO VIDEO-LANGUAGE ALIGNMENT;** Hongwei Xue et al
- **Masked Video Distillation: Rethinking Masked Feature Modeling for Self-supervised Video Representation Learning;** Rui Wang et al
- **Long-Form Video-Language Pre-Training with Multimodal Temporal Contrastive Learning;** Yuchong Sun et al
- **Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning;** Antoine Yang et al
- **InternVideo: General Video Foundation Models via Generative and Discriminative Learning;** Yi Wang et al
- **MINOTAUR: Multi-task Video Grounding From Multimodal Queries;** Raghav Goyal et al
- **VideoLLM: Modeling Video Sequence with Large Language Models;** Guo Chen et al
- **COSA: Concatenated Sample Pretrained Vision-Language Foundation Model;** Sihan Chen et al
- **VALLEY: VIDEO ASSISTANT WITH LARGE LANGUAGE MODEL ENHANCED ABILITY;** Ruipu Luo et al
- **Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models;** Muhammad Maaz et al
- **Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding;** Hang Zhang et al
- **InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation;** Yi Wang et al
- **VideoCon: Robust Video-Language Alignment via Contrast Captions;** Hritik Bansal et al
- **PG-Video-LLaVA: Pixel Grounding Large Video-Language Models;** Shehan Munasinghe et al
- **VLM-Eval: A General Evaluation on Video Large Language Models;** Shuailin Li et al
- **Video-LLaVA: Learning United Visual Representation by Alignment Before Projection;** Bin Lin et al
- **MVBench: A Comprehensive Multi-modal Video Understanding Benchmark;** Kunchang Li et al
- **LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models;** Yanwei Li et al
- **LANGUAGEBIND: EXTENDING VIDEO-LANGUAGE PRETRAINING TO N-MODALITY BY LANGUAGE-BASED SEMANTIC ALIGNMENT;** Bin Zhu et al
- **TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding;** Shuhuai Ren et al
- **Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers;** Tsai-Shien Chen et al
- **INTERNVIDEO2: SCALING VIDEO FOUNDATION MODELS FOR MULTIMODAL VIDEO UNDERSTANDING;** Yi Wang et al

## Key Frame Detection

- **Self-Supervised Learning to Detect Key Frames in Videos;** Xiang Yan et al
- **Towards Generalisable Video Moment Retrieval: Visual-Dynamic Injection to Image-Text Pre-Training;** Dezhao Luo et al
- **Localizing Moments in Long Video Via Multimodal Guidance;** Wayner Barrios et al

## Vision Model

- **PIX2SEQ: A LANGUAGE MODELING FRAMEWORK FOR OBJECT DETECTION;** Ting Chen et al
- **Scaling Vision Transformers to 22 Billion Parameters;** Mostafa Dehghani et al
- **CLIPPO: Image-and-Language Understanding from Pixels Only;** Michael Tschannen et al
- **Segment Anything;** Alexander Kirillov et al
- **InstructDiffusion: A Generalist Modeling Interface for Vision Tasks;** Zigang Geng et al
- **RMT: Retentive Networks Meet Vision Transformers;** Qihang Fan et al
- **INSTRUCTCV: INSTRUCTION-TUNED TEXT-TO-IMAGE DIFFUSION MODELS AS VISION GENERALISTS;** Yulu Gan et al
- **Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks;** Micah Goldblum et al
- **RECOGNIZE ANY REGIONS;** Haosen Yang et al
- **AiluRus: A Scalable ViT Framework for Dense Prediction;** Jin Li et al
- **T-Rex: Counting by Visual Prompting;** Qing Jiang et al
- **Visual In-Context Prompting;** Feng Li et al
- **SAM-CLIP: Merging Vision Foundation Models towards Semantic and Spatial Understanding;** Haoxiang Wang et al
- **Sequential Modeling Enables Scalable Learning for Large Vision Models;** Yutong Bai et al
- **Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks;** Bin Xiao et al
- **4M: Massively Multimodal Masked Modeling;** David Mizrahi et al
- **InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks;** Zhe Chen et al
- **Scalable Pre-training of Large Autoregressive Image Models;** Alaaeldin El-Nouby et al
- **When Do We Not Need Larger Vision Models?;** Baifeng Shi et al

## Pretraining

- **MDETR - Modulated Detection for End-to-End Multi-Modal Understanding;** Aishwarya Kamath et al
- **SGEITL: Scene Graph Enhanced Image-Text Learning for Visual Commonsense Reasoning;** Zhecan Wang et al; Incorporating scene graphs in pretraining and fine-tuning improves performance of VCR tasks.
- **ERNIE-ViL: Knowledge Enhanced Vision-Language Representations through Scene Graphs;** Fei Yu et al
- **KB-VLP: Knowledge Based Vision and Language Pretraining;** Kezhen Chen et al; Propose to distill the object knowledge in VL pretraining for object-detector-free VL foundation models; Pretraining tasks include predicting the RoI features, category, and learning the alignments between phrases and image regions.
- **Large-Scale Adversarial Training for Vision-and-Language Representation Learning;** Zhe Gan et al
- **Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts;** Yan Zeng et al
- **BEIT: BERT Pre-Training of Image Transformers;** Hangbo Bao et al; Pre-trained CV model.
- **BEIT V2: Masked Image Modeling with Vector-Quantized Visual Tokenizers;** Zhiliang Peng et al; Pre-trained CV model.
- **VirTex: Learning Visual Representations from Textual Annotations;** Karan Desai et al; Pretraining CV models through the dense image captioning task.
- **Florence: A New Foundation Model for Computer Vision;** Lu Yuan et al; Pre-trained CV model.
- **Grounded Language-Image Pre-training;** Liunian Harold Li et al; Learning object-level, language-aware, and semantic-rich visual representations. Introducing phrase grounding to the pretraining task and focusing on object detection as the downstream task; Propose GLIP.
- **VLMixer: Unpaired Vision-Language Pre-training via Cross-Modal CutMix;** Teng Wang et al; Using unpaired data for pretraining.
- **Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone;** Zi-Yi Dou et al
- **WRITE AND PAINT: GENERATIVE VISION-LANGUAGE MODELS ARE UNIFIED MODAL LEARNERS;** Shizhe Diao et al
- **VILA: Learning Image Aesthetics from User Comments with Vision-Language Pretraining;** Junjie Ke et al
- **CONTRASTIVE ALIGNMENT OF VISION TO LANGUAGE THROUGH PARAMETER-EFFICIENT TRANSFER LEARNING;** Zaid Khan et al
- **The effectiveness of MAE pre-pretraining for billion-scale pretraining;** Mannat Singh et al
- **Retrieval-based Knowledge Augmented Vision Language Pre-training;** Jiahua Rao et al

**Visual-augmented LM**

- **Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision;** Hao Tan et al
- **Imagination-Augmented Natural Language Understanding;** Yujie Lu et al
- **Visually-augmented language modeling;** Weizhi Wang et al
- **Effect of Visual Extensions on Natural Language Understanding in Vision-and-Language Models;** Taichi Iki et al
- **Is BERT Blind? Exploring the Effect of Vision-and-Language Pretraining on Visual Language Understanding;** Morris Alper et al
- **TextMI: Textualize Multimodal Information for Integrating Non-verbal Cues in Pre-trained Language Models;** Md Kamrul Hasan et al
- **Learning to Imagine: Visually-Augmented Natural Language Generation;** Tianyi Tang et al

**Novel techniques.**

- **CM3: A CAUSAL MASKED MULTIMODAL MODEL OF THE INTERNET;** Armen Aghajanyan et al; Propose to pretrain on large corpus of structured multi-modal documents (CC-NEWS & En-Wikipedia) that can contain both text and image tokens.
- **PaLI: A Jointly-Scaled Multilingual Language-Image Model;** Xi Chen et al; Investigate the scaling effect of multi-modal models; Pretrained on WebLI that contains text in over 100 languages.
- **Retrieval-Augmented Multimodal Language Modeling;** Michihiro Yasunaga et al; Consider text generation and image generation tasks.
- **Re-ViLM: Retrieval-Augmented Visual Language Model for Zero and Few-Shot Image Captioning;** Zhuolin Yang et al
- **Teaching Structured Vision & Language Concepts to Vision & Language Models;** Sivan Doveh et al
- **MATCHA : Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering;** Fangyu Liu et al
- **Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training;** Filip Radenovic et al; Propose methods to improve zero-shot performance on retrieval and classification tasks through large-scale pre-training.
- **Prismer: A Vision-Language Model with An Ensemble of Experts;** Shikun Liu et al
- **REVEAL: Retrieval-Augmented Visual-Language Pre-Training with Multi-Source Multimodal Knowledge Memory;** Ziniu Hu et al

## Adaptation of Foundation Model

- **owards General Purpose Vision Systems: An End-to-End Task-Agnostic Vision-Language Architecture;** Tanmay Gupta et al
- **Language Models with Image Descriptors are Strong Few-Shot Video-Language Learners;** Zhenhailong Wang et al
- **Multimodal Few-Shot Learning with Frozen Language Models;** Maria Tsimpoukelli et al; Use prefix-like image-embedding to stear the text generation process to achieve few-shot learning.
- **Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language;** Andy Zeng et al
- **UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes;** Alexander Kolesnikov et al
- **META LEARNING TO BRIDGE VISION AND LANGUAGE MODELS FOR MULTIMODAL FEW-SHOT LEARNING;** Ivona Najdenkoska et al
- **RAMM: Retrieval-augmented Biomedical Visual Question Answering with Multi-modal Pre-training;** Zheng Yuan et al
- **Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners;** Renrui Zhang et al
- **F-VLM: OPEN-VOCABULARY OBJECT DETECTION UPON FROZEN VISION AND LANGUAGE MODELS;** Weicheng Kuo et al
- **eP-ALM: Efficient Perceptual Augmentation of Language Models;** Mustafa Shukor et al
- **Transfer Visual Prompt Generator across LLMs;** Ao Zhang et al
- **Multimodal Web Navigation with Instruction-Finetuned Foundation Models;** Hiroki Furuta et al

## Prompting

- **Learning to Prompt for Vision-Language Models;** Kaiyang Zhou et al; Soft prompt tuning. Useing few-shot learning to improve performance on both in-distribution and out-of-distribution data. Few-shot setting.
- **Unsupervised Prompt Learning for Vision-Language Models;** Tony Huang et al; Soft prompt tuning. Unsupervised setting.
- **Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling;** Renrui Zhang et al; Few-shot setting.
- **CLIP-Adapter: Better Vision-Language Models with Feature Adapters;** Peng Gao et al; Few-shot setting.
- **Neural Prompt Search;** Yuanhan Zhang et al; Explore the combination of LoRA, Adapter, Soft prompt tuning. In full-data, few-shot, and domain shift settings.
- **Visual Prompt Tuning;** Menglin Jia et al; Soft prompt tuning + head tuning. Show better performance in few-shot and full-data settings than full-parameters tuning. Quite different from the NLP field.
- **Prompt Distribution Learning;** Yuning Lu et al; Soft prompt tuning. Few-shot setting.
- **Conditional Prompt Learning for Vision-Language Models;** identify a critical problem of CoOp: the learned context is not generalizable to wider unseen classes within the same dataset; Propose to learn a DNN that can generate for each image an input-conditional token (vector).
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
- **What Makes Good Examples for Visual In-Context Learning?;** Yuanhan Zhang et al
- **Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery;** Yuxin Wen et al
- **PLOT: PROMPT LEARNING WITH OPTIMAL TRANSPORT FOR VISION-LANGUAGE MODELS;** Guangyi Chen et al
- **What does CLIP know about a red circle? Visual prompt engineering for VLMs;** Aleksandar Shtedritski et al

## Efficiency

- **M3SAT: A SPARSELY ACTIVATED TRANSFORMER FOR EFFICIENT MULTI-TASK LEARNING FROM MULTIPLE MODALITIES;** Anonymous
- **Prompt Tuning for Generative Multimodal Pretrained Models;** Hao Yang et al; Implement prefix-tuning in OFA. Try full-data setting and demonstrate comparable performance.
- **Fine-tuning Image Transformers using Learnable Memory;** Mark Sandler et al; Add soft prompts in each layer. full-data.
- **Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks;** Jeffrey O. Zhang et al; Transfer learning.
- **Polyhistor: Parameter-Efficient Multi-Task Adaptation for Dense Vision Tasks;** Yen-Cheng Liu et al
- **Task Residual for Tuning Vision-Language Models;** Tao Yu et al
- **UniAdapter: Unified Parameter-Efficient Transfer Learning for Cross-modal Modeling;** Haoyu Lu et al

## Analysis

- **What Does BERT with Vision Look At?** Liunian Harold Li et al
- **Visual Referring Expression Recognition: What Do Systems Actually Learn?;** Volkan Cirik et al
- **Characterizing and Overcoming the Greedy Nature of Learning in Multi-modal Deep Neural Networks;** Nan Wu et al; Study the problem of only relying on one certain modality in training when using multi-modal models.
- **Behind the Scene: Revealing the Secrets of Pre-trained Vision-and-Language Models;** Jize Cao et al
- **Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning** Weixin Liang et al
- **How Much Can CLIP Benefit Vision-and-Language Tasks?;** Sheng Shen et al; Explore two scenarios: 1) plugging CLIP into task-specific fine-tuning; 2) combining CLIP with V&L pre-training and transferring to downstream tasks. Show the boost in performance when using CLIP as the image encoder.
- **Vision-and-Language or Vision-for-Language? On Cross-Modal Influence in Multimodal Transformers;** Stella Frank et al
- **Controlling for Stereotypes in Multimodal Language Model Evaluation;** Manuj Malik et al
- **Beyond Instructional Videos: Probing for More Diverse Visual-Textual Grounding on YouTube;** Jack Hessel et al
- **What is More Likely to Happen Next? Video-and-Language Future Event Prediction;** Jie Lei et al

## Grounding

- **Flickr30k Entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models;** Bryan A. Plummer et al; A new benchmark dataset, annotating phrase-region correspondences.
- **Connecting Vision and Language with Localized Narratives;** Jordi Pont-Tuset et al
- **MAF: Multimodal Alignment Framework for Weakly-Supervised Phrase Grounding;** Qinxin Wang et al
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
- **Multimodal retrieval-augmented generator for open question answering over images and text;** Wenhu Chen et al
- **Towards a Unified Model for Generating Answers and Explanations in Visual Question Answering;** Chenxi Whitehouse et al
- **Modularized Zero-shot VQA with Pre-trained Models;** Rui Cao et al
- **Generate then Select: Open-ended Visual Question Answering Guided by World Knowledge;** Xingyu Fu et al
- **Using Visual Cropping to Enhance Fine-Detail Question Answering of BLIP-Family Models;** Jiarui Zhang et al
- **Zero-shot Visual Question Answering with Language Model Feedback;** Yifan Du et al
- **Learning to Ask Informative Sub-Questions for Visual Question Answering;** Kohei Uehara et al
- **Why Did the Chicken Cross the Road? Rephrasing and Analyzing Ambiguous Questions in VQA;** Elias Stengel-Eskin et al
- **Investigating Prompting Techniques for Zero- and Few-Shot Visual Question Answering;** Rabiul Awal et al

## VQA Dataset

- **VQA: Visual Question Answering;** Aishwarya Agrawal et al
- **Towards VQA Models That Can Read;** Amanpreet Singh et al
- **Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering;** Yash Goyal et al; VQA-V2.
- **MULTIMODALQA: COMPLEX QUESTION ANSWERING OVER TEXT, TABLES AND IMAGES;** Alon Talmor et al
- **WebQA: Multihop and Multimodal QA;** Yingshan Chang et al
- **FunQA: Towards Surprising Video Comprehension;** Binzhu Xie et al; Used for video foundation model evaluation.
- **Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering;** Pan Lu et al
- **Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions?;** Yang Chen et al

**Cognition**

- **Inferring the Why in Images;** Hamed Pirsiavash et al
- **Visual Madlibs: Fill in the blank Image Generation and Question Answering;** Licheng Yu et al
- **From Recognition to Cognition: Visual Commonsense Reasoning;** Rowan Zellers et al; Benchmark dataset, requiring models to go beyond the recognition level to cognition. Need to reason about a still image and give rationales.
- **VisualCOMET: Reasoning about the Dynamic Context of a Still Image;** Jae Sung Park et al
- **The Abduction of Sherlock Holmes: A Dataset for Visual Abductive Reasoning;** Jack Hessel et al

**Knowledge**

- **Explicit Knowledge-based Reasoning for Visual Question Answering;** Peng Wang et al
- **FVQA: Fact-based Visual Question Answering;** Peng Wang;
- **OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge;** Kenneth Marino et al

## Social Good

- **The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes;** Douwe Kiela et al; Multi-modal hate-speech detection.
- **Detecting Cross-Modal Inconsistency to Defend Against Neural Fake News;** Reuben Tan et al; Multi-modal fake news dedetection.
- **InfoSurgeon: Cross-Media Fine-grained Information Consistency Checking for Fake News Detection;** Yi R. Fung et al; Cross-modal fake news detection.
- **EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection;** Yaqing Wang et al
- **End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models;** Barry Menglong Yao et al
- **SAFE: Similarity-Aware Multi-Modal Fake News Detection;** Xinyi Zhou et al
- **r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection;** Kai Nakamura et al; Fake news detection dataset.
- **Fact-Checking Meets Fauxtography: Verifying Claims About Images;** Dimitrina Zlatkova et al; Claim-Images pairs.
- **Prompting for Multimodal Hateful Meme Classification;** Rui Cao et al

## Application

- **MSMO: Multimodal Summarization with Multimodal Output;** Junnan Zhu et al
- **Re-imagen: Retrieval-augmented text-to-image generator;** Wenhu Chen et al
- **Large Scale Multi-Lingual Multi-Modal Summarization Dataset;** Yash Verma et al
- **Retrieval-augmented Image Captioning;** Rita Ramos et al
- **SYNTHETIC MISINFORMERS: GENERATING AND COMBATING MULTIMODAL MISINFORMATION;** Stefanos-Iordanis Papadopoulos et al
- **The Dialog Must Go On: Improving Visual Dialog via Generative Self-Training;** Gi-Cheon Kang et al
- **CapDet: Unifying Dense Captioning and Open-World Detection Pretraining;** Yanxin Long et al
- **DECAP: DECODING CLIP LATENTS FOR ZERO-SHOT CAPTIONING VIA TEXT-ONLY TRAINING;** Wei Li et al
- **Align and Attend: Multimodal Summarization with Dual Contrastive Losses;** Bo He et al

## Benchmark & Evaluation

- **Multimodal datasets: misogyny, pornography, and malignant stereotypes;** Abeba Birhane et al
- **Understanding ME? Multimodal Evaluation for Fine-grained Visual Commonsense;** Zhecan Wang et al
- **Probing Image–Language Transformers for Verb Understanding;** Lisa Anne Hendricks et al
- **VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations;** Tiancheng Zhao et al
- **WHEN AND WHY VISION-LANGUAGE MODELS BEHAVE LIKE BAGS-OF-WORDS, AND WHAT TO DO ABOUT IT?;** Mert Yuksekgonul et al
- **GRIT: General Robust Image Task Benchmark;** Tanmay Gupta et al
- **MULTIMODALQA: COMPLEX QUESTION ANSWERING OVER TEXT, TABLES AND IMAGES;** Alon Talmor et al
- **Test of Time: Instilling Video-Language Models with a Sense of Time;** Piyush Bagad et al

## Dataset

- **Visual Entailment: A Novel Task for Fine-Grained Image Understanding;** Ning Xie et al; Visual entailment task. SNLI-VE.
- **A Corpus for Reasoning About Natural Language Grounded in Photographs;** Alane Suhr et al; NLVR2.
- **VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models;** Wangchunshu Zhou et al; VLUE.
- **Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning;** Piyush Sharma et al
- **Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts;** Soravit Changpinyo et al
- **LAION-5B: An open large-scale dataset for training next generation image-text models;** Christoph Schuhmann et al
- **Bloom Library: Multimodal Datasets in 300+ Languages for a Variety of Downstream Tasks;** Colin Leong et al
- **Find Someone Who: Visual Commonsense Understanding in Human-Centric Grounding;** Haoxuan You et al
- **MULTIINSTRUCT: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning;** Zhiyang Xu et al
- **UKnow: A Unified Knowledge Protocol for Common-Sense Reasoning and Vision-Language Pre-training;** Biao Gong et al
- **HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips;** Antoine Miech et al
- **Connecting Vision and Language with Video Localized Narratives;** Paul Voigtlaender et al
- **LAION-5B: An open large-scale dataset for training next generation image-text models;** Christoph Schuhmann et al
- **MAD: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions;** Mattia Soldan et al
- **CHAMPAGNE: Learning Real-world Conversation from Large-Scale Web Videos;** Seungju Han et al
- **WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning;** Krishna Srinivasan et al
- **Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved With Text;** Wanrong Zhu et al
- **OpenAssistant Conversations - Democratizing Large Language Model Alignment;** Andreas Köpf et al
- **TheoremQA: A Theorem-driven Question Answering dataset;** Wenhu Chen et al
- **MetaCLUE: Towards Comprehensive Visual Metaphors Research;** Arjun R. Akula et al
- **CAPSFUSION: Rethinking Image-Text Data at Scale;** Qiying Yu et al
- **RedCaps: Web-curated image-text data created by the people, for the people;** Karan Desai et al
- **OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents;** Hugo Laurençon et al

## Robustness

- **Domino: Discovering Systematic Errors with Cross-Modal Embeddings;** Sabri Eyuboglu et al
- **Learning Visually-Grounded Semantics from Contrastive Adversarial Samples;** Haoyue Shi et al
- **Visually Grounded Reasoning across Languages and Cultures;** Fangyu Liu et al
- **A Closer Look at the Robustness of Vision-and-Language Pre-trained Models;** Linjie Li et al; Compile a list of robustness-VQA datasets.
- **ROBUSTNESS ANALYSIS OF VIDEO-LANGUAGE MODELS AGAINST VISUAL AND LANGUAGE PERTURBATIONS;** Madeline C. Schiappa et al
- **Context-Aware Robust Fine-Tuning;** Xiaofeng Mao et al
- **Task Bias in Vision-Language Models;** Sachit Menon et al
- **Are Multimodal Models Robust to Image and Text Perturbations?;** Jielin Qiu et al
- **CPL: Counterfactual Prompt Learning for Vision and Language Models;** Xuehai He et al
- **Improving Zero-shot Generalization and Robustness of Multi-modal Models;** Yunhao Ge et al
- **DIAGNOSING AND RECTIFYING VISION MODELS USING LANGUAGE;** Yuhui Zhang et al
- **Multimodal Prompting with Missing Modalities for Visual Recognition;** Yi-Lun Lee et al

## Hallucination&Factuality

- **Object Hallucination in Image Captioning;** Anna Rohrbach et al
- **Learning to Generate Grounded Visual Captions without Localization Supervision;** Chih-Yao Ma et al
- **On Hallucination and Predictive Uncertainty in Conditional Language Generation;** Yijun Xiao et al
- **Consensus Graph Representation Learning for Better Grounded Image Captioning;** Wenqiao Zhang et al
- **Relational Graph Learning for Grounded Video Description Generation;** Wenqiao Zhang et al
- **Let there be a clock on the beach: Reducing Object Hallucination in Image Captioning;** Ali Furkan Biten et al
- **Plausible May Not Be Faithful: Probing Object Hallucination in Vision-Language Pre-training;** Wenliang Dai et al
- **Models See Hallucinations: Evaluating the Factuality in Video Captioning;** Hui Liu et al
- **Evaluating and Improving Factuality in Multimodal Abstractive Summarization;** David Wan et al
- **Evaluating Object Hallucination in Large Vision-Language Models;** Yifan Li et al
- **Do Language Models Know When They’re Hallucinating References?;** Ayush Agrawal et al
- **Detecting and Preventing Hallucinations in Large Vision Language Models;** Anisha Gunjal et al
- **DOLA: DECODING BY CONTRASTING LAYERS IMPROVES FACTUALITY IN LARGE LANGUAGE MODELS;** Yung-Sung Chuang et al
- **A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models;** S.M Towhidul Islam Tonmoy et al
- **Inference-Time Intervention: Eliciting Truthful Answers from a Language Model;** Kenneth Li et al
- **FELM: Benchmarking Factuality Evaluation of Large Language Models;** Shiqi Chen et al
- **Unveiling the Siren’s Song: Towards Reliable Fact-Conflicting Hallucination Detection;** Xiang Chen et al
- **ANALYZING AND MITIGATING OBJECT HALLUCINATION IN LARGE VISION-LANGUAGE MODELS;** Yiyang Zhou et al
- **Woodpecker: Hallucination Correction for Multimodal Large Language Models;** Shukang Yin et al
- **AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation;** Junyang Wang et al
- **Fine-tuning Language Models for Factuality;** Katherine Tian et al
- **Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization;** Zhiyuan Zhao et al
- **RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback;** Tianyu Yu et al
- **RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models;** Yuanhao Wu et al
- **Learning to Trust Your Feelings: Leveraging Self-awareness in LLMs for Hallucination Mitigation;** Yuxin Liang et al
- **Don’t Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration;** Shangbin Feng et al

## Cognitive NeuronScience & Machine Learning

- **Mind Reader: Reconstructing complex images from brain activities;** Sikun Lin et al
- **Joint processing of linguistic properties in brains and language models;** Subba Reddy Oota et al
- **Is the Brain Mechanism for Hierarchical Structure Building Universal Across Languages? An fMRI Study of Chinese and English;** Xiaohan Zhang et al
- **TRAINING LANGUAGE MODELS FOR DEEPER UNDERSTANDING IMPROVES BRAIN ALIGNMENT;** Khai Loong Aw et al
- **Abstract Visual Reasoning with Tangram Shapes;** Anya Ji et al
- **DISSOCIATING LANGUAGE AND THOUGHT IN LARGE LANGUAGE MODELS: A COGNITIVE PERSPECTIVE;** Kyle Mahowald et al
- **Language Cognition and Language Computation Human and Machine Language Understanding;** Shaonan Wang et al
- **From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought;** Lionel Wong et al
- **DIVERGENCES BETWEEN LANGUAGE MODELS AND HUMAN BRAINS;** Yuchen Zhou et al
- **Do Language Models Exhibit the Same Cognitive Biases in Problem Solving as Human Learners?;** Andreas Opedal et al

## Theory of Mind

- **Do Large Language Models know what humans know?;** Sean Trott et al
- **Few-shot Language Coordination by Modeling Theory of Mind;** Hao Zhu et al
- **Few-Shot Character Understanding in Movies as an Assessment to Meta-Learning of Theory-of-Mind;** Mo Yu et al
- **Neural Theory-of-Mind? On the Limits of Social Intelligence in Large LMs;** Maarten Sap et al
- **A Cognitive Evaluation of Instruction Generation Agents tl;dr They Need Better Theory-of-Mind Capabilities;** Lingjun Zhao et al
- **MINDCRAFT: Theory of Mind Modeling for Situated Dialogue in Collaborative Tasks;** Cristian-Paul Bara et al
- **TVSHOWGUESS: Character Comprehension in Stories as Speaker Guessing;** Yisi Sang et al
- **Theory of Mind May Have Spontaneously Emerged in Large Language Models;** Michal Kosinski
- **COMPUTATIONAL LANGUAGE ACQUISITION WITH THEORY OF MIND;** Andy Liu et al
- **Speaking the Language of Your Listener: Audience-Aware Adaptation via Plug-and-Play Theory of Mind;** Ece Takmaz et al
- **Understanding Social Reasoning in Language Models with Language Models;** Kanishk Gandhi et al
- **HOW FAR ARE LARGE LANGUAGE MODELS FROM AGENTS WITH THEORY-OF-MIND?;** Pei Zhou et al

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

## World Model

- **Recurrent World Models Facilitate Policy Evolution;** David Ha et al
- **TRANSFORMERS ARE SAMPLE-EFFICIENT WORLD MODELS;** Vincent Micheli et al
- **Language Models Meet World Models: Embodied Experiences Enhance Language Models;** Jiannan Xiang et al
- **Reasoning with Language Model is Planning with World Model;** Shibo Hao et al
- **Learning to Model the World with Language;** Jessy Lin et al
- **Learning Interactive Real-World Simulators;** Mengjiao Yang et al
- **Diffusion World Model;** Zihan Ding et al
- **Genie: Generative Interactive Environments;** Jake Bruce et al
- **Learning and Leveraging World Models in Visual Representation Learning;** Quentin Garrido et al

## Resource

- **LAVIS-A One-stop Library for Language-Vision Intelligence;** https://github.com/salesforce/LAVIS
- **MULTIVIZ: TOWARDS VISUALIZING AND UNDERSTANDING MULTIMODAL MODELS;** Paul Pu Liang et al
- **TorchScale - A Library for Transformers at (Any) Scale;** Shuming Ma et al
- **Video pretraining;** https://zhuanlan.zhihu.com/p/515175476
- **Towards Complex Reasoning: the Polaris of Large Language Models;** Yao Fu
- **Prompt Engineering;** Lilian Weng
- **Memory in human brains;** https://qbi.uq.edu.au/brain-basics/memory
- **Bloom's Taxonomy;** https://cft.vanderbilt.edu/guides-sub-pages/blooms-taxonomy/#:~:text=Familiarly%20known%20as%20Bloom's%20Taxonomy,Analysis%2C%20Synthesis%2C%20and%20Evaluation.
- **Chain-of-Thought Hub: A Continuous Effort to Measure Large Language Models’ Reasoning Performance;** Yao Fu et al
- **LLM Powered Autonomous Agents;** Lilian Weng
- **Retrieval-based Language Models and Applications;** Tutorial; https://github.com/ACL2023-Retrieval-LM/ACL2023-Retrieval-LM.github.io
- **Recent Advances in Vision Foundation Models;** Tutorial; https://vlp-tutorial.github.io/
- **LM-Polygraph: Uncertainty Estimation for Language Models;** Ekaterina Fadeeva et al
- **The Q\* hypothesis: Tree-of-thoughts reasoning, process reward models, and supercharging synthetic data;** Nathan Lambert
- **Data-Juicer: A One-Stop Data Processing System for Large Language Models;** Daoyuan Chen et al
- **Designing, Evaluating, and Learning from Human-AI Interactions;** Sherry Tongshuang Wu et al
- **Reinforcement Learning from Human Feedback: Progress and Challenges;** John Schulman
- **Our approach to alignment research;** OpenAI
- **AI Alignment: A Comprehensive Survey;** Jiaming Ji et al; https://alignmentsurvey.com/
- **Alignment Workshop;** https://www.alignment-workshop.com/nola-2023
- **AI Alignment Research Overview;** Jacob Steinhardt
- **Proxy objectives in reinforcement learning from human feedback;** John Schulman
- **AgentLite: A Lightweight Library for Building and Advancing Task-Oriented LLM Agent System;** Zhiwei Liu et al
- **Training great LLMs entirely from ground up in the wilderness as a startup;** Yi Tay

