# General-purpose Models

## Table of Contents 
- [Survey](#survey)
- [Structure](#structure)
- [Planning](#planning)
- [Reasoning](#reasoning)
- [Generation](#generation)
- [Representation Learning](#representation-learning)
- [LLM Analysis](#llm-analysis)
- [LLM Evaluation](#llm-evaluation)
- [LLM Reasoning](#llm-reasoning)
- [LLM Application](#llm-application)
- [LLM with Memory](#llm-with-memory)
- [LLM with Human](#llm-with-human)
- [MoE](#moe)
- [Vision-Language Foundation Model](#vision-language-foundation-model)
- [Multimodal Foundation Model](#multimodal-foundation-model)
- [Document Understanding](#document-understanding)
- [External Tool](#external-tool)
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
- [Benchmark & Evaluation](#benchmark-&-evaluation)
- [Dataset](#dataset)
- [Robustness](#robustness)
- [Hallucination](#hallucination)
- [Cognitive NeuronScience & Machine Learning](#cognitive-neuronscience-&-machine-learning)
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
- **Eight Things to Know about Large Language Models;** Samuel R. Bowman et al
- **Retrieving Multimodal Information for Augmented Generation: A Survey;** Ruochen Zhao et al
- **Is Prompt All You Need? No. A Comprehensive and Broader View of Instruction Learning;** Renze Lou et al
- **A Survey of Large Language Models;** Wayne Xin Zhao et al
- **Tool Learning with Foundation Models;** Yujia Qin et al
- **A Cookbook of Self-Supervised Learning;** Randall Balestriero et al
- **Foundation Models for Decision Making: Problems, Methods, and Opportunities;** Sherry Yang et al
- **Bridging the Gap: A Survey on Integrating (Human) Feedback for Natural Language Generation;** Patrick Fernandes et al
- **Reasoning with Language Model Prompting: A Survey;** Shuofei Qiao et al 
- **Towards Reasoning in Large Language Models: A Survey;** Jie Huang et al
- **A PhD Student’s Perspective on Research in NLP in the Era of Very Large Language Models;** Oana Ignat et al
- **Beyond One-Model-Fits-All: A Survey of Domain Specialization for Large Language Models;** Chen Ling et al
- **Unifying Large Language Models and Knowledge Graphs: A Roadmap;** Shirui Pan et al


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





## Planning
- **Multimedia Generative Script Learning for Task Planning;** Qingyun Wang et al; Next step prediction.
- **PlaTe: Visually-Grounded Planning with Transformers in Procedural Tasks;** Jiankai Sun et al; Procedure planning. 
- **P3IV: Probabilistic Procedure Planning from Instructional Videos with Weak Supervision;** He Zhao et al; Procedure planning. Using text as weak supervision to replace video clips. 
- **Procedure Planning in Instructional Videos;** Chien-Yi Chang et al; Procedure planning.
- **ViLPAct: A Benchmark for Compositional Generalization on Multimodal Human Activities;** Terry Yue Zhuo et al
- **Actional Atomic-Concept Learning for Demystifying Vision-Language Navigation;** Bingqian Lin et al




## Reasoning
- **VisualCOMET: Reasoning about the Dynamic Context of a Still Image;** Jae Sung Park et al; Benchmark dataset, requiring models to reason about a still iamge (what happen past & next). 
- **Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering;** Pan Lu et al
- **See, Think, Confirm: Interactive Prompting Between Vision and Language Models for Knowledge-based Visual Reasoning;** Zhenfang Chen et al
- **An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA;** Zhengyuan Yang et al
- **Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering;** Pan Lu et al
- **Multimodal Chain-of-Thought Reasoning in Language Models;** Zhuosheng Zhang et al
- **LAMPP: Language Models as Probabilistic Priors for Perception and Action;** Belinda Z. Li et al
- **Visual Chain of Thought: Bridging Logical Gaps with Multimodal Infillings;** Daniel Rose et al


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
- **A Categorical Archive of ChatGPT Failures;** Ali Borji et al
- **Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling;** Stella Biderman  et al
- **Are Emergent Abilities of Large Language Models a Mirage?;** Rylan Schaeffer et al
- **A Drop of Ink may Make a Million Think: The Spread of False Information in Large Language Models;** Ning Bian et al
- **Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting;** Miles Turpin et al
- **SYMBOL TUNING IMPROVES IN-CONTEXT LEARNING IN LANGUAGE MODELS;** Jerry Wei et al
- **What In-Context Learning “Learns” In-Context: Disentangling Task Recognition and Task Learning;** Jane Pan et al
- **Measuring the Knowledge Acquisition-Utilization Gap in Pretrained Language Models;** Amirhossein Kazemnejad et al
- **Knowledge of Knowledge: Exploring Known-Unknowns Uncertainty with Large Language Models;** Alfonso Amayuelas et al
- **Scaling Data-Constrained Language Models;** Niklas Muennighoff et al
- **The False Promise of Imitating Proprietary LLMs;** Arnav Gudibande et al
- **Counterfactual reasoning: Testing language models’ understanding of hypothetical scenarios;** Jiaxuan Li et al
- **Inverse Scaling: When Bigger Isn’t Better;** Ian R. McKenzie et al
- **DECODINGTRUST: A Comprehensive Assessment of Trustworthiness in GPT Models;** Boxin Wang et al
- **Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs;** Miao Xiong et al
- **Lost in the Middle: How Language Models Use Long Contexts;** Nelson F. Liu et al
- **Won’t Get Fooled Again: Answering Questions with False Premises;** Shengding Hu et al
- **Jailbroken: How Does LLM Safety Training Fail? Content Warning: This paper contains examples of harmful language;** Alexander Wei et al
- **Generating Benchmarks for Factuality Evaluation of Language Models;** Dor Muhlgay et al



## LLM Evaluation
- **IS CHATGPT A GENERAL-PURPOSE NATURAL LANGUAGE PROCESSING TASK SOLVER?;** Chengwei Qin et al
- **AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models;** Wanjun Zhong et al
- **A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity;** Yejin Bang et al
- **On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective;** Jindong Wang et al
- **A Comprehensive Capability Analysis of GPT-3 and GPT-3.5 Series Models;** Junjie Ye et al
- **KoLA: Carefully Benchmarking World Knowledge of Large Language Models;** Jifan Yu et al



## LLM Reasoning
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
- **Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate;** Tian Liang et al
- **Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters;** Boshi Wang et al
- **Recursion of Thought: A Divide-and-Conquer Approach to Multi-Context Reasoning with Language Models;** Soochan Lee et al
- **Large Language Model Is Semi-Parametric Reinforcement Learning Agent;** Danyang Zhang et al
- **Large Language Models Are Reasoning Teachers;** Namgyu Ho et al
- **Meta-Reasoning: Semantics-Symbol Deconstruction For Large Language Models;** Yiming Wang et al
- **SWIFTSAGE: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks;** Bill Yuchen Lin et al
- **BeamSearchQA: Large Language Models are Strong Zero-Shot QA Solver;** Hao Sun et al
- **Improving Factuality and Reasoning in Language Models through Multiagent Debate;** Yilun Du et al
- **AdaPlanner: Adaptive Planning from Feedback with Language Models;** Haotian Sun et al
- **ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models;** Binfeng Xu et al




**Self-consistency**
- **Enhancing Self-Consistency and Performance of Pre-Trained Language Models through Natural Language Inference;** Eric Mitchell et al
- **Two Failures of Self-Consistency in the Multi-Step Reasoning of LLMs;** Angelica Chen et al
- **Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation;** Niels Mündler et al
- **Measuring and Narrowing the Compositionality Gap in Language Models;** Ofir Press et al



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
- **Generative Agents: Interactive Simulacra of Human Behavior;** Joon Sung Park et al
- **The Role of Summarization in Generative Agents: A Preliminary Perspective;** Xiachong Feng et al
- **CAMEL: Communicative Agents for “Mind” Exploration of Large Scale Language Model Society;** Guohao Li et al
- **ArK: Augmented Reality with Knowledge Interactive Emergent Ability;** Qiuyuan Huang et al
- **Can Large Language Models Be an Alternative to Human Evaluation?;** Cheng-Han Chiang et al
- **Few-shot In-context Learning for Knowledge Base Question Answering;** Tianle Li et al
- **Plan, Eliminate, and Track-Language Models are Good Teachers for Embodied Agents;** Yue Wu et al
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
- **Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents;** Zihao Wang et al
- **Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory;** Xizhou Zhu et al
- **Mindstorms in Natural Language-Based Societies of Mind;** Mingchen Zhuge et al
- **Responsible Task Automation: Empowering Large Language Models as Responsible Task Automators;** Zhizheng Zhang et al
- **Large Language Models as General Pattern Machines;** Suvir Mirchandani et al


## LLM with Memory
- **Neural Turing Machines;** Alex Graves et al
- **Narrative Question Answering with Cutting-Edge Open-Domain QA Techniques: A Comprehensive Study;** Xiangyang Mou et al
- **Memory and Knowledge Augmented Language Models for Inferring Salience in Long-Form Stories;** David Wilmot et al
- **MemPrompt: Memory-assisted Prompt Editing with User Feedback;** Aman Madaan et al
- **LANGUAGE MODEL WITH PLUG-IN KNOWLEDGE MEMORY;** Xin Cheng et al
- **Generative Agents: Interactive Simulacra of Human Behavior;** Joon Sung Park et al
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


**Retrieval-augmented LLM**
- **Training Language Models with Memory Augmentation;** Zexuan Zhong et al
- **Enabling Large Language Models to Generate Text with Citations;** Tianyu Gao et al
- **Multiview Identifiers Enhanced Generative Retrieval;** Yongqi Li et al
- **Meta-training with Demonstration Retrieval for Efficient Few-shot Learning;** Aaron Mueller et al


  

## LLM with Human
- **CoAuthor: Designing a Human-AI Collaborative Writing Dataset for Exploring Language Model Capabilities;** Mina Lee et al
- **RewriteLM: An Instruction-Tuned Large Language Model for Text Rewriting;** Lei Shu et al
- **LeanDojo: Theorem Proving with Retrieval-Augmented Language Models;** Kaiyu Yang et al
- **Evaluating Human-Language Model Interaction;** Mina Lee et al



## MoE
- **OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER;** Noam Shazeer et al
- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity;** William Fedus et al
- **DEMIX Layers: Disentangling Domains for Modular Language Modeling;** Suchin Gururangan et al
- **ModuleFormer: Learning Modular Large Language Models From Uncurated Data;** Yikang Shen et al
- **Mixture-of-Experts Meets Instruction Tuning: A Winning Combination for Large Language Models;** Sheng Shen et al




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



### Analysis & Evaluation
- **What Makes for Good Visual Tokenizers for Large Language Models?;** Guangzhi Wang et al
- **LVLM-eHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models;** Peng Xu et al
- **MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models;** Chaoyou Fu et al
- **JourneyDB: A Benchmark for Generative Image Understanding;** Junting Pan et al


### Others
- **Unified Vision-Language Pre-Training for Image Captioning and VQA;** Luowei Zhou et al
- **Unifying Vision-and-Language Tasks via Text Generation;** Jaemin Cho et al
- **MERLOT Reserve: Neural Script Knowledge through Vision and Language and Sound;** Rowan Zellers et al
- **CLIP-Event: Connecting Text and Images with Event Structures;** Manling Li et al; The new model CLIP-Event, specifically designed for multi-modal event extraction. Introducing new pretraining tasks to enable strong zero-shot performances. From object-centric representations to event-centric representations. 
- **Scaling Vision-Language Models with Sparse Mixture of Experts;** Sheng Shen et al
- **MaMMUT: A Simple Architecture for Joint Learning for MultiModal Tasks;** Weicheng Kuo et al

## Multimodal Foundation Model
- **MotionGPT: Human Motion as a Foreign Language;** Biao Jiang et al

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


**Dataset**
- **A Diagram Is Worth A Dozen Images;** Aniruddha Kembhavi et al
- **ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning;** Ahmed Masry et al
- **PDF-VQA: A New Dataset for Real-World VQA on PDF Documents;** Yihao Ding et al



***Table***
- **Visual Understanding of Complex Table Structures from Document Images;** Sachin Raja et al
- **Improving Table Structure Recognition with Visual-Alignment Sequential Coordinate Modeling;** Yongshuai Huang et al



## External Tool
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
- **Aligning Large Multi-Modal Model with Robust Instruction Tuning;** Fuxiao Liu et al
- **M3IT: A Large-Scale Dataset towards Multi-Modal Multilingual Instruction Tuning;** Lei Li et al
- **InstructEval: Systematic Evaluation of Instruction Selection Methods;** Anirudh Ajith et al
- **LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark;** Zhenfei Yin et al
- **Instruction Mining: High-Quality Instruction Data Selection for Large Language Models;** Yihan Cao et al

  

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
- **Let’s Verify Step by Step;** Hunter Lightman et al
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
- **BEAVERTAILS: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset;** Jiaming Ji et al
  

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



## Key Frame Detection
- **Self-Supervised Learning to Detect Key Frames in Videos;** Xiang Yan et al
- **Towards Generalisable Video Moment Retrieval: Visual-Dynamic Injection to Image-Text Pre-Training;** Dezhao Luo et al
- **Localizing Moments in Long Video Via Multimodal Guidance;** Wayner Barrios et al



## Vision Model
- **PIX2SEQ: A LANGUAGE MODELING FRAMEWORK FOR OBJECT DETECTION;** Ting Chen et al
- **Scaling Vision Transformers to 22 Billion Parameters;** Mostafa Dehghani et al
- **CLIPPO: Image-and-Language Understanding from Pixels Only;** Michael Tschannen et al
- **Segment Anything;** Alexander Kirillov et al




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


## Hallucination
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




## Cognitive NeuronScience & Machine Learning
- **Mind Reader: Reconstructing complex images from brain activities;** Sikun Lin et al
- **Joint processing of linguistic properties in brains and language models;** Subba Reddy Oota et al
- **Is the Brain Mechanism for Hierarchical Structure Building Universal Across Languages? An fMRI Study of Chinese and English;** Xiaohan Zhang et al
- **TRAINING LANGUAGE MODELS FOR DEEPER UNDERSTANDING IMPROVES BRAIN ALIGNMENT;** Khai Loong Aw et al
- **Abstract Visual Reasoning with Tangram Shapes;** Anya Ji et al
- **DISSOCIATING LANGUAGE AND THOUGHT IN LARGE LANGUAGE MODELS: A COGNITIVE PERSPECTIVE;** Kyle Mahowald et al
- **Language Cognition and Language Computation Human and Machine Language Understanding;** Shaonan Wang et al
- **From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought;** Lionel Wong et al



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
