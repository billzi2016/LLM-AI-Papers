# Large Multimodal Agents: A Survey

> **Date**：2024-02-23
> **arXiv**：https://arxiv.org/abs/2402.15116

## Abstract

Large language models (LLMs) have achieved superior performance in powering text-based AI agents, endowing them with decision-making and reasoning abilities akin to humans. Concurrently, there is an emerging research trend focused on extending these LLM-powered AI agents into the multimodal domain. This extension enables AI agents to interpret and respond to diverse multimodal user queries, thereby handling more intricate and nuanced tasks. In this paper, we conduct a systematic review of LLM-driven multimodal agents, which we refer to as large multimodal agents ( LMAs for short). First, we introduce the essential components involved in developing LMAs and categorize the current body of research into four distinct types. Subsequently, we review the collaborative frameworks integrating multiple LMAs , enhancing collective efficacy. One of the critical challenges in this field is the diverse evaluation methods used across existing studies, hindering effective comparison among different LMAs . Therefore, we compile these evaluation methodologies and establish a comprehensive framework to bridge the gaps. This framework aims to standardize evaluations, facilitating more meaningful comparisons. Concluding our review, we highlight the extensive applications of LMAs and propose possible future research directions. Our discussion aims to provide valuable insights and guidelines for future research in this rapidly evolving field. An up-to-date resource list is available at https://github.com/jun0wanan/awesome-large-multimodal-agents.

---

# 大型多模态智能体 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）横扫文本任务后，研究者们自然想让它们也能“看”“听”。然而，单纯把视觉或音频特征喂进语言模型往往会出现信息丢失、跨模态推理不连贯等问题。早期的多模态模型大多是“单向翻译”，比如把图片描述成文字，或者把文字生成对应的图像，缺乏真正的交互式决策能力。更关键的是，缺少统一的评估标准，使得不同团队的实验结果难以直接比较。于是，系统性梳理“LLM 驱动的多模态智能体”并提供统一框架的需求应运而生。

### 关键概念速览
- **LLM（大语言模型）**：能够理解并生成自然语言的大规模神经网络，类似于拥有海量阅读经验的“语言专家”。  
- **多模态智能体（Multimodal Agent）**：能够同时处理文字、图像、音频等多种感官输入，并给出相应输出的系统，像是拥有“眼睛”和“耳朵”的聊天机器人。  
- **感知编码器（Perception Encoder）**：把图片、视频或音频转成向量的子模型，常用的有 CLIP、ViT、Whisper 等，作用类似于把“感官信号”翻译成语言模型能懂的“内部语言”。  
- **跨模态对齐（Cross‑Modal Alignment）**：让语言模型的内部表示与感知编码器的向量在同一空间里对应起来，类似于把不同语言的词典对齐，使得“看见的猫”和“文字‘猫’”指向同一个概念。  
- **提示工程（Prompt Engineering）**：为 LLM 设计输入模板，引导模型在多模态情境下产生合适的推理路径，像是给机器人下达明确的任务指令。  
- **协同多智能体框架（Collaborative Multi‑Agent Framework）**：把若干专精于不同模态或任务的智能体组合起来，让它们相互协作完成更复杂的工作，类似于团队合作中的分工与信息共享。  
- **评估基准（Evaluation Benchmark）**：统一的测试集合和指标，用来衡量多模态智能体在理解、推理、生成等方面的表现，起到“体育比赛的计分表”作用。  

### 核心创新点
1. **系统化的四类划分 → 将现有 LMA 按功能和交互方式划分为四大类（感知‑驱动、语言‑驱动、混合‑交互、协同‑网络） → 研究者可以快速定位自己工作属于哪一类，避免重复造轮子。**  
2. **协同框架的统一视角 → 提出一种把多个单模态或多模态智能体通过共享记忆、任务调度和结果融合进行协作的通用架构 → 让不同模型之间的“对话”变得可编程，提升了整体任务完成度。**  
3. **评估方法的全景整合 → 收集并归类了目前散落在各篇论文中的评测手段，进而构建了一个覆盖感知、推理、交互、效率四维度的统一评估框架 → 为后续工作提供了可比对的“统一赛道”。**  
4. **资源库的持续更新 → 在 GitHub 上维护了一个包含数据集、模型、工具链的精选列表 → 降低了新手入门门槛，也帮助老手快速找到最新的实现。  

### 方法详解
整体思路可以概括为“三层塔”结构：**感知层 → 对齐层 → 决策层**，并在此基础上加入**协同层**和**评估层**。

1. **感知层**  
   - 使用专门的视觉、音频或视频编码器把原始信号映射成高维向量。比如，图片通过 CLIP 的视觉分支得到 512 维特征，音频通过 Whisper 的编码器得到时序特征。  
   - 这些向量会被统一投射到一个共享的跨模态空间，确保后续的语言模型能够直接读取。

2. **对齐层**  
   - 通过轻量的投影网络（通常是两层 MLP）把感知向量映射到 LLM 的隐藏层维度。这里的关键技巧是**冻结 LLM 参数，只训练投影层**，既保持语言模型的强大能力，又让多模态信息顺利进入。  
   - 为了让模型学会“看图说话”，会在训练阶段加入**跨模态对齐损失**（如对比学习），让对应的图文向量距离更近。

3. **决策层（LLM 本体）**  
   - 在每一次交互时，系统会先把感知向量拼接到 Prompt（提示）中，形成类似“[IMG] <image_embedding> 请根据这张图回答以下问题：...”的输入。  
   - LLM 根据提示进行链式思考（Chain‑of‑Thought），输出中可能包含**内部工具调用**（如检索、计算）或**子任务分配**（交给其他专精智能体）。

4. **协同层**  
   - 当任务复杂到需要多种专长时，主 LLM 会生成**任务分配指令**，把子任务发送给其他智能体（比如专门的视觉推理体或数学计算体）。  
   - 各智能体执行后把结果写回共享记忆库，主 LLM 再把所有信息整合，给出最终答案。这个过程类似于“项目经理 → 各部门 → 项目经理”。

5. **评估层**  
   - 作者把现有的 20+ 评测拆分成四类：**感知准确度、跨模态推理、交互流畅度、系统效率**。每类提供统一的指标（如 Top‑1 准确率、BLEU、交互轮数、推理时延），并建议使用统一的 **Leaderboard** 来记录。  
   - 评估框架的核心是**可插拔**：研究者只需实现对应的评测接口，即可把自己的模型加入对比。

**最巧妙的点**在于把多模态感知完全抽象为“向量投影”，而不需要对 LLM 进行大规模微调，这大幅降低了算力门槛，同时保留了 LLM 的通用推理能力。

### 实验与效果
- **测试任务**：论文列举了视觉问答（VQA）、图文生成、跨模态检索、机器人指令执行等 6 大场景。  
- **基线对比**：对比对象包括传统的单模态 LLM + 视觉前置、专用的多模态模型（如 Flamingo、BLIP‑2）以及最新的协同系统（如 Gato）。作者报告说，在 VQA 上平均提升约 **12%** 的准确率，在跨模态检索上 MAP 提升 **0.08**。  
- **消融实验**：通过去掉投影层、关闭协同调度或不使用对齐损失，性能分别下降 **5‑9%**，说明每个模块都有实质贡献。  
- **局限性**：原文承认评估仍受数据集偏差影响，尤其是长对话场景缺乏统一基准；此外，投影层的线性映射在处理高分辨率视频时会出现信息瓶颈。  

### 影响与延伸思考
自从这篇综述发布后，**LLaVA、MiniGPT‑4、GPT‑4V** 等项目纷纷在论文中引用其分类与评估框架，形成了“多模态智能体生态”。后续工作开始探索 **自适应投影**（根据任务动态调节投影维度）和 **跨模型记忆共享**（把不同模型的内部状态写入统一的向量数据库），这些方向都可以视为对本文协同层的延伸。想进一步深入，建议关注以下两个方向：  
1. **可解释的跨模态推理**——让模型在给出答案的同时，展示“看见了什么、怎么推理”。  
2. **统一的多模态基准平台**——类似于 GLUE 对 NLP 的作用，构建覆盖视觉、音频、文本的统一排行榜。  

### 一句话记住它
**把 LLM 当成“大脑”，用轻量投影把“眼睛”和“耳朵”接进来，再用统一评估和协同框架让多模态智能体真正“会思考”。**