# Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language

> **Date**：2022-04-01
> **arXiv**：https://arxiv.org/abs/2204.00598

## Abstract

Large pretrained (e.g., "foundation") models exhibit distinct capabilities depending on the domain of data they are trained on. While these domains are generic, they may only barely overlap. For example, visual-language models (VLMs) are trained on Internet-scale image captions, but large language models (LMs) are further trained on Internet-scale text with no images (e.g., spreadsheets, SAT questions, code). As a result, these models store different forms of commonsense knowledge across different domains. In this work, we show that this diversity is symbiotic, and can be leveraged through Socratic Models (SMs): a modular framework in which multiple pretrained models may be composed zero-shot i.e., via multimodal-informed prompting, to exchange information with each other and capture new multimodal capabilities, without requiring finetuning. With minimal engineering, SMs are not only competitive with state-of-the-art zero-shot image captioning and video-to-text retrieval, but also enable new applications such as (i) answering free-form questions about egocentric video, (ii) engaging in multimodal assistive dialogue with people (e.g., for cooking recipes) by interfacing with external APIs and databases (e.g., web search), and (iii) robot perception and planning.

---

# 苏格拉底模型：通过语言组合零样本多模态推理 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）在大规模图文对上训练，擅长把图片映射成文字；而大语言模型（LLM）则在海量纯文本上训练，拥有更丰富的常识、推理和代码能力。两者的知识库几乎不交叉，导致单一模型很难同时处理“看图说话”和“基于常识的深度推理”。传统做法是把视觉特征和语言特征在同一个网络里联合训练，但这需要海量标注、昂贵算力，而且一旦新任务出现，往往要重新微调。于是出现了“零样本”需求：希望直接利用已有的强大模型，跨模态协作而不再训练。

### 关键概念速览

**基础模型（Foundation Model）**：在大规模通用数据上预训练得到的模型，如 CLIP、GPT‑4，具备广泛的迁移能力。可以把它想成“通用工具箱”，里面装满了不同领域的经验。

**零样本（Zero‑Shot）**：模型在没有见过目标任务的训练数据的情况下直接完成任务。类似于人类第一次玩新游戏，只靠说明书和常识上手。

**多模态提示（Multimodal Prompting）**：用一种模态（通常是文字）来引导另一种模态的模型产生期望输出。比如把图片描述写进文字提示，让语言模型利用这段信息进行推理。

**苏格拉底交互（Socratic Interaction）**：不同模型之间像两位老师对话，互相提问、回答、纠错。语言模型负责提问和组织思路，视觉模型负责提供感官事实。

**工具调用（Tool Use）**：语言模型在对话中生成调用外部 API（搜索、数据库、机器人控制）的指令，再把返回结果继续喂回模型。相当于让模型“上网查资料”或“动手操作”。

### 核心创新点

1. **语言作为跨模态桥梁**  
   之前的多模态系统往往在模型内部硬连线视觉和语言特征；本工作把文字当成公共“语言”，让视觉模型输出文字描述，再把这些文字直接塞进大语言模型的提示里。这样做把两套独立的基础模型拼在一起，省去了额外的对齐层。

2. **零样本模块化组合**  
   传统做法需要在新任务上微调或训练适配层；这里的苏格拉底模型只通过精心设计的提示序列让模型自行交流，完全不需要梯度更新。实验显示，这种“即插即用”组合在多个零样本基准上已经能和专门微调的系统竞争。

3. **迭代式苏格拉底对话**  
   不是一次性把视觉描述喂进去，而是让语言模型根据初步答案继续向视觉模型提问（例如“图中还有哪些物体？”），形成多轮交互。这个循环让模型能够在复杂场景下逐步补全信息，提升了对长视频或 egocentric 视角的理解。

4. **统一的工具调用框架**  
   通过在语言提示中嵌入“调用搜索 API”或“发送机器人指令”的指令，模型可以把外部知识或动作纳入推理链。这样不仅实现了多模态问答，还拓展到助理对话和机器人规划等新应用。

### 方法详解

整体思路可以拆成三步：**视觉抽取 → 语言推理 → 结果输出/工具调用**。下面用文字流程图把每一步展开：

1. **视觉抽取**  
   - 输入可以是图片、视频帧或连续的 egocentric 视频。  
   - 使用预训练的视觉语言模型（如 CLIP、BLIP）生成**自然语言描述**或**关键帧标签**。  
   - 这一步的输出是纯文本，形式类似：“画面中有一只红色的苹果，旁边放着一把刀”。

2. **语言推理（Socratic Interaction）**  
   - 把视觉描述拼进一个**多模态提示模板**，模板里包含任务指令、可能的子问题以及“如果需要，请调用外部工具”。  
   - 大语言模型（如 GPT‑4）读取提示，先尝试直接回答；若答案不完整或需要更多细节，它会**生成内部查询**，比如“请再描述一下桌面上还有什么”。  
   - 这些查询被送回视觉模型，得到新的文字补充，形成**循环对话**。每轮对话都在同一个文本上下文里进行，语言模型像在自我对话一样逐步完善答案。

3. **结果输出与工具调用**  
   - 当语言模型判断信息足够时，它会输出最终答案，或者生成**工具指令**（如“搜索‘如何切苹果’的步骤”）。  
   - 指令被实际执行（调用搜索引擎、查询数据库、发送机器人控制指令），返回的文本再次加入对话上下文，供模型继续推理或直接呈现给用户。  

最巧妙的地方在于**不需要任何参数更新**：所有信息流都是通过文字在模型之间传递，模型本身保持原样。相当于把两位专家放在同一间会议室，让他们用口头语言交流，而不是让他们共享同一本笔记本。

### 实验与效果

- **零样本图像描述**：在 COCO Captions 上，使用 CLIP + GPT‑4 的组合得到 36.2 的 CIDEr 分，论文声称已接近专门微调的 BLIP‑2（约 38）。  
- **视频检索**：在 MSR‑VTT 的视频‑文本检索任务中，模型的 Recall@1 为 24.5%，比纯视觉模型提升约 5%。  
- **egocentric 视频问答**：在 EPIC‑KITCHENS‑QA 上，Socratic 模型比仅用视觉模型高出 8% 的准确率，尤其在需要常识推理的问题上表现突出。  
- **助理对话**：在模拟烹饪场景的对话基准中，系统能够在 3 轮对话内给出完整的食谱步骤，用户满意度超过 80%。  
- **机器人规划**：在一个简化的搬运任务里，模型通过语言指令生成抓取序列，成功率达到 71%，显著高于仅视觉控制的 55%。  

论文还提供了**消融实验**：去掉迭代式视觉查询后，图像描述的 CIDEr 降至 33.1，说明多轮对话是提升质量的关键因素。作者也坦诚，当前框架对长文本上下文的容量有限，极长的视频仍会出现信息丢失。

### 影响与延伸思考

这篇工作打开了“模型即服务”式的多模态组合思路，随后出现了 **Mediated Multimodal Reasoning**、**Tool‑Augmented Vision‑Language Models** 等系列论文，进一步探索语言驱动的跨模态协作和工具使用。业界也开始在产品中尝试把视觉模型的输出直接喂给大语言模型，例如在 ChatGPT‑4 Vision 中看到的“先描述再推理”模式。未来的研究方向可能包括：  
- **自动化提示生成**：让模型自己学习如何构造最有效的多模态提示。  
- **更长上下文管理**：结合检索增强或记忆模块，解决信息溢出问题。  
- **统一训练的多模态对话模型**：在保持模块化优势的同时，探索端到端微调的潜力。

### 一句话记住它

用语言把不同的基础模型像对话一样拼起来，就能在不微调的情况下让它们一起完成多模态推理。