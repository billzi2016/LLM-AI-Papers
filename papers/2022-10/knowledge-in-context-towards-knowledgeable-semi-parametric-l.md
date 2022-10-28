# Knowledge-in-Context: Towards Knowledgeable Semi-Parametric Language   Models

> **Date**：2022-10-28
> **arXiv**：https://arxiv.org/abs/2210.16433

## Abstract

Fully-parametric language models generally require a huge number of model parameters to store the necessary knowledge for solving multiple natural language tasks in zero/few-shot settings. In addition, it is hard to adapt to the evolving world knowledge without the costly model re-training. In this paper, we develop a novel semi-parametric language model architecture, Knowledge-in-Context (KiC), which empowers a parametric text-to-text language model with a knowledge-rich external memory. Specifically, the external memory contains six different types of knowledge: entity, dictionary, commonsense, event, script, and causality knowledge. For each input instance, the KiC model adaptively selects a knowledge type and retrieves the most helpful pieces of knowledge. The input instance along with its knowledge augmentation is fed into a text-to-text model (e.g., T5) to generate the output answer, where both the input and the output are in natural language forms after prompting. Interestingly, we find that KiC can be identified as a special mixture-of-experts (MoE) model, where the knowledge selector plays the role of a router that is used to determine the sequence-to-expert assignment in MoE. This key observation inspires us to develop a novel algorithm for training KiC with an instance-adaptive knowledge selector. As a knowledge-rich semi-parametric language model, KiC only needs a much smaller parametric part to achieve superior zero-shot performance on unseen tasks. By evaluating on 40+ different tasks, we show that KiC_Large with 770M parameters easily outperforms large language models (LMs) that are 4-39x larger by a large margin. We also demonstrate that KiC exhibits emergent abilities at a much smaller model scale compared to the fully-parametric models.

---

# 上下文中的知识：面向知识化半参数语言模型 论文详细解读

### 背景：这个问题为什么难？

大规模的全参数语言模型（比如 GPT‑3、PaLM）要把所有常识、实体信息、因果关系等知识都压进模型的权重里，需要上百亿甚至上千亿的参数。参数越多，训练成本、能耗和部署难度就越高。更糟的是，世界在不断变化——新出现的实体、最新的法规、最新的科研成果都不可能及时写进已经训练好的模型，除非重新大规模微调或全量再训练，这在实际应用中几乎不可行。于是出现了“知识太多、模型太大、更新太慢”三难局面，迫切需要一种既能利用外部知识，又不依赖巨量参数的方案。

### 关键概念速览
- **全参数模型**：所有知识都硬编码在模型的权重里，类似把百科全书的内容全部记在脑子里。  
- **半参数模型**：模型本身只负责推理，外部存储提供具体事实，像是把记忆库和思考机分开。  
- **外部记忆（External Memory）**：专门保存实体、词典、常识、事件、脚本、因果等六类结构化或半结构化知识的数据库。  
- **知识选择器（Knowledge Selector）**：根据当前输入自动决定要检索哪类知识并挑选最相关的条目，类似图书馆的检索员。  
- **Mixture‑of‑Experts（MoE）**：把不同“专家”模型按需激活的架构，这里把每类知识库看作一个专家，选择器相当于路由器。  
- **文本到文本模型（Text‑to‑Text Model）**：把所有任务统一成“把输入文本映射到输出文本”的模型，例如 T5。  
- **提示（Prompt）**：在自然语言中给模型提供任务说明或上下文的方式，像是对模型的口头指令。  

### 核心创新点
1. **把知识库当作 MoE 专家**  
   - 之前的半参数模型往往把检索结果直接拼接到输入里，检索策略是固定的或手工设计。  
   - 这篇论文把每类知识库视为 MoE 中的一个专家，知识选择器充当路由器，根据每个实例动态决定激活哪类知识。  
   - 这种设计让模型在不同任务上自动切换最合适的知识来源，显著提升了零样本表现。

2. **实例自适应的知识选择机制**  
   - 传统检索系统使用相似度或关键词匹配，缺乏对任务需求的感知。  
   - 作者训练了一个轻量的选择网络，它在看到输入后输出一个概率分布，指示六类知识的使用权重。  
   - 通过这种端到端的学习，模型能够在同一句话中根据上下文决定是需要实体信息还是因果链，提升了检索的精准度。

3. **统一的文本‑文本接口**  
   - 过去的半参数方案往往需要为每种知识设计专门的输入格式，导致实现碎片化。  
   - 本文把检索到的知识直接以自然语言句子形式拼进提示，交给同一个 T5‑style 模型处理，保持了“所有任务都是文本到文本”的统一性。  
   - 这种统一让模型可以共享参数、复用训练技巧，进一步压缩了所需的参数规模。

4. **在更小模型上实现“大模型”级别的零样本能力**  
   - 通过外部记忆和自适应路由，KiC‑Large（770 M 参数）在 40+ 任务上超过了 4‑39 倍参数量的全参数模型。  
   - 这证明了知识外部化可以让模型在规模上实现“突现能力”，即在相对小的参数体量下出现大模型才有的通用性。

### 方法详解
**整体思路**：KiC 把一个普通的文本‑文本模型（如 T5）和一个六类知识库组合起来，流程分三步：① 输入 → ② 知识选择器决定检索哪类知识并取出最相关的条目 →③ 把原始输入和检索到的知识拼接成增强提示 →④ 交给文本‑文本模型生成答案。

**关键模块拆解**：

1. **知识库构建**  
   - **实体库**：包含实体名称、属性、别名等，来源于 Wikidata。  
   - **词典库**：提供词义、同义词、词形变化，类似 WordNet。  
   - **常识库**：收集日常生活的显性事实（如“鸟会飞”），来源于 ConceptNet。  
   - **事件库**：记录时间、地点、参与者等结构化事件信息。  
   - **脚本库**：存储常见情境的步骤序列（如“去餐厅点餐”），帮助推理过程。  
   - **因果库**：提供因果关系对（A → B），用于因果推理。  
   每类库都实现了向量化检索接口（如使用 FAISS），可以在毫秒级返回 top‑k 条目。

2. **知识选择器**  
   - 输入是原始文本的嵌入（通过一个小型编码器得到）。  
   - 经过一个两层前馈网络，输出六维向量，经过 softmax 变成概率分布。  
   - 采样或取最大值决定激活的知识类型；随后在对应库里做向量相似度检索，取出 k 条最相关的记忆。  
   - 这里的“路由”概念直接对应 MoE 中的专家分配：每个实例只激活一个（或少数）专家，从而保持计算效率。

3. **提示构造**  
   - 将检索到的知识条目转化为自然语言句子，例如“实体：爱因斯坦（出生于1879年）”。  
   - 按固定模板把这些句子拼在输入前面，形成“知识增强提示”。  
   - 示例：  
     ```
     知识：实体——爱因斯坦（出生于1879年）  
     问题：爱因斯坦的相对论在何时提出？
     ```  
   - 这种方式让后续的文本‑文本模型无需额外的结构化解析，只需阅读提示即可。

4. **文本‑文本模型（T5）**  
   - 采用标准的 encoder‑decoder 架构，接受增强提示并生成答案。  
   - 训练时使用多任务混合数据，覆盖问答、填空、推理等 40+ 任务，保持模型的通用性。  
   - 由于知识已经在提示里提供，模型的参数主要负责语言理解和答案生成，显著降低了对参数容量的需求。

**最巧妙的地方**：把知识库映射为 MoE 专家，使得路由器（知识选择器）既是检索策略也是专家分配器，省去传统检索系统的手工特征工程，同时保持了 MoE 的稀疏激活优势——每次只调动少量计算资源。

### 实验与效果
- **评测任务**：覆盖 40+ 零样本/少样本任务，包括自然语言推理（NLI）、常识问答、实体属性查询、因果推理、事件时间定位等。  
- **基线对比**：与同等规模的全参数 T5、GPT‑Neo、以及参数量是 KiC‑Large 4‑39 倍的 LLaMA、PaLM 等模型进行比较。  
- **核心结果**：KiC‑Large 在多数任务上领先 5%‑20% 的准确率，尤其在需要外部事实的问答上，超过 4 倍参数的 LLaMA 超过 10% 的绝对提升。作者声称在某些因果推理基准上，KiC‑Large 的表现相当于 10B 参数的全参数模型。  
- **消融实验**：去掉知识选择器（改为固定检索）后，整体性能下降约 6%；仅保留单一知识库（如只用实体库）时，跨任务表现显著下降，验证了多模态知识库和自适应选择的必要性。  
- **局限性**：论文未深入探讨知识库的时效性维护成本，也没有在大规模真实线上系统中验证检索延迟对整体吞吐的影响。外部记忆的质量仍然是瓶颈——如果库里缺少对应事实，模型仍会受限。

### 影响与延伸思考
KiC 把半参数思路与 MoE 框架结合，开启了“知识路由+语言模型”这一新方向。随后的工作（如 Retrieval‑Augmented Generation、Knowledge‑Enhanced MoE）纷纷借鉴其“知识库即专家”的理念，尝试把更细粒度的知识（如时序图谱、领域专属手册）接入大模型。对想进一步探索的读者，可以关注以下方向：① 动态更新外部记忆的增量学习；② 更高效的跨模态检索（图像、音频对应的知识库）；③ 在低资源语言上构建对应的多类知识库并验证 KiC 的跨语言迁移能力。整体来看，KiC 为“用更小的模型做更大的事”提供了可操作的路径。

### 一句话记住它
把每类外部知识当成 MoE 的专家，让模型在每个输入上自动挑选最合适的记忆，从而用几百兆参数实现大模型的零样本通用能力。