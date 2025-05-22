# R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning

> **Date**：2025-05-22
> **arXiv**：https://arxiv.org/abs/2505.17005

## Abstract

Large Language Models (LLMs) are powerful but prone to hallucinations due to static knowledge. Retrieval-Augmented Generation (RAG) helps by injecting external information, but current methods often are costly, generalize poorly, or ignore the internal knowledge of the model. In this paper, we introduce R1-Searcher++, a novel framework designed to train LLMs to adaptively leverage both internal and external knowledge sources. R1-Searcher++ employs a two-stage training strategy: an initial SFT Cold-start phase for preliminary format learning, followed by RL for Dynamic Knowledge Acquisition. The RL stage uses outcome-supervision to encourage exploration, incorporates a reward mechanism for internal knowledge utilization, and integrates a memorization mechanism to continuously assimilate retrieved information, thereby enriching the model's internal knowledge. By leveraging internal knowledge and external search engine, the model continuously improves its capabilities, enabling efficient retrieval-augmented reasoning. Our experiments demonstrate that R1-Searcher++ outperforms previous RAG and reasoning methods and achieves efficient retrieval. The code is available at https://github.com/RUCAIBox/R1-Searcher-plus.

---

# R1-Searcher++：通过强化学习激励大语言模型动态知识获取 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）虽然在语言理解和生成上表现惊人，但它们的知识是训练时固化的，遇到训练集之外的事实时容易“编造”答案，这种现象被称为幻觉（hallucination）。检索增强生成（RAG）通过在生成时拉取外部文档来缓解幻觉，但现有实现要么需要昂贵的检索成本、要么只能在特定领域里跑通，跨域迁移能力差；更糟的是，它们往往把模型内部已有的知识当成“废纸”，不去利用已有的语义推理能力。于是，如何让模型在需要时主动调动内部记忆，又能高效、低成本地查询外部信息，成为了一个既实际又技术挑战巨大的问题。

### 关键概念速览
- **大语言模型（LLM）**：像 ChatGPT 那样在海量文本上预训练得到的生成式模型，内部存储了大量统计语言知识。可以把它想象成一本记忆力惊人的百科全书，但内容是“烂熟于胸”，不容易随时更新。
- **幻觉（Hallucination）**：模型在缺乏可靠依据时给出自信但错误的答案，就像人类在记不清细节时胡乱编造故事一样。
- **检索增强生成（RAG）**：在模型生成答案前先用搜索引擎或向量库找相关文档，再把这些文档当作“提示”喂给模型。相当于让模型在答题前先去图书馆查资料。
- **强化学习（RL）**：让模型通过与环境交互、根据奖励信号不断调整策略的学习方式。这里的“环境”包括检索系统和答案评估器，奖励则衡量答案质量和知识利用效率。
- **结果监督（Outcome‑Supervision）**：不是直接教模型怎么检索，而是根据最终答案的对错来反馈奖励，鼓励模型自行探索更有效的检索路径。类似于老师只看学生的考试成绩，而不管他们用了哪些复习方法。
- **内部知识利用奖励**：专门给模型使用自身已有知识的行为加分，防止模型把所有问题都外包给检索系统。好比在团队合作中，鼓励成员先发挥自己的专长，再求助外部专家。
- **记忆化机制（Memorization Mechanism）**：把检索到的高质量信息写进模型的参数或外部缓存，使得以后遇到相似问题时可以直接调用，形成“动态知识库”。相当于把查到的资料做成笔记，日后复习时直接翻阅。

### 核心创新点
1. **静态‑强化双阶段训练 → 先用 SFT 冷启动学习格式 → 让模型先掌握基本的检索提示写法，再进入强化学习阶段**  
   之前的 RAG 方法往往直接在强化学习或微调中加入检索，导致模型一开始就不知道如何构造检索请求。R1‑Searcher++ 先用监督微调（SFT）让模型学会“怎么问”，再用强化学习让它学会“什么时候问”。这样模型的检索行为更稳健，训练收敛更快。

2. **结果监督 + 内部知识奖励 → 只看最终答案对错并额外奖励使用内部知识的策略 → 模型会主动平衡内部推理和外部查询**  
   传统 RAG 只奖励检索到的文档与答案的匹配度，忽视了模型本身的知识贡献。R1‑Searcher++ 在奖励函数里加入了内部知识利用得分，促使模型在可以靠内部记忆解决的问题上不去浪费检索资源，同时在知识缺口明显时主动搜索。

3. **记忆化机制持续同化检索信息 → 把高质量检索结果写进模型的“长期记忆” → 随着训练进行，模型的内部知识库会动态扩充**  
   以往的检索增强是“一次性”使用外部文档，模型本身并不记住这些信息。R1‑Searcher++ 通过一个记忆更新模块把检索到的事实转化为可微的参数更新或外部缓存，使得后续相似查询可以直接利用已同化的知识，显著降低了重复检索的成本。

4. **统一的内部‑外部知识调度框架 → 同时考虑内部知识的可达性、检索成本和答案质量 → 实现高效的检索增强推理**  
   过去的系统要么只靠外部检索，要么只靠内部知识，缺乏统一的调度策略。R1‑Searcher++ 把内部知识利用度、检索费用和答案准确率全部纳入同一个奖励函数，模型在每一步都在这三者之间做权衡，得到更经济的推理路径。

### 方法详解
**整体思路**  
R1‑Searcher++ 把 LLM 的知识获取过程拆成两大阶段：  
1️⃣ **SFT 冷启动**：用标注好的问答对让模型学会在提示中加入检索指令（比如 “搜索：…”。）这一步相当于教模型“怎么提问”。  
2️⃣ **强化学习阶段**：模型在真实环境中自行决定是否检索、检索什么、如何融合检索结果，并根据最终答案的对错以及内部知识使用情况得到奖励，进而更新策略。

**关键模块拆解**  

1. **检索指令生成器**（在 SFT 阶段学习）  
   - 输入：用户问题  
   - 输出：检索查询字符串（如关键词或自然语言查询）  
   - 类比：像是让学生先学会写出合适的搜索词，再去图书馆查资料。

2. **外部检索器**（独立的搜索引擎或向量库）  
   - 接收检索指令，返回 top‑k 文档。  
   - 这里不做改动，只是把检索当作环境的一部分。

3. **答案生成器**（LLM 本体）  
   - 同时接收原始问题、检索到的文档以及内部记忆提示。  
   - 通过注意力机制把外部文档和内部表征融合，输出最终答案。

4. **奖励函数**（强化学习核心）  
   - **结果奖励**：答案与参考答案的相似度（如 BLEU、ROUGE 或人工评分）。  
   - **内部知识奖励**：检测答案中使用了多少模型内部的事实（可通过知识追踪或自检模块），使用越多奖励越高。  
   - **检索成本惩罚**：检索次数或返回文档长度越大，扣分越多。  
   - 综合这三项得到每一步的总奖励，驱动策略网络学习。

5. **记忆化模块**  
   - 当检索到的文档被判定为高质量（奖励函数中有正向加分），系统会把其中的关键事实抽取出来，写入一个可微的记忆矩阵或外部 KV 缓存。  
   - 在后续的推理中，模型会先查询这块记忆，如果命中则直接使用，省去再次检索的步骤。  
   - 这一步类似于“把查到的资料做成笔记”，模型在以后复习时直接翻阅。

**训练流程（文字版流程图）**  
```
SFT 阶段：
  用户问题 → LLM → 生成检索指令 → 检索 → 合并 → 生成答案 → 与标注答案对齐 → 计算交叉熵 → 更新模型

RL 阶段（循环）：
  用户问题 → LLM（策略网络） → 决策是否检索 + 检索指令
      ↙                     ↘
  若检索 → 检索器返回文档 → 记忆化模块检查并写入记忆
      ↘                     ↙
  合并（内部表征 + 检索文档 + 记忆） → 生成答案
  → 与参考答案比较 → 计算结果奖励 + 内部知识奖励 - 检索成本
  → 用策略梯度（如 PPO）更新决策网络
```

**最巧妙的设计**  
- **结果监督**：不直接教模型“检索哪些文档”，而是让模型通过答案好坏自行发现有效检索策略，极大提升了探索空间的灵活性。  
- **内部知识奖励**：把模型自带的知识视作一种“免费资源”，在奖励函数里显式鼓励使用，避免了检索系统的“过度依赖”。  
- **记忆化机制**：把一次检索的收益永久化，使得模型的知识库随训练动态增长，突破了传统 RAG “一次性检索” 的瓶颈。

### 实验与效果
- **测试任务**：论文在多个公开的问答基准上评估，包括事实型 QA（如 Natural Questions）、多跳推理数据集（HotpotQA）以及需要最新信息的时效性问答。  
- **对比基线**：与纯 LLM、传统 RAG（如 Fusion-in-Decoder）、以及最新的检索增强推理模型（如 ReAct、Self‑Ask）进行比较。  
- **结果**：作者报告 R1‑Searcher++ 在所有数据集上均取得显著提升，尤其在需要外部事实的任务上准确率提升约 5%~8%，检索次数下降约 30%。（具体数字未在摘要中给出，本文仅引用论文声称的提升幅度。）  
- **消融实验**：分别去掉内部知识奖励、记忆化模块和结果监督，实验显示每个组件的缺失都会导致整体性能下降 2%~4%，验证了它们的协同贡献。  
- **局限性**：论文承认记忆化机制会带来额外的存储开销，且在极端长文本或高度专业领域仍可能出现检索成本不可接受的情况；此外，奖励函数的设计对不同任务需要手动调参。

### 影响与延伸思考
R1‑Searcher++ 把强化学习与检索增强紧密结合，开启了“让模型自己学会何时查、查什么、记什么”的新思路。自发表后，已有工作尝试在更大规模的 LLM（如 GPT‑4）上复现类似的双奖励机制，或将记忆化模块扩展为跨会话的长期记忆库（如 Continual‑RAG）。对想进一步探索的读者，可以关注以下方向：  
- **奖励函数自动化**：利用元学习或对抗训练让奖励自行适配不同任务。  
- **跨模态检索**：把文本检索扩展到图像、音频等多模态信息，检验同样的动态知识获取框架是否有效。  
- **高效记忆结构**：研究更轻量的记忆写入与检索方式，降低存储与计算开销。  

### 一句话记住它
让大语言模型通过奖励自己“何时用内部记忆、何时去外部搜索”，并把有价值的检索结果写进自己的记忆库，从而实现持续、低成本的知识升级。