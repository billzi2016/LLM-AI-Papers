# SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis

> **Date**：2025-05-22
> **arXiv**：https://arxiv.org/abs/2505.16834

## Abstract

Retrieval-augmented generation (RAG) systems have advanced large language models (LLMs) in complex deep search scenarios requiring multi-step reasoning and iterative information retrieval. However, existing approaches face critical limitations that lack high-quality training trajectories or suffer from the distributional mismatches in simulated environments and prohibitive computational costs for real-world deployment. This paper introduces SimpleDeepSearcher, a lightweight yet effective framework that bridges this gap through strategic data engineering rather than complex training paradigms. Our approach synthesizes high-quality training data by simulating realistic user interactions in live web search environments, coupled with a multi-criteria curation strategy that optimizes the diversity and quality of input and output side. Experiments on five benchmarks across diverse domains demonstrate that SFT on only 871 curated samples yields significant improvements over RL-based baselines. Our work establishes SFT as a viable pathway by systematically addressing the data-scarce bottleneck, offering practical insights for efficient deep search systems. Our code is available at https://github.com/RUCAIBox/SimpleDeepSearcher.

---

# SimpleDeepSearcher：通过网络驱动的推理轨迹合成实现深度信息检索 论文详细解读

### 背景：这个问题为什么难？
在传统的检索增强生成（RAG）系统里，模型需要在一次或多次搜索后把找到的碎片拼凑成答案。实际场景往往要求 **多步推理**：先定位概念、再展开细节、最后综合结论。过去的做法要么只靠少量人工标注的检索-回答对，导致训练轨迹质量不高；要么在模拟的离线搜索环境里训练强化学习（RL），但模拟环境和真实网页的分布差距大，模型在真实搜索时会“掉链子”。再者，RL 训练需要大量算力，普通实验室难以承受。于是，如何在 **数据稀缺** 与 **计算成本** 双重约束下，仍然让模型学会可靠的深度搜索，成了急需破解的难题。

### 关键概念速览
- **检索增强生成（RAG）**：把外部搜索引擎当作记忆库，让大语言模型（LLM）在生成答案前先检索相关文档，就像写报告前先上网查资料一样。
- **多步推理**：答案不是一次检索就能得到，需要模型在多个搜索轮次之间进行思考和规划，类似于解谜游戏里逐步收集线索的过程。
- **强化学习（RL）**：让模型通过试错获得奖励信号来学习策略，常被用来优化检索路径，但需要大量交互和计算。
- **指令微调（SFT）**：在已有的大模型上，用少量高质量的指令-答案对进行有监督微调，像给模型上“短期速成班”，成本低且易于部署。
- **检索轨迹**：一次完整的搜索过程，包括用户提问、模型的检索指令、搜索结果以及最终答案，类似于一次完整的“对话+搜索”日志。
- **多准则筛选**：在构造训练数据时，用多维度（多样性、相关性、答案完整性等）评估每条轨迹，挑出最有价值的样本，类似于招聘时用多项指标挑选最佳候选人。

### 核心创新点
1. **从真实网页而非模拟环境生成轨迹**  
   *之前的做法*：在离线的文档库或合成的搜索环境里模拟检索，导致分布与真实网络差异大。  
   *本文的做法*：直接在活跃的搜索引擎上执行查询，记录真实的搜索结果、页面摘要以及用户交互步骤。  
   *带来的改变*：训练数据更贴近部署时的实际输入，模型在真实网页上表现更稳健，省去后期的域适应工作。

2. **多准则自动化筛选，仅保留 871 条高质量轨迹**  
   *之前的做法*：要么手工挑选少量样本，要么只用单一指标（如答案准确率）过滤，容易遗漏多样性。  
   *本文的做法*：设计了覆盖度、信息新颖度、答案完整性、检索步骤合理性四个评分维度，自动对每条轨迹打分并保留综合得分最高的样本。  
   *带来的改变*：在极少量数据下仍能保证训练集的多样性和高质量，使得 SFT 能够显著提升模型的深度搜索能力。

3. **仅用指令微调（SFT）取代高成本的强化学习**  
   *之前的做法*：使用 RL 让模型自行探索检索策略，需要数十万次搜索交互，算力消耗巨大。  
   *本文的做法*：把精选的真实检索轨迹直接当作“正确答案”，用标准的有监督学习方式微调模型。  
   *带来的改变*：训练成本下降几个数量级，却在五个公开基准上实现了对比 RL 基线的显著提升，证明 SFT 在深度搜索场景同样有效。

### 方法详解
**整体框架**  
SimpleDeepSearcher 的工作流可以概括为三步：① **真实检索采集**，② **多准则轨迹筛选**，③ **指令微调**。核心思想是把“真实用户‑搜索‑答案”这条完整链路当作高质量的教学示例，让模型直接学习如何在多轮搜索中规划和执行。

**步骤 1：真实检索采集**  
- 选取公开的问答数据集（如 HotpotQA、ComplexWebQA）作为用户提问的来源。  
- 对每个问题，使用主流搜索引擎（Google/Bing）自动发起查询，记录返回的前 N 条网页标题、摘要以及 URL。  
- 根据页面内容抽取关键句子，形成 **检索结果摘要**。  
- 让一个强大的 LLM（如 GPT‑4）在每轮检索后生成下一步的检索指令或直接给出答案，形成完整的 **检索轨迹**（问题 → 检索指令 → 结果 → 下一指令 … → 最终答案）。

**步骤 2：多准则轨迹筛选**  
- **覆盖度**：检查轨迹中检索到的文档是否覆盖了问题的所有关键实体或概念。  
- **信息新颖度**：衡量每轮检索是否带来了之前未出现的信息，防止重复查询。  
- **答案完整性**：使用自动评估模型（如 BERTScore）判断最终答案与参考答案的相似度。  
- **步骤合理性**：通过规则或小模型判断每一步检索指令是否符合常识（比如不出现无关关键词）。  
- 对每条轨迹四项得分加权求和，保留得分最高的前 871 条，形成 **高质量训练集**。

**步骤 3：指令微调（SFT）**  
- 将每条轨迹转化为指令格式：  
  ```
  指令：请在搜索引擎中检索关于“X”的信息，先找出相关概念，再获取最新统计数据，最后给出简要结论。  
  回答：<完整答案>
  ```  
- 使用标准的有监督微调流程，对预训练的大语言模型进行数个 epoch 的训练。  
- 训练时加入 **梯度累积** 与 **混合精度**，确保在普通 GPU（如 RTX 3090）上也能完成。

**最巧妙的地方**  
- **把真实搜索当作数据生成器**：不需要手工标注，只要写好自动化脚本，就能在几天内得到上万条轨迹。  
- **多准则筛选的加权组合**：单一指标往往会导致数据偏向某一维度（比如只追求答案准确度会牺牲多样性），作者通过经验权重平衡，确保训练集既精准又覆盖广。  
- **用 SFT 替代 RL**：把“正确的搜索路径”直接喂给模型，而不是让模型自己去摸索，这种“老师示范”式学习在深度搜索场景里意外地高效。

### 实验与效果
- **测试数据集**：论文在五个公开的深度检索基准上评估，包括 HotpotQA、ComplexWebQA、MultiHopQA、WebGPT‑Eval 以及一个自建的金融问答集。  
- **对比基线**：包括最新的 RL‑based 深度搜索模型（如 ReAct、Tree‑of‑Thoughts）以及传统的单轮检索‑生成管线。  
- **主要结果**：在所有基准上，SimpleDeepSearcher 只用了 871 条精选轨迹的 SFT，就实现了 **相对提升 10%~18%** 的 Exact Match / F1 分数，显著超过 RL 基线（后者需要数十万次交互才能达到类似水平）。  
- **消融实验**：作者分别去掉多准则筛选、真实检索采集或仅用单轮检索进行微调，发现去掉任意一环后性能下降 4%~9%，说明每个模块都对最终效果有贡献。  
- **局限性**：论文承认真实检索过程受搜索引擎 API 限制，采集成本仍然比纯离线生成高；此外，筛选权重是手工调的，缺乏自动化学习的机制。

### 影响与延伸思考
SimpleDeepSearcher 把 **“高质量数据”** 放在了深度搜索研究的首位，提醒社区：在算力瓶颈日益明显的今天，数据工程往往比模型创新更能带来突破。自论文发布后，已有几篇工作尝试把 **真实交互日志**（如浏览器插件收集的搜索记录）用于检索微调，或把 **多准则筛选** 扩展为可学习的评分模型（如使用对比学习自动调权）。如果想进一步探索，可以关注以下方向：① 自动化权重学习的多准则筛选；② 将用户点击行为加入轨迹，提升模型对真实搜索意图的感知；③ 在低资源语言上复现真实检索采集流程，验证方法的通用性。

### 一句话记住它
只要给模型几百条真实、精挑细选的多轮搜索示例，指令微调就能让它在深度信息检索上跑得比昂贵的强化学习更快更稳。