# PaSa: An LLM Agent for Comprehensive Academic Paper Search

> **Date**：2025-01-17
> **arXiv**：https://arxiv.org/abs/2501.10120

## Abstract

We introduce PaSa, an advanced Paper Search agent powered by large language models. PaSa can autonomously make a series of decisions, including invoking search tools, reading papers, and selecting relevant references, to ultimately obtain comprehensive and accurate results for complex scholar queries. We optimize PaSa using reinforcement learning with a synthetic dataset, AutoScholarQuery, which includes 35k fine-grained academic queries and corresponding papers sourced from top-tier AI conference publications. Additionally, we develop RealScholarQuery, a benchmark collecting real-world academic queries to assess PaSa performance in more realistic scenarios. Despite being trained on synthetic data, PaSa significantly outperforms existing baselines on RealScholarQuery, including Google, Google Scholar, Google with GPT-4o for paraphrased queries, ChatGPT (search-enabled GPT-4o), GPT-o1, and PaSa-GPT-4o (PaSa implemented by prompting GPT-4o). Notably, PaSa-7B surpasses the best Google-based baseline, Google with GPT-4o, by 37.78% in recall@20 and 39.90% in recall@50, and exceeds PaSa-GPT-4o by 30.36% in recall and 4.25% in precision. Model, datasets, and code are available at https://github.com/bytedance/pasa.

---

# PaSa：面向全面学术论文检索的 LLM 代理 论文详细解读

### 背景：这个问题为什么难？

学术研究常常需要在海量会议论文和期刊文章中快速定位最相关的工作。传统搜索引擎只能靠关键词匹配，面对长句子、专业术语或隐含需求时容易漏掉关键文献。即使是带有学术过滤的 Google Scholar，也缺乏对检索结果的二次筛选和引用链的主动追踪。更糟的是，研究者往往需要手动阅读摘要、判断引用价值，这一步骤在大规模文献调研时耗时巨大。于是，如何让机器像人一样“思考”检索路径、主动追踪引用、并在多轮交互中不断优化结果，成为了一个亟待突破的难题。

### 关键概念速览
- **LLM（大语言模型）**：能够理解并生成自然语言的深度学习模型，类似于“会说话的百科全书”。在本工作中，它负责决定何时搜索、搜索什么以及如何评估文献。
- **Agent（智能体）**：把 LLM 当作“大脑”，配合外部工具（搜索引擎、文献库）执行一系列动作，就像人类使用浏览器、打开 PDF、做笔记一样。
- **Crawler（爬取模块）**：负责生成检索词、调用搜索工具并把找到的论文放进待评估队列，还会沿着文献的参考文献链继续抓取，类似于“图书管理员”不断把相关书籍搬到手边。
- **Selector（筛选模块）**：对爬取到的每篇论文判断是否满足用户的查询意图，像是“审稿人”给每篇稿件打分。
- **强化学习（RL）**：让模型通过试错获得奖励的训练方式，这里奖励是检索的准确率和召回率，类似于游戏中得分越高，策略越好。
- **Synthetic Dataset（合成数据集）**：人工生成的训练数据，本文的 AutoScholarQuery 包含 35k 细粒度学术查询和对应的目标论文，像是“模拟考试”帮助模型练习。
- **Real-world Benchmark（真实基准）**：真实用户提出的查询集合，本文的 RealScholarQuery 收录了 50 条实际研究需求，用来检验模型在真实场景下的表现。
- **Recall@k / Precision**：检索评价指标，Recall@k 表示在前 k 条返回结果中找到了多少目标文献，Precision 表示返回结果中有多少是真正相关的。

### 核心创新点
1. **合成查询‑文献对的 AutoScholarQuery → 通过 GPT‑4o 自动生成高质量学术查询并配对对应论文 → 为 LLM 提供了大规模、细粒度的监督信号，使得即使没有真实标注也能学会精准检索。**  
   以前的文献检索模型大多只能利用少量手工标注的数据，规模受限；本工作用大模型自行生成查询，突破了数据瓶颈。

2. **两阶段训练的 Crawler：模仿学习 → 强化学习 → 首先让模型模仿专家的搜索行为，再通过奖励（召回率、Selector 辅助的准确率）进行自我优化 → 让模型在搜索路径上既能快速收敛，又能在复杂查询中发现更深层次的引用链。**  
   传统检索系统只靠一次性查询，缺乏多轮决策；这里引入了“先学会基本技巧，再自行探索”的训练范式。

3. **Selector 的辅助奖励机制 → 在强化学习阶段，Selector 为每一步爬取提供即时的相关性评分 → Crawler 能够根据 Selector 的反馈动态调整搜索策略，而不是盲目扩展文献队列。**  
   这相当于在搜索过程中加入了“实时审稿”，显著提升了检索的精度和效率。

4. **LLM‑Agent 与传统搜索引擎的对比实验 → 在 RealScholarQuery 上，PaSa‑7B 的 Recall@20 超过 Google+GPT‑4o 37.78%，Recall@50 超过 39.90%，并且在 Precision 上也领先 4.25%。 → 证明即使只在合成数据上训练，模型也能在真实场景中显著超越最强基线。**  
   之前的 LLM‑augmented 搜索往往只能在特定任务上略有提升，本工作实现了跨任务的大幅跃迁。

### 方法详解
#### 整体框架
PaSa 把学术检索看作一次“搜索‑评估‑扩展”的循环。给定用户的自然语言查询，系统首先让 **Crawler** 产生检索词并调用外部搜索工具（如 Google Scholar API）抓取初始文献集合。抓到的文献会进入 **论文队列**，随后 **Selector** 对每篇文献打分，决定是否保留并进一步追踪其参考文献。整个循环在 **强化学习** 框架下进行，模型每完成一次搜索‑筛选‑扩展的迭代，就会根据最终召回率和 Selector 给出的即时奖励更新策略。

#### 关键模块拆解
1. **查询生成与搜索调用（Crawler）**  
   - 输入：用户原始查询 Q。  
   - LLM 先把 Q 重写成若干检索词 S₁…Sₙ（类似于“把长句拆成关键词”），每个检索词会被送入搜索工具。  
   - 搜索返回的每篇论文记为 P，加入待评估的 **候选池**。  

2. **引用链追踪**  
   - 对候选池中的每篇 P，Crawler 读取其参考文献列表，生成新的检索词（如“引用了 P 的工作”），再一次调用搜索工具。  
   - 这样形成了“深度优先”或“广度优先”的文献扩散，类似于在学术网络中沿着边走动。  

3. **文献筛选（Selector）**  
   - Selector 也是一个 LLM，接收用户查询 Q 与候选文献的标题、摘要等信息。  
   - 它输出一个相关性分数 r∈[0,1]，并根据阈值决定是否把文献标记为“有效”。  
   - 同时，Selector 会把分数反馈给 Crawler，作为即时奖励信号。  

4. **强化学习优化**  
   - **状态**：当前查询 Q、已检索的文献集合、Selector 的最新分数。  
   - **动作**：生成下一个检索词、决定是否继续追踪引用。  
   - **奖励**：主要由两部分组成——（a）最终召回率（在所有目标文献中找到了多少），（b）Selector 在每一步给出的分数累计。  
   - 训练分两阶段：  
     - **模仿学习**：使用 AutoScholarQuery 中的“专家”检索轨迹，让模型先学会基本的检索套路。  
     - **强化学习**：在模拟环境中让模型自行探索，依据奖励函数微调策略。  

#### 巧妙之处
- **奖励信号的双层设计**：把全局召回率和局部 Selector 评分结合，使得模型既关注整体目标，又能在每一步做出细粒度的判断。  
- **引用链的递归抓取**：不像传统搜索只停留在第一层结果，PaSa 能像学者追根溯源一样，自动沿着引用网络深入，显著提升了召回深度。  
- **合成数据的高质量**：利用 GPT‑4o 生成的查询覆盖了最新会议的前沿主题，保证了训练数据的时效性和多样性。

### 实验与效果
- **训练数据**：AutoScholarQuery，35,000 条由 GPT‑4o 生成的细粒度学术查询，每条都配有对应的目标论文集合。  
- **评估数据**：RealScholarQuery，收集了 50 条真实研究者提出的查询，这些查询都是作者实际使用 PaSa 得到的结果。  
- **对比基线**：Google、Google Scholar、Google + GPT‑4o（对查询进行改写后再搜索）、ChatGPT（开启搜索功能的 GPT‑4o）、GPT‑o1、以及 PaSa‑GPT‑4o（用 GPT‑4o 直接实现的 PaSa 版本）。  
- **核心指标**：Recall@20、Recall@50、Precision。  
- **主要结果**：  
  - PaSa‑7B 在 Recall@20 上比 Google + GPT‑4o 高出 37.78%，在 Recall@50 上高出 39.90%。  
  - 与 PaSa‑GPT‑4o 相比，Recall 提升 30.36%，Precision 提升 4.25%。  
  - 在所有基线中，PaSa‑7B 的整体表现最为均衡，尤其在深度召回（@50）上优势明显。  
- **消融实验**：作者分别去掉（1）引用链追踪、（2）Selector 奖励、（3）强化学习阶段，仅保留模仿学习。实验显示，去掉任意一项都会导致 Recall@20 下降至少 12%，说明每个模块对最终性能都有实质贡献。  
- **局限性**：论文承认 RealScholarQuery 规模仅为 50 条，尚不足以覆盖所有学科；此外，模型在极其专业的细分领域（如医学、材料科学）仍可能受限于训练语料的覆盖度。

### 影响与延伸思考
PaSa 的开源代码和数据集在发布后迅速被学术检索社区引用，催生了几类后续工作：  
- **跨模态文献检索**：把代码、实验数据等非文本资源加入检索链路。  
- **领域专属 Agent**：在医学、法律等高风险领域训练专门的 AutoScholarQuery，以提升专业度。  
- **大规模引用图强化学习**：把整个学术网络视作强化学习的环境，进一步探索长期奖励（如提升论文影响力）的策略。  
如果想更深入，可以关注近期在 “LLM‑driven tool use” 方向的会议论文，尤其是关于 **Toolformer**、**ReAct** 等框架的改进，它们与 PaSa 的思路高度相似，只是应用场景不同。

### 一句话记住它
PaSa 用 LLM 主导的爬取‑筛选循环，在学术检索上把合成训练和强化学习结合，轻松把搜索准确率甩开传统搜索引擎。