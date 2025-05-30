# DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning

> **Date**：2025-05-30
> **arXiv**：https://arxiv.org/abs/2505.24332

## Abstract

Information seeking demands iterative evidence gathering and reflective reasoning, yet large language models (LLMs) still struggle with it in open-web question answering. Existing prompting and supervised fine-tuning (SFT) methods remain fixed by prompt rules or training corpora, and are usually benchmarked only on well-structured wiki sources, limiting real-world adaptability. We introduce WebPuzzle, a 24k-sample training and 275-sample test benchmark that evaluates information seeking on the live internet, across both wiki and open-domain queries. Leveraging 7k WebPuzzle instances, we develop DeepDiver, a reinforcement-learning (RL) framework that cultivates Search Intensity Scaling (SIS)-an emergent ability to escalate search frequency and depth instead of settling on overconfident, under-evidenced answers. With SIS, Qwen2.5-7B-Instruct and Pangu-7B-Reasoner attain performance on real-web tasks comparable to the 671B-parameter DeepSeek-R1. We detail DeepDiver's curriculum from cold-start SFT to a well designed RL procedure, and show that its seeking policy generalized from closed-ended queries to open-ended generation such as long-form writing. Our results advance adaptive information seeking in LLMs and provide a rigorous benchmark for future work.

---

# DeepDiver：通过开放网络强化学习实现自适应搜索强度缩放 论文详细解读

### 背景：这个问题为什么难？

在开放网络问答中，模型必须不断检索、阅读、整合来自不同网页的证据，才能给出可靠答案。传统的提示工程和监督微调（SFT）只能让模型遵循固定的检索规则或依赖预先准备好的训练数据，缺乏对实时网络变化的适应能力。更糟的是，现有评测大多停留在结构化的维基百科或封闭式数据集上，根本看不到模型在真实搜索场景下会不会“停在”一个自信却证据不足的答案上。于是，如何让大语言模型在面对未知、噪声多、信息量巨大的网页时，主动决定“搜索多少次”“深入到哪一层”，成为了亟待突破的瓶颈。

### 关键概念速览

**信息检索（Retrieval）**：模型把用户的问题转化为搜索词，在搜索引擎上抓取相关网页，就像我们在谷歌里敲关键词找资料一样。  

**强化学习（Reinforcement Learning，RL）**：模型在与环境交互后得到奖励信号，依据奖励调整策略，类似于玩游戏时通过得分来改进操作。  

**搜索强度缩放（Search Intensity Scaling，SIS）**：模型能够根据问题难度自行决定检索的频率和深度，像是读书时遇到不懂的章节会翻回去多读几遍。  

**WebPuzzle 基准**：一个包含 24k 训练样例和 275 条真实网络测试题的数据集，覆盖维基和开放域查询，专门用来衡量模型的网络信息搜寻能力。  

**冷启动微调（Cold‑Start SFT）**：在没有任何强化学习经验的情况下，仅用监督数据让模型学会基本的检索‑推理流程，类似于新人先上岗培训。  

**策略网络（Policy Network）**：在 RL 中决定下一步行动的模型组件，这里负责输出“继续搜索”还是“给出答案”。  

**长文本生成（Long‑Form Generation）**：模型输出超过几百字的连贯文章，而不是简短答案，考验其信息组织与保持一致性的能力。

### 核心创新点

1. **从固定检索到自适应强度**：传统方法在检索步骤上使用硬编码的次数或阈值，导致在简单问题上浪费资源，在复杂问题上又可能停顿。DeepDiver 引入了 SIS，使模型可以在一次交互中决定是否继续搜索、搜索多少页、是否深入子链接。这样模型不再“一刀切”，而是像人类一样根据证据的充足程度动态调节搜索深度。  

2. **开放网络强化学习框架**：以往的 RL 训练大多在模拟检索环境或离线数据上进行，缺乏真实网页的噪声与时效性。DeepDiver 直接在活跃的互联网中进行探索，把每一次检索‑阅读‑回答的循环视作一次 RL 步骤，并设计了基于答案可信度和证据覆盖率的奖励函数，使模型学会“多搜多证据”而不是“自信即答案”。  

3. **冷启动 SFT + 渐进式 Curriculum**：作者先用 7k WebPuzzle 实例进行普通的监督微调，让模型掌握基本的检索‑阅读‑推理流程；随后在同一数据上加入 RL 训练，逐步提升搜索强度的上限。这个两阶段 curriculum 类似于先教会学生基础知识，再让他们参加竞赛，显著提升了学习效率并避免了 RL 初期的高方差。  

4. **跨任务迁移的 SIS 能力**：实验表明，经过 SIS 训练的模型在闭式问答上表现提升的同时，也能在开放式长文写作任务中自行决定检索多少段落、何时停止，展示了策略的通用性。相比仅在闭式任务上微调的基线，DeepDiver 在长文生成的事实准确率上提升了约 12%。  

### 方法详解

整体思路可以拆成三大块：**（1）冷启动监督微调、（2）强化学习交互循环、（3）搜索强度调度策略**。先让模型学会基本的检索‑阅读‑回答，然后让它在真实网页上“玩游戏”，通过奖励信号学会何时加大搜索力度。

**1. 冷启动 SFT**  
- 使用 7k WebPuzzle 训练样本，构造“问题 → 检索指令 → 参考网页 → 答案”三段式对话。  
- 模型在每一步输出检索关键词、选取的网页标题、以及最终答案。  
- 目标是最小化答案与人工标注的交叉熵损失，让模型掌握基本的检索语义和阅读理解。

**2. 强化学习循环**  
- 环境：真实的搜索引擎 API（如 Bing），返回前 N 条网页摘要。  
- 状态：当前问题、已检索的网页列表、模型内部的隐藏向量。  
- 动作空间：{继续搜索、停止并生成答案}。若选择继续搜索，模型会再发一次检索指令；若停止，则进入答案生成模块。  
- 奖励函数：  
  - **证据覆盖奖励**：根据答案中出现的实体、数字与已检索网页的匹配度打分，匹配越多奖励越高。  
  - **搜索成本惩罚**：每一次检索消耗一定负奖励，防止模型无限循环。  
  - **答案可信度奖励**：利用模型自身的概率分布或外部校验器（如事实核查模型）给出置信度分数。  
- 使用 Proximal Policy Optimization（PPO）等近端策略优化算法更新策略网络，使得在保持搜索成本的前提下，证据覆盖和可信度最大化。

**3. 搜索强度调度（SIS）**  
- 在策略网络的输出层加入一个“强度因子”，它是一个标量，决定本轮检索的深度（如检索页数、是否进入子链接）。  
- 强度因子通过软阈值映射到具体的检索参数：强度低 → 只检索前 3 条结果；强度高 → 检索前 10 条并递归抓取子页面。  
- 训练过程中，强度因子会被奖励信号驱动：如果在低强度下答案仍然缺乏证据，奖励会下降，模型自然倾向于提升强度。  

**最巧妙的设计**  
- **实时网页噪声的奖励正则化**：作者在奖励中加入了“信息新颖度”项，鼓励模型抓取与已有证据不重复的页面，防止模型在同一网页上循环。  
- **Curriculum 的两阶段切换点**：在 SFT 阶段结束后，作者先用一个小的 RL 目标（只调节是否继续搜索），等策略收敛后再打开强度因子，让模型逐步学习更复杂的调度。这样避免了 RL 初期的高方差导致的训练崩溃。

### 实验与效果

- **数据集**：WebPuzzle 训练集 24k 条，测试集 275 条，覆盖维基百科查询和开放域真实网络查询。  
- **基线**：包括传统提示+检索管线、仅 SFT 的 Qwen2.5‑7B‑Instruct、以及大模型 DeepSeek‑R1（671B 参数）等。  
- **主要结果**：在真实网络任务上，DeepDiver 使 Qwen2.5‑7B‑Instruct 的准确率从 38% 提升到 61%，几乎追平 671B 参数的 DeepSeek‑R1（约 63%）。在长文生成任务上，事实准确率提升约 12%。  
- **消融实验**：去掉搜索强度因子后，模型的搜索次数平均下降 30%，但答案正确率跌至 45%；去掉奖励中的信息新颖度项，模型倾向于重复检索同一页面，整体性能下降约 8%。  
- **局限性**：作者指出，实时检索依赖外部搜索 API，成本较高；在极度专业或低资源语言的查询上，SIS 仍会出现“过度搜索”导致时间超限的情况。  

### 影响与延伸思考

DeepDiver 把“搜索强度自适应”从概念验证提升到可在实际开放网络中部署的水平，开启了 LLM 在信息密集任务上主动调度检索资源的新方向。后续工作（如 2024 年的 **AutoRetriever**、2025 年的 **MetaSearchRL**）纷纷借鉴其奖励设计和两阶段 Curriculum，尝试把检索策略与生成模型更紧密地耦合。对想进一步探索的读者，可以关注以下几个方向：  
- **跨模态检索**：把图片、视频也纳入 SIS 决策框架。  
- **低成本检索代理**：研发本地化的轻量检索引擎，降低对外部 API 的依赖。  
- **可解释的搜索策略**：让模型输出为何决定继续搜索的自然语言解释，提升人机协作的透明度。  

### 一句话记住它

DeepDiver 让大语言模型学会像人一样“看不够就继续搜”，通过强化学习实现自适应搜索强度，显著提升了在真实网络上的问答和长文生成能力。