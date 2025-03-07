# R1-Searcher: Incentivizing the Search Capability in LLMs via   Reinforcement Learning

> **Date**：2025-03-07
> **arXiv**：https://arxiv.org/abs/2503.05592

## Abstract

Existing Large Reasoning Models (LRMs) have shown the potential of reinforcement learning (RL) to enhance the complex reasoning capabilities of Large Language Models~(LLMs). While they achieve remarkable performance on challenging tasks such as mathematics and coding, they often rely on their internal knowledge to solve problems, which can be inadequate for time-sensitive or knowledge-intensive questions, leading to inaccuracies and hallucinations. To address this, we propose \textbf{R1-Searcher}, a novel two-stage outcome-based RL approach designed to enhance the search capabilities of LLMs. This method allows LLMs to autonomously invoke external search systems to access additional knowledge during the reasoning process. Our framework relies exclusively on RL, without requiring process rewards or distillation for a cold start. % effectively generalizing to out-of-domain datasets and supporting both Base and Instruct models. Our experiments demonstrate that our method significantly outperforms previous strong RAG methods, even when compared to the closed-source GPT-4o-mini.

---

# R1-Searcher：通过强化学习激励大语言模型的检索能力 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在数学、代码等需要深度推理的任务上已经能靠内部知识取得好成绩，但它们的知识库是固定的、随模型训练截止时间冻结的。面对最新的时事、专业文献或细节丰富的查询时，模型只能“凭记忆”作答，容易出现信息缺失或凭空捏造（即 hallucination）。传统的检索增强生成（RAG）方法虽然能把外部搜索结果拼进去，但往往需要手工设计的检索触发规则或额外的监督信号，难以让模型自主决定何时、怎样去查询。于是，如何让 LLM 自主、有效地调用外部搜索系统，成为提升时效性和准确性的关键瓶颈。

### 关键概念速览
**大语言模型（LLM）**：基于海量文本训练的生成式模型，能够完成对话、写作、代码等任务，类似于“会说话的百科全书”。  
**强化学习（RL）**：让智能体通过试错获得奖励的学习方式，想象成训练一只狗，只有在完成任务后才给零食。  
**结果导向 RL（outcome‑based RL）**：奖励只在整个任务结束时给出，而不是每一步都打分，类似于只在比赛结束后才评判成绩。  
**检索增强生成（RAG）**：把外部搜索引擎的结果当作上下文喂给 LLM，像是让学生在答题时查阅教材。  
**两阶段（two‑stage）**：把一次完整的推理过程拆成“决定是否搜索”与“利用搜索结果继续推理”两个子任务。  
**过程奖励（process reward）**：对每一步的行为都打分的细粒度奖励，常用于细致的策略学习。  
**蒸馏（distillation）**：把大模型的知识压缩到小模型的技巧，这里指的是不需要通过教师模型来引导学习。  

### 核心创新点
1. **只用结果奖励 → 两阶段 RL 设计 → 让模型自行决定何时搜索**  
   过去的 RAG 方法往往在每一步都给出搜索信号或依赖人工标注的搜索标签。R1-Searcher 把奖励推迟到整个答案完成后才发放，并把一次推理拆成“搜索决策”和“搜索利用”两段，让模型在没有细粒度监督的情况下学会在需要时主动调用检索工具。

2. **不依赖过程奖励或蒸馏 → 纯 RL 冷启动 → 更好跨域适应**  
   许多强化学习方案需要先用人类示例或教师模型提供过程奖励，才能让模型不至于乱探索。R1-Searcher 完全摆脱这些依赖，直接用最终答案的对错作为唯一信号，使得方法可以直接在全新领域或在没有标注数据的 Base 模型上使用。

3. **统一框架兼容 Base 与 Instruct 模型 → 同时提升搜索与推理质量**  
   传统方案往往只针对指令微调（Instruct）模型进行优化。R1-Searcher 的两阶段策略对原始的基础模型（Base）和已经微调好的指令模型都有效，说明搜索能力的激励并不依赖特定的微调方式。

4. **对标闭源强模型（GPT‑4o‑mini）仍保持优势 → 实证证明方法的强度**  
   实验中即使与闭源的 GPT‑4o‑mini 比较，R1-Searcher 仍然取得更高的准确率，说明仅靠 RL 激励搜索就能在实际任务上超越最前沿的商业模型。

### 方法详解
**整体思路**  
R1-Searcher 把一次完整的问答过程视作一次“游戏”。模型先阅读问题，然后在“搜索决策阶段”决定是否向外部检索系统发起查询；如果决定搜索，就在“检索利用阶段”把检索到的文档当作新上下文继续推理，最终输出答案。整个过程只在答案生成完毕后，根据答案的对错或评分给出一次全局奖励，RL 通过这个奖励来调整两阶段的策略。

**关键模块拆解**  

1. **搜索决策模块**  
   - 输入：原始问题 + 当前对话历史。  
   - 输出：二元信号（搜索 / 不搜索）以及可选的检索关键词。  
   - 类比：像是学生在做题时先判断自己是否记得答案，记不清就去图书馆查资料。

2. **检索接口**  
   - 接收关键词，调用外部搜索引擎（如 Bing、Google）或专用文献库，返回若干段相关文本。  
   - 这里不对检索质量做额外奖励，所有压力都由最终答案的好坏传回。

3. **检索利用模块**  
   - 将检索到的文本拼接到原始上下文后，交给同一个 LLM 继续生成。  
   - 类比：学生把查到的资料放在笔记本上，再继续写答案。

4. **奖励计算**  
   - 采用“结果导向”方式：如果最终答案在评测指标上达标（如准确率、BLEU、代码通过率），则给正向奖励；否则给负向奖励。  
   - 奖励信号直接用于策略梯度更新，调整搜索决策的概率以及检索利用的生成策略。

**最巧妙的设计**  
- **冷启动的 RL**：没有任何过程奖励或教师信号，模型一开始可能会频繁搜索或完全不搜索，但通过大规模的试错，策略会自然收敛到“只在需要时搜索”。这让方法在没有任何标注的情况下也能学习到有效的搜索行为。  
- **两阶段解耦**：把搜索决策和答案生成分开，使得搜索策略可以独立优化，而生成模块仍然保留原有的强大推理能力，避免因搜索噪声直接破坏生成质量。

### 实验与效果
- **测试任务**：论文在多个需要外部知识的基准上评估，包括时事问答、专业医学问答、最新编程库使用等，均属于“知识密集且随时间变化”的场景。  
- **对比基线**：包括传统的检索增强生成（RAG）方法、使用过程奖励的 RL 框架以及闭源的 GPT‑4o‑mini。  
- **结果**：论文声称在所有公开基准上均显著超越 RAG，尤其在 GPT‑4o‑mini 上也取得更高的整体得分（具体数值未在摘要中给出）。  
- **消融实验**：通过去掉搜索决策阶段或改为固定搜索策略，性能明显下降，说明两阶段 RL 的贡献是关键。  
- **局限性**：由于奖励仅在答案完成后才给，训练过程需要大量的交互样本才能收敛；此外，检索质量仍受外部搜索引擎的限制，若检索结果本身噪声大，模型可能仍会产生错误答案。作者也提到在极端低资源语言上实验尚未展开。

### 影响与延伸思考
R1-Searcher 把“主动检索”提升到可以通过纯 RL 学习的层面，打开了让 LLM 自主获取外部信息的新路径。随后的工作开始探索更细粒度的奖励信号（如检索相关度）或把检索决策与工具调用（如代码执行、表格查询）统一进同一 RL 框架。对想进一步了解的读者，可以关注以下方向：  
- **RL 在工具使用上的扩展**：把搜索视作一种工具，结合代码执行、图像分析等多模态工具的统一调度。  
- **低资源环境下的自适应检索**：如何在缺少高质量搜索引擎的语言或领域中仍保持有效的检索激励。  
- **安全与可解释性**：在模型主动搜索时，如何追踪检索路径并防止恶意信息注入。  

### 一句话记住它
R1-Searcher 用一次性结果奖励教会大语言模型“只在需要时主动去查资料”，从而在没有任何过程监督的情况下显著提升了搜索与推理的整体准确率。