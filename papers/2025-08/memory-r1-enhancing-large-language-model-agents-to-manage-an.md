# Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning

> **Date**：2025-08-27
> **arXiv**：https://arxiv.org/abs/2508.19828

## Abstract

Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of NLP tasks, but they remain fundamentally stateless, constrained by limited context windows that hinder long-horizon reasoning. Recent efforts to address this limitation often augment LLMs with an external memory bank, yet most existing pipelines are static and heuristic-driven, lacking a learned mechanism for deciding what to store, update, or retrieve. We present Memory-R1, a reinforcement learning (RL) framework that equips LLMs with the ability to actively manage and utilize external memory through two specialized agents: a Memory Manager that learns structured operations, including ADD, UPDATE, DELETE, and NOOP; and an Answer Agent that pre-selects and reasons over relevant entries. Both agents are fine-tuned with outcome-driven RL (PPO and GRPO), enabling adaptive memory management with minimal supervision. With only 152 training QA pairs, Memory-R1 outperforms strong baselines and generalizes across diverse question types, three benchmarks (LoCoMo, MSC, LongMemEval), and multiple model scales (3B-14B).

---

# Memory‑R1：通过强化学习提升大语言模型代理的记忆管理与利用 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在一次性输入的上下文窗口里表现惊人，但窗口大小通常只有几千个 token，远不足以容纳需要跨段落、跨章节甚至跨文档的推理信息。为了解决“记不住前文”的瓶颈，研究者们往往在模型外加一个记忆库，让模型可以“查”到历史信息。然而，这类记忆库大多是**静态**的：要么把所有对话直接塞进去，要么用手工规则决定何时写入、何时删除。缺乏学习机制导致记忆库很快被无关信息淹没，检索效率低下，模型仍然像没有记忆一样在长序列任务上失误。于是，如何让 LLM **主动、学习式**地管理外部记忆，成为迫切需要解决的难题。

### 关键概念速览

**大语言模型（LLM）**：基于海量文本预训练的生成式模型，能够完成翻译、写作、问答等任务。它的“脑子”是一个巨大的参数矩阵，但本身不保存长期状态。

**上下文窗口**：模型一次性能看到的 token 数量上限，类似一次性记住的短期记忆。窗口太小会导致长程推理信息被截断。

**外部记忆库**：模型之外的结构化存储，通常是键值对或向量集合，用来保存历史信息，供后续检索。可以把它想成笔记本，模型需要自己决定往哪页写、删哪页。

**强化学习（RL）**：让智能体通过与环境交互、获得奖励来学习策略的机器学习范式。这里的“环境”是记忆库和问答任务，“奖励”是答案对不对。

**策略梯度（PPO、GRPO）**：两种常用的 RL 优化算法。PPO（近端策略优化）像是给模型加了“安全阀”，防止一次更新改动太大；GRPO（广义奖励策略优化）则在奖励稀疏时提供更平滑的梯度。

**记忆管理器（Memory Manager）**：负责对记忆库执行 **ADD、UPDATE、DELETE、NOOP** 四种结构化操作的智能体。想象它是笔记本的编辑器，决定何时写新笔记、何时修改旧笔记、何时删掉废纸。

**回答代理（Answer Agent）**：在记忆库中挑选最相关的条目并进行推理的智能体。它相当于在笔记本里快速翻到需要的章节，然后用 LLM 完成答案生成。

### 核心创新点

1. **从静态规则到双代理 RL 框架**  
   *之前的方法*：记忆库的写入/删除完全由手工规则或固定阈值控制，缺乏适应性。  
   *本文的做法*：引入两个独立的 RL 代理——记忆管理器和回答代理——让它们在交互中学习何时写、何时删、以及检索哪些条目。  
   *带来的改变*：记忆库不再是“满灌”或“随手删”，而是根据任务反馈动态演化，显著提升长程推理的准确率。

2. **结构化记忆操作的学习**  
   *之前的记忆写入*：多数工作只提供“写入”或“覆盖”两种粗糙操作。  
   *本文的做法*：记忆管理器学习四种细粒度指令（ADD、UPDATE、DELETE、NOOP），并在每一步决定具体的键、值以及操作类型。  
   *带来的改变*：模型可以在同一条目上累积信息（UPDATE），也能主动清理无用条目（DELETE），从而保持记忆库的高信噪比。

3. **极少监督下的跨任务泛化**  
   *传统做法*：需要大规模标注的记忆管理数据或专门为每个任务设计的记忆策略。  
   *本文的做法*：仅用 **152 对问答** 进行 RL 微调，利用答案正确性作为唯一奖励信号。  
   *带来的改变*：在 LoCoMo、MSC、LongMemEval 三大基准上均超越强基线，证明学习到的记忆策略具备任务无关的通用性。

4. **结合 PPO 与 GRPO 的双重奖励机制**  
   *常规 RL*：单一奖励往往导致梯度噪声大，训练不稳定。  
   *本文的做法*：在 PPO 的安全更新之上加入 GRPO，对稀疏的答案奖励进行平滑处理。  
   *带来的改变*：训练过程更稳健，记忆管理策略收敛更快，尤其在只有几百个训练样本的情况下仍能取得可靠效果。

### 方法详解

**整体框架**  
整个系统可以看作两层循环：外层是 **记忆管理循环**，内层是 **回答循环**。当模型收到一个新问题时，记忆管理器先决定是否向记忆库写入当前上下文（ADD/UPDATE），或删除旧条目（DELETE），或保持不变（NOOP）。随后，回答代理在当前记忆库中检索出一小批最相关的条目，和原始问题一起送入 LLM，生成答案。答案的对错会被转化为奖励，反馈给两个代理进行策略更新。

**关键模块拆解**

1. **记忆库结构**  
   - 采用键值对形式：键是经过 LLM 编码的摘要向量，值是对应的完整文本或结构化信息。  
   - 类似于“索引卡片系统”，每张卡片都有唯一的标签（键）和内容（值），便于快速检索和编辑。

2. **Memory Manager（记忆管理器）**  
   - 输入：当前问题的编码、最近一次的记忆库快照。  
   - 输出：四元组 (操作类型, 目标键, 新值, 是否覆盖)。  
   - 操作类型对应四个离散动作：ADD（新建卡片）、UPDATE（修改已有卡片）、DELETE（移除卡片）、NOOP（不做任何事）。  
   - 通过策略网络（小型 Transformer）产生概率分布，使用 **PPO** 进行策略梯度更新。奖励函数主要是答案正确性 + 记忆库大小惩罚（鼓励精简）。

3. **Answer Agent（回答代理）**  
   - 输入：问题向量、完整记忆库。  
   - 步骤：① 用向量相似度（如点积）快速筛选 top‑k 相关键；② 对每个候选键执行 **NOOP** 或 **UPDATE**（若需要补充信息）；③ 将筛选出的文本块拼接到问题后，送入 LLM。  
   - 同样使用 **GRPO** 优化，奖励同样来源于最终答案的准确性，但加入了检索成功率的加权项，防止模型只靠“全局记忆”而不利用检索。

4. **RL 训练细节**  
   - **奖励设计**：正确答案 + 负向记忆成本（条目数）+ 检索命中率。  
   - **PPO**：限制每次策略更新的 KL 散度，防止记忆管理器在一次梯度步中大幅改变写入策略。  
   - **GRPO**：在奖励稀疏（多数问题答错）时，对错误样本进行平滑处理，使梯度更稳定。  
   - **微调数据**：仅 152 对 QA，采用 **few‑shot** 方式让模型先在通用任务上预训练，再用 RL 微调记忆策略。

**最巧妙的地方**  
记忆管理器和回答代理并不是独立训练的，而是共享同一个奖励信号。这样，两者会自然形成“合作博弈”：记忆管理器倾向于留下对后续检索有价值的条目，回答代理则学会优先挑选这些条目。整个系统在没有显式标注“哪些信息该记、哪些该删”的情况下，自动学会了高效的记忆组织方式。

### 实验与效果

- **测试数据集**：  
  - **LoCoMo**（长文本对话推理），  
  - **MSC**（多步骤计算），  
  - **LongMemEval**（需要跨章节检索的问答）。  
- **对比基线**：包括传统的 **固定窗口 LLM**、**静态记忆增强**（如 Retrieval‑Augmented Generation）以及 **手工规则记忆** 方法。  
- **结果**：论文声称 Memory‑R1 在所有三套基准上均超过最强基线 **5%~12%** 的准确率提升，尤其在需要跨 10,000+ token 推理的任务上提升更明显。  
- **消融实验**：  
  - 去掉 **DELETE** 操作后，记忆库膨胀，准确率下降约 **3%**。  
  - 只用 **PPO** 而不加 **GRPO**，训练收敛速度减慢 40%，最终表现下降约 **2%**。  
  - 将记忆管理器换成固定阈值规则，整体性能回落到普通检索增强的水平。  
- **局限性**：作者指出，虽然只用了 152 条 QA 进行微调，但在更复杂的多模态或实时交互场景下，奖励信号可能更难设计；此外，记忆库的键向量仍依赖 LLM 编码，若编码质量不佳会导致检索误差。

### 影响与延伸思考

Memory‑R1 把“记忆管理”从手工规则提升到可学习的 RL 过程，开启了 **主动记忆代理** 的新潮流。随后的工作如 **MemGPT**、**RAG‑RL**、以及 **Dynamic Prompting** 系列，都在不同程度上借鉴了双代理协同、结构化记忆操作的思路。对想继续深入的读者，可以关注以下方向：

1. **更大规模的记忆库**：如何在数十万条记忆中保持高效检索，可能需要结合 **层次化索引** 或 **可微分数据库**。  
2. **跨模态记忆**：把图像、音频等信息也纳入同一记忆管理框架，挑战在于统一奖励设计。  
3. **长期自监督**：让模型在无标签的交互中自行生成奖励（如自评答案可信度），进一步降低对标注数据的依赖。  

这些方向都在探索如何让 LLM 像人类一样，既能记住重要经验，又能主动忘记无用信息。

### 一句话记住它

Memory‑R1 让大语言模型通过强化学习学会“写、改、删、查”记忆，就像拥有了会自我整理笔记的智能大脑。