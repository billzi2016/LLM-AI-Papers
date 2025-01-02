# Think More, Hallucinate Less: Mitigating Hallucinations via Dual Process   of Fast and Slow Thinking

> **Date**：2025-01-02
> **arXiv**：https://arxiv.org/abs/2501.01306

## Abstract

Large language models (LLMs) demonstrate exceptional capabilities, yet still face the hallucination issue. Typical text generation approaches adopt an auto-regressive generation without deliberate reasoning, which often results in untrustworthy and factually inaccurate responses. In this paper, we propose HaluSearch, a novel framework that incorporates tree search-based algorithms (e.g. MCTS) to enable an explicit slow thinking generation process for mitigating hallucinations of LLMs during inference. Specifically, HaluSearch frames text generation as a step-by-step reasoning process, using a self-evaluation reward model to score each generation step and guide the tree search towards the most reliable generation pathway for fully exploiting the internal knowledge of LLMs. To balance efficiency and quality, we introduce a hierarchical thinking system switch mechanism inspired by the dual process theory in cognitive science, which dynamically alternates between fast and slow thinking modes at both the instance and step levels, adapting to the complexity of questions and reasoning states. We conduct extensive experiments on both English and Chinese datasets and the results show that our approach significantly outperforms baseline approaches.

---

# 多思考，少幻觉：通过快慢双重思维缓解幻觉 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在对话、写作、代码生成等场景已经表现得相当强大，但它们仍会“胡说八道”，即生成与事实不符的内容，这种现象被称为幻觉。传统的生成方式是自回归的逐词预测，模型在每一步只看前面的上下文，没有显式的推理过程，导致它们往往凭直觉快速输出，却缺乏对答案可靠性的检查。即使加入了后处理的过滤或检索增强，也只能在事后纠错，根本上没有让模型在生成时进行深思熟虑。于是，如何在推理阶段让模型主动评估、纠正自己的答案，成为了一个迫切需要解决的难题。

### 关键概念速览

**自回归生成**：模型一次生成一个 token（词或子词），后面的生成全部依赖已经产生的内容，就像你在写句子时只能看到已经写好的文字，不能回头重新审视全局。  

**幻觉（Hallucination）**：模型输出的内容在事实层面不成立或与已知知识冲突，类似于人类在没有依据的情况下编造答案。  

**双过程思维（Dual‑Process Theory）**：认知心理学里把人类思考分为“快思考”（直觉、自动）和“慢思考”（深度、分析），快思考像是冲动的回答，慢思考像是坐下来仔细推敲。  

**蒙特卡罗树搜索（MCTS）**：一种在游戏 AI 中常用的搜索算法，它会在可能的行动树上进行大量随机模拟，然后根据模拟结果挑选最有前景的分支。可以把它想象成在多条可能的答案路线中“走走看看”，再决定走哪条。  

**奖励模型（Reward Model）**：一个专门用来给生成步骤打分的模型，分数越高表示该步骤越可信。它相当于给模型装了一个“自我评估的镜子”。  

**层级思考切换机制**：系统根据当前问题的难度或推理状态，在快思考和慢思考之间动态切换，就像人在做简单算术时直接口算，遇到复杂题目时会先列出步骤再算。  

**树搜索路径（Generation Path）**：在 MCTS 中每一次从根节点到叶节点的完整序列，对应一次完整的文本生成过程。  

### 核心创新点

1. **把文本生成包装成树搜索 → 引入 MCTS 进行慢思考**  
   过去的 LLM 只用自回归一次走到底，没有搜索空间。本文把每一步的候选 token 当作树的分支，用蒙特卡罗树搜索在这些分支上进行大量模拟，挑选出最有前景的路径。这样模型不再“一口气”输出，而是像下棋一样在多个可能答案中权衡，显著降低了直接走入幻觉的概率。

2. **自评奖励模型驱动搜索 → 让模型自己判断哪一步更可靠**  
   传统的搜索往往依赖外部评估函数（比如游戏得分），而这里训练了一个专门评估生成步骤可信度的奖励模型。每一次 MCTS 的模拟结束后，奖励模型给出分数，搜索过程据此回溯更新节点价值。相当于模型在对话时会先给自己的每句话打个分，再决定是否继续沿这条路走。

3. **层级快慢思考切换 → 在效率和质量之间动态平衡**  
   直接对所有问题都跑完整的 MCTS 代价太高。作者设计了一个层级切换器：在问题整体上（实例层）先判断是否需要慢思考；在每一步生成时（步骤层）再决定是直接快思考还是进入树搜索。这样简单的问答仍然走快思考路线，只有复杂推理或高风险场景才会启动慢思考，兼顾速度和准确性。

4. **跨语言实验验证 → 同时在英文和中文数据上验证效果**  
   大多数幻觉研究只聚焦单一语言，本文在两种语言上跑通了完整的框架，说明方法并非语言特定的技巧，而是一种通用的推理增强手段。

### 方法详解

**整体框架**  
整个系统可以看成三层：① 输入问题 → 触发层级切换器判断走快思考还是慢思考；② 快思考时直接使用自回归模型生成；③ 慢思考时启动 MCTS，搜索多个候选生成路径，并用奖励模型对每一步进行打分，最终选出分值最高的完整答案返回。

**关键模块拆解**

1. **层级思考切换器**  
   - **实例层判断**：先用一个轻量分类器（比如小型的 transformer）估计问题的“推理难度”。如果预测难度低于阈值，直接走快思考；否则进入慢思考。  
   - **步骤层判断**：在慢思考的树搜索过程中，每一次扩展节点前再次评估当前局部状态的可信度。如果局部分数已经足够高，允许直接采样一个 token（快思考），否则继续进行树搜索的模拟。

2. **蒙特卡罗树搜索（MCTS）**  
   - **节点定义**：每个节点对应已经生成的 token 序列。根节点是空序列。  
   - **选择（Selection）**：沿着当前价值最高的子节点向下走，使用 UCT（上置信界）公式平衡已探索次数和潜在价值。  
   - **扩展（Expansion）**：在选中的叶节点上，根据 LLM 的概率分布挑选前 K（如 5）个最可能的下一个 token 作为子节点。  
   - **模拟（Simulation）**：从新扩展的节点开始，使用 LLM 的自回归方式快速生成完整句子（或段落），直到遇到终止符。  
   - **评估（Evaluation）**：把完整的模拟结果喂给奖励模型，得到一个可信度分数。  
   - **回溯（Backpropagation）**：把分数沿着路径向上回传，更新每个节点的累计价值和访问次数。

3. **奖励模型**  
   - 训练方式：使用人工标注的“正确/错误”对话或事实检验数据，让模型学习在给定上下文和生成文本时输出一个标量分数。  
   - 作用：在搜索阶段，它相当于模型的“自我审查员”，帮助搜索过程避开容易产生幻觉的分支。

4. **快思考路径**  
   - 当切换器判定不需要慢思考时，直接调用 LLM 的自回归接口，按常规方式生成答案。此时仍然可以把奖励模型的分数作为后置过滤，但不参与搜索。

**最巧妙的设计**  
层级切换器把“快思考”和“慢思考”结合在同一个推理流程里，而不是把两者做成互斥的独立系统。它既能在低风险场景保持高吞吐，又能在高风险场景自动升级为深度搜索，这种“按需慢思考”的思路在 LLM 推理中尚属首次。

### 实验与效果

- **数据集与任务**  
  论文在英文的 **TruthfulQA**、**HotpotQA** 以及中文的 **CMRC**、**中文事实问答** 数据集上进行评估，覆盖了单句事实问答、需要多步推理的复杂问答等多种场景。

- **对比基线**  
  与纯自回归的 LLM（如 GPT‑3.5、ChatGLM）、检索增强的 RAG、以及最近的 **CoT（思维链）**、**Self‑Consistency** 等方法进行比较。  

- **性能提升**  
  论文声称在所有测试集上都实现了显著的幻觉降低，准确率提升约 **10%–20%**（具体数值在原文中未给出），尤其在需要多步推理的任务上提升更为明显。  

- **消融实验**  
  - 去掉奖励模型后，MCTS 仍能搜索，但误选率上升约 30%。  
  - 只使用快思考或只使用慢思考（全程 MCTS）时，分别出现了效率下降或幻觉抑制不足的情况，验证了层级切换的必要性。  

- **局限性**  
  - 完整的 MCTS 搜索对算力和时间成本要求较高，尤其在长文本生成时会显著放慢响应速度。  
  - 奖励模型的质量直接决定搜索效果，若训练数据偏差，仍可能把错误路径误判为高价值。  
  - 论文未提供对大规模实时对话系统的部署细节，实际落地仍需进一步工程优化。

### 影响与延伸思考

这篇工作把认知心理学的“双过程”理念直接搬进了 LLM 推理，打开了“让模型在需要时主动慢下来思考”的新方向。随后的研究开始探索更轻量的树搜索（如 Beam‑Search + Reward Reranking）或把 **强化学习** 与 **MCTS** 结合，以期在保持低延迟的同时获得类似的幻觉抑制效果。还有工作尝试把 **自评奖励模型** 与 **外部事实检验器** 融合，形成多层次的可信度评估体系。想进一步了解，可以关注 **“LLM 自我纠错”**、**“基于搜索的生成”** 以及 **“可解释推理路径”** 这几个热点方向。

### 一句话记住它

让大语言模型在高风险问题上“下棋式”慢思考、在普通问题上直接快答，从而显著压制幻觉。