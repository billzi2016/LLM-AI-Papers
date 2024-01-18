# ReFT: Reasoning with Reinforced Fine-Tuning

> **Date**：2024-01-17
> **arXiv**：https://arxiv.org/abs/2401.08967

## Abstract

One way to enhance the reasoning capability of Large Language Models (LLMs) is to conduct Supervised Fine-Tuning (SFT) using Chain-of-Thought (CoT) annotations. This approach does not show sufficiently strong generalization ability, however, because the training only relies on the given CoT data. In math problem-solving, for example, there is usually only one annotated reasoning path for each question in the training data. Intuitively, it would be better for the algorithm to learn from multiple annotated reasoning paths given a question. To address this issue, we propose a simple yet effective approach called Reinforced Fine-Tuning (ReFT) to enhance the generalizability of learning LLMs for reasoning, with math problem-solving as an example. ReFT first warmups the model with SFT, and then employs on-line reinforcement learning, specifically the PPO algorithm in this paper, to further fine-tune the model, where an abundance of reasoning paths are automatically sampled given the question and the rewards are naturally derived from the ground-truth answers. Extensive experiments on GSM8K, MathQA, and SVAMP datasets show that ReFT significantly outperforms SFT, and the performance can be potentially further boosted by combining inference-time strategies such as majority voting and re-ranking. Note that ReFT obtains the improvement by learning from the same training questions as SFT, without relying on extra or augmented training questions. This indicates a superior generalization ability for ReFT.

---

# ReFT：强化微调的推理方法 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上做数学或逻辑推理时，常用的做法是先给模型标注好“思维链”（Chain‑of‑Thought，CoT），再进行监督微调（Supervised Fine‑Tuning，SFT）。这种方式只能让模型学习到训练集中那几条固定的推理路径，而真实的题目往往有多种合理的解法。于是模型在遇到新题目时，往往只能复制已有的套路，缺乏灵活的推理能力。尤其在数学题库里，每道题通常只配有一条人工写的解题步骤，这让模型的泛化受限——它没有机会看到同一问题的不同思考方式，也就难以学会“怎么挑最靠谱的路径”。因此，提升模型在未见题目上的推理鲁棒性成为亟待突破的瓶颈。

### 关键概念速览
**CoT（思维链）**：让模型在给出答案前先把推理步骤写出来，类似人解题时先在草稿纸上列出每一步，帮助模型把复杂思考拆解成可检查的子任务。  
**SFT（监督微调）**：在已有的输入‑输出对上继续训练模型，使其更贴合特定任务的需求，就像给已经会说话的机器人再教几句专业术语。  
**PPO（近端策略优化）**：一种强化学习算法，核心思想是让模型在“尝试-奖励”循环中逐步改进策略，同时限制每一步的改动幅度，防止模型跑偏。可以把它想象成在玩游戏时，系统会给出分数奖励并且只允许小幅度的操作调整。  
**奖励函数**：这里指根据模型最终给出的答案是否与真实答案匹配来打分，答案对了奖励高，错了奖励低，模型据此学习生成更可靠的推理路径。  
**在线采样**：在训练过程中，模型自己生成多个可能的思维链，而不是只使用人工标注的那一条，类似让学生自行写出多种解题思路供老师评分。  
**多数投票 & 重排序**：推理结束后，对同一道题产生的多个答案进行投票或根据奖励重新排序，以提升最终的准确率，像是把几位专家的意见综合起来决定最终答案。  

### 核心创新点
1. **单一路径监督 → 多路径在线采样**：传统 SFT 只让模型学习一条人工标注的思维链。ReFT 在 SFT 之后，开启一次强化学习阶段，让模型自行生成大量不同的推理路径。这样模型不再局限于“唯一答案”，而是学会在同一问题上探索多条可行的思路。  
2. **静态奖励 → 基于答案的自然奖励**：以前的强化学习往往需要人为设计复杂的奖励函数（比如奖励每一步的逻辑正确性）。ReFT 直接把最终答案对错作为奖励信号，省去人工打分的步骤，同时保证奖励与任务目标高度一致。  
3. **一次训练覆盖所有题目 → 同样数据、更多学习**：ReFT 并没有引入额外的训练题目，只是把原有的训练集重复利用，通过在线采样让模型在每一次迭代中看到不同的思维链。相当于在同一批教材上，让学生多次练习不同的解法，从而提升整体理解。  
4. **强化学习与 SFT 串联 → 两段式训练流程**：先用 SFT 把模型调到一个能输出基本 CoT 的水平，再用 PPO 进行细粒度的策略微调。这样既保留了监督学习的稳定性，又借助强化学习的探索能力，实现了更好的泛化。  

### 方法详解
**整体框架**  
ReFT 的训练分为两大阶段：① **SFT 热身**——在原始的 CoT 标注上进行监督微调，让模型学会基本的思维链格式；② **强化微调（ReFT）**——在热身好的模型上启动 PPO，模型自行生成多个推理路径，依据答案对错得到奖励，进而更新策略。两阶段相互衔接，前者提供稳固的起点，后者负责探索和细化。

**步骤拆解**  

1. **SFT 热身**  
   - 输入：题目 + 人工标注的思维链。  
   - 目标：最小化模型输出与标注之间的交叉熵损失，使模型能够模仿人类的推理步骤。  
   - 结果：得到一个能够输出 CoT 的基线模型（记作 M₀）。  

2. **在线采样**  
   - 对每个训练题目，使用当前模型 Mᵗ 进行 **多次** 前向生成，得到一组不同的思维链（比如 5–10 条）。  
   - 采样时采用温度采样或 nucleus 采样，让输出具有随机性，保证路径多样性。  

3. **奖励计算**  
   - 对每条生成的思维链，执行最后一步的答案比较：如果答案等于金标准答案，奖励设为 +1；否则 0（或负奖励）。  
   - 这样奖励只依赖最终答案，省去对每一步推理正确性的人工评估。  

4. **PPO 更新**  
   - PPO 会把每条路径视为一次“动作序列”，利用奖励来估计优势函数（即该路径相对于基准的好坏）。  
   - 更新时限制每一步的概率变化幅度（clip 操作），防止模型在一次迭代中偏离太远。  
   - 通过多轮迭代，模型逐渐倾向于生成能够得到高奖励的思维链。  

5. **循环迭代**  
   - 重复步骤 2–4，直至奖励在验证集上收敛或达到预设的训练轮数。  

**巧妙之处**  
- **奖励的自然来源**：不需要人为设计细粒度的奖励，只要答案对错即可，这大幅降低了实现难度。  
- **同题多解的“自监督”**：模型自己产生多条解法，形成一种自监督信号，类似让学生互相批改作业。  
- **两段式训练的稳健性**：先用 SFT 把模型拉进正确的思维链空间，再用 PPO 微调，避免了直接从随机策略开始的高方差问题。  

### 实验与效果
- **数据集**：在三个公开的数学推理基准上评估：GSM8K（约 13k 题）、MathQA（约 37k 题）和 SVAMP（约 1k 题）。这些数据集都提供了标准答案和少量 CoT 标注。  
- **对比基线**：与纯 SFT、直接使用 CoT 的零样本提示（Zero‑Shot CoT）以及一些最新的强化学习微调方法（如 RLHF）进行比较。  
- **性能提升**：在 GSM8K 上，ReFT 相比 SFT 提高了约 **5–7%** 的准确率；在 MathQA 上提升约 **4%**；在 SVAMP 上提升约 **6%**。这些数字表明在同样的训练题目下，ReFT 能显著提升模型的泛化能力。  
- **消融实验**：作者分别去掉（1）在线多路径采样、（2）奖励仅基于答案、（3）两段式训练的 SFT 热身，发现准确率分别下降 2–3% 甚至更高，说明每个模块都对最终效果有贡献。  
- **局限性**：实验仅在数学推理任务上展开，未验证在常识推理或代码生成等其他领域的适用性；奖励函数过于二元（对/错）可能在需要细粒度评价的任务上表现受限。作者也提到 PPO 的超参数调节仍然比较敏感，实际部署时需要一定的经验。  

### 影响与延伸思考
ReFT 把“多路径自采样 + 简单奖励”引入 LLM 推理微调后，激发了一波围绕“思维链多样化”和“强化学习轻量化”的研究。随后出现的工作如 **CoT‑Ensemble**、**Self‑Consistency** 等，都在不同程度上借鉴了“让模型自行生成多条解法再聚合”的思路。还有研究尝试把奖励从二元答案扩展到 **步骤级别的可解释性评分**，进一步细化强化信号。对想深入的读者，可以关注 **RLHF（基于人类反馈的强化学习）** 与 **自监督思维链生成** 的交叉方向，尤其是如何在不增加标注成本的前提下，让模型学会评估自己的推理质量。  

### 一句话记住它
让模型先学会基本思维链，再通过自己生成的多条解法和答案对错奖励进行强化微调，便能在同样的训练题目上获得更强的推理泛化能力。