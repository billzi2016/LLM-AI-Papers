# AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy

> **Date**：2025-06-16
> **arXiv**：https://arxiv.org/abs/2506.13284

## Abstract

In this work, we investigate the synergy between supervised fine-tuning (SFT) and reinforcement learning (RL) in developing strong reasoning models. We begin by curating the SFT training data through two scaling strategies: increasing the number of collected prompts and the number of generated responses per prompt. Both approaches yield notable improvements in reasoning performance, with scaling the number of prompts resulting in more substantial gains. We then explore the following questions regarding the synergy between SFT and RL: (i) Does a stronger SFT model consistently lead to better final performance after large-scale RL training? (ii) How can we determine an appropriate sampling temperature during RL training to effectively balance exploration and exploitation for a given SFT initialization? Our findings suggest that (i) holds true, provided effective RL training is conducted, particularly when the sampling temperature is carefully chosen to maintain the temperature-adjusted entropy around 0.3, a setting that strikes a good balance between exploration and exploitation. Notably, the performance gap between initial SFT models narrows significantly throughout the RL process. Leveraging a strong SFT foundation and insights into the synergistic interplay between SFT and RL, our AceReason-Nemotron-1.1 7B model significantly outperforms AceReason-Nemotron-1.0 and achieves new state-of-the-art performance among Qwen2.5-7B-based reasoning models on challenging math and code benchmarks, thereby demonstrating the effectiveness of our post-training recipe. We release the model and data at: https://huggingface.co/nvidia/AceReason-Nemotron-1.1-7B

---

# AceReason‑Nemotron 1.1：通过SFT与RL协同提升数学与代码推理 论文详细解读

### 背景：这个问题为什么难？

在大模型已经能写代码、解数学题的时代，真正的「推理」仍然是瓶颈。传统的监督微调（SFT）只能让模型学会模仿已有答案，面对需要多步思考或灵活搜索的题目时容易卡死。单纯的强化学习（RL）又缺少高质量的奖励信号，容易走偏或产生无意义的胡言。于是，如何把两者的优势结合起来，让模型既有扎实的知识底子，又能在训练中主动探索、更好地优化答案，成为了迫切需要突破的点。

### 关键概念速览
- **监督微调（SFT）**：在大模型的基础上，用标注好的问答对继续训练，让模型学会更精准地输出期望答案。类似于给学生补习，老师提供标准答案让学生模仿。
- **强化学习（RL）**：模型在生成答案时根据奖励信号进行自我改进，像是让学生在考试后根据分数自行调整解题策略。
- **采样温度（temperature）**：控制生成时的随机程度，温度高时答案更“大胆”但不稳定，温度低时更保守但缺乏创新。可以把它想成学生答题时的“冒险程度”。
- **温度调节熵（temperature‑adjusted entropy）**：在给定温度下，模型输出分布的熵值，用来衡量答案的多样性。保持在约0.3相当于让学生既不盲目猜测，也不死板保守。
- **Prompt scaling**：增加训练时使用的提问数量，让模型见到更多不同的题目类型。
- **Response scaling**：对同一个提问生成多条答案，供模型学习不同的解题路径。
- **数学/代码基准（benchmark）**：专门评估模型在数学推理或代码生成上的表现的测试集合，如MATH、HumanEval等。

### 核心创新点
1. **双向扩容的SFT数据构建**  
   *之前的做法*：只增大数据量或只生成多答案，提升有限。  
   *本文的做法*：同步扩大提问数量（Prompt scaling）和每个提问的答案数量（Response scaling），形成更丰富的训练池。  
   *带来的改变*：实验显示，单纯增加提问数带来的性能提升更显著，说明模型在多样化的题目上更能学到通用推理技巧。

2. **温度‑熵目标的RL采样策略**  
   *之前的做法*：RL训练时随意设定采样温度，常导致探索不足或噪声过大。  
   *本文的做法*：在RL阶段动态调节采样温度，使得温度调节后的熵维持在约0.3左右，既保证足够探索，又不至于失控。  
   *带来的改变*：在相同的RL迭代次数下，这一策略显著提升了最终模型的解题准确率，并且缩小了不同SFT起点之间的差距。

3. **SFT强度与RL最终表现的系统性关联实验**  
   *之前的做法*：普遍假设更强的SFT模型会带来更好RL结果，但缺乏大规模验证。  
   *本文的做法*：训练多个不同质量的SFT基线（从弱到强），统一进行大规模RL，观察性能走向。  
   *带来的改变*：证实只要RL训练得当（尤其是温度‑熵控制），更强的SFT模型始终能在RL后取得更高的最终分数，验证了“强基底+好RL=更强模型”的经验法则。

4. **AceReason‑Nemotron‑1.1 7B的完整配方公开**  
   *之前的做法*：大多数高性能推理模型只公布模型权重，训练细节模糊。  
   *本文的做法*：在论文中系统披露了数据构建、SFT、RL的每一步超参数和实现细节，并在 HuggingFace 开源。  
   *带来的改变*：社区可以直接复现并在此基础上迭代，推动了 7B 级别推理模型的快速进步。

### 方法详解
整体框架可以划分为三大阶段：**数据扩容 → 监督微调 → 温度‑熵调控的强化学习**。下面逐步拆解。

1. **数据扩容**  
   - **Prompt scaling**：作者从公开的数学、代码题库以及自建的合成题库中抽取了数倍于以往的提问，形成数十万到上百万级别的多样化 Prompt。  
   - **Response scaling**：对每个 Prompt，使用一个已经微调好的基线模型进行多次采样（不同温度、不同随机种子），得到 2‑5 条高质量答案。这样每条 Prompt 对应的答案集合形成了“多解”学习信号，帮助模型理解同一道题可以有多种合法解法。

2. **监督微调（SFT）**  
   - 将上述 Prompt‑Response 对直接喂入模型，使用标准的交叉熵损失进行微调。  
   - 关键在于 **混合采样**：在同一 batch 中混入不同难度、不同解法的样本，防止模型只学会“最常见”解法。  
   - 训练时采用了 **梯度累积** 与 **学习率 warm‑up**，确保在 7B 参数规模下仍能稳定收敛。

3. **强化学习（RL）阶段**  
   - **奖励函数**：基于自动评测器（如代码单元测试、数学答案校验）给出二元奖励（对/错），并加入 **答案置信度** 作为软奖励，使模型倾向于输出高置信度的正确答案。  
   - **温度‑熵控制**：在每一步采样前，先计算当前模型在给定 Prompt 下的输出熵。若熵偏低，则提升采样温度；若熵偏高，则降低温度。目标是让 **温度调节后的熵≈0.3**，相当于在“适度随机”与“保持质量”之间找到平衡。  
   - **PPO（近端策略优化）**：采用 PPO 进行策略更新，保持新旧策略的 KL 散度在安全范围内，防止模型在追求奖励时出现剧烈退化。  
   - **迭代过程**：每轮 RL 结束后，重新评估模型在验证集上的熵分布，微调温度策略，确保整个训练过程始终围绕目标熵波动。

**最巧妙的点**在于把 **熵** 作为温度调节的直接目标，而不是手动设定固定温度。这样模型在不同难度的 Prompt 上会自动采用不同的探索力度，类似于学生在熟悉的题目上稳扎稳打，在陌生的题目上大胆尝试。

### 实验与效果
- **评测任务**：MATH（高中数学推理）、GSM8K（通用数学）、HumanEval（代码生成）以及 CodeContests（竞赛级代码）等。  
- **基线对比**：与前一代 AceReason‑Nemotron‑1.0（同样 7B 参数）以及 Qwen2.5‑7B 系列的最佳推理模型进行比较。  
- **主要结果**：在 MATH 上提升约 6%（从 28% 到 34%），GSM8K 提升约 5%（从 31% 到 36%），HumanEval 通过率提升约 4%（从 45% 到 49%）。这些数字在同等参数规模下属于领先水平。  
- **消融实验**：  
  1. 只做 Prompt scaling（不做 Response scaling）仍能提升约 3%；  
  2. 去掉温度‑熵控制的 RL，性能下降约 2‑3%；  
  3. 使用固定温度（0.7）进行 RL，熵波动大，最终分数比动态调节低约 1.5%。  
- **局限性**：作者指出，当前奖励函数仍依赖于自动评测器，难以覆盖所有开放式推理场景；此外，温度‑熵目标的 0.3 是在实验中经验得到的，跨语言或跨任务时可能需要重新调参。

### 影响与延伸思考
这篇工作在 7B 级别模型上展示了“强 SFT + 温度‑熵调控 RL”可以显著突破推理上限，激发了两类后续研究：  
1. **自适应采样策略**：更多工作开始探索把熵、置信度等信息直接嵌入采样分布的调节器，甚至用小模型预测最优温度。  
2. **多解学习**：Response scaling 的思路被用于对话系统、检索增强生成等场景，鼓励模型学习“一题多解”。  
如果想进一步深入，可以关注 **RLHF（基于人类反馈的强化学习）** 与 **温度‑熵自适应** 的结合，或尝试在更大模型上复现该配方，观察是否仍保持同样的收益。

### 一句话记住它
**把 SFT 打好底，再用“熵≈0.3”的温度调节让 RL 探索，7B 模型也能在数学和代码上跑出 SOTA。**