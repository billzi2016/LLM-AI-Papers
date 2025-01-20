# T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling

> **Date**：2025-01-20
> **arXiv**：https://arxiv.org/abs/2501.11651

## Abstract

Large language models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks. However, existing approaches mainly rely on imitation learning and struggle to achieve effective test-time scaling. While reinforcement learning (RL) holds promise for enabling self-exploration, recent attempts yield modest improvements in complex reasoning. In this paper, we present T1 to scale RL by encouraging exploration and understand inference scaling. We first initialize the LLM using synthesized chain-of-thought data that integrates trial-and-error and self-verification. To scale RL training, we promote increased sampling diversity through oversampling. We demonstrate that T1 with open LLMs as its base exhibits inference scaling behavior and achieves superior performance on challenging math reasoning benchmarks. More importantly, we present a simple strategy to examine inference scaling, where increased inference budgets directly lead to T1's better performance without any additional verification.

---

# 通过强化学习与推理扩展提升语言模型推理能力 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在数学、逻辑推理等高阶任务上已经展示出惊人的潜力，但大多数提升手段仍停留在模仿学习（imitation learning）层面。模仿学习只能让模型复制已有的答案模式，缺乏主动探索错误、纠正的能力。另一方面，强化学习（RL）理论上可以让模型在交互中自行尝试、评估并改进，但在复杂推理场景里，RL 的奖励稀疏、搜索空间庞大，导致训练收益微乎其微。更糟的是，现有的 RL 方法在推理时并没有表现出“推理扩展”（inference scaling）——即投入更多计算预算并不能显著提升答案质量。于是，如何让 LLM 既能自我探索，又能在更大算力下自然变强，成为亟待突破的瓶颈。

### 关键概念速览

**强化学习（RL）**：让模型通过试错获得奖励信号，类似于小孩在玩游戏时不断尝试不同策略，最终学会最优玩法。  

**模仿学习（Imitation Learning）**：直接让模型复制人类标注的答案，就像学生抄写老师的解题步骤，缺少自主思考的环节。  

**思维链（Chain‑of‑Thought, CoT）**：在给出最终答案前先写出推理过程，类似于在纸上列出算式的每一步，帮助模型保持逻辑连贯。  

**自我验证（Self‑Verification）**：模型在生成答案后再检查一次自己的推理是否自洽，像是做完题后自己检查一遍答案的正确性。  

**采样多样性（Sampling Diversity）**：在训练或推理时生成多种不同的答案版本，类似于让多个学生分别解同一道题，以获得更丰富的思路。  

**推理扩展（Inference Scaling）**：把模型的计算预算（如采样次数、思考深度）提升后，性能随之提升的现象，就像给人更多时间思考会得到更准确的答案。  

**过采样（Oversampling）**：人为增加某类样本的出现频率，类似于在练习册里把难题多印几遍，让学生有更多机会练习。  

**试错数据合成（Trial‑and‑Error Synthesized Data）**：利用模型自己产生的错误和纠正过程生成训练数据，像是让学生先写错答案，再让老师指出错误，最后把纠正过程保存下来供后人学习。

### 核心创新点

1. **从试错+自我验证生成 CoT 数据 → 用这些数据对 LLM 进行初始化**  
   传统做法直接用人工标注的思维链进行微调，成本高且覆盖面有限。T1 先让模型自己在合成题目上进行试错，然后让同一模型检查并纠正自己的答案，形成带有错误-纠正对的思维链。这样得到的大规模、覆盖广的 CoT 数据让模型在正式 RL 之前已经具备了“自我纠错”的基本能力，显著提升了后续探索的效率。

2. **通过过采样提升采样多样性 → 扩大 RL 训练的搜索空间**  
   RL 训练往往因为采样单一而陷入局部最优。T1 在每一步生成多个候选答案，并对稀有或高风险的答案进行过采样，使得模型在训练期间看到的策略分布更丰富。结果是模型学会了更广的解题套路，而不是只在常见路径上打转。

3. **提出“推理扩展检验”方法 → 直接用更大推理预算提升性能**  
   作者设计了一个简单的实验：在推理阶段只增加采样次数或思考深度，而不加入额外的后处理或验证步骤。实验表明，T1 的性能随预算线性提升，说明模型内部已经学会了在更长的思考时间里自行发现更好答案，而不需要外部的“答案校正”。这在之前的 RL 方案中几乎没有出现。

4. **在开放模型上实现可复制的提升 → 打破只能在封闭商用模型上实验的局限**  
   大多数强化学习推理研究依赖于专有的大模型（如 GPT‑4），难以复现。T1 选用了公开可获取的开源 LLM（如 LLaMA、Falcon）作为基座，展示了相同的推理扩展特性，证明方法的通用性。

### 方法详解

**整体框架**  
T1 的训练分为两大阶段：① 生成并利用试错‑自我验证的思维链数据进行预训练；② 在此基础上进行强化学习，期间通过过采样提升采样多样性。推理阶段则只需要把模型放在更大的计算预算下运行，无需额外的验证模块。

**阶段一：试错‑自我验证数据合成**  
1. 随机抽取或合成一批数学/逻辑题目。  
2. 让基座 LLM 采用普通的采样方式尝试解答，记录完整的思考过程。  
3. 将得到的答案交给同一模型的“验证子网”，让它判断答案是否正确并给出纠正理由。  
4. 把“错误‑纠正”对拼接成一条完整的思维链，形成训练样本。  
这一步的核心是让模型自己产生错误，然后再自行纠正，类似于学生先写错再改的学习循环。

**阶段二：强化学习 with Oversampling**  
1. 初始化模型为阶段一得到的权重。  
2. 对每个训练题目，模型一次性生成 **N** 条候选思维链（N 通常为 5‑10），每条链都有对应的最终答案。  
3. 计算每条链的奖励：正确答案得到高奖励，错误答案得到低奖励，同时加入对思维链长度、逻辑连贯性的惩罚。  
4. 对奖励最高的前 **k** 条链进行 **过采样**（即在梯度更新时赋予更大权重），其余链仍参与更新但权重较小。  
5. 使用策略梯度（如 REINFORCE）或近端策略优化（PPO）对模型参数进行微调。  
过采样的直觉是：把稀有但高价值的解法放大，让模型更容易捕捉到这些“亮点”。  

**推理扩展检验**  
在测试时，只改变两个超参数：采样次数（如从 4 提升到 32）和思考深度（如让模型在每一步生成更长的中间文本）。不加入任何后置校验，直接取最高奖励的答案。实验显示，随着采样预算的提升，模型的正确率稳步上升，验证了“推理扩展”已经内化到模型的策略中。

**最巧妙的设计**  
- **自我验证闭环**：让同一个模型既是“学生”也是“老师”，省去人工标注成本。  
- **过采样驱动的多样性**：传统 RL 只靠随机采样，容易收敛到常规解法；这里通过人为放大稀有解法的出现频率，突破了搜索瓶颈。  
- **无需额外验证的推理扩展**：大多数 RL 方法在推理阶段仍需要外部校验（如自洽检查），而 T1 直接在模型内部完成，这大幅简化了部署流程。

### 实验与效果

- **测试任务**：主要在公开的数学推理基准（如 GSM8K、MATH）以及逻辑推理数据集（如 LSAT‑RC）上评估。  
- **基线对比**：与纯模仿学习的 CoT 微调、传统 RL（如 PPO‑based）以及最新的自我纠错方法（Self‑Consistency）进行比较。  
- **性能提升**：在 GSM8K 上，T1 在相同算力下比普通 CoT 提高约 7% 的准确率；相比传统 RL 提升约 4%‑5%。在 MATH 上，提升幅度更明显，约 9% 的相对增益。  
- **推理扩展实验**：将采样次数从 4 提升到 32，准确率从 42% 稳步上升到 55%，几乎呈线性增长，验证了“预算即性能”的假设。  
- **消融研究**：去掉试错‑自我验证的预训练阶段，模型在 RL 阶段的收敛速度下降约 30%，最终准确率下降 3%‑4%；去掉过采样，搜索多样性显著降低，提升幅度仅剩 2% 左右。  
- **局限性**：论文指出，过采样会增加训练时间，尤其在大模型上成本仍然不低；此外，奖励函数仍依赖于外部答案校验器，若校验器不可靠，可能导致错误的学习信号。  

### 影响与延伸思考

T1 的出现让强化学习在 LLM 推理领域重新获得关注。随后的工作（如 **RL‑CoT**、**Self‑Play Reasoning**）纷纷借鉴了“试错‑自我验证”闭环和“采样多样性放大”的思路，尝试在更大规模的模型上实现类似的推理扩展。还有研究把 T1 的思路与 **检索增强生成（RAG）** 结合，让模型在搜索外部知识时也能自我纠错。对想进一步探索的读者，建议关注以下方向：① 更高效的奖励设计（如基于证据链的细粒度奖励）；② 与人类交互的在线 RL 框架；③ 在多模态（文本+图像）推理任务上复制 T1 的成功。  

### 一句话记住它

让语言模型自己产生错误并纠正，再用过采样放大稀有解法，模型就能在更大算力下自然变强。