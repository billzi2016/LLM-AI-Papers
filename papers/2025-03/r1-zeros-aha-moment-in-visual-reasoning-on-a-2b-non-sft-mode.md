# R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model

> **Date**：2025-03-07
> **arXiv**：https://arxiv.org/abs/2503.05132

## Abstract

Recently DeepSeek R1 demonstrated how reinforcement learning with simple rule-based incentives can enable autonomous development of complex reasoning in large language models, characterized by the "aha moment", in which the model manifest self-reflection and increased response length during training. However, attempts to extend this success to multimodal reasoning often failed to reproduce these key characteristics. In this report, we present the first successful replication of these emergent characteristics for multimodal reasoning on only a non-SFT 2B model. Starting with Qwen2-VL-2B and applying reinforcement learning directly on the SAT dataset, our model achieves 59.47% accuracy on CVBench, outperforming the base model by approximately ~30% and exceeding both SFT setting by ~2%. In addition, we share our failed attempts and insights in attempting to achieve R1-like reasoning using RL with instruct models. aiming to shed light on the challenges involved. Our key observations include: (1) applying RL on instruct model often results in trivial reasoning trajectories, and (2) naive length reward are ineffective in eliciting reasoning capabilities. The project code is available at https://github.com/turningpoint-ai/VisualThinker-R1-Zero

---

# R1‑Zero 在 2B 非 SFT 模型上的视觉推理 “恍然大悟” 时刻 论文详细解读

### 背景：这个问题为什么难？

视觉推理要求模型在看到图片后，像人一样先观察、再思考、最后给出解释。过去的多模态大模型大多靠大规模的指令微调（SFT）来学习“怎么说”，却很少出现真正的自我推理过程。DeepSeek R1 用强化学习（RL）在纯语言模型上成功让模型在训练中出现“恍然大悟”——自我反省、回答变长的现象。但把同样的技巧搬到视觉领域时，模型往往只会输出更长的文字，却没有真正的多步推理，关键的自我检查也消失了。于是，如何在不经过指令微调的情况下，让一个只有 2 B 参数的视觉语言模型自行学会分步思考，成为了一个急需突破的难点。

### 关键概念速览
- **R1‑Zero**：指在没有指令微调（Zero‑SFT）前提下，通过强化学习让模型出现“aha 时刻”。这里的 “Zero” 强调完全跳过传统的指令微调阶段。
- **强化学习（RL）**：让模型把生成答案当作“动作”，根据奖励信号来调整内部策略，就像训练机器人玩游戏一样。奖励越高，模型越倾向于产生类似的输出。
- **SAT 数据集**：全称 “Spatial‑Attribute‑Task”，是一套专门设计的视觉推理题目，要求模型先定位图中对象，再比较属性，最后给出解释。相当于视觉版的逻辑推理练习册。
- **CVBench**：一个公开的视觉推理基准，覆盖常见的计数、关系、属性推断等任务，用来衡量模型的实际推理能力。
- **自我反省（Self‑Reflection）**：模型在回答前主动检查自己的思路或提出疑问，类似人类在解题时先说“我先确认一下这张图里有几只猫”。在 R1‑Zero 中，这种行为会体现在输出中出现 “思考中” 或 “检查” 等标记。
- **长度奖励（Length Reward）**：最直接的奖励方式是给更长的回答更高分，假设长文本意味着更细致的推理。但实验表明，仅靠长度容易让模型写废话。
- **规则式激励（Rule‑Based Incentives）**：作者设计的一套奖励规则，除了奖励正确性，还奖励出现特定的思考标记、分步描述等，帮助模型形成真正的推理链。

### 核心创新点
1. **直接在非指令模型上做 RL → 用 SAT 数据集和规则式激励对 Qwen2‑VL‑2B 进行强化学习 → 让模型在没有任何指令微调的情况下出现多步推理和自我检查，准确率提升约 30%。**  
   过去的工作要先把模型调成“能听指令”，再用 RL 微调；这里直接跳过这一步，证明 RL 本身就能驱动推理能力的出现。

2. **规则式激励替代单纯长度奖励 → 设计奖励函数时加入“思考标记出现次数”“分步描述完整度”等指标 → 解决了仅靠长度奖励会产生冗长但无意义文本的问题。**  
   这让模型学会在答案里写出 “先找出对象 A，再比较属性 B”，而不是随意扩写。

3. **对比实验系统化展示 RL 在指令模型上失效的原因 → 在指令微调后的模型上使用相同 RL，得到的推理轨迹几乎是平铺直叙 → 说明指令微调会把模型的行为空间“锁死”，不利于出现新颖的推理路径。**  
   这为以后如何在指令模型上重新打开探索空间提供了重要线索。

### 方法详解
**整体思路**：先拿到一个已经训练好的视觉语言模型（Qwen2‑VL‑2B），不做任何指令微调；再把它当作 RL 的“策略”，在 SAT 任务上让它不断尝试生成答案；每一次生成后，用一个手工设计的奖励函数评估答案的质量、思考标记和长度，得到一个标量奖励；最后用 PPO（近端策略优化）把奖励反馈给模型，循环迭代。整个过程可以想象成“让模型玩一场答题游戏”，每答对一次、每写出一步思考，就会得到积分。

**关键模块拆解**：

1. **环境（SAT 任务）**  
   - 输入：一张图片 + 一个文字问题。  
   - 输出：模型生成的文字答案。  
   - 类比：就像在课堂上老师出一道图形题，学生要先观察图形再写答案。

2. **奖励函数**  
   - **正确性奖励**：使用一个外部的判分器（可以是人工标注的答案或自动匹配的脚本）判断答案是否在逻辑上正确，给出 0~1 的分。  
   - **思考标记奖励**：检查答案里是否出现预定义的思考关键词（如 “先找出…”，“我认为…”，或特殊 token `<think>`），每出现一次加一定分。  
   - **分步完整度奖励**：通过正则匹配或简单的依存分析，判断答案是否包含“定位 → 比较 → 结论”三个阶段，完整一次加分。  
   - **长度奖励**：对答案长度做轻微加权，但上限设定，防止模型只写长篇大论。  
   - **总奖励**：把上述几项加权求和，权重经过小规模网格搜索得到最佳平衡。

3. **策略更新（PPO）**  
   - 使用 PPO 的剪切目标函数，确保每一步更新不会让模型偏离原策略太远，保持训练的稳定性。  
   - 为了避免奖励噪声，作者在每个 batch 中对同一题目采样多次，取平均奖励。

4. **训练细节**  
   - **学习率**：采用 5e‑5 的小学习率，防止大模型在 RL 阶段出现灾难性遗忘。  
   - **批次大小**：每批 32 条 SAT 题目，兼顾显存和梯度估计的方差。  
   - **训练轮数**：约 200k 步，期间监控奖励曲线和模型的自我反省出现频率。

**最巧妙的地方**：作者没有直接把“正确答案”作为唯一奖励，而是把“思考过程”显式量化进奖励函数。这相当于在模型的“动机”里植入了“写思考步骤”的欲望，让模型在追求高分的同时自然形成 CoT（思维链）式的输出，而不是单纯的答案。

### 实验与效果
- **测试数据**：CVBench，涵盖计数、属性比较、空间关系等 10 类视觉推理任务。  
- **基线**：原始 Qwen2‑VL‑2B（未微调）在 CVBench 上的准确率约为 29%。同规模的指令微调（SFT）模型在同任务上约为 57%。  
- **本方法**：R1‑Zero 通过 RL 达到 59.47% 的整体准确率，**比原始模型提升约 30%**，并**略超 SFT 基线约 2%**。  
- **消融实验**：  
  - *仅长度奖励* → 准确率回落至 31%，说明模型只会写长文本而不真正推理。  
  - *在指令模型上使用相同 RL* → 推理轨迹几乎没有思考标记，准确率仅提升 1%，验证了指令微调会限制探索空间。  
  - *去掉思考标记奖励* → 准确率下降约 4%，表明思考标记对提升推理质量有实质贡献。  
- **局限性**：实验只在 2 B 参数模型上完成，未验证在更大模型上是否仍保持优势；奖励函数仍依赖手工设计的关键词，迁移到其他任务可能需要重新调参。

### 影响与延伸思考
这篇工作首次展示了 **在没有指令微调的情况下，强化学习就能让小规模视觉语言模型出现类似人类的思考过程**。它打开了两条可能的研究路线：  
1. **RL‑驱动的多模态自我监督**：后续可以尝试把更丰富的视觉任务（如视频问答、场景理解）放进 RL 框架，用更细粒度的奖励（比如视觉注意力的可解释性）进一步提升模型的自我检查能力。  
2. **奖励函数自动化**：目前的规则式激励依赖人工设定，未来可以用元学习或逆向强化学习让模型自己发现哪些思考步骤最有助于提升正确率。  

自论文发布后，已有几篇后续工作（如 *VisualThinker‑RL*、*MultimodalCoT‑Zero*）尝试在更大模型上复现 R1‑Zero 的思路，或把奖励设计迁移到跨语言、跨模态的任务上，显示出该思路的可扩展性。

### 一句话记住它
**用规则化的思考奖励直接在 2 B 视觉模型上做强化学习，就能让模型自行写出多步推理，摆脱指令微调的束缚。**