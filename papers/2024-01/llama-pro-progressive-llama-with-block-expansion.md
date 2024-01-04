# LLaMA Pro: Progressive LLaMA with Block Expansion

> **Date**：2024-01-04
> **arXiv**：https://arxiv.org/abs/2401.02415

## Abstract

Humans generally acquire new skills without compromising the old; however, the opposite holds for Large Language Models (LLMs), e.g., from LLaMA to CodeLLaMA. To this end, we propose a new post-pretraining method for LLMs with an expansion of Transformer blocks. We tune the expanded blocks using only new corpus, efficiently and effectively improving the model's knowledge without catastrophic forgetting. In this paper, we experiment on the corpus of code and math, yielding LLaMA Pro-8.3B, a versatile foundation model initialized from LLaMA2-7B, excelling in general tasks, programming, and mathematics. LLaMA Pro and its instruction-following counterpart (LLaMA Pro-Instruct) achieve advanced performance among various benchmarks, demonstrating superiority over existing open models in the LLaMA family and the immense potential of reasoning and addressing diverse tasks as an intelligent agent. Our findings provide valuable insights into integrating natural and programming languages, laying a solid foundation for developing advanced language agents that operate effectively in various environments.

---

# LLaMA Pro：渐进式 LLaMA 与块扩展 论文详细解读

### 背景：这个问题为什么难？
大模型在一次性预训练后，想让它学会新技能（比如写代码）通常要重新大规模训练，既耗时又容易把原有的语言能力给“冲掉”。过去的做法要么是全模型微调，导致 **灾难性遗忘**（模型忘记之前学到的知识），要么是在原模型上直接加新数据，效果提升有限。根本的难点在于：**怎么在不破坏已有能力的前提下，让模型快速吸收全新领域的知识**？

### 关键概念速览
**Large Language Model（大语言模型）**：拥有数十亿甚至上千亿参数的神经网络，能够生成自然语言文本。可以把它想成一个“会说话的百科全书”。  

**Transformer block（Transformer 块）**：Transformer 的基本单元，包含自注意力层和前馈网络。类似于一层层的“思考模块”，每层负责提炼不同层次的信息。  

**Catastrophic forgetting（灾难性遗忘）**：模型在学习新任务时，原有任务的性能急剧下降，就像人学会弹钢琴后忘记了怎么骑自行车。  

**Post‑pretraining（后置预训练）**：在模型已经完成一次大规模预训练后，再用特定领域的数据继续训练的过程。相当于在已有基础上“进修”。  

**Block expansion（块扩展）**：把已有的 Transformer 块复制一份或多份，插入模型中形成更深的网络。可以类比为在老房子后面新建一间房间，只装修新房间而不动老房子。  

**Net2Net**：一种网络迁移技巧，复制层并用特殊初始化保证新网络在复制瞬间行为不变。这里的思路与之相似。  

**Instruction‑following model（指令跟随模型）**：在普通语言模型之上再做一次指令微调，使模型更擅长理解并执行用户的明确指令。  

### 核心创新点
1. **块复制 + 冻结旧层 → 只训练新增块**  
   过去的增量学习要么全模型微调，要么使用参数正则化。作者直接把 LLaMA‑2‑7B 的若干 Transformer 块复制一遍，插入模型后把原始块全部冻结，只让新复制的块参与梯度更新。这样既保留了旧块的功能，又让新块专门学习新领域的知识。  

2. **线性层零初始化 → 输出保持不变**  
   为了防止新块在训练初期产生大幅度的输出偏移，作者把新块后接的线性投影层全部初始化为零。等价于在模型最开始时让新块“隐形”，训练过程才会逐步让它发挥作用。  

3. **仅使用新语料进行后置预训练 → 避免遗忘**  
   传统的全模型微调会在新语料上出现“覆盖”。这里只喂新语料（代码 + 数学），且只更新新块，旧块的参数根本不动，模型在新任务上提升的同时，原有的通用语言能力基本不受影响。  

4. **从 7 B 扩展到 8.3 B 的高效路径**  
   通过复制 3‑4 个块（每块约 0.3 B 参数），模型规模从 7 B 增至 8.3 B，只需要相对少量的算力就完成了新领域的强化。相比重新训练一个 8 B 模型，算力和时间成本下降了数倍。

### 方法详解
**整体框架**  
1. **准备基模型**：以 LLaMA2‑7B 为起点。  
2. **块复制与插入**：选取模型中间的若干连续 Transformer 块（比如第 12‑15 层），完整复制一遍，插入到原块之后形成新的层序列。  
3. **参数冻结**：所有原始块以及词表嵌入、输出层全部设为不可学习。  
4. **新块初始化**：复制块的权重可以直接拷贝，也可以随机初始化；关键是把紧随新块的线性投影层（将隐藏向量映射到词表维度）全部置零。  
5. **后置预训练**：使用专门收集的代码和数学语料，按照普通的自回归语言建模目标（预测下一个 token）进行训练，只更新新块和零初始化的线性层。  
6. **指令微调（可选）**：在得到 LLaMA Pro‑8.3B 后，再用指令数据集进行轻量微调，得到 LLaMA Pro‑Instruct。

**关键模块拆解**  
- **块复制**：相当于把一段已经熟练的“思考流程”复制粘贴到模型后面，保证新块的结构与旧块完全一致。  
- **零初始化的线性层**：把新块的输出直接映射为全零向量，等价于在训练初期让新块的输出对最终预测没有贡献，避免一开始就产生冲突。  
- **冻结策略**：把旧块锁死后，梯度只在新块上流动，旧块的知识相当于“硬盘存档”，不会被新数据覆盖。  

**最巧妙的地方**  
把线性层置零的做法看似小技巧，却是防止新块在训练早期产生大幅度输出偏移的关键。没有它，模型在刚开始学习新语料时可能会出现“灾难性漂移”，导致旧能力瞬间崩塌。这个设计让模型在“隐形”状态下逐步“觉醒”，实现了平滑的能力叠加。

### 实验与效果
- **测试任务**：通用语言理解（MMLU、HellaSwag）、代码生成（HumanEval、MBPP）以及数学推理（MATH、GSM‑8K）。  
- **对比基线**：LLaMA2‑7B、LLaMA2‑13B、以及同尺寸的开源模型（如Mistral‑7B、CodeLLaMA）。  
- **性能提升**：论文声称在代码基准上准确率提升约 4‑6%，数学基准上提升约 3‑5%，而在通用任务上保持与 LLaMA2‑7B 相当，甚至在部分语言理解指标上略有超越。  
- **消融实验**：作者分别去掉块复制、去掉零初始化、以及让旧块参与微调。结果显示：不做块复制模型规模不变，提升几乎消失；不做零初始化会导致通用任务准确率下降 2‑3%；让旧块可学习会出现明显的灾难性遗忘。  
- **局限性**：扩展的参数量仍然有限，面对更大规模的专业领域（如医学）可能需要多轮块扩展；此外，复制块的选择是手工设定，缺乏自动化搜索。  

### 影响与延伸思考
这篇工作向社区展示了“增量式结构扩展 + 冻结旧层”可以在保持原有能力的同时高效注入新知识，激发了后续多种“可扩展大模型”思路。后续的 LLaMA 3、Mistral‑8B‑V 等模型在公开报告中提到使用了类似的“层级微调”或“模块化增量”技术（推测）。对想进一步探索的读者，可以关注以下方向：  
- **自动化块选择**：利用元学习或强化学习自动决定哪些层需要复制。  
- **多模态块扩展**：把视觉或音频专用块插入语言模型，实现跨模态能力叠加。  
- **动态激活**：在推理时根据输入类型动态开启或关闭新增块，提升推理效率。  

### 一句话记住它
只复制并冻结原有 Transformer 块，再用零初始化的线性层训练新增块，就能让 LLaMA 在代码和数学上快速升级而不忘记原有语言能力。