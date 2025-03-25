# Overtrained Language Models Are Harder to Fine-Tune

> **Date**：2025-03-24
> **arXiv**：https://arxiv.org/abs/2503.19206

## Abstract

Large language models are pre-trained on ever-growing token budgets under the assumption that better pre-training performance translates to improved downstream models. In this work, we challenge this assumption and show that extended pre-training can make models harder to fine-tune, leading to degraded final performance. We term this phenomenon catastrophic overtraining. For example, the instruction-tuned OLMo-1B model pre-trained on 3T tokens leads to over 2% worse performance on multiple standard LLM benchmarks than its 2.3T token counterpart. Through controlled experiments and theoretical analysis, we show that catastrophic overtraining arises from a systematic increase in the broad sensitivity of pre-trained parameters to modifications, including but not limited to fine-tuning. Our findings call for a critical reassessment of pre-training design that considers the downstream adaptability of the model.

---

# 过度训练的语言模型更难微调 论文详细解读

### 背景：这个问题为什么难？
在大模型时代，研究者普遍相信，只要把模型在海量文本上训练得更久，后续的下游任务表现就会更好。于是预训练的 token 数量（即模型看到的词语总量）一直在突破 2 T、3 T，甚至更高。可是，这种“一味加量”的思路忽视了模型在预训练阶段已经形成的内部表征是否仍然容易被微调所调整。实际使用中，很多团队发现，越是“超大”预训练的模型，微调时需要的学习率、梯度步数甚至调参技巧都变得更苛刻，最终的任务成绩反而不如稍微“少训练”一点的版本。这个现象在之前的工作里没有系统的解释，也没有对应的理论支撑，导致大家在追求更大预训练规模时，往往会踩到性能瓶颈，却不自知。

### 关键概念速览
**预训练（pre‑training）**：在大规模未标注文本上让模型学习语言规律，就像让学生先读大量教材，为后面的专业课打基础。  
**微调（fine‑tuning）**：在特定任务的数据上继续训练模型，让它把通用语言能力转化为具体任务的技能，类似学生在掌握基础后选修专业课程。  
**Token**：文本被切分成的最小单元，常是词或子词，模型的每一次学习都是对这些单元的预测。  
**Catastrophic overtraining（灾难性过度训练）**：指预训练时间过长导致模型对后续任何参数修改都变得异常敏感，微调效果反而下降的现象。可以把它想象成一块已经被雕刻得非常精细的石像，想再改动一点细节就会导致整体结构失衡。  
**Broad sensitivity（广泛敏感性）**：模型参数对外部扰动（如微调梯度）的响应范围扩大，等价于把原本“弹性适中”的橡皮筋拉得太紧，一点点拉伸就会断裂。  
**Instruction‑tuned（指令微调）**：在预训练后加入大量指令式数据，让模型学会遵循自然语言指令，类似给学生额外的“使用手册”。  

### 核心创新点
1. **从经验假设到系统实验 → 通过控制预训练 token 数量进行对比实验 → 揭示了更长预训练不一定带来更好下游表现**。作者分别训练了同一模型架构在 2.3 T、3 T token 上的版本，随后在多个公开 LLM 基准上进行指令微调，发现 3 T 版本在所有指标上平均下降超过 2%。这一步直接打破了“越大越好”的经验法则。  
2. **提出“灾难性过度训练”概念 → 用参数敏感性度量工具（如梯度范数、Hessian 近似）量化预训练后模型对微调的易损程度 → 证明预训练越久，参数对微调的响应幅度越大**。通过对比不同预训练阶段的梯度分布，作者发现后期模型的梯度更稀疏且幅度更高，说明微调时容易产生剧烈参数跳动。  
3. **理论分析解释广泛敏感性的根源 → 引入“参数平滑度”概念，证明在极大 token 规模下，模型在损失曲面上趋向于形成更陡峭的局部极小点 → 这些极小点对微调的扰动极为敏感**。这一步提供了从优化理论角度解释为何过度预训练会导致微调困难。  
4. **提供设计建议 → 强调在预训练阶段加入“适度正则化”或“梯度噪声注入”可以抑制敏感性增长 → 为后续的预训练策略提供了可操作的方向**。虽然这部分在实验中仅作初步验证，但已经指明了可能的缓解路径。

### 方法详解
整体思路可以拆成三大块：① 构造对照实验，② 量化参数敏感性，③ 理论解释与缓解建议。

**第一步：对照实验**  
作者选取了同一模型（OLMo‑1B）在两个不同的预训练规模上进行训练：一个在 2.3 T token，另一个在 3 T token。两者在模型结构、优化器、学习率调度等超参数上保持完全一致，只是训练步数不同。随后对两者进行相同的指令微调流程，使用公开的 LLM 基准（如 MMLU、TruthfulQA 等）评估微调后的性能。这样做的好处是排除了模型容量、数据质量等变量的干扰，直接把“预训练时长”作为唯一自变量。

**第二步：敏感性度量**  
为了捕捉模型在微调时的“易动”程度，作者设计了两类指标：  
- **梯度范数分布**：在微调的第一批梯度更新中，统计每层参数梯度的 L2 范数。若范数普遍偏大，说明微调需要更大的参数调整。  
- **Hessian 近似（曲率）**：使用随机子空间方法近似二阶导数矩阵的特征值，评估损失曲面的陡峭程度。特征值越大，表示参数所在的局部极小点越尖锐，对微调扰动更敏感。  
实验发现，3 T 预训练模型的梯度范数整体提升约 30%，Hessian 最大特征值也显著增大，形成了“广泛敏感性”。

**第三步：理论解释**  
作者把预训练过程抽象为在高维参数空间中寻找一个局部最小点。随着 token 数量增大，优化过程会在更细的噪声梯度下进行更长时间，导致参数逐渐收敛到更窄、更深的谷底。作者用“参数平滑度”来描述这个谷底的宽度：平滑度低意味着曲面更陡，微小的梯度更新会导致参数跳出谷底，产生显著性能波动。基于此，他们推导出一个不等式，说明在固定模型容量下，预训练 token 超过某个阈值后，平滑度会呈指数下降，从而解释了灾难性过度训练的出现。

**缓解建议**  
在实验的延伸部分，作者尝试在 3 T 预训练过程中加入额外的正则化项（如权重衰减增强）和梯度噪声（Gaussian noise injection），观察敏感性指标的变化。结果显示，噪声注入能够轻微提升参数平滑度，降低微调时的梯度范数，但仍未完全恢复到 2.3 T 规模的水平。作者把这视为一种“预防性”手段，提示后续工作可以进一步探索更有效的正则化策略。

### 实验与效果
- **测试任务**：指令微调后在多个标准 LLM 基准上评估，包括 MMLU（多学科知识测验）、TruthfulQA（事实性问答）以及 GSM‑8K（数学推理）等。  
- **对比基线**：2.3 T 预训练模型（同架构）以及公开的同等规模模型（如 LLaMA‑7B 的子模型）作为参考。  
- **主要发现**：在所有基准上，3 T 预训练模型的得分均比 2.3 T 版本低约 2%–3%（论文声称），且在微调过程中的收敛速度更慢，需要更小的学习率才能避免梯度爆炸。  
- **消融实验**：作者分别去掉梯度噪声、正则化等干预手段，观察敏感性指标的回升，证实这些因素对抑制广泛敏感性有一定贡献。  
- **局限性**：实验仅在单一模型架构（OLMo‑1B）和少数指令微调设置下进行，尚未验证在更大模型（如 7 B、30 B）或不同预训练数据分布上的普适性。作者也承认，当前的理论分析仍是粗略的近似，缺少对实际训练噪声分布的精细建模。

### 影响与延伸思考
这篇工作在社区里掀起了对“预训练规模无限增长”假设的反思。随后出现的几篇论文（如《Scaling Laws with Adaptability Constraints》《Regularized Pre‑training for Better Fine‑tuning》）直接引用了灾难性过度训练的概念，尝试在更大模型上加入自适应正则化或多任务预训练，以保持参数的平滑度。还有一些研究把“敏感性度量”作为模型评估的新指标，帮助选取更易微调的预训练检查点。想进一步了解这条线索的读者可以关注 **适应性预训练（adaptive pre‑training）** 和 **梯度噪声注入（gradient noise injection）** 两个方向，尤其是近期在大模型训练框架中加入噪声的实现细节。

### 一句话记住它
预训练越久，模型越“硬”，微调越难——适度的预训练才是下游适应性的关键。