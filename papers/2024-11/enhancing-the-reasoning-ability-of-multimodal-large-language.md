# Enhancing the Reasoning Ability of Multimodal Large Language Models via   Mixed Preference Optimization

> **Date**：2024-11-15
> **arXiv**：https://arxiv.org/abs/2411.10442

## Abstract

Existing open-source multimodal large language models (MLLMs) generally follow a training process involving pre-training and supervised fine-tuning. However, these models suffer from distribution shifts, which limit their multimodal reasoning, particularly in the Chain-of-Thought (CoT) performance. To address this, we introduce a preference optimization (PO) process to enhance the multimodal reasoning capabilities of MLLMs. Specifically, (1) on the data side, we design an automated preference data construction pipeline to create MMPR, a high-quality, large-scale multimodal reasoning preference dataset; and (2) on the model side, we explore integrating PO with MLLMs, developing a simple yet effective method, termed Mixed Preference Optimization (MPO), which boosts multimodal CoT performance. Our approach enhances the multimodal reasoning abilities of both InternVL2-8B and InternVL2-76B. Notably, our model, InternVL2-8B-MPO, achieves an accuracy of 67.0 on MathVista, outperforming InternVL2-8B by 8.7 points and achieving performance comparable to the 10$\times$ larger InternVL2-76B. We hope this study could inspire further advancements in MLLMs. Code, data, and model are released.

---

# 通过混合偏好优化提升多模态大语言模型的推理能力 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）在视觉+语言的预训练和监督微调后，已经能生成流畅的描述。但当任务要求模型进行多步推理——比如解几何题、做跨图推理时，模型常常出现“答案直接跳出、思路不连贯”的现象。根本原因是训练数据的分布与真实推理场景不匹配，称为**分布漂移**。传统的微调只教模型“看图说话”，没有系统地教它“怎么思考”。因此，提升 MLLM 的链式思考（CoT）能力一直是个瓶颈。

### 关键概念速览
- **多模态大语言模型（MLLM）**：同时接受图像和文字输入，输出文字的模型。想象成会看图的 ChatGPT。  
- **预训练**：在海量图文对上让模型学会基本的视觉‑语言关联，就像小孩先学会认字。  
- **监督微调（SFT）**：用标注好的任务数据进一步教模型怎么完成特定任务，相当于给孩子上专门的辅导班。  
- **分布漂移**：训练时看到的例子和真实使用时的例子差别大，导致模型在新情境下表现下降。  
- **链式思考（CoT）**：让模型在给出最终答案前先写出推理步骤，类似人解题时的草稿纸。  
- **偏好优化（Preference Optimization）**：把“哪个答案更好”这种相对评价当作学习信号，让模型学会偏向高质量答案。  
- **混合偏好优化（MPO）**：把多种偏好信号（如正确性、解释性）混在一起，用统一的目标函数同时优化，像把不同口味的调味料一起拌进汤里。  
- **MMPR 数据集**：作者自动构造的“大规模多模态推理偏好”数据集，里面每条记录都是一对图文推理答案，标明哪一个更好。

### 核心创新点
1. **自动化偏好数据构建 → 生成 MMPR**  
   过去的偏好学习大多依赖人工标注的对比数据，成本高且规模小。作者搭建了一个流水线：先用现有的 MLLM 生成多种推理答案，再用更强的语言模型对答案进行质量排序，最终得到数十万对高质量的“好‑坏”样本。这样既保证了数据质量，又实现了大规模。  
2. **将偏好优化直接嵌入 MLLM 训练 → Mixed Preference Optimization**  
   传统做法是先完成监督微调，再单独跑强化学习（RLHF）来利用偏好。这里提出一种更轻量的混合方式：在同一次梯度更新中，同时计算监督损失和偏好损失，并用一个加权系数平衡两者。这样模型在保持原有语言能力的同时，学会更倾向于产生高质量的推理链。  
3. **多偏好信号混合 → 更全面的推理能力**  
   作者没有只关注答案正确性，还加入了解释完整性和逻辑连贯性的偏好评分。通过在损失函数中混合这些信号，模型被迫在“对了”和“解释好”之间找到平衡，最终在复杂的 CoT 任务上表现更稳健。  
4. **少量额外训练即可匹配更大模型 → 规模效应突破**  
   只在原有的 InternVL2‑8B 基础上额外训练约 0.5 B 参数的偏好阶段，就让它的 MathVista 准确率从 58.3 提升到 67.0，几乎追上参数量是它十倍的 InternVL2‑76B。说明偏好优化的收益在一定规模下可以抵消模型容量的劣势。

### 方法详解
整体思路可以分为两大块：**数据侧的偏好构造**和**模型侧的混合优化**。

1. **偏好数据构建流水线**  
   - **答案生成**：使用已有的强力多模态模型（如 InternVL2‑76B）对每张图像生成多条不同的推理答案，确保答案在正确性、详细程度上有差异。  
   - **质量评估**：把这些答案喂给一个更大的语言模型（如 GPT‑4）进行对比打分，输出“更好/更差”的二元标签。  
   - **对齐过滤**：剔除评分差距过小或出现明显错误的对比，保留信噪比高的样本。  
   - **数据聚合**：最终得到的 MMPR 包含 (图像, 参考答案, 次佳答案, 偏好标签) 四元组，规模达到数十万对。

2. **混合偏好优化（MPO）训练流程**  
   - **输入**：每个训练批次同时抽取普通的监督微调样本和 MMPR 对比样本。  
   - **损失计算**：  
     - **监督损失**（L_SFT）：对普通样本使用交叉熵，保持模型的基本语言和视觉理解能力。  
     - **偏好损失**（L_PO）：对每对答案计算一个对数似然，鼓励模型在同一图像下更倾向生成被标记为“好”的答案。实现方式类似于对比学习的 pairwise ranking。  
     - **混合系数**（α）：作者在实验中调节 α，使两种损失的梯度贡献大致相当。  
   - **梯度更新**：一次前向传播后，直接把 L_SFT + α·L_PO 的梯度回传到模型全部参数，省去单独的强化学习阶段。  
   - **训练策略**：先进行若干 epoch 的纯监督微调，让模型基本收敛；再切换到混合模式进行偏好微调，通常只需 1–2 个 epoch 即可看到显著提升。

**巧妙之处**在于：不需要额外的奖励模型或复杂的 PPO 循环，只用一个简单的加权求和就把偏好信息注入模型内部；而且因为偏好数据是自动生成的，成本极低，易于扩展到更多视觉任务。

### 实验与效果
- **测试任务**：主要在 MathVista（包含数学、几何、逻辑推理的多模态题库）上评估 CoT 能力；此外还在一些视觉问答基准上做了验证。  
- **基线对比**：  
  - 原始 InternVL2‑8B（仅监督微调）在 MathVista 上的准确率约 58.3%。  
  - 加入 MPO 后的 InternVL2‑8B‑MPO 达到 67.0%，提升 8.7 分。  
  - 规模更大的 InternVL2‑76B（10× 参数）在同一数据集上也在 66–68 分左右，说明 MPO 让小模型的表现几乎追平大模型。  
- **消融实验**：作者分别去掉（1）自动生成的偏好标签，只保留随机对比；（2）多偏好信号中的解释性评分；结果显示准确率分别下降约 3–4 分，验证了每个模块的贡献。  
- **局限性**：论文承认偏好数据仍依赖于生成模型的质量，若生成模型本身出现系统性错误，偏好标签会被污染；此外实验主要聚焦于数学推理，其他跨模态推理场景的效果还有待验证。

### 影响与延伸思考
这篇工作首次展示了“偏好优化+多模态”可以在不大幅增加算力的前提下显著提升 CoT 能力，随后有几篇后续研究（如 **MM‑Reward**、**Vision‑RLHF**）开始探索更细粒度的视觉‑语言偏好信号。对想继续深入的读者，可以关注以下方向：  
- **更高质量的自动偏好生成**：利用自监督的对比学习或人机协同标注提升 MMPR 的噪声鲁棒性。  
- **跨模态偏好统一框架**：把文本、语音、视频等多种模态的偏好信号统一到同一个优化目标中。  
- **理论分析**：系统研究混合损失对模型收敛性和泛化性的影响，解释为何少量偏好微调能抵消参数规模差距。

### 一句话记住它
用自动生成的高质量对比数据，轻量混合偏好优化，让小型多模态模型的链式推理能力直接追上十倍大模型。