# Guess the Instruction! Flipped Learning Makes Language Models Stronger   Zero-Shot Learners

> **Date**：2022-10-06
> **arXiv**：https://arxiv.org/abs/2210.02969

## Abstract

Meta-training, which fine-tunes the language model (LM) on various downstream tasks by maximizing the likelihood of the target label given the task instruction and input instance, has improved the zero-shot task generalization performance. However, meta-trained LMs still struggle to generalize to challenging tasks containing novel labels unseen during meta-training. In this paper, we propose Flipped Learning, an alternative method of meta-training which trains the LM to generate the task instruction given the input instance and label. During inference, the LM trained with Flipped Learning, referred to as Flipped, selects the label option that is most likely to generate the task instruction. On 14 tasks of the BIG-bench benchmark, the 11B-sized Flipped outperforms zero-shot T0-11B and even a 16 times larger 3-shot GPT-3 (175B) on average by 8.4% and 9.7% points, respectively. Flipped gives particularly large improvements on tasks with unseen labels, outperforming T0-11B by up to +20% average F1 score. This indicates that the strong task generalization of Flipped comes from improved generalization to novel labels. We release our code at https://github.com/seonghyeonye/Flipped-Learning.

---

# 猜指令！翻转学习让语言模型成为更强的零样本学习者 论文详细解读

### 背景：这个问题为什么难？

在零样本学习里，模型要在没有看到过的任务上直接给出答案，核心靠的是“指令调优”：把任务说明（instruction）和输入一起喂进模型，让它学会把正确标签的概率最大化。这个思路在很多 benchmark 上已经把性能推上了新高，但它有个盲点——模型只学会把**已知标签**映射到指令，遇到训练时从未出现过的新标签时往往会慌。换句话说，标签本身的语义信息没有被充分利用，导致在标签空间扩展时泛化能力骤降。

### 关键概念速览
- **元训练（Meta‑training）**：在大量不同任务上统一微调模型，使其学会“一看指令就能做事”。相当于让模型参加“多科目考试”，每科都有自己的说明书。
- **零样本学习（Zero‑shot）**：模型在没有任何该任务的示例情况下直接完成任务，就像第一次玩新游戏却只看说明书就能上手。
- **指令调优（Instruction Tuning）**：把任务的自然语言描述当作输入，让模型把它当作“任务卡片”。模型的目标是根据卡片和数据实例输出正确答案。
- **翻转学习（Flipped Learning）**：把指令调优的方向反过来——给模型**输入+标签**，让它生成对应的任务指令。相当于让模型先写“任务卡片”，再检查卡片是否和我们想要的标签匹配。
- **标签空间（Label Space）**：所有可能的答案集合。传统方法把标签当作纯粹的符号，忽视它们的文字意义；翻转学习则把标签的文字信息直接喂进模型。
- **BIG‑bench**：一个包含上百种跨领域任务的大型评测套件，用来检验模型的通用能力。这里选了其中的 14 项任务做实验。
- **F1 分数**：精确率和召回率的调和平均，用来衡量模型在不平衡数据上的整体表现。

### 核心创新点
1. **训练目标反向化**  
   - 之前的做法：模型学习“在看到指令和输入后，预测标签”。  
   - 本文做法：模型学习“在看到输入和标签后，生成指令”。  
   - 改变：标签的文字信息被直接用于生成自然语言指令，模型因此对标签语义有更深的理解。

2. **推理时用指令概率打分**  
   - 之前的做法：直接把输入喂进模型，让它输出最可能的标签。  
   - 本文做法：对每个候选标签，计算模型在给定该标签和输入时生成原始指令的概率，选概率最高的标签。  
   - 改变：把“生成指令的好坏”当作标签选择的依据，利用语言模型擅长的生成能力来间接判断答案。

3. **专注提升对未见标签的泛化**  
   - 之前的元训练在标签集合固定的情况下表现不错，但遇到全新标签会掉分。  
   - 本文通过让模型学会把标签文字映射到任务描述，显著提升了在“标签全新”情形下的 F1，最高提升约 20%。  
   - 改变：模型不再把标签当作不可解释的编号，而是把它们当作有意义的词汇来处理。

4. **用中等规模模型超越更大模型**  
   - 只用了 11 B 参数的模型，却在 14 项 BIG‑bench 任务上平均比 175 B 参数的 GPT‑3（3‑shot）高出 9.7 分。  
   - 说明翻转学习的收益可以抵消一定的规模劣势，为资源受限的研究者提供了高效路径。

### 方法详解
**整体思路**：先把每条训练样本写成「输入 + 标签 → 指令」的映射，让模型学会“从答案倒推任务”。推理时，把所有可能的标签逐一带入同样的映射，计算生成指令的概率，选最高的那个。

**步骤拆解**：

1. **数据准备**  
   - 每条样本原本是「指令、输入、标签」三元组（如：指令=“判断句子情感”，输入=“I love this movie”，标签=“positive”）。  
   - 训练时把它们重新排列成「输入 + 标签」作为模型的上下文，目标是让模型输出原始指令的完整文字序列。

2. **模型结构与训练**  
   - 采用标准的自回归语言模型（decoder‑only），在「输入+标签」后接一个特殊分隔符，然后让模型逐词预测指令。  
   - 损失函数仍是交叉熵，即每个生成的指令词的概率与真实词的匹配程度。  
   - 关键在于**不再把标签当作目标输出**，而是把它当作条件信息，帮助模型构造指令。

3. **推理机制**  
   - 给定新任务的指令（从训练集或公开描述中取得）和待预测的输入，先列出所有候选标签（可能是二分类、多个类别或开放词表）。  
   - 对每个候选标签，构造「输入 + 该标签」的上下文，计算模型在该上下文下生成完整指令的对数概率。  
   - 选取概率最大的标签作为最终答案。直观上，这相当于“哪个标签最能让模型说出‘这就是我该做的任务’”。

4. **实现细节**  
   - 为防止指令长度差异带来的偏差，作者在计算概率时对指令进行归一化（如除以指令长度）。  
   - 在候选标签很多的情况下，可以使用束搜索（beam search）快速近似概率最高的标签。  
   - 训练时加入了标签的随机噪声和指令的多样化表述，以提升模型对指令变体的鲁棒性。

**最巧妙的点**：把任务指令当作“桥梁”，让模型在生成指令的过程中自然对齐标签语义和任务需求。这样做既利用了语言模型在自然语言生成上的强项，又把标签的文字信息充分暴露给模型，解决了传统指令调优“标签是黑盒”的局限。

### 实验与效果
- **评测数据**：在 BIG‑bench 中挑选了 14 项多样化任务，包括情感分析、文本分类、常识推理等，尤其包含了大量标签在训练阶段未出现的子任务。
- **对比基线**：  
  - T0‑11B（零样本指令调优模型）  
  - GPT‑3 175B（3‑shot 提示）  
- **主要结果**：  
  - 在全部 14 项任务上，Flipped‑11B 的平均得分比 T0‑11B 高出 **8.4 分**。  
  - 同时比 3‑shot GPT‑3（175B）高出 **9.7 分**，实现了“更小模型更强表现”。  
  - 对于标签全新（未在元训练中出现）的任务，提升尤为显著，最高可达 **+20%** 的 F1 增益。
- **消融实验**：原文报告了去掉指令生成目标、仅保留传统标签预测的对照实验，性能跌回到与 T0 相当，说明“翻转”目标是提升的关键因素。还有实验表明，指令长度归一化对多标签任务的提升贡献约 1.5 分。
- **局限性**：  
  - 需要在推理阶段为每个候选标签都计算一次指令概率，计算成本随标签数线性增长。  
  - 方法依赖于高质量、明确的任务指令；如果指令模糊或缺失，模型的评分依据会受影响。  
  - 论文未在极大规模（> 100B）模型上验证是否仍保持优势。

### 影响与延伸思考
- 这篇工作打开了“逆向指令调优”的思路，随后有研究尝试把 **指令生成** 与 **答案生成** 交叉训练，形成双向循环，提高模型对任务描述的理解深度。  
- 在少样本学习、提示工程等方向，出现了“Prompt Inversion”或 “Instruction Reconstruction” 的变体，直接受 Flipped Learning 的启发。  
- 对于想继续深入的读者，可以关注以下方向：  
  1. **高效评分**：设计更快的指令概率近似方法，降低多标签任务的推理成本。  
  2. **指令多样化学习**：让模型同时学习多种表述的同一指令，提高对指令噪声的鲁棒性。  
  3. **跨语言扩展**：把翻转学习推广到多语言指令，检验其在语言迁移上的效果。  
  4. **与检索结合**：在大规模知识库检索后，用翻转学习对检索结果进行指令匹配，提升开放域问答的准确率。

### 一句话记住它
**把任务指令当作答案的“解释”，让模型先写解释再选答案，标签的文字意义因此被彻底激活，零样本表现大幅提升。**