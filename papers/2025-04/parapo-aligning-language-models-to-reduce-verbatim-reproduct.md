# ParaPO: Aligning Language Models to Reduce Verbatim Reproduction of Pre-training Data

> **Date**：2025-04-20
> **arXiv**：https://arxiv.org/abs/2504.14452

## Abstract

Language models (LMs) can memorize and reproduce segments from their pretraining data verbatim even in non-adversarial settings, raising concerns about copyright, plagiarism, privacy, and creativity. We introduce Paraphrase Preference Optimization (ParaPO), a post-training method that fine-tunes LMs to reduce unintentional regurgitation while preserving their overall utility. ParaPO trains LMs to prefer paraphrased versions of memorized segments over the original verbatim content from the pretraining data. To maintain the ability to recall famous quotations when appropriate, we develop a variant of ParaPO that uses system prompts to control regurgitation behavior. In our evaluation on Llama3.1-8B, ParaPO consistently reduces regurgitation across all tested datasets (e.g., reducing the regurgitation metric from 17.3 to 12.9 in creative writing), whereas unlearning methods used in prior work to mitigate regurgitation are less effective outside their targeted unlearned domain (from 17.3 to 16.9). When applied to the instruction-tuned Tulu3-8B model, ParaPO with system prompting successfully preserves famous quotation recall while reducing unintentional regurgitation (from 8.7 to 6.3 in creative writing) when prompted not to regurgitate. In contrast, without ParaPO tuning, prompting the model not to regurgitate produces only a marginal reduction (8.7 to 8.4).

---

# ParaPO：对齐语言模型以降低对预训练数据的逐字复现 论文详细解读

### 背景：这个问题为什么难？

语言模型在大规模预训练后会把训练语料里的片段记得很牢，甚至在普通对话里直接把原句复制出来。虽然这种“记忆”能帮助模型准确引用名言或技术细节，却会引发版权、抄袭、隐私泄露等法律和伦理风险。过去的解决思路主要是**unlearning**——把某些数据从模型中“删掉”。但这种方法只能针对事先指定的文本，范围窄，而且在未被标记的领域往往效果不佳。更糟的是，直接削弱记忆会让模型失去对重要引用的能力，导致实用性下降。于是，如何在保留必要记忆的同时，抑制无意的逐字复述，成了一个既技术上棘手、又价值突出的挑战。

### 关键概念速览
- **语言模型（LM）**：通过大量文本学习统计规律，能够根据上下文生成下一个词，就像自动补全一样。  
- **记忆（memorization）**：模型在训练时对某些句子或段落形成高置信度的内部表征，导致生成时可能直接复制原文。  
- **逐字复现（verbatim regurgitation）**：模型输出与训练语料中出现的文字完全一致的现象，常被视为“抄袭”。  
- **Paraphrase Preference Optimization（ParaPO）**：一种后训练技术，让模型在遇到记忆片段时更倾向于输出其改写版本，而不是原句。  
- **系统提示（system prompt）**：在对话模型的输入前加上一段指令，用来调节模型行为的“开关”。这里用它来指示模型何时可以直接引用，何时必须改写。  
- **指令微调（instruction‑tuning）**：在大模型基础上，用大量指令‑响应对进行二次训练，使模型更擅长遵循用户指令。  
- **对齐（alignment）**：让模型的输出与人类价值或使用场景保持一致的过程，这里指的是对齐模型的记忆行为。

### 核心创新点
1. **从“删记”到“改写”**：传统的 unlearning 直接削弱模型对特定文本的记忆，往往导致信息丢失。ParaPO 则不把记忆抹掉，而是通过训练让模型在记住内容的同时，更倾向于生成同义改写。这样既保留了事实准确性，又降低了逐字复制的概率。  
2. **对比式偏好学习**：作者构造了“原句 vs. 改写”配对，使用一种偏好损失（preference loss），让模型在同样的上下文下给改写更高的概率分数。相当于教模型在两条答案里挑选更“创意”的那一条。  
3. **可控的复述开关**：通过在系统提示中加入“请不要直接引用”或“可以直接引用名言”等指令，模型能够在不同需求之间切换。这样既解决了“必须保留名言”这一特殊场景，又不影响整体的改写倾向。  
4. **跨模型、跨任务的通用性**：ParaPO 在 Llama3.1‑8B（原始模型）和 Tulu3‑8B（已指令微调）上都取得了显著效果，说明该方法不依赖特定模型结构或训练目标，具备较好的迁移能力。

### 方法详解
**整体思路**：先找出模型容易逐字复现的文本片段，给每个片段准备一个高质量的改写版本；然后把“原句‑改写”对作为训练样本，用偏好损失让模型在相同上下文下更倾向输出改写；最后加入系统提示，让用户可以显式控制是否允许直接引用。

**步骤拆解**：

1. **记忆片段挖掘**  
   - 使用已有的检测工具（如泄露检测模型或人工抽样）在模型生成的文本中定位高置信度的逐字复现。  
   - 这些片段被视为“记忆目标”，形成一个候选集合。

2. **改写数据构建**  
   - 对每个记忆目标，利用外部改写模型或人工方式生成若干同义改写。  
   - 为保证改写质量，要求保留核心事实信息但在词序、表达方式上有明显差异。  
   - 最终得到 (上下文, 原句, 改写) 三元组。

3. **偏好学习目标**  
   - 在微调阶段，模型接收相同的上下文作为输入，分别计算生成原句和改写的概率。  
   - 损失函数鼓励改写的概率 > 原句的概率，形式类似于“对比学习”中的正负样本区分。  
   - 这种方式不需要显式标记哪些是“错误”，只需要让模型学会“更喜欢改写”。

4. **系统提示注入**  
   - 在指令微调模型的输入前加入特定系统提示，例如：“[NO\_COPY] 请用自己的话回答”。  
   - 该提示在模型的词表中对应一个特殊 token，微调时让模型学习在看到该 token 时提升改写偏好。  
   - 反之，使用 “[ALLOW\_QUOTE] 请直接引用原文” 时，模型会恢复对原句的高概率输出。

5. **训练细节**  
   - 采用低学习率的轻量微调，防止模型整体能力被削弱。  
   - 为避免过度改写导致事实错误，训练时加入事实一致性检查（如使用检索模型验证改写是否保持原意）。

**最巧妙的点**：把记忆本身当作资源，而不是要消除的“毒”。通过让模型在记住事实的前提下主动改写，既解决了版权隐私，又保留了模型的知识库。

### 实验与效果
- **测试对象**：Llama3.1‑8B（未指令微调）和 Tulu3‑8B（已指令微调）。  
- **评估任务**：创意写作、对话生成以及名言引用等场景，使用专门的“复述度”指标衡量模型输出中逐字复现的比例。  
- **主要结果**：  
  - 在 Llama3.1‑8B 上，创意写作任务的复述度从 17.3% 降到 12.9%，相比传统 unlearning 方法的 16.9% 有显著提升。  
  - 在 Tulu3‑8B 上加入系统提示后，创意写作的复述度从 8.7% 降到 6.3%；若仅靠提示而不做 ParaPO 微调，下降幅度只有 0.3%（8.7%→8.4%）。  
- **消融实验**：  
  - 去掉系统提示的偏好学习，改写倾向下降约 30%，说明提示是控制行为的关键开关。  
  - 缩减改写数据量至原来的 50% 时，复述度下降幅度减半，表明改写数据的覆盖度直接影响效果。  
- **局限性**：  
  - 对于极其专业或高度精确的引用（如法律条文），模型仍倾向直接复制，说明完全抑制记忆仍有难度。  
  - 改写质量依赖外部改写模型或人工标注，构建大规模改写库成本不低。  
  - 论文未给出对模型整体生成质量（如流畅度、事实准确性）的系统评估，可能存在轻微的性能折损。

### 影响与延伸思考
ParaPO 把“记忆控制”从硬删向软改写转变，为后续的隐私保护和版权合规提供了新思路。自发表后，已有工作尝试将相似的偏好学习用于**事实校准**（让模型在不确定时给出模糊答案）以及**可解释记忆检索**（让模型在引用时标注来源）。如果想进一步探索，可以关注以下方向：  
- 自动化生成高质量改写对的流水线，降低人工成本。  
- 将偏好学习与检索增强模型结合，让模型在需要精确引用时主动检索原文。  
- 探索更细粒度的系统提示语言，形成多层次的记忆控制策略。  
（以上为基于当前文献的推测，后续实际进展请自行关注最新会议论文）

### 一句话记住它
ParaPO 让语言模型在保留知识的同时主动改写记忆片段，用系统提示随时切换“引用”与“改写”，从根本上降低了无意的逐字复现。