# Language Models Are Greedy Reasoners: A Systematic Formal Analysis of   Chain-of-Thought

> **Date**：2022-10-03
> **arXiv**：https://arxiv.org/abs/2210.01240

## Abstract

Large language models (LLMs) have shown remarkable reasoning capabilities given chain-of-thought prompts (examples with intermediate reasoning steps). Existing benchmarks measure reasoning ability indirectly, by evaluating accuracy on downstream tasks such as mathematical reasoning. However, it is unclear how these models obtain the answers and whether they rely on simple heuristics rather than the generated chain-of-thought. To enable systematic exploration of the reasoning ability of LLMs, we present a new synthetic question-answering dataset called PrOntoQA, where each example is generated from a synthetic world model represented in first-order logic. This allows us to parse the generated chain-of-thought into symbolic proofs for formal analysis. Our analysis on InstructGPT and GPT-3 shows that LLMs are quite capable of making correct individual deduction steps, and so are generally capable of reasoning, even in fictional contexts. However, they have difficulty with proof planning: When multiple valid deduction steps are available, they are not able to systematically explore the different options.

---

# 语言模型是贪婪推理者：对思维链的系统形式化分析 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）出现之前，机器推理大多依赖显式的符号系统或专门的数学求解器，缺乏对自然语言的灵活理解。CoT（思维链）提示让模型先写出推理步骤，再给出答案，的确在数学、逻辑等任务上提升了准确率，但我们并不知道模型到底是“真的在推理”，还是只是在套用训练时学到的模式。缺少能够把模型的文字推理转化为可验证的形式化证明的工具，使得评估模型的推理深度变得模糊——这正是这篇论文要破解的难点。

### 关键概念速览

**大语言模型（LLM）**：像 GPT‑3、InstructGPT 这样通过海量文本预训练得到的生成式模型，能够在给定提示后输出连贯的自然语言。  
**思维链（CoT）**：在回答前让模型写出一步步的推理过程，类似人做数学题时的草稿，目的是让模型的中间思考可见。  
**合成世界模型**：作者自行构造的虚构“世界”，用一阶逻辑的事实和规则描述，所有问题都可以在这个世界里用逻辑推理得到答案。  
**一阶逻辑**：一种表达对象、属性和关系的形式语言，能够写出诸如“所有猫都是动物”之类的通用规则。  
**符号证明**：把自然语言的推理步骤映射到一阶逻辑的推理规则上，得到一串可机器检查的逻辑演绎。  
**证明规划**：在多条可行推理路径中挑选并组织步骤，使得最终能够到达目标结论的过程。  
**贪婪推理**：模型倾向于选取当前看起来最直接的推理一步，而不去系统地搜索所有可能的路径，像只吃眼前糖果的孩子。

### 核心创新点

1. **合成数据集 PrOntoQA**：之前的评测大多使用真实世界的数学或常识题，难以得到完整的推理过程。作者构造了一个全由一阶逻辑生成的问答库，每一道题都有对应的完整符号证明。这样就能把模型的文字 CoT 直接对照到形式化的推理链上。  
   *之前 → 只能间接评估（准确率） → 现在提供可解析的完整证明，能够逐步检查每一步是否正确。*

2. **CoT → 符号证明的自动解析器**：论文设计了一个规则驱动的解析器，把模型输出的自然语言步骤映射到具体的逻辑规则（如 modus ponens、全称实例化等）。这一步让原本模糊的文字推理变得可机器验证。  
   *之前 → 没有办法把文字转成逻辑 → 现在用确定性映射把每一步转成符号推理，得到可量化的正确率。*

3. **系统化的推理能力剖析**：作者把模型的表现拆分为“单步正确率”和“全局证明规划成功率”。实验发现模型在单步推理上表现不错，但在需要选择多条可能路径时常常停在局部最优，形成“贪婪”行为。  
   *之前 → 把整体准确率当作唯一指标 → 现在区分局部推理和全局规划，揭示模型的真实弱点。*

4. **“贪婪推理者”概念的提出**：通过对比不同模型（InstructGPT、GPT‑3）在同一合成世界中的表现，作者给出了“语言模型倾向于贪婪选择当前最显眼的推理步骤”的概念，为后续研究提供了新的解释框架。  
   *之前 → 没有专门的行为标签 → 现在用“贪婪推理”来描述模型的系统性缺陷，帮助定位改进方向。*

### 方法详解

整体思路可以分为三大步骤：**构造合成世界 → 生成 CoT 示例 → 解析并评估**。

1. **合成世界的搭建**  
   - 作者先定义一套一阶逻辑的常量、谓词和规则（比如“所有 A 都是 B”“如果 X 是 Y 且 Y 是 Z，则 X 是 Z”）。  
   - 随机抽取若干事实并用这些规则推导出新的事实，形成一个闭合的知识库。  
   - 再从知识库中随机挑选一个目标结论，使用自动定理证明器逆向生成一条完整的推理路径，这条路径即为“金标准”符号证明。  
   - 最后把每一步的逻辑规则用自然语言模板重新表述，得到对应的 CoT 文本。

2. **Few‑shot CoT 提示**  
   - 选取若干已经有金标准的例子（通常 4–8 条），把它们的自然语言 CoT 与答案一起放进提示。  
   - 把待测问题的描述放在提示末尾，让模型在同样的格式下生成自己的 CoT。这里使用的是 InstructGPT 和 GPT‑3 的指令式版本，以保证模型倾向于遵循提示。

3. **CoT → 符号证明的解析**  
   - 解析器先做句法分块，识别出关键的谓词、对象和逻辑连接词。  
   - 再匹配预定义的自然语言模式（比如“因为 … 所以 …”对应 modus ponens），把每句话映射到具体的逻辑规则编号。  
   - 生成的序列就是模型的符号证明草稿，随后与金标准逐步对比，统计**单步匹配率**（模型每一步是否对应正确规则）和**全局成功率**（模型的整条证明是否完整到达目标）。

**最巧妙的地方**在于把自然语言的自由表达压缩成一套有限的模式，使得解析过程几乎是确定性的。这样既保留了模型生成的多样性，又避免了传统自然语言理解的歧义。

### 实验与效果

- **数据集**：全部实验在新建的 PrOntoQA 上进行，规模约数万条问答，每条都有金标准的符号证明。  
- **对比基线**：包括（1）直接让模型输出答案不写 CoT 的零提示方式；（2）使用相同 few‑shot 但不提供 CoT 示例的普通提示。  
- **主要发现**：  
  - 在单步推理上，InstructGPT 的正确率约在 80% 左右，GPT‑3 稍低但仍超过 70%。这说明模型能够把自然语言描述的推理步骤映射到正确的逻辑规则。  
  - 在需要多条可选路径的题目中，完整证明的成功率只有约 30%–40%，远低于单步正确率，验证了“贪婪”现象。  
  - 与不写 CoT 的基线相比，使用 CoT 提示整体答案准确率提升约 10%–15%，但提升主要来源于模型在容易的单步推理上表现更好，而不是在复杂规划上取得突破。  
- **消融实验**：作者去掉了 few‑shot CoT 示例，仅保留指令提示，发现单步正确率下降约 5%，说明示例对模型学习推理规则仍有帮助。  
- **局限性**：论文主要在完全合成的逻辑世界里评估，真实世界的语言噪声、常识冲突等因素未被覆盖；解析器依赖手工设计的模式，若模型产生未覆盖的表达会导致解析失败。

### 影响与延伸思考

这篇工作把“模型到底在推理还是在套话”这个疑问搬上了实验台，随后出现了多篇围绕**可解释推理评测**的研究，例如使用程序合成、图结构证明或是强化学习搜索来补足模型的规划能力。还有人基于 PrOntoQA 的思路，构造了更大规模、包含概率规则的合成数据集，以检验模型在不确定推理中的表现。对想进一步探索的读者，可以关注**“语言模型的证明搜索（proof search）”**和**“基于符号约束的 CoT 纠错”**这两个方向，那里已经有不少开源工具和最新论文。

### 一句话记住它

语言模型在思维链上像贪吃的孩子，只会把眼前能写的推理写出来，却不擅长规划整条证明路线。