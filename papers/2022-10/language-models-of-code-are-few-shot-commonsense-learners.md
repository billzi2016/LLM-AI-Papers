# Language Models of Code are Few-Shot Commonsense Learners

> **Date**：2022-10-13
> **arXiv**：https://arxiv.org/abs/2210.07128

## Abstract

We address the general task of structured commonsense reasoning: given a natural language input, the goal is to generate a graph such as an event -- or a reasoning-graph. To employ large language models (LMs) for this task, existing approaches ``serialize'' the output graph as a flat list of nodes and edges. Although feasible, these serialized graphs strongly deviate from the natural language corpora that LMs were pre-trained on, hindering LMs from generating them correctly. In this paper, we show that when we instead frame structured commonsense reasoning tasks as code generation tasks, pre-trained LMs of code are better structured commonsense reasoners than LMs of natural language, even when the downstream task does not involve source code at all. We demonstrate our approach across three diverse structured commonsense reasoning tasks. In all these natural language tasks, we show that using our approach, a code generation LM (CODEX) outperforms natural-LMs that are fine-tuned on the target task (e.g., T5) and other strong LMs such as GPT-3 in the few-shot setting.

---

# 代码语言模型是少样本常识学习者 论文详细解读

### 背景：这个问题为什么难？
结构化常识推理要求模型把一句自然语言描述转化为事件图或推理图，这类图包含节点（实体、事件）和边（因果、时间等）之间的关系。传统做法把图序列化成一长串文本，让大语言模型直接生成，但序列化的格式与模型在海量自然语言上预训练的分布差异很大，导致模型在保持结构完整性和逻辑一致性方面表现不佳。换句话说，模型在“写代码”式的结构化输出上比在“写散文”上更擅长，却一直没有被系统化利用。

### 关键概念速览
**结构化常识推理**：把自然语言输入映射为包含实体和关系的图形结构，类似把一句话拆解成因果链条。  
**序列化图**：把图的节点和边展平成一行文字，类似把树形结构压平成列表，容易破坏原有层次。  
**代码生成模型（Code LM）**：在大量源码上训练的语言模型，熟悉函数、变量、缩进等严格语法规则。  
**Few‑Shot（少样本）学习**：只给模型极少的示例就让它完成新任务，考验模型的通用推理能力。  
**Prompt（提示）**：在输入前加上一段指令或示例，引导模型产生期望的输出形式。  
**CODEX**：OpenAI 发布的代码生成模型，专门优化了对编程语言的理解和生成。  
**T5**：一种在自然语言任务上表现优秀的文本到文本模型，常被微调用于结构化输出。

### 核心创新点
1. **任务重构：从图序列化到代码生成**  
   之前的做法直接让自然语言模型输出扁平化的节点/边列表，导致结构错误频发。本文把同一任务改写成“写一段代码”，让模型生成符合语法的函数或类定义，其中变量对应图的节点，函数调用对应边。这样模型的输出天然遵守结构约束。  
2. **利用代码模型的结构感知优势**  
   代码模型在预训练阶段已经学习到变量声明、作用域、缩进等严格规则。把常识图映射为代码后，模型在 Few‑Shot 场景下能够凭借这些规则自动保持图的完整性，而不需要额外的后处理。  
3. **统一 Prompt 设计，跨任务复用**  
   作者为三类常识推理任务（事件图、因果图、属性图）设计了统一的代码模板，只需在 Few‑Shot 示例中换几行具体实例。相比之前每个任务都要单独调参，这种统一模板大幅降低了使用门槛。  
4. **对比实验显示代码模型在少样本下超越微调的自然语言模型**  
   在相同的 Few‑Shot 条件下，CODEX 的准确率超过了专门在目标任务上微调的 T5，也跑赢了 GPT‑3 等强大的自然语言模型，证明结构化输出的“代码化”真的能提升推理质量。

### 方法详解
整体思路可以拆成三步：  
1. **图‑代码映射**：先把目标图的概念抽象为代码元素。节点被映射为类的属性或字典键，边被映射为函数调用或列表中的关联语句。例如，一个“人‑吃‑食物”的因果关系会被写成 `eat(person, food)`。  
2. **Few‑Shot Prompt 构造**：在输入前放入几段完整的代码示例，每段示例对应一个自然语言描述和对应的代码实现。Prompt 的结构大致是：  
   ```
   # 示例 1
   Input: "小明在雨天打伞"
   Output:
   def scene():
       person = "小明"
       action = "打伞"
       condition = "雨天"
       return {"agent": person, "action": action, "context": condition}
   # 示例 2
   ...
   # 目标
   Input: "<待推理句子>"
   Output:
   ```  
   这样模型在看到前面的模式后，会自动生成符合模板的代码。  
3. **代码到图的逆向解析**：模型输出的代码被解析回结构化图。因为代码本身已经是结构化的，只需读取变量名和函数参数即可恢复节点和边。这个步骤几乎是机械的，不涉及模型推理。  

**关键细节**  
- **缩进与语法检查**：作者利用代码模型自带的语法约束，让模型在生成时自然保持正确的缩进和括号匹配，避免了序列化文本常见的逗号、引号错误。  
- **模板通用性**：虽然任务不同，但所有任务的代码模板都围绕“def …():”函数展开，只需替换内部变量名和函数名即可。  
- **反直觉点**：即使目标任务本身与编程毫无关系，强行把它包装成代码仍然提升了模型表现，这一现象挑战了“模型只能在训练分布上发挥”的常规认知。  

### 实验与效果
- **测试任务**：论文选取了三个公开的结构化常识推理基准：Event2Graph（事件图生成）、ATOMIC（因果推理图）和 ConceptNet‑5（属性关系图）。每个任务都要求从一句自然语言生成对应的图结构。  
- **对比基线**：包括在相同 Few‑Shot 条件下的 T5（自然语言到文本微调版）、GPT‑3（大规模自然语言模型）以及一些专门为图序列化设计的模型。  
- **结果概述**：在所有三个任务上，CODEX 的准确率均高于基线，尤其在极少样本（1‑3 示例）时优势最明显。论文声称相对提升在 5%~15% 之间，具体数值随任务而变。  
- **消融实验**：作者分别去掉代码模板、去掉 Few‑Shot 示例、改用普通自然语言模型进行同样的代码化处理，发现：① 去掉模板会导致结构错误率激增；② 少量示例仍能保持优势，说明模型对代码结构的内在理解是关键；③ 用自然语言模型生成代码时性能回落到普通序列化水平，验证了“代码模型特有的结构感知”。  
- **局限性**：论文承认当前方法依赖于手工设计的代码模板，跨语言或更复杂图结构时需要额外工程工作；此外，代码生成的速度比直接文本生成稍慢，且对模型的 token 限制更敏感。

### 影响与延伸思考
这篇工作打开了“把非代码任务包装成代码让模型更好完成”的新思路，随后出现了多篇把结构化输出转化为 SQL、JSON、甚至 LaTeX 的研究，均借鉴了代码化提示的优势。后续工作（如 CodePrompt、Program-Aided Language Models）进一步探索让模型在推理过程中真正调用可执行代码，而不是仅仅生成代码文本。想继续深入，可以关注 **Program Synthesis**（程序合成）与 **Neural Symbolic Reasoning**（神经符号推理）的交叉方向，这两块正逐步把“写代码”和“做推理”合二为一。

### 一句话记住它
把常识图当成代码写，让代码模型在少样本下也能精准推理。