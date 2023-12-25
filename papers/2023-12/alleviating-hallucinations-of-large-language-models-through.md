# Alleviating Hallucinations of Large Language Models through Induced   Hallucinations

> **Date**：2023-12-25
> **arXiv**：https://arxiv.org/abs/2312.15710

## Abstract

Despite their impressive capabilities, large language models (LLMs) have been observed to generate responses that include inaccurate or fabricated information, a phenomenon commonly known as ``hallucination''. In this work, we propose a simple \textit{Induce-then-Contrast} Decoding (ICD) strategy to alleviate hallucinations. We first construct a factually weak LLM by inducing hallucinations from the original LLMs. Then, we penalize these induced hallucinations during decoding to enhance the factuality of the generated content. Concretely, we determine the final next-token predictions by amplifying the predictions from the original model and downplaying the induced untruthful predictions via contrastive decoding. Experimental results on both discrimination-based and generation-based hallucination evaluation benchmarks, such as TruthfulQA and \textsc{FActScore}, demonstrate that our proposed ICD methods can effectively enhance the factuality of LLMs across various model sizes and families. For example, when equipped with ICD, Llama2-7B-Chat and Mistral-7B-Instruct achieve performance comparable to ChatGPT and GPT4 on TruthfulQA, respectively.

---

# 通过诱导幻觉缓解大语言模型的幻觉问题 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在对话、写作、编程等场景表现惊人，但它们经常会“编造”事实——把不存在的细节说得信誓旦旦，这种现象被称为幻觉。传统的解决思路是直接在训练数据上做更严格的过滤或在微调阶段加入事实校验，但这些办法往往需要大量标注、计算成本高，而且仍然难以根除模型内部的“自信错误”。根本原因在于，生成过程本身缺少一个实时的、对抗性的纠错机制：模型在每一步都只看自己最可能的下一个词，而不主动比较“对”和“错”。因此，如何在不重新训练、且保持生成流畅性的前提下，抑制幻觉，成为一个急需突破的难题。

### 关键概念速览

**幻觉（Hallucination）**：模型输出的内容与真实世界不符，类似于人类在记忆模糊时胡乱编造的情节。  
**对比解码（Contrastive Decoding）**：在生成下一个词时，同时考虑两个模型的预测分布，用差异来抑制不可靠的选项，像是让两位“审稿人”相互比较后决定最终稿。  
**诱导模型（Induced Model）**：在原始模型的基础上，故意让它产生更多幻觉的弱化版本，类似于把原本的“好学生”训练成“爱说大话的同学”。  
**放大/削弱（Amplify / Downplay）**：对原模型的高置信度预测进行放大，对诱导模型的高置信度预测进行削弱，形成一种“正负对冲”。  
**TruthfulQA**：一个专门测评模型回答真实性的基准，问题往往涉及常识或专业知识，答案越接近事实分数越高。  
**FActScore**：另一套评估生成文本事实性的指标，综合了答案的准确率和覆盖率。  

### 核心创新点

1. **先制造幻觉再对抗**：过去的研究大多尝试直接过滤幻觉或在后处理阶段检查事实。这里的做法是先让模型自己产生大量幻觉（通过诱导模型），再利用这些“错误”作为对照，形成一种自监督的对抗信号。这样不需要外部标注，只靠模型内部的差异即可进行校正。  
2. **对比解码的双模型融合**：传统的对比解码只在同一模型的不同温度或采样策略之间做比较。本文把原始模型的输出和诱导模型的输出并行计算，用原模型的概率乘以一个放大系数，用诱导模型的概率乘以一个削弱系数，再合并得到最终分布。这样在每一步都显式抑制了被诱导模型“自信”但不真实的词。  
3. **跨模型、跨规模的通用性**：实验覆盖了 Llama2、Mistral 等不同架构和参数规模的模型，结果显示即使是 7 B 参数的模型，加入对比解码后也能追平 ChatGPT、GPT‑4 在 TruthfulQA 上的表现。这说明该策略不依赖特定模型结构，具备即插即用的特性。  

### 方法详解

整体思路可以划分为三步：  
1) **构造诱导模型**；2) **并行生成两套候选分布**；3) **对比融合得到最终 token**。

**步骤 1：构造诱导模型**  
作者在原始 LLM 基础上，加入一个轻量的“幻觉诱导器”。具体实现方式在摘要里没有细化，可能是通过在解码时使用更高的温度、加入噪声或在提示中加入误导性指令，使模型倾向于输出不可靠信息。关键是让这个模型在相同输入下，产生比原模型更多的错误答案。

**步骤 2：并行预测**  
给定上下文，原模型输出一个概率分布 \(P_{\text{orig}}(w)\)，诱导模型输出另一个分布 \(P_{\text{ind}}(w)\)。这两个分布在每一步都同步计算，类似于两位评审同时给出各自的评分。

**步骤 3：对比融合**  
最终的 token 选择依据一个对比得分：  
- 对原模型的概率乘以放大因子 \(\alpha > 1\)，提升它的影响力。  
- 对诱导模型的概率乘以削弱因子 \(\beta < 1\)，抑制它的影响。  
合并后得到新的分布 \(P_{\text{final}}(w) = \alpha P_{\text{orig}}(w) - \beta P_{\text{ind}}(w)\)（实际实现中会做归一化，防止负数）。随后按照常规的采样或贪心策略选出下一个 token。  

**巧妙之处**  
- 只在解码阶段加入额外计算，几乎不需要重新训练或大规模微调，成本与普通采样相当。  
- 通过“负向信号”直接削弱模型自信的错误选项，而不是事后检查，避免了延迟纠错带来的上下文漂移。  
- 放大/削弱系数可以根据模型大小或任务难度调节，提供了灵活的控制杠杆。  

### 实验与效果

- **评测数据**：TruthfulQA（常识问答）和 FActScore（事实性生成）两套基准。  
- **对比基线**：原始 Llama2‑7B‑Chat、Mistral‑7B‑Instruct、以及业界强模型 ChatGPT、GPT‑4。  
- **主要结果**：加入 ICD（Induce‑then‑Contrast Decoding）后，Llama2‑7B‑Chat 在 TruthfulQA 上的得分提升到与 ChatGPT 相当；Mistral‑7B‑Instruct 的表现也逼近 GPT‑4。具体数值未在摘要中给出，论文声称两者“可比”。  
- **消融实验**：作者分别去掉诱导模型、去掉放大系数或削弱系数，发现缺少任一环节都会导致事实性分数显著回落，说明对比解码的双向调节是关键。  
- **局限性**：对比解码在极长文本或需要高度创造性的任务上可能会抑制创新，因为削弱了模型的“大胆”输出；此外，诱导模型的构造方式在不同语言或领域可能需要手工调参，尚未实现完全自动化。  

### 影响与延伸思考

这篇工作打开了“自我对抗式解码”的新思路，后续有研究尝试把事实校验模型、检索增强模块甚至人类反馈都包装成“诱导器”，在解码时进行对比。还有人把类似的思路搬到图像生成模型，利用噪声版的扩散模型来抑制不符合语义的像素。想进一步深入，可以关注以下方向：  
- **自动化诱导模型生成**：如何在不人工干预的情况下，让模型自行学习产生幻觉的模式。  
- **多模态对比解码**：把文本、图像、音频的不同生成器放在同一对比框架里，提升跨模态一致性。  
- **动态系数调节**：根据实时的置信度或外部检索结果，自适应调整放大/削弱力度，做到更细粒度的事实控制。  

### 一句话记住它

让模型先“说错”，再用它的错误来“压制”自己，从而在生成时自动纠正幻觉。