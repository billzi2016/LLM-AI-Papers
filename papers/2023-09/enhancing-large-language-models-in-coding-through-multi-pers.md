# Enhancing Large Language Models in Coding Through Multi-Perspective   Self-Consistency

> **Date**：2023-09-29
> **arXiv**：https://arxiv.org/abs/2309.17272

## Abstract

Large language models (LLMs) have exhibited remarkable ability in code generation. However, generating the correct solution in a single attempt still remains a challenge. Prior works utilize verification properties in software engineering to verify and re-rank solutions in a majority voting manner. But the assumption behind them that generated verification properties have better qualities than solutions may not always hold. In this paper, we treat them equally as different perspectives of LLMs' reasoning processes. We propose the Multi-Perspective Self-Consistency (MPSC) framework incorporating both inter- and intra-consistency across outputs from multiple perspectives. Specifically, we prompt LLMs to generate diverse outputs from three perspectives, Solution, Specification and Test case, constructing a 3-partite graph. With two measure functions of consistency, we embed both inter- and intra-consistency information into the graph. The optimal choice of solutions is then determined based on analysis in the graph. MPSC significantly boosts performance of foundation models (ChatGPT in this paper) on various benchmarks, including HumanEval (+15.91%), MBPP (+6.43%) and CodeContests (+9.37%), even surpassing GPT-4.

---

# 通过多视角自一致性提升大型语言模型的代码生成能力 论文详细解读

### 背景：这个问题为什么难？
代码生成看似只要让模型一次性输出完整程序，但实际情况是模型经常会出现细微的语法错误、逻辑漏洞或不符合题目要求的实现。早期的自一致性（Self‑Consistency）方法通过让模型多次生成答案，然后用多数投票挑出最常见的解答，取得了一定提升。然而，这类方法默认“验证属性”（比如测试用例或规格说明）比直接的代码更可靠，实际上生成的测试或规格本身也会出现错误，导致投票过程把错误的答案误认为是正确的。于是，单纯依赖同一视角的多数投票难以突破正确率的瓶颈。

### 关键概念速览
**大语言模型（LLM）**：能够理解并生成自然语言或代码的深度学习模型，如ChatGPT、GPT‑4。  
**自一致性（Self‑Consistency）**：让模型多次独立生成答案，再通过一致性原则挑选最可信的结果，类似于让多个专家独立给出诊断后取多数意见。  
**视角（Perspective）**：模型在同一道题目上可以从不同的切入点输出内容，例如完整代码、功能规格或测试用例。把这些不同的输出看作“不同的眼睛”。  
**三部图（3‑partite graph）**：一种把三类节点（解答、规格、测试）分别放在三个层次上，并用边表示它们之间的匹配度的结构，像是把三组人分别坐在圆桌的三侧，互相检查对方的说法。  
**跨视角一致性（Inter‑Consistency）**：不同视角之间相互验证的程度，例如代码是否满足生成的规格或能通过生成的测试。  
**同视角一致性（Intra‑Consistency）**：同一视角内部多次生成结果之间的相似度，类似于同一专家多次给出的诊断是否相近。

### 核心创新点
1. **把规格和测试视作与代码同等重要的视角**  
   之前的工作把生成的规格或测试当作“更可靠的校验器”，只在投票时给它们更高权重。本文把三者平等看待，认为它们都是模型思考的不同切片。这样做让错误的规格或测试不会因为权重高而主导最终决定。

2. **构建三部图并在图上嵌入一致性信息**  
   传统自一致性只在同一维度上统计出现频次。这里先把所有代码、规格、测试分别放进三个节点集合，再用两套度量函数（跨视角匹配度、同视角相似度）在图的边上打分。相当于在“谁和谁说得更像”上做了双层过滤。

3. **基于图的全局优化挑选答案**  
   不是简单多数投票，而是对整个图进行分析，寻找能够最大化跨视角和同视角一致性的子图。直观上是找出那组代码、规格、测试三者最配合、最自洽的组合，从而得到更可靠的代码实现。

4. **在主流代码生成基准上实现显著提升**  
   将该框架套在ChatGPT上，HumanEval、MBPP、CodeContests等数据集的通过率分别提升了约16%、6%和9%，甚至超过了GPT‑4的原始表现。说明多视角自一致性在实际编程任务中具备强大实用价值。

### 方法详解
**整体思路**  
1）给定编程题目，分别让模型从“解答（Solution）”“规格（Specification）”“测试用例（Test case）”三个视角各自生成若干候选。  
2）把所有候选组织成三部图：每个解答节点连到所有规格节点，解答节点也连到所有测试节点，规格节点之间以及测试节点之间不直接相连。  
3）计算两类一致性得分：跨视角一致性衡量解答与规格、解答与测试之间的匹配度；同视角一致性衡量同一视角内部不同候选之间的相似度。  
4）在图上执行一次全局搜索，选出使总一致性得分最高的解答节点对应的代码，即为最终输出。

**关键模块拆解**  
- **多视角生成**：使用同一 LLM，分别给出三类提示词。例如，“请写出满足以下需求的 Python 程序”， “请给出该程序的功能规格”，以及 “请为上述程序生成 5 条测试用例”。每类提示生成 N（如 5）个不同答案，确保多样性。  
- **一致性度量函数**：  
  - *跨视角*：把生成的代码与规格做文本相似度（如 BLEU）或更直接的功能匹配（如运行代码检查是否满足规格中的约束），把代码与测试用例做执行通过率。  
  - *同视角*：对同一视角的 N 条答案使用嵌入相似度或编辑距离，得到内部一致性分数。  
- **图构建与打分**：每条跨视角边的权重 = 跨视角度量分数；每条同视角边的权重 = 同视角度量分数。这样每个解答节点拥有两套边：指向所有规格和测试的跨视角边，以及与同类解答的同视角边。  
- **全局优化**：作者采用一种基于加权投票的启发式搜索：先对每个解答节点累计其跨视角边的分数，再加上该节点所在同视角子图的平均一致性。分数最高的节点即为最自洽的代码。该过程不需要遍历所有子图组合，计算复杂度保持在可接受范围。

**最巧妙的地方**  
把“规格”和“测试”从“辅助校验”提升为“同等视角”，并用图结构把它们的相互关系显式化，使得错误的规格或测试不会因为单独的高分而误导整体决策；而是必须在整体一致性中得到支持，才能被采纳。

### 实验与效果
- **测试数据**：HumanEval（Python 编程题目集合），MBPP（中等规模的代码生成基准），以及 CodeContests（竞赛级别的挑战）。  
- **基线对比**：直接使用 ChatGPT 单次生成的结果、传统自一致性（多数投票）以及 GPT‑4 原始表现。  
- **提升幅度**：HumanEval 上提升约 15.91%（从约 45% 到 60%），MBPP 提升约 6.43%，CodeContests 提升约 9.37%。在 HumanEval 上的最终得分甚至超过了 GPT‑4 的基准。  
- **消融实验**：作者分别去掉跨视角一致性、去掉同视角一致性以及只保留单一视角生成，结果显示跨视角一致性贡献最大，但同视角一致性仍能提供约 2% 的额外提升，证明两者相辅相成。  
- **局限性**：方法依赖于能够成功生成高质量规格和测试用例的模型；在极其复杂或开放式任务中，规格/测试的生成质量仍是瓶颈。实验中只在 ChatGPT 上验证，未在更小模型上做广泛评估。

### 影响与延伸思考
这篇工作把“多视角自洽”引入代码生成，打开了把软件工程常用的需求、实现、测试三要素统一建模的思路。随后有研究尝试将同理扩展到自然语言问答、数学推理等领域，加入“解释”“证据”等额外视角。未来可以探索：  
- 用更专业的验证工具（如静态分析、形式化验证）替代或补充模型生成的规格/测试。  
- 在更小的开源模型上进行轻量化实现，验证方法的模型规模鲁棒性。  
- 将图结构改为可学习的神经图网络，让模型自行学习跨视角一致性的权重。

### 一句话记住它
把代码、规格、测试视作同等的“思考角度”，用图上的跨视角和同视角一致性挑出最自洽的答案，代码生成准确率因此大幅跃升。