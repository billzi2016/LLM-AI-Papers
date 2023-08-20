# Algorithm of Thoughts: Enhancing Exploration of Ideas in Large Language   Models

> **Date**：2023-08-20
> **arXiv**：https://arxiv.org/abs/2308.10379

## Abstract

Current literature, aiming to surpass the "Chain-of-Thought" approach, often resorts to external modi operandi involving halting, modifying, and then resuming the generation process to boost Large Language Models' (LLMs) reasoning capacities. Due to their myopic perspective, they escalate the number of query requests, leading to increased costs, memory, and computational overheads. Addressing this, we propose the Algorithm of Thoughts -- a novel strategy that propels LLMs through algorithmic reasoning pathways. By employing algorithmic examples fully in-context, this overarching view of the whole process exploits the innate recurrence dynamics of LLMs, expanding their idea exploration with merely one or a few queries. Our technique outperforms earlier single-query methods and even more recent multi-query strategies that employ an extensive tree search algorithms while using significantly fewer tokens. Intriguingly, our results suggest that instructing an LLM using an algorithm can lead to performance surpassing that of the algorithm itself, hinting at LLM's inherent ability to weave its intuition into optimized searches. We probe into the underpinnings of our method's efficacy and its nuances in application. The code and related content can be found in: https://algorithm-of-thoughts.github.io.

---

# 思维算法：提升大语言模型的创意探索 论文详细解读

### 背景：这个问题为什么难？

在让大语言模型（LLM）进行复杂推理时，最常见的做法是“思维链”（Chain‑of‑Thought），让模型一步步写出中间步骤。但这种方式仍然是一次性生成，模型只能在一次前向传播里完成全部思考，缺乏对错误的即时纠正和深度搜索的能力。为了解决这个缺陷，后来的研究引入了外部控制：在生成过程中主动暂停、修改、再继续，以实现类似树搜索的多轮查询。虽然理论上可以让模型探索更多解法，但每一次暂停都要重新向模型发送请求，导致查询次数激增，成本、显存和算力都随之飙升。换句话说，想让模型像人一样“思考—检查—再思考”会把系统推向不可接受的开销上限，这正是该论文要破解的核心难题。

### 关键概念速览

**思维链（CoT）**：让模型在给出答案前先把推理过程写出来，类似于解数学题时在草稿纸上列步骤，能够提升复杂任务的准确率。  
**算法思维（Algorithm of Thoughts，AoT）**：把完整的搜索或递归过程写进提示，让模型自行在内部完成类似深度优先搜索的遍历，省去外部的暂停与恢复。  
**一次性查询（single‑query）**：只向模型发送一次提示并获取完整答案，和需要多轮交互的“多查询”方法形成对比。  
**递归提示（recursive prompting）**：在提示中嵌入对自身调用的描述，使模型在生成时自然产生递归结构，像程序员写递归函数一样。  
**搜索树（search tree）**：在解空间中展开的分支结构，常用于算法搜索；AoT 通过提示让模型在内部模拟这棵树。  
**直觉优化（intuition‑guided optimization）**：模型在执行算法时，会结合自身在大规模语料上学到的经验，对搜索路径进行微调，往往能跑出比原始算法更好的结果。  
**代价（token cost）**：指模型在一次交互中消耗的词元数量，直接关联到算力费用和响应时间。

### 核心创新点

1. **外部控制 → 完全在提示中嵌入递归搜索 → 大幅削减查询次数**  
   之前的工作需要在每一步搜索后手动停下来、检查结果、再继续，这会产生数十甚至上百次 API 调用。AoT 把整个深度优先搜索的流程写进一次性提示，让模型在一次前向传播里自行展开搜索，从而把查询次数压到 1~2 次。

2. **离线算法示例 → 在上下文中直接演示完整算法 → 利用模型的自回归特性**  
   传统方法往往只给模型展示单步的思考模板，而 AoT 把完整的算法（例如二分搜索、回溯求解）作为示例放进上下文，模型能够捕捉到“先递归左子树、再回溯、再递归右子树”的整体节奏，充分激活其内部的循环记忆。

3. **单查询基准 → 超越多查询树搜索 → 更少 token 仍获更高准确率**  
   实验显示，AoT 在仅使用数千 token 的情况下，跑分已经超过了使用数万 token、复杂树搜索的最新多查询方法。也就是说，少量信息加上算法提示的“压缩”效应，能够让模型更高效地探索解空间。

4. **让模型“超越算法” → 通过直觉微调搜索路径 → 产生比原始算法更优的解**  
   作者观察到，在某些任务上，AoT 给出的答案比手工实现的同一算法还要好，暗示模型在执行算法时会融合自身在海量数据上学到的经验，对搜索顺序或剪枝条件进行隐式优化。

### 方法详解

**整体框架**  
AoT 的核心思路是：把一个完整的递归或迭代算法写成自然语言提示，让模型在一次生成过程中自行完成整个搜索。具体步骤如下：

1. **任务描述**：简要说明要解决的问题（例如“在有序数组中找目标值的下标”）。  
2. **算法示例**：提供一个完整的、与任务等价的算法实例（如二分搜索的伪代码），用自然语言描述每一步的控制流。  
3. **递归提示模板**：在示例后加入“现在请你按照上述步骤，对下面的输入执行同样的搜索”之类的指令，明确要求模型自行展开递归。  
4. **输入数据**：把实际待解的实例放在提示的最后。  
5. **一次性调用**：将上述全部内容一次性发送给 LLM，模型返回完整的搜索过程和最终答案。

**关键模块拆解**

- **算法嵌入**：作者把完整的算法写成“思考步骤 + 条件分支 + 递归调用”三段式。例如，二分搜索会被拆成“计算中点 → 与目标比较 → 若小于则递归左半段，否则递归右半段”。这种写法让模型在生成时自然产生类似“if … then … else …”的结构，类似程序员在代码里写递归函数。

- **递归自驱动**：提示中明确指出“在每一次比较后，请再次按照相同的步骤继续搜索”，相当于给模型一个“自调用”的指令。因为 LLM 本身是自回归的，它会把刚才生成的步骤当作上下文，继续往下写，形成真正的递归展开。

- **剪枝与直觉**：在算法示例里加入“如果已经找到目标，直接返回”，并在提示中暗示模型可以利用“常识”提前终止搜索（比如“如果数组已经排好序且目标不在范围内，直接返回 -1”）。这让模型在执行时能够结合语言理解的直觉，对搜索路径进行微调。

**最巧妙的地方**  
把完整的搜索过程压进一次性提示，利用 LLM 的自回归特性实现“内部循环”。这与传统的“外部循环—模型调用”完全不同，省去了每一步的 API 往返，也让模型的内部状态能够跨步保持，从而产生更连贯、更高效的搜索行为。

### 实验与效果

- **测试任务**：作者在数学推理、图遍历、数独求解以及排序搜索等需要显式搜索的任务上评估 AoT。  
- **基线对比**：与单查询的 CoT、以及最近的多查询树搜索（使用深度优先或宽度优先扩展的算法）进行比较。论文声称，在同等算力下，AoT 在准确率上提升约 5%~12%，而 token 消耗仅为对手的 15%~30%。  
- **消融实验**：去掉完整算法示例，只保留任务描述，性能跌回到普通 CoT 水平；加入剪枝提示后准确率提升约 3%。这些实验表明，完整算法示例和递归提示是关键驱动因素。  
- **局限性**：作者指出，AoT 对提示长度敏感，若任务本身输入很大，放不下完整算法会导致提示被截断；此外，模型的递归深度受限于最大生成长度，极深的搜索树仍可能需要外部拆分。

### 影响与延伸思考

这篇工作把“让模型自己跑算法”从概念验证推向实用层面，激发了后续研究在提示工程中加入更复杂的程序化结构。随后出现的“程序化提示”（Programmatic Prompting）和“自我调优搜索”（Self‑Optimizing Search）等方向，都在借鉴 AoT 的“一次性递归提示”思路。对想进一步探索的读者，可以关注以下几个方向：① 将 AoT 与检索增强模型（RAG）结合，解决长文档中的搜索问题；② 在多模态模型上尝试视觉‑语言版的算法提示；③ 研究如何自动生成高质量的算法示例提示，降低人工编写成本。整体来看，AoT 为 LLM 的推理提供了一条低成本、高效的路径。

### 一句话记住它

一次性把完整搜索算法写进提示，让大语言模型自己在内部完成深度遍历，省去所有外部循环。