# Chain-of-Thought Reasoning Without Prompting

> **Date**：2024-02-15
> **arXiv**：https://arxiv.org/abs/2402.10200

## Abstract

In enhancing the reasoning capabilities of large language models (LLMs), prior research primarily focuses on specific prompting techniques such as few-shot or zero-shot chain-of-thought (CoT) prompting. These methods, while effective, often involve manually intensive prompt engineering. Our study takes a novel approach by asking: Can LLMs reason effectively without prompting? Our findings reveal that, intriguingly, CoT reasoning paths can be elicited from pre-trained LLMs by simply altering the \textit{decoding} process. Rather than conventional greedy decoding, we investigate the top-$k$ alternative tokens, uncovering that CoT paths are frequently inherent in these sequences. This approach not only bypasses the confounders of prompting but also allows us to assess the LLMs' \textit{intrinsic} reasoning abilities. Moreover, we observe that the presence of a CoT in the decoding path correlates with a higher confidence in the model's decoded answer. This confidence metric effectively differentiates between CoT and non-CoT paths. Extensive empirical studies on various reasoning benchmarks show that the proposed CoT-decoding effectively elicits reasoning capabilities from language models, which were previously obscured by standard greedy decoding.

---

# 无需提示的链式思考推理 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在解答需要多步推理的题目时，往往直接给出答案，错误率很高。研究者通过在提示（prompt）里加入“思考链”（Chain‑of‑Thought, CoT）或提供少量示例，让模型先写出推理步骤，再给出结论，效果明显提升。但这种做法把模型的能力和提示工程捆绑在一起：  
1）需要人工设计或搜索合适的示例，成本高；  
2）不同任务、不同模型的最佳提示差异大，难以通用；  
3）提示本身可能引入偏差，掩盖模型到底能否自行产生推理过程。于是，是否可以让模型在不改动提示的情况下自行展现思考链，成为一个迫切的研究点。

### 关键概念速览
**Chain‑of‑Thought（思考链）**：模型在给出最终答案前，先把每一步推理写出来，类似人做数学题时的草稿，帮助模型保持逻辑连贯性。  
**Prompt（提示）**：向模型提供的输入文本，常用来引导模型产生特定格式或行为。这里的提示指的是不做任何特殊设计的普通问题描述。  
**Greedy Decoding（贪婪解码）**：每一步都选概率最高的词作为输出，速度快但容易错过潜在的有价值分支。  
**Top‑k Decoding（Top‑k 解码）**：在每一步保留概率最高的 k 个候选词，随后继续扩展，这相当于在搜索树上保留多条可能的路径。  
**Intrinsic Reasoning Ability（内在推理能力）**：模型在没有外部提示帮助时，自己能够产生合理推理步骤的潜力。  
**Confidence Metric（置信度度量）**：模型对最终答案的自信程度，通常通过输出概率或对数似然来衡量，这里用来区分是否走了思考链路径。  

### 核心创新点
1. **从提示工程转向解码策略**：以前的工作通过在输入端加入 CoT 示例或指令来激活推理；这篇工作直接改动输出阶段，使用 Top‑k 解码来挖掘模型内部已经存在的思考链。这样做把“让模型思考”从人工提示搬到了搜索策略上。  
2. **把思考链视作解码路径的自然子集**：研究者发现，在 Top‑k 的候选序列里，常常能找到完整的推理步骤，而这些步骤在贪婪解码时被直接跳过。换句话说，模型已经学会了写草稿，只是默认的解码方式不展示它们。  
3. **利用置信度区分 CoT 与非 CoT 路径**：实验表明，出现完整思考链的解码路径往往伴随更高的答案置信度。作者提出用置信度阈值来筛选出可信的 CoT 结果，既提升了准确率，又提供了一个自动判断模型是否真正推理的信号。  
4. **无需额外微调或数据**：整个流程只在推理阶段改动解码超参数，保持模型权重不变，因而可以直接在已有的公开模型上复现，极大降低了实验成本。

### 方法详解
整体思路可以概括为三步：  
1）**保持原始提示不变**，直接把问题交给预训练好的 LLM；  
2）**在生成答案时采用 Top‑k 解码**，记录下每一步的 k 条候选词及其对应的概率；  
3）**遍历这些候选序列，检测是否出现完整的思考链**，若出现则计算该路径的整体置信度，最终选取置信度最高的答案。

**步骤拆解**  
- **候选树构建**：把每一步的 Top‑k 词视为树的分支，向下展开直到生成结束标记或达到预设的最大长度。与传统的束搜索（beam search）不同，这里不对整条路径做归一化比较，而是保留所有可能的分支，以免提前剪掉潜在的思考链。  
- **思考链检测**：作者使用一个轻量的规则检测器，搜索候选序列中是否出现常见的推理关键词（如“因此”“所以”“可以得到”）以及数学符号或逻辑连接词。只要满足这些模式，就认为该分支已经形成了 CoT。  
- **置信度计算**：对每条被标记为 CoT 的路径，累乘（或累加对数）其每一步的概率，得到该路径的整体置信度。置信度越高，说明模型在整个推理过程中越自信。  
- **答案选取**：最终从所有 CoT 路径中挑选置信度最高的答案；如果没有任何 CoT 被检测到，则回退到最高概率的贪婪答案，保证系统始终能输出。

**最巧妙的地方**  
- 把“思考链”从显式的提示转化为解码过程的隐式属性，实际上是把模型内部的潜在能力“挖掘”出来，而不是重新教模型怎么写草稿。  
- 使用置信度作为筛选标准，既避免了人工设定的阈值，也让模型的自信程度直接决定是否采用 CoT，形成了一个自洽的闭环。

### 实验与效果
- **测试任务**：论文在多个公开的推理基准上评估，包括数学推理（如 GSM8K）、逻辑推理（如 LogicalDeduction）以及常识推理（如 CommonsenseQA）。  
- **对比基线**：与传统的贪婪解码、Few‑shot CoT 提示、Zero‑shot CoT 提示以及最新的束搜索方法进行比较。  
- **主要结果**：论文声称在大多数基准上，CoT‑decoding 相比贪婪解码提升了 5%~12% 的准确率；在与 Few‑shot CoT 提示的对比中，差距在 1%~3% 之间，甚至在某些任务上略有超越。  
- **消融实验**：作者分别关闭 Top‑k 采样、思考链检测和置信度筛选，发现去掉任何一环都会导致性能回落到接近贪婪解码的水平，说明三者缺一不可。  
- **局限性**：由于 Top‑k 会显著增加解码时间和显存占用，尤其在大模型上成本仍然不低；此外，思考链检测依赖于手工规则，可能在语言风格变化时漏检或误检。

### 影响与延伸思考
这篇工作打开了“解码层面激活模型潜能”的新视角，后续有不少研究尝试把类似的思路推广到其他隐式能力，如事实抽取、代码生成等。还有人把 Top‑k 替换成更高效的采样策略（如 nucleus sampling）并结合自适应置信度阈值，进一步降低计算开销。对想深入的读者，可以关注以下方向：  
- **自适应解码**：根据实时置信度动态调整 k 的大小；  
- **跨语言 CoT 挖掘**：在多语言模型中探索是否同样存在未显露的思考链；  
- **提示与解码的协同设计**：研究如何在最小化提示工程的同时，利用解码策略最大化模型的内在推理。  

### 一句话记住它
只要把解码从“只选最高”改成“在多个候选中寻找思考链”，大模型的推理能力就会自行显现。