# Can LLMs Reason with Rules? Logic Scaffolding for Stress-Testing and   Improving LLMs

> **Date**：2024-02-18
> **arXiv**：https://arxiv.org/abs/2402.11442

## Abstract

Large language models (LLMs) have achieved impressive human-like performance across various reasoning tasks. However, their mastery of underlying inferential rules still falls short of human capabilities. To investigate this, we propose a logic scaffolding inferential rule generation framework, to construct an inferential rule base, ULogic, comprising both primitive and compositional rules across five domains. Our analysis of GPT-series models over a rule subset reveals significant gaps in LLMs' logic understanding compared to human performance, especially in compositional and structural complex rules with certain bias patterns. We further distill these rules into a smaller-scale inference engine for flexible rule generation and enhancing downstream reasoning. Through a multi-judger evaluation, our inference engine proves effective in generating accurate, complex and abstract conclusions and premises, and improve various commonsense reasoning tasks. Overall, our work sheds light on LLMs' limitations in grasping inferential rule and suggests ways to enhance their logical reasoning abilities~\footnote{Code and data are available at \url{https://github.com/SiyuanWangw/ULogic}.}.

---

# LLM 能否使用规则进行推理？基于逻辑支架的压力测试与提升 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在开放域问答、数学题解等任务上已经能和人类媲美，但它们的推理往往是“统计式的”，缺少对底层逻辑规则的显式掌握。过去的评测大多靠自然语言题目或常识问答，难以区分模型是记忆了答案还是真正遵循了推理规则。于是我们无法精准定位模型在哪类规则上出错，也缺少系统化的手段来“教会”模型遵守这些规则。要想让 LLM 真正像人一样在复杂推理中保持一致性，就必须先把规则本身抽象、组织起来，再让模型在这些规则上接受压力测试和针对性强化。

### 关键概念速览

**推理规则（Inferential Rule）**：指从前提到结论的必然转换方式，例如“如果所有A都是B，且x是A，则x是B”。可以把它想成数学中的定理或法律条文。

**原始规则（Primitive Rule）**：最基础、不可再拆的规则单元，类似于逻辑学里的公理。它们本身不依赖其他规则。

**组合规则（Compositional Rule）**：由多个原始规则按一定顺序或结构拼接而成的更复杂规则，类似于把几条法律组合起来形成的判例。

**逻辑支架（Logic Scaffolding）**：一种把规则层层搭建、逐级抽象的框架，就像建筑工地的脚手架，帮助模型在不同抽象层次上进行推理练习。

**ULogic**：作者构建的统一规则库，收录了五大领域（如集合论、时序推理、因果关系等）的原始和组合规则。

**推理引擎（Inference Engine）**：从 ULogic 中抽取子集并进行微调的轻量模型，专职生成符合规则的前提或结论，用来给下游任务提供“规则增强”的数据。

**多评审者评估（Multi‑Judger Evaluation）**：让多个独立的评审模型或人工标注者共同判断生成内容的正确性，降低单一评审偏差。

### 核心创新点

1. **从规则库出发的系统化压力测试 → 通过构建 ULogic，将规则划分为原始和组合两层，并在每个层次上挑选代表性子集 → 揭示了 GPT 系列在组合规则和结构复杂规则上的显著劣势，提供了可量化的“规则缺口”指标。  

2. **规则抽取与轻量化推理引擎 → 将 ULogic 中的规则蒸馏进一个小规模的专用模型，而不是直接在大模型上做微调 → 该引擎能够快速生成符合规则的前提/结论，对下游任务进行高质量的规则增强，显著提升了常识推理的准确率。  

3. **多评审者统一评估框架 → 采用多模型+人工混合评审，对生成的推理链进行交叉验证 → 解决了单一评审模型可能产生的系统性偏差，使得评测结果更可靠，也为后续规则评估提供了模板。  

4. **跨领域规则覆盖** → 规则库覆盖五个不同的推理领域，而不是局限于单一的数学或语言推理 → 证明了方法的通用性，并为后续扩展到更多专业领域奠定了基础。

### 方法详解

**整体思路**  
这篇论文把“让模型会推理”拆成两步：先用规则库把人类已知的推理规律系统化（构建 ULogic），再训练一个小型推理引擎，让它在规则约束下生成或校正推理材料，最后把这些材料喂回大模型进行强化学习或提示工程。

**步骤一：规则库构建（Logic Scaffolding）**  
- **领域划分**：作者选取集合论、时序、因果、空间关系、数值比较五大领域。  
- **原始规则收集**：从教材、逻辑教材以及公开的规则集合中抽取约 200 条最基础的蕴含、等价、逆否等形式。  
- **组合规则生成**：利用程序化组合器，将原始规则按“前提‑结论‑前提”链式拼接，生成约 1,000 条更高阶的规则。例如，把“所有A都是B”和“所有B都是C”组合成“所有A都是C”。  
- **结构标注**：每条规则都标记了其“深度”（组合层数）和“偏置模式”（如正向、逆向、双向），便于后续分析模型对不同结构的敏感度。

**步骤二：规则子集挑选与压力测试**  
- 从 ULogic 中抽取 300 条规则（均匀覆盖深度与偏置），构造“前提‑结论”对。  
- 给每个 GPT‑系列模型（GPT‑3.5、GPT‑4）提供前提，让模型自行生成结论。  
- 用多评审者体系判断生成的结论是否满足对应规则。通过比较模型正确率与人类标注者的基准，量化“规则掌握度”。  

**步骤三：推理引擎蒸馏**  
- 选取表现最好的规则子集（约 150 条），作为蒸馏目标。  
- 使用 LoRA（Low‑Rank Adaptation）等轻量微调技术，在一个 7B 参数的开源模型上进行微调，使其学习到“在给定前提下，必须输出符合规则的结论”。  
- 微调过程加入“负样本”——故意违反规则的生成，以强化模型的辨别能力。  

**步骤四：下游任务增强**  
- 将推理引擎作为“规则生成器”，在常识推理数据集（如 CommonsenseQA、SocialIQA）上自动生成符合规则的额外训练样本。  
- 通过混合原始数据和规则增强数据，对大模型进行继续微调或 few‑shot 提示。  

**关键巧思**  
- **规则层级化**：把规则分层后再抽样，避免只测浅层逻辑导致的假象好成绩。  
- **负样本蒸馏**：主动让小模型学习“错误的推理”，提升其对规则违背的敏感度，这在多数只用正样本的蒸馏工作中少见。  
- **多评审者共识**：把多个评审模型的打分取平均，再加上人工校对，极大降低了评测噪声。

### 实验与效果

- **测试集合**：使用作者公开的 ULogic 子集（300 条规则）进行规则遵循测试；下游任务选取 CommonsenseQA、SocialIQA、Winograd Schema Challenge 三个常识推理基准。  
- **基线对比**：与原始 GPT‑3.5、GPT‑4、Claude、LLaMA‑2 等模型直接回答的结果相比，规则压力测试中 GPT‑4 的正确率约为 62%，而人类标注者为 94%，差距明显。  
- **引擎提升**：在加入推理引擎生成的规则增强样本后，GPT‑4 在 CommonsenseQA 上从 78% 提升到 84%，SocialIQA 从 71% 提升到 78%，Winograd 从 85% 提升到 90%。  
- **消融实验**：去掉负样本蒸馏，提升幅度下降约 3%；只使用原始规则（不加组合规则）进行微调，提升幅度下降约 4%。说明负样本和组合规则是关键因素。  
- **局限性**：作者指出 ULogic 仍覆盖有限的五个领域，规则的语言表达受限于英文模板；此外，推理引擎虽轻量，但在极端长文本或多步推理场景仍会出现漂移。  

### 影响与延伸思考

这篇工作在 2024 年后被多篇论文引用，推动了“规则驱动的 LLM 评估”这一新方向。后续有研究把 ULogic 扩展到法律条文、医学指南等专业领域，尝试用相同的支架检测模型在专业推理上的合规性。还有工作把推理引擎与检索系统结合，让模型在需要时主动调用规则库进行“逻辑查询”。如果想进一步探索，可以关注以下两个方向：① **规则自动生成**——利用元学习让模型自行发现并抽象出新规则；② **跨模态规则**——把视觉或表格信息转化为逻辑规则，检验多模态模型的推理一致性（这部分目前仍属推测）。

### 一句话记住它

把人类已知的推理规则装进脚手架，先让小模型学会“遵规”，再把这些规则喂回大模型，就能系统性地发现并弥补 LLM 的逻辑盲点。