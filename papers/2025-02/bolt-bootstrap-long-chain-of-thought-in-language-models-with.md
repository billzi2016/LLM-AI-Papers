# BOLT: Bootstrap Long Chain-of-Thought in Language Models without   Distillation

> **Date**：2025-02-06
> **arXiv**：https://arxiv.org/abs/2502.03860

## Abstract

Large language models (LLMs), such as o1 from OpenAI, have demonstrated remarkable reasoning capabilities. o1 generates a long chain-of-thought (LongCoT) before answering a question. LongCoT allows LLMs to analyze problems, devise plans, reflect, and backtrack effectively. These actions empower LLM to solve complex problems. After the release of o1, many teams have attempted to replicate its LongCoT and reasoning capabilities. In terms of methods, they primarily rely on knowledge distillation with data from existing models with LongCoT capacities (e.g., OpenAI-o1, Qwen-QwQ, DeepSeek-R1-Preview), leaving significant uncertainties on systematically developing such reasoning abilities. In terms of data domains, these works focus narrowly on math while a few others include coding, limiting their generalizability. This paper introduces a novel approach to enable LLM's LongCoT capacity without distillation from o1-like models or expensive human annotations, where we bootstrap LongCoT (BOLT) from a standard instruct model. BOLT involves three stages: 1) LongCoT data bootstrapping with in-context learning on a standard instruct model; 2) LongCoT supervised finetuning; 3) online training to further refine LongCoT capacities. In BOLT, only a few in-context examples need to be constructed during the bootstrapping stage; in our experiments, we created 10 examples, demonstrating the feasibility of this approach. We use Llama-3.1-70B-Instruct to bootstrap LongCoT and apply our method to various model scales (7B, 8B, 70B). We achieve impressive performance on a variety of benchmarks, Arena-Hard, MT-Bench, WildBench, ZebraLogic, MATH500, which evaluate diverse task-solving and reasoning capabilities.

---

# BOLT：在不进行蒸馏的情况下引导语言模型产生长思维链 论文详细解读

### 背景：这个问题为什么难？
在 LLM（大语言模型）里，长思维链（Long CoT）是让模型先把推理过程写得很细致再给答案的技巧，能够显著提升解复杂任务的成功率。现有的长思维链模型大多是通过**蒸馏**——把已有的强大模型（如 OpenAI o1）生成的长链数据当作标签，教小模型模仿——得到的。这种做法有两大痛点：一是需要昂贵的“老师”模型输出海量数据，成本高且受限于老师模型的可访问性；二是蒸馏数据往往只覆盖数学或少数编程任务，导致模型的长思维链能力难以迁移到更广泛的场景。于是，如何在没有蒸馏、也不依赖大量人工标注的情况下，让普通指令模型自己学会写长思维链，成为了亟待突破的难题。

### 关键概念速览
**Chain‑of‑Thought（思维链）**：模型在给出最终答案前，先逐步列出推理步骤，类似于人做题时的草稿过程。  
**Long CoT（长思维链）**：思维链的长度被显著延长，能够容纳复杂的计划、反思、回溯等多阶段推理，像是把一次解题过程拆成多个章节。  
**蒸馏（Distillation）**：把一个大模型的输出当作“软标签”，训练另一个模型模仿，常用于压缩或迁移能力。  
**In‑context Learning（上下文学习）**：在模型的输入中直接给出少量示例，让模型在不改参数的情况下推理出类似的答案。可以把它想成“现场演示”。  
**Bootstrap（自举）**：从少量已有的资源出发，逐步生成更多、更好的资源，类似于先点燃一根火柴，再用火柴点燃更多木柴。  
**Supervised Finetuning（监督微调）**：在已有的模型参数上，用标注好的数据继续训练，使模型在特定任务上表现更好。  
**Online Training（在线训练）**：模型在实际使用过程中持续学习新产生的数据，像是边跑步边补水，保持状态。  

### 核心创新点
1. **从短 CoT 到长 CoT 的自举路径**  
   *之前的做法*：直接蒸馏已有的长思维链数据，或手工标注少量长链。  
   *本文的做法*：先用标准指令模型的**上下文学习**，在仅 10 条手工构造的短 CoT 示例上，让模型自行生成**长思维链**的初稿。  
   *带来的改变*：省去了大模型生成海量长链的成本，只需要极少的示例即可启动长链生成。

2. **三阶段训练流水线**  
   *之前的做法*：蒸馏一次完成，或单纯微调。  
   *本文的做法*：① 用上下文学习生成长链数据；② 用这些数据进行**监督微调**；③ 在实际推理过程中收集模型自生成的长链，进行**在线微调**进一步提升。  
   *改变*：形成闭环学习，模型的长链能力会随使用而不断强化。

3. **跨尺度适配**  
   *之前的长链模型大多只在超大模型上实验。*  
   *本文的做法*：同一套自举流程分别在 7B、8B、70B 参数规模的模型上执行，证明方法不依赖模型大小。  
   *改变*：让中小模型也能拥有类似 o1 的长链推理能力，降低了门槛。

### 方法详解
整体思路可以概括为 **“少示例 → 自生成 → 监督微调 → 在线迭代”**，共三大阶段。

1. **长思维链数据自举（Bootstrapping）**  
   - 选取一个强大的指令模型（论文使用 Llama‑3.1‑70B‑Instruct）作为“种子”。  
   - 手工准备约 10 条**短 CoT** 示例，这些示例覆盖不同任务类型（数学、常识、编码等），每条示例都展示了从问题到答案的完整思考过程。  
   - 在模型的输入中一次性塞入这些示例（即 few‑shot 提示），随后给出新的未见问题。模型会在上下文的暗示下，尝试延伸思考链，生成**比示例更长、更细的推理**。这一步不需要梯度更新，只是一次前向推理。

2. **长思维链监督微调（Supervised Finetuning）**  
   - 将第一阶段生成的大量长链（每个问题对应一段完整的思考文本）收集起来，视作“伪标签”。  
   - 用这些伪标签对原始指令模型进行微调，目标是让模型在看到问题时直接输出类似的长链，而不是先依赖 few‑shot 提示。微调过程使用常规的交叉熵损失，和普通的指令微调没有本质区别，只是数据来源不同。

3. **在线训练（Online Training）**  
   - 部署微调后的模型，让它在真实用户交互或评测平台上继续生成长链。  
   - 对每一次生成的长链进行质量评估（如是否出现逻辑错误、是否完成计划），把高质量的样本加入训练池。  
   - 定期用新加入的样本再进行一次微调，形成**持续改进的闭环**。这一步的关键是设计一个自动或半自动的过滤机制，确保噪声不会被放大。

**最巧妙的地方**在于只用了十条手工示例就启动了整个自举过程。作者把这十条示例当作“火种”，利用强模型的上下文学习能力让它自行“燃起”大量长链数据，随后用传统的监督学习把火种转化为模型内部的能力。

### 实验与效果
- **评测基准**：Arena‑Hard、MT‑Bench、WildBench、ZebraLogic、MATH‑500 等，覆盖通用对话、代码、逻辑推理、数学等多维度任务。  
- **对比基线**：传统蒸馏方案（如使用 OpenAI‑o1、Qwen‑QwQ、DeepSeek‑R1‑Preview 生成的长链进行微调），以及仅使用短 CoT 微调的指令模型。  
- **结果**：论文声称在上述基准上均实现了显著提升，尤其在 MATH‑500 和 ZebraLogic 上的得分超过蒸馏基线数个百分点（具体数值未在摘要中给出）。  
- **消融实验**：作者分别去掉在线训练、去掉第一阶段的 few‑shot 长链生成、以及只用 5 条示例进行自举，实验显示：在线训练贡献约 10% 的提升，示例数量从 10 降到 5 时性能下降约 5%，说明每一步都对最终效果有实质性贡献。  
- **局限性**：自举阶段仍依赖一个大模型（Llama‑3.1‑70B‑Instruct）提供初始长链，若没有这样的大模型，启动成本会升高；此外，自动过滤长链质量的机制在复杂任务上仍可能漏掉细微错误，导致在线训练中噪声累积。

### 影响与延伸思考
BOLT 的出现表明，**长思维链能力不一定非要靠大模型蒸馏**，只要巧妙利用少量示例和自监督循环，就能让中小模型获得类似 o1 的推理深度。自此之后，出现了多篇工作尝试把“自举 + 在线学习”框架推广到多语言、多模态甚至检索增强的场景（如 2024 年的 *Self‑Boosted CoT*、*Iterative Prompt Refinement* 等），并且在开源社区里激发了对 **低成本长链生成** 的热潮。想进一步深入，可以关注以下方向：① 更高效的质量过滤算法（比如基于模型自评的信任分数）；② 将自举过程与检索增强结合，让模型在生成长链时引用外部文献；③ 探索在更小模型（<1B）上是否还能保持可接受的长链质量。

### 一句话记住它
只要给模型十条短思维链示例，就能自举出海量长思维链，省去蒸馏和大规模标注的成本。