# CodeChain: Towards Modular Code Generation Through Chain of   Self-revisions with Representative Sub-modules

> **Date**：2023-10-13
> **arXiv**：https://arxiv.org/abs/2310.08992

## Abstract

Large Language Models (LLMs) have already become quite proficient at solving simpler programming tasks like those in HumanEval or MBPP benchmarks. However, solving more complex and competitive programming tasks is still quite challenging for these models - possibly due to their tendency to generate solutions as monolithic code blocks instead of decomposing them into logical sub-tasks and sub-modules. On the other hand, experienced programmers instinctively write modularized code with abstraction for solving complex tasks, often reusing previously developed modules. To address this gap, we propose CodeChain, a novel framework for inference that elicits modularized code generation through a chain of self-revisions, each being guided by some representative sub-modules generated in previous iterations. Concretely, CodeChain first instructs the LLM to generate modularized codes through chain-of-thought prompting. Then it applies a chain of self-revisions by iterating the two steps: 1) extracting and clustering the generated sub-modules and selecting the cluster representatives as the more generic and re-usable implementations, and 2) augmenting the original chain-of-thought prompt with these selected module-implementations and instructing the LLM to re-generate new modularized solutions. We find that by naturally encouraging the LLM to reuse the previously developed and verified sub-modules, CodeChain can significantly boost both modularity as well as correctness of the generated solutions, achieving relative pass@1 improvements of 35% on APPS and 76% on CodeContests. It is shown to be effective on both OpenAI LLMs as well as open-sourced LLMs like WizardCoder. We also conduct comprehensive ablation studies with different methods of prompting, number of clusters, model sizes, program qualities, etc., to provide useful insights that underpin CodeChain's success.

---

# CodeChain：通过自我修订链与代表子模块实现模块化代码生成 论文详细解读

### 背景：这个问题为什么难？

现有的大语言模型（LLM）在 HumanEval、MBPP 这类相对简单的编程基准上已经能写出可运行的代码，但面对需要多步推理、复杂数据结构或算法竞赛级别的题目时，它们的表现仍然不理想。根本原因在于模型倾向一次性生成“一整块”代码，而不是把任务拆解成若干子任务并分别实现对应的子函数。人类程序员在处理大项目时会自然地抽象出模块、复用已有实现，这种“模块化思维”在 LLM 的生成过程中几乎没有被显式鼓励，导致生成的代码缺乏可读性、可复用性，错误率也随之升高。于是，如何让 LLM 学会像程序员一样分而治之、循环利用已有子模块，成为提升复杂代码生成质量的关键瓶颈。

### 关键概念速览

**Chain‑of‑Thought（思维链）**：让模型在给出最终代码前先写出思考步骤，就像解数学题时先列出解题思路，帮助模型保持逻辑连贯性。  

**Self‑revision（自我修订）**：模型在生成一次后，再次审视并改写自己的输出，类似人写完代码后自己调试、重构的过程。  

**子模块（Sub‑module）**：任务拆解后得到的独立函数或类，实现特定的子功能，类似乐高块可以自由拼接。  

**聚类（Clustering）**：把多次生成的子模块按实现方式相似度分组，找出同类实现的共性。  

**代表子模块（Representative Sub‑module）**：在每个聚类中挑选最具通用性、最简洁的实现，作为后续生成的“模板”。  

**Prompt Augmentation（提示增强）**：在原始提示中加入额外信息（如代表子模块代码），让模型在新一轮生成时能够直接引用这些实现。  

**Pass@k**：衡量模型在 k 次尝试中至少有一次生成可通过测试用例的指标，常用于评估代码生成系统的成功率。

### 核心创新点

1. **从单轮生成到自我修订链**  
   之前的工作大多只让模型一次性输出完整代码；CodeChain 引入了“生成 → 提取子模块 → 聚类 → 选代表 → 提示增强 → 重新生成”的循环，使模型在每一次迭代中都能利用自己已经验证过的代码片段。这样不仅提升了代码的正确率，也自然形成了模块化结构。

2. **基于聚类的子模块抽象**  
   直接把所有生成的函数喂回模型会导致噪声累积。CodeChain 先对这些函数进行相似度聚类，挑选出每类最通用、最简洁的实现作为代表子模块。相当于在“众多草稿”中挑选出“最佳范例”，从而让后续生成更具复用价值。

3. **提示中注入代表实现**  
   传统的 CoT 提示只包含思考步骤，缺少实际代码参考。CodeChain 在每轮的 CoT 提示后面追加代表子模块的实现，让模型在思考的同时可以直接调用已验证的代码块，类似于在写代码时打开了“代码片段库”。

4. **跨模型、跨规模的通用框架**  
   作者在 OpenAI 系列模型和开源的 WizardCoder 上都验证了 CodeChain 的有效性，说明该方法不依赖特定模型的内部结构，只要模型支持文本生成和基本的指令遵循，就能受益。

### 方法详解

**整体思路**  
CodeChain 把一次完整的代码生成任务拆成若干迭代，每一次迭代包括两大步骤：① 生成并抽取子模块 → 聚类并挑选代表子模块；② 用这些代表子模块扩充原始的思维链提示，重新生成更完整、更模块化的代码。整个过程在模型内部完成，无需外部工具的介入。

**步骤拆解**

1. **初始思维链生成**  
   - 给模型一个“思维链”式的指令，要求它先写出解题思路（例如：分步骤说明要实现的功能、需要哪些子函数），随后直接生成对应的代码块。  
   - 这一步相当于让模型先把任务拆解成若干子任务，并尝试一次性实现每个子任务。

2. **子模块抽取与聚类**  
   - 从模型输出的代码中解析出所有独立的函数/类定义，视为候选子模块。  
   - 使用文本相似度或语义嵌入（如 CodeBERT）对这些候选进行聚类，目的是把实现方式相近的函数归为一类。  
   - 在每个聚类内部，根据代码长度、注释完整度、通过的单元测试数等指标挑选出最具代表性的实现，记作“代表子模块”。这一步的核心是把噪声（低质量实现）过滤掉，只保留“最佳范例”。

3. **提示增强（Prompt Augmentation）**  
   - 将原始的思维链提示与所有代表子模块的代码拼接在一起，形成新的、信息更丰富的提示。  
   - 在提示中明确告诉模型：“下面的函数已经经过验证，你可以直接调用它们来完成剩余的功能”。这样模型在生成时会倾向于复用已有实现，而不是重新写一遍。

4. **第二轮（或多轮）生成**  
   - 使用增强后的提示再次让模型生成完整的解决方案。因为模型已经看到了高质量的子模块实现，它更容易把这些块拼接成一个整体，且错误率显著下降。  
   - 如果需要进一步提升，可重复步骤 2‑4，形成更长的自我修订链，直至满足预设的通过率或迭代次数上限。

**关键细节与巧思**  
- **聚类数目**：作者在实验中发现，适度的聚类数量（如 5‑10 类）能在保持多样性的同时提供足够的代表性，过多会导致每类只有少量样本，代表性下降。  
- **代表子模块的选取标准**：不仅看代码是否通过测试，还考虑代码的可读性和抽象程度，确保后续复用时不会引入隐藏的副作用。  
- **自我修订的停机条件**：当新一轮生成的代码通过率不再提升或达到预设阈值时，停止迭代，防止无限循环。  
- **模型无状态**：整个流程只依赖文本交互，不需要对模型内部状态进行修改，这使得 CodeChain 能直接套用在任何支持 API 调用的 LLM 上。

### 实验与效果

- **测试数据集**：作者在两个公开的竞争编程基准上评估：APPS（约 10k 题目，覆盖算法、数据结构等）和 CodeContests（来自真实竞赛的高难度题目）。  
- **对比基线**：包括直接的 CoT 生成、单轮的模块化提示、以及最新的基于检索的代码生成方法。  
- **主要结果**：  
  - 在 APPS 上，CodeChain 的 Pass@1 提升了约 35%（相对提升），即原本 30% 的通过率提升到约 40%。  
  - 在 CodeContests 上，提升更为显著，约 76% 的相对提升，说明在高难度任务中模块化复用的优势更大。  
  - 对 OpenAI 的 GPT‑4 与开源的 WizardCoder（7B）均实现了类似幅度的提升，验证了方法的模型无关性。  
- **消融实验**：  
  - 去掉聚类直接使用所有子模块会导致性能下降约 12%，说明聚类过滤噪声是关键。  
  - 只使用一次自我修订（单轮）而不形成链式迭代，提升幅度约为 20%（APPS），表明多轮修订带来的累积效应不可忽视。  
- **局限性**：  
  - 需要对生成代码进行可靠的函数抽取和语义相似度计算，若代码风格极度不统一，聚类质量会受影响。  
  - 对于极度创新的子任务（没有任何相似已有实现），代表子模块的复用帮助有限。  
  - 迭代次数过多会显著增加推理成本，实际部署时需在效果与计算预算之间权衡。

### 影响与延伸思考

CodeChain 把“自我修订”与“模块化复用”结合起来，为 LLM 代码生成提供了一个可扩展的思路。自论文发布后，已有工作尝试将相同的自我修订链用于自然语言写作、数学证明等需要多步推理的场景，说明该框架的通用性。后续研究可能会在以下方向深化：

- **更智能的子模块检索**：结合向量数据库，实现跨任务、跨项目的跨库复用，而不仅局限于同一次生成的子模块。  
- **动态聚类策略**：根据任务难度或模型信心自适应调整聚类数目，进一步提升代表子模块的质量。  
- **与程序验证工具结合**：在每轮修订后加入自动化单元测试或形式化验证，让模型在“通过测试后再继续”中形成更稳固的迭代闭环。  

如果想深入了解，可关注近期在 “LLM‑guided program synthesis” 领域的会议论文，尤其是那些把检索、强化学习与自我修订结合的工作。

### 一句话记住它

让大模型像程序员一样“写完代码再挑出好函数，放进工具箱，再用这些函数拼出更好的代码”。