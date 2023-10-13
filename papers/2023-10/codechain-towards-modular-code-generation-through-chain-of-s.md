# CodeChain: Towards Modular Code Generation Through Chain of   Self-revisions with Representative Sub-modules

> **Date**：2023-10-13
> **arXiv**：https://arxiv.org/abs/2310.08992

## Abstract

Large Language Models (LLMs) have already become quite proficient at solving simpler programming tasks like those in HumanEval or MBPP benchmarks. However, solving more complex and competitive programming tasks is still quite challenging for these models - possibly due to their tendency to generate solutions as monolithic code blocks instead of decomposing them into logical sub-tasks and sub-modules. On the other hand, experienced programmers instinctively write modularized code with abstraction for solving complex tasks, often reusing previously developed modules. To address this gap, we propose CodeChain, a novel framework for inference that elicits modularized code generation through a chain of self-revisions, each being guided by some representative sub-modules generated in previous iterations. Concretely, CodeChain first instructs the LLM to generate modularized codes through chain-of-thought prompting. Then it applies a chain of self-revisions by iterating the two steps: 1) extracting and clustering the generated sub-modules and selecting the cluster representatives as the more generic and re-usable implementations, and 2) augmenting the original chain-of-thought prompt with these selected module-implementations and instructing the LLM to re-generate new modularized solutions. We find that by naturally encouraging the LLM to reuse the previously developed and verified sub-modules, CodeChain can significantly boost both modularity as well as correctness of the generated solutions, achieving relative pass@1 improvements of 35% on APPS and 76% on CodeContests. It is shown to be effective on both OpenAI LLMs as well as open-sourced LLMs like WizardCoder. We also conduct comprehensive ablation studies with different methods of prompting, number of clusters, model sizes, program qualities, etc., to provide useful insights that underpin CodeChain's success.

---

# CodeChain：通过自我修订链与代表性子模块实现模块化代码生成 论文详细解读

### 背景：这个问题为什么难？
现有的大语言模型（LLM）在 HumanEval、MBPP 这类相对简单的编程基准上已经能写出可运行的代码，但面对需要多步推理、复杂数据结构或算法竞赛级别的题目时，模型往往一次性输出一大段“单块”代码。缺乏模块化思路导致代码可读性差、错误难以定位，也让模型难以复用已有的子功能。传统的提示（prompt）技术只能让模型在一次生成中考虑全部细节，无法像人类程序员那样先拆解任务、写出可复用的函数库。因此，提升 LLM 在复杂编程任务上的表现，需要一种机制让模型主动进行任务分解并在后续迭代中重用已验证的子模块。

### 关键概念速览
**Chain‑of‑Thought（思维链）**：在给出最终答案前，让模型把思考过程写出来，类似于人在解题时先列出步骤再求解，帮助模型保持逻辑连贯性。  
**自我修订（Self‑revision）**：模型在生成完一次答案后，再次审视并改写自己的输出，就像人写完代码后自己调试、重构。  
**子模块（Sub‑module）**：代码中的独立函数或类，承担特定子任务，可被其他代码块调用，类似于乐高积木的单块。  
**代表性子模块（Representative Sub‑module）**：在多次生成的子模块中，通过聚类挑选出的最具通用性、可复用的实现，类似于从一堆相似工具中挑出最标准的那把螺丝刀。  
**聚类（Clustering）**：把相似的子模块放在同一组，常用的相似度依据是代码结构或语义相似，类似于把相同颜色的积木放在一起。  
**Prompt Augmentation（提示增强）**：在原始提示的基础上加入额外信息（如已选的子模块实现），相当于给模型提供了“参考答案”来引导后续生成。  

### 核心创新点
1. **单次生成 → 多轮自我修订**：传统方法只让模型一次性输出完整代码，而 CodeChain 让模型在每轮生成后提取子模块、聚类并挑选代表实现，再把这些实现写回提示，迫使模型在下一轮“站在已有代码的肩膀上”继续构建。这样显著提升了代码的可复用性和错误纠正机会。  
2. **随机子模块 → 代表性子模块**：以前的模块化尝试往往直接使用所有生成的函数，导致冗余和噪声。CodeChain 用聚类把相似函数归类，并只保留每类的代表实现，使提示中只出现最通用、最干净的代码片段，降低了提示长度并提升了质量。  
3. **思维链 + 模块化提示**：单纯的思维链只能帮助模型理清思路，但仍会产生单块代码。CodeChain 把思维链与模块化提示结合，在思考步骤中明确要求“把任务拆成子函数”，并在后续迭代中实际提供这些子函数的实现，形成闭环。  
4. **跨模型通用性**：大多数模块化生成方法依赖特定模型的内部结构或微调数据，CodeChain 只通过提示层面的操作实现模块化，实验表明它在 OpenAI 系列模型和开源的 WizardCoder 上都能带来同样的提升，展示了方法的模型无关性。

### 方法详解
**整体框架**  
CodeChain 将一次代码生成拆成两层循环：外层是“思维链 + 子模块抽取”，内层是“聚类 → 代表子模块 → 提示增强 → 重新生成”。整个过程可以想象成一次“写草稿—挑好句—再写稿”的写作循环，只不过这里的“句子”是函数实现。

**步骤拆解**  

1. **思维链提示生成初稿**  
   - 给模型一个任务描述，并在提示中加入“先用思维链说明解题思路，每一步对应一个子函数”。模型先输出类似  
     ```
     思考步骤：
     1. 读取输入 → parse_input()
     2. 构建图 → build_graph()
     3. 求最短路 → dijkstra()
     4. 输出结果 → format_output()
     ```  
   - 随后模型在同一次生成里给出每个子函数的代码实现，形成一个完整但仍然是“单块”代码的集合。

2. **子模块抽取与聚类**  
   - 从生成的代码中解析出所有函数定义（子模块）。  
   - 使用代码相似度度量（如基于 AST 的结构相似或语义嵌入）把函数划分到若干簇。每个簇对应一种“实现思路”。  
   - 对每个簇执行 **代表性选择**：挑选最简洁、最少依赖外部变量的函数作为该簇的代表，实现类似“标准库函数”。如果有多个候选，优先选择通过单元测试的那个。

3. **提示增强**  
   - 将选出的代表子模块代码直接拼接到原始思维链提示的后面，形成 **增强提示**。此时提示里已经包含了“已经验证的子函数实现”。  
   - 同时在提示中加入指令：“在下面的代码中复用已有子模块，不要重新实现相同功能”。这一步相当于给模型提供了“代码片段库”。

4. **第二轮（或多轮）生成**  
   - 使用增强提示让模型重新生成完整解答。因为提示里已经有了可直接调用的函数实现，模型倾向于在高层逻辑上调用这些函数，而不是重新写一遍。  
   - 若仍有未覆盖的子任务，继续回到步骤 2，进行新的聚类与代表选择，形成新的增强提示，循环若干次（实验中 2~3 轮已足够）。

**关键细节**  
- **聚类数目**：作者在实验中发现 5~10 个簇在大多数任务上能平衡多样性与噪声。  
- **代表性判定**：除了代码简洁度，还会运行一次轻量化的单元测试（使用题目提供的样例），确保代表子模块是“可运行的”。  
- **提示长度控制**：因为 LLM 的上下文窗口有限，聚类后只保留代表子模块，避免提示爆炸。  
- **最巧妙的点**：把“代码复用”这件人类编程的常识搬进了纯提示层面，而不需要对模型进行额外的微调或结构改动，几乎可以直接套用在任何支持长上下文的 LLM 上。

### 实验与效果
- **测试数据集**：APPS（包含 10k+ 编程题）和 CodeContests（竞赛级别题目）。两者都要求较高的算法深度和代码组织能力。  
- **基线对比**：与原始 GPT‑4、GPT‑3.5、以及开源 WizardCoder（7B/13B）在相同的 zero‑shot / few‑shot 设置下比较。  
- **提升幅度**：在 APPS 上，CodeChain 的 pass@1 提升了约 35%；在 CodeContests 上提升了约 76%。这些数字是相对于不使用任何模块化提示的直接生成结果。  
- **消融实验**：  
  - 去掉聚类只保留所有子模块 → 提升幅度下降约 12%。  
  - 只使用思维链不进行自我修订 → 提升约 8%。  
  - 改变聚类数目（从 3 到 15）显示 5~8 簇效果最佳，过多或过少都会削弱收益。  
- **模型规模影响**：在 7B、13B、以及 175B 参数的模型上均有正向提升，说明方法对模型大小不敏感。  
- **局限性**：  
  - 对于需要大量全局状态共享的题目（如全局变量大量交叉），代表子模块的抽离会导致接口不匹配，需要手工调节提示。  
  - 当前聚类依赖于预先定义的相似度度量，若度量不准可能选出不恰当的代表实现。  
  - 提示长度仍受限于模型上下文窗口，极端长代码仍可能被截断。

### 影响与延伸思考
CodeChain 把“模块化思维”搬进了 LLM 的提示层面，开启了“提示驱动的代码复用”新方向。随后的工作（如 **ModularCoder**、**PromptChain**）开始探索更细粒度的函数抽象、跨任务的子模块库构建，甚至尝试把聚类过程外部化为专门的代码检索系统。对想进一步研究的读者，可以关注以下几个方向：  
1. **自动化子模块库**：把多轮生成得到的代表子模块存入长期向量库，实现跨任务检索。  
2. **更鲁棒的相似度度量**：结合图神经网络或大模型嵌入提升代码相似度的准确性。  
3. **交互式自我修订**：让模型在每轮生成后主动提出“我这里有不确定的实现，是否需要改进？”形成人机协同的迭代。  
4. **提示压缩技术**：在上下文受限的模型上，研究如何把大量子模块信息压缩成摘要或符号化表示。  

### 一句话记住它
让大模型像程序员一样“先拆任务、写库函数、再组装”，只需在提示里循环加入已验证的子函数，就能显著提升复杂代码生成的正确率。