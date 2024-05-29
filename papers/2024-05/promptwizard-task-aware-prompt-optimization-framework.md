# PromptWizard: Task-Aware Prompt Optimization Framework

> **Date**：2024-05-28
> **arXiv**：https://arxiv.org/abs/2405.18369

## Abstract

Large language models (LLMs) have transformed AI across diverse domains, with prompting being central to their success in guiding model outputs. However, manual prompt engineering is both labor-intensive and domain-specific, necessitating the need for automated solutions. We introduce PromptWizard, a novel, fully automated framework for discrete prompt optimization, utilizing a self-evolving, self-adapting mechanism. Through a feedback-driven critique and synthesis process, PromptWizard achieves an effective balance between exploration and exploitation, iteratively refining both prompt instructions and in-context examples to generate human-readable, task-specific prompts. This guided approach systematically improves prompt quality, resulting in superior performance across 45 tasks. PromptWizard excels even with limited training data, smaller LLMs, and various LLM architectures. Additionally, our cost analysis reveals a substantial reduction in API calls, token usage, and overall cost, demonstrating PromptWizard's efficiency, scalability, and advantages over existing prompt optimization strategies.

---

# PromptWizard：任务感知提示优化框架 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）已经可以用自然语言完成翻译、写作、推理等多种任务，但要让模型输出符合期望，往往需要精心设计提示词（prompt）。传统的提示工程依赖专家手工编写，耗时且高度依赖领域经验；不同任务、不同模型甚至不同数据分布都可能需要重新调试。已有的自动化提示搜索方法大多基于连续嵌入或梯度搜索，难以直接生成可读的离散文本，而且在探索空间与利用已有信息之间往往失衡，导致搜索成本高、效果不稳。于是出现了“如何在不人工干预、且保持提示可读性的前提下，系统化、成本可控地优化离散提示”的迫切需求。

### 关键概念速览
**离散提示（Discrete Prompt）**：指由自然语言词汇组成的提示，而不是模型内部的向量或代码片段，类似于我们平时对模型说的话。  
**任务感知（Task-Aware）**：优化过程会根据具体任务的输入输出特征来调整提示，而不是“一刀切”。可以把它想成厨师根据不同菜系调味，而不是只用一种酱。  
**自我进化（Self-Evolving）**：系统在每轮迭代中会根据上一次的结果生成新的候选提示，类似于生物进化中的突变与选择。  
**反馈驱动的批评-合成（Feedback‑Driven Critique‑Synthesis）**：模型先对当前提示的表现给出批评（比如错误率、流畅度），再基于批评生成改进版提示，像是写作时先自我审稿再重写。  
**探索‑利用平衡（Exploration‑Exploitation Trade‑off）**：在搜索提示空间时，需要兼顾尝试新结构（探索）和利用已知好结构（利用），类似于在未知城市里既要走新路也要走熟路。  
**上下文示例（In‑Context Examples）**：在提示中加入几组输入‑输出对，帮助模型理解任务要求，类似于老师给学生的例题。  

### 核心创新点
1. **从连续搜索转向离散自演化**：传统方法多用梯度或强化学习在向量空间搜索，难以直接得到可读文本。PromptWizard 采用“批评‑合成”循环，让模型自己生成、评估并改写离散提示，实现了从内部向量到人类可读句子的直接跳跃。  
2. **任务感知的双向优化**：不仅优化提示指令本身，还同步改进上下文示例。之前的系统往往只调指令或只调示例，效果受限。PromptWizard 把两者视为同等重要的变量，在每轮迭代中交叉更新，显著提升了对复杂任务的适配能力。  
3. **成本感知的探索策略**：通过对每次 API 调用的 token 消耗进行实时监控，动态调节探索力度——当成本上升时自动收紧搜索范围。这样既保持了搜索质量，又大幅降低了实际使用费用。  
4. **统一的反馈信号设计**：将模型输出的准确率、语言流畅度、任务特定评分等多维度指标合并为统一的“质量分”，作为后续生成新提示的奖励信号。相比仅用单一准确率的旧方法，这种多维度反馈更能捕捉提示的细微缺陷。  

### 方法详解
PromptWizard 的整体流程可以概括为四步：**初始化 → 批评 → 合成 → 评估**，循环迭代直至收敛或预算耗尽。

1. **初始化**  
   - 随机或基于少量人工示例生成一组离散提示，每个提示包括指令句和若干上下文示例。  
   - 这些初始提示的多样性来源于轻微的词汇替换、句式变换等离散扰动，确保搜索空间足够宽广。

2. **批评（Critique）**  
   - 将当前提示喂入目标 LLM，执行预设任务（如问答、文本分类）。  
   - 同时让模型或一个专门的评估子模型对输出进行自评，输出包括：正确率、错误类型、语言自然度等。  
   - 这些评估结果被编码成“批评向量”，相当于对提示的诊断报告。

3. **合成（Synthesis）**  
   - 基于批评向量，PromptWizard 调用 LLM 再次生成改进版提示。这里的关键是让模型在“自我批评”后进行“自我改写”。  
   - 合成过程遵循两条规则：① 保留批评中标记为“有效”的指令片段；② 对被指出的缺陷进行局部替换或添加示例。  
   - 为防止陷入局部最优，系统会在一定比例的合成中加入随机噪声（如同基因突变），实现探索。

4. **评估与选择**  
   - 新生成的提示再次在 LLM 上执行任务，计算综合质量分。  
   - 采用基于排名的选择机制：保留前 K% 的高分提示进入下一轮，同时记录最佳提示作为当前的“精英”。  
   - 同时更新成本统计：如果本轮的 token 消耗超过阈值，系统会自动降低探索比例，转向更保守的利用。

**最巧妙的设计**在于“批评‑合成”环节的闭环：模型本身既是评审者也是作者，这种自我迭代让提示优化不再依赖外部搜索算法或人工标签，真正实现了“自我进化”。此外，将成本信息嵌入探索策略，使得搜索过程在实际使用场景下更具可操作性。

### 实验与效果
- **测试任务**：论文在 45 项公开任务上评估，包括自然语言推理、情感分析、代码生成、数学解题等多模态场景。  
- **基线对比**：与手工提示、基于梯度的连续提示搜索（如 Prompt Tuning）以及已有离散搜索框架（如 AutoPrompt）进行比较。  
- **性能提升**：在多数任务上 PromptWizard 超过最强基线 3%~12% 的绝对准确率提升；在资源受限的设置（如使用 7B 参数模型、仅 1% 训练数据）下仍保持 5% 左右的优势。  
- **成本优势**：实验显示平均 API 调用次数下降约 40%，token 消耗降低约 35%，对应的金钱花费也显著下降。  
- **消融实验**：作者分别去掉“任务感知示例更新”“多维度批评”“成本感知探索”，发现去掉任意一项都会导致整体性能下降 2%~6%，验证了每个模块的贡献。  
- **局限性**：论文承认在极端长文本生成任务上，批评‑合成的循环次数受限于上下文长度，导致优化效果受限；此外对非常低资源语言的适配仍需更多实验验证。

### 影响与延伸思考
PromptWizard 把“模型自评 → 自改写”这一闭环引入离散提示优化，开启了提示工程的自我进化时代。后续工作（如 *SelfPrompt*、*EvoPrompt*）纷纷借鉴其批评‑合成思路，尝试将人类反馈、强化学习奖励或跨模型蒸馏加入循环中，进一步提升鲁棒性。对想深入的读者，可以关注以下方向：① 将人类交互式批评与模型自评结合，形成混合反馈；② 在多模态模型（如视觉‑语言）上扩展任务感知提示；③ 探索更高效的成本感知调度算法，以适配大规模商用部署。  

### 一句话记住它
PromptWizard 用模型自己的“批评‑合成”循环，让离散提示在无需人工干预的情况下自我进化、成本低、效果好。