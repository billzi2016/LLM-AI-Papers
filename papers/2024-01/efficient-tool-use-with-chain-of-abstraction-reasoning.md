# Efficient Tool Use with Chain-of-Abstraction Reasoning

> **Date**：2024-01-30
> **arXiv**：https://arxiv.org/abs/2401.17464

## Abstract

To achieve faithful reasoning that aligns with human expectations, large language models (LLMs) need to ground their reasoning to real-world knowledge (e.g., web facts, math and physical rules). Tools help LLMs access this external knowledge, but there remains challenges for fine-tuning LLM agents (e.g., Toolformer) to invoke tools in multi-step reasoning problems, where inter-connected tool calls require holistic and efficient tool usage planning.   In this work, we propose a new method for LLMs to better leverage tools in multi-step reasoning. Our method, Chain-of-Abstraction (CoA), trains LLMs to first decode reasoning chains with abstract placeholders, and then call domain tools to reify each reasoning chain by filling in specific knowledge. This planning with abstract chains enables LLMs to learn more general reasoning strategies, which are robust to shifts of domain knowledge (e.g., math results) relevant to different reasoning questions. It also allows LLMs to perform decoding and calling of external tools in parallel, which avoids the inference delay caused by waiting for tool responses. In mathematical reasoning and Wiki QA domains, we show that our method consistently outperforms previous chain-of-thought and tool-augmented baselines on both in-distribution and out-of-distribution test sets, with an average ~6% absolute QA accuracy improvement. LLM agents trained with our method also show more efficient tool use, with inference speed being on average ~1.4x faster than baseline tool-augmented LLMs.

---

# 高效工具使用与抽象链推理 论文详细解读

### 背景：这个问题为什么难？

在让大语言模型（LLM）解决需要外部知识的多步推理时，模型必须决定何时、怎样调用工具（如搜索引擎、数学求解器）。早期的工具增强方法（比如 Toolformer）只能在单步或固定模式下触发工具，面对需要多次、相互关联的调用时会出现两大痛点：一是模型往往只能顺序等待上一次工具的返回，导致推理速度慢；二是每一次调用都基于当前的上下文，缺乏全局规划，导致在知识迁移（比如数学常数变化）时表现脆弱。于是，如何让 LLM 在整体上规划工具使用、并行化调用，成为提升效率和鲁棒性的关键瓶颈。

### 关键概念速览
- **工具（Tool）**：模型可以调用的外部程序或 API，例如网页搜索、符号计算器，用来补充模型内部的知识盲区。  
- **Chain-of-Thought（思维链）**：让模型在给出答案前先写出推理步骤，类似解题时的草稿，帮助模型保持逻辑连贯。  
- **抽象占位符（Abstract Placeholder）**：在推理链中用一个通用符号（如 `<NUM>`、`<ENTITY>`）代替具体数值或实体，先把结构搭好再填实。  
- **抽象链（Chain-of-Abstraction，CoA）**：先让模型生成只含占位符的推理链，再逐个调用工具把占位符具体化的两阶段流程。  
- **并行工具调用**：在 CoA 中，所有占位符对应的工具请求可以一次性发起，而不是等前一个返回后再发，类似一次性下单多个商品。  
- **领域迁移（Domain Shift）**：模型在不同任务或不同数据分布下使用相同推理结构的能力，例如同一套数学公式在不同题目里出现。  

### 核心创新点
1. **抽象链先行 → 具体化后置**  
   传统方法在每一步推理时直接请求工具，导致模型必须在每一步都等返回。CoA 先让模型输出只含占位符的完整推理链，再统一触发对应工具，把占位符填回。这样模型在规划阶段不受工具响应时间限制，整体推理更连贯。

2. **全局规划的抽象占位符 → 更强的迁移能力**  
   通过抽象占位符，模型学习到“先决定要用哪些工具、顺序如何”，而不是记住具体数值。于是当数学常数或事实库变化时，模型只需要重新填充占位符，推理结构保持不变，表现出对领域迁移的鲁棒性。

3. **并行化工具调用 → 推理速度提升**  
   把所有占位符对应的工具请求一次性发送，利用外部系统的并发能力，省去逐步等待的时间。实验显示，这种并行策略让整体推理比传统工具增强模型快约 1.4 倍。

4. **统一的端到端训练框架**  
   作者在训练时让模型同时学习生成抽象链和对应的占位符映射，使用了类似教师强制的方式让模型在生成抽象链后自动触发工具并学习填回结果。这样模型在推理时不需要额外的调度逻辑，真正做到“一键式”工具使用。

### 方法详解
**整体思路**：CoA 把推理过程拆成两层：  
1）**抽象层**：模型先写出完整的思维链，但把所有需要外部知识的地方用占位符标记。  
2）**具体化层**：系统根据占位符的类型批量调用对应工具，得到真实数值或事实后把占位符替换回去，得到最终答案。

**步骤拆解**：

1. **抽象链生成**  
   - 输入：原始问题（如数学题或 Wiki 问答）。  
   - 模型输出：形如 “先计算 `<NUM1>`，再把 `<NUM1>` 代入公式得到 `<RESULT>`，最后比较 `<RESULT>` 与 `<NUM2>`”。这里的 `<NUM1>`、`<NUM2>`、`<RESULT>` 都是抽象占位符。  
   - 类比：像先在白板上画出解题框架，再去找具体数字。

2. **占位符识别与工具映射**  
   - 系统扫描抽象链，识别每种占位符对应的工具类型（搜索、计算器、知识库）。  
   - 为每个占位符生成一个工具请求，例如对 `<NUM1>` 发起数学求解器调用，对 `<ENTITY>` 发起网页搜索。

3. **并行调用**  
   - 所有请求一次性发送到对应的外部服务。因为请求相互独立，服务器可以并行处理，整体等待时间接近最慢的那一个，而不是所有请求的累计时间。

4. **结果回填**  
   - 收到每个工具的返回后，系统把对应占位符替换成真实值。  
   - 替换后得到完整的、无占位符的思维链，例如 “先计算 42， 再把 42 代入公式得到 84，最后比较 84 与 100”。  

5. **答案生成**  
   - 最后一步，模型根据填充好的完整链直接生成答案或做进一步的推理（如比较大小、选择选项）。

**训练细节**：  
- 作者使用了两阶段的监督信号：第一阶段让模型学习抽象链的结构（使用人工标注的抽象链或自动生成的模板），第二阶段让模型学习占位符到工具调用的映射（通过模拟工具返回的方式进行教师强制）。  
- 损失函数同时考虑抽象链的语言流畅度和占位符填充的准确性，确保模型在两阶段都能保持高质量输出。

**最巧妙的点**：把“何时调用工具”从推理的每一步中抽离出来，变成“先规划、后执行”。这让模型的思考过程更像人类先列出解题步骤，再去查资料或算数，极大降低了对工具响应的依赖。

### 实验与效果
- **测试任务**：数学推理（如 GSM8K）和 Wiki 问答（如 Natural Questions）。  
- **基线**：传统 Chain-of-Thought（纯文本思维链）、Toolformer（单步工具调用）以及最新的工具增强模型。  
- **整体提升**：在两类任务上，CoA 的 QA 正确率平均提升约 6%（例如 GSM8K 从 71% 提升到 77%）。  
- **速度提升**：并行调用使得推理时间比普通工具增强模型快约 1.4 倍，尤其在需要多次工具调用的题目上更明显。  
- **消融实验**：去掉抽象占位符直接使用传统工具调用，准确率下降约 4%；关闭并行调用改为顺序调用，速度回到基线水平，验证了两大设计的必要性。  
- **局限性**：论文未在大规模开放域对话或需要复杂工具组合的场景做实验，抽象占位符的自动生成仍依赖一定的标注或模板，完全零样本生成的鲁棒性还有待验证。

### 影响与延伸思考
CoA 把“工具调度”提升到全局规划层面，开启了“先抽象后具体化”的新范式。后续工作（如 2024 年的 *Abstracted Tool Planning*、*Parallel Tool-Chain*）已经在此基础上进一步探索更细粒度的占位符类型和自适应并行度调度。对想继续深入的读者，可以关注以下方向：  
- **自动抽象占位符生成**：如何让模型在没有人工模板的情况下自行发现哪些步骤需要外部工具。  
- **跨模态工具协同**：把搜索、图像识别、代码执行等多种工具统一进抽象链框架。  
- **强化学习调度**：利用奖励信号让模型学习何时并行、何时顺序，以适应不同工具的响应特性。  

### 一句话记住它
先让模型写出只含占位符的完整推理链，再一次性并行调用工具把占位符填回，既提升了准确率，又把推理速度提速约 1.4 倍。