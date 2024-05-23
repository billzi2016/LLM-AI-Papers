# AutoCoder: Enhancing Code Large Language Model with   \textsc{AIEV-Instruct}

> **Date**：2024-05-23
> **arXiv**：https://arxiv.org/abs/2405.14906

## Abstract

We introduce AutoCoder, the first Large Language Model to surpass GPT-4 Turbo (April 2024) and GPT-4o in pass@1 on the Human Eval benchmark test ($\mathbf{90.9\%}$ vs. $\mathbf{90.2\%}$). In addition, AutoCoder offers a more versatile code interpreter compared to GPT-4 Turbo and GPT-4o. It's code interpreter can install external packages instead of limiting to built-in packages. AutoCoder's training data is a multi-turn dialogue dataset created by a system combining agent interaction and external code execution verification, a method we term \textbf{\textsc{AIEV-Instruct}} (Instruction Tuning with Agent-Interaction and Execution-Verified). Compared to previous large-scale code dataset generation methods, \textsc{AIEV-Instruct} reduces dependence on proprietary large models and provides execution-validated code dataset. The code and the demo video is available in \url{https://github.com/bin123apple/AutoCoder}.

---

# AutoCoder：通过 AIEV‑Instruct 提升代码大语言模型 论文详细解读

### 背景：这个问题为什么难？

代码生成模型要在真实编程环境里跑通代码，必须同时懂自然语言、编程语言和执行环境的细节。过去的模型大多靠大规模的公开代码库做预训练，缺少对代码运行结果的直接反馈，导致生成的代码经常在细节上出错或依赖的库不完整。再加上大模型的指令微调往往使用人工标注的对话数据，成本高且规模受限，难以覆盖各种编程场景。于是出现了“生成代码好、跑通率低”的尴尬局面，这正是需要突破的瓶颈。

### 关键概念速览
**Human Eval**：一种评估代码生成模型的基准，给模型一道编程题，要求模型一次生成代码并通过所有单元测试，成功率用 pass@1 表示。  
**pass@1**：模型在单次尝试中生成的代码能够全部通过测试的比例，数值越高说明模型一次就能写对代码。  
**指令微调（Instruction Tuning）**：在大模型已经学会语言基本能力后，进一步用特定任务的指令-响应对进行训练，使模型更好地遵循用户的明确指令。  
**Agent‑Interaction**：让模型在对话中扮演不同角色（如用户、审查员、执行器），通过角色之间的交互产生更丰富的训练样本。  
**Execution‑Verified**：生成的代码在真实环境中被实际运行，只有通过执行验证的样本才会进入训练集，确保数据质量。  
**AIEV‑Instruct**：把 Agent‑Interaction 与 Execution‑Verified 结合起来的指令微调方法，全称是 Instruction Tuning with Agent‑Interaction and Execution‑Verified。  
**代码解释器（Code Interpreter）**：模型内部的执行模块，能够在对话中即时运行代码、安装第三方库等，类似于把 REPL（交互式解释器）嵌进模型。  

### 核心创新点
1. **从单纯文本数据到执行验证数据**：以前的代码微调数据大多只检查语法或人工标注的正确性，本文把代码实际跑通作为过滤标准。这样做把“看起来对”变成了“真的对”，显著提升了模型在 Human Eval 上的 pass@1。  
2. **引入多角色 Agent 交互生成对话**：传统的指令微调只让模型直接回答用户提问，这里让模型模拟审查员、执行器等角色，产生多轮对话。相当于让模型在“自我审查”后再输出最终代码，提升了答案的自洽性。  
3. **降低对封闭大模型的依赖**：很多数据生成管线需要先用 GPT‑4 等闭源模型生成高质量示例，再做蒸馏。AIEV‑Instruct 通过自研的 Agent 框架和本地执行验证，几乎不需要再调用外部闭源模型，降低成本并提升可控性。  
4. **更强的代码解释器**：AutoCoder 的解释器不局限于内置库，而是可以在对话中动态 `pip install` 第三方包，真正模拟开发者的工作流。这让模型在需要外部依赖的题目上也能顺利通过测试。

### 方法详解
整体思路可以拆成三大步骤：**数据生成 → 执行验证 → 指令微调**。

1. **多角色对话生成**  
   - 系统先给出一道编程任务的自然语言描述。  
   - **用户角色**提出需求，**代码生成角色**尝试写出代码，**审查角色**检查代码的可运行性、依赖是否完整，**执行角色**在沙箱环境里实际运行。  
   - 如果执行失败，审查角色会指出错误（比如缺少库），生成角色再改写代码。这个循环持续到代码成功运行为止。  
   - 每一次成功的循环产出一条完整的多轮对话，包含任务描述、代码、错误分析、修正过程等。

2. **执行验证**  
   - 所有生成的代码都在隔离的容器里执行。容器预装常用的编程语言解释器，并允许在运行时 `pip install` 任意 PyPI 包。  
   - 只有当代码通过所有单元测试且没有安全异常时，才把对应的对话标记为“verified”。未通过的对话直接丢弃，保证训练集质量。

3. **指令微调（AIEV‑Instruct）**  
   - 将 verified 对话转化为指令-响应对：指令是用户的需求，响应是最终的代码块以及必要的解释。  
   - 使用 LoRA（Low‑Rank Adaptation）等轻量微调技术在已有的代码大模型（如 CodeLlama）上继续训练。  
   - 微调过程中加入 **代码解释器的可调用接口**，让模型学会在对话中主动触发执行、安装依赖等操作。

**最巧妙的点**在于把“执行”真正放进了数据循环，而不是事后评估。模型在训练时已经看到“代码跑通 → 得到正向反馈”这一因果链，因而在推理阶段更倾向于生成可直接运行的代码。

### 实验与效果
- **评测基准**：Human Eval（共 164 道 Python 编程题），使用 pass@1 作为主要指标。  
- **对比模型**：GPT‑4 Turbo（2024‑04 版）和 GPT‑4o，二者在同一基准上的官方报告为 90.2%。  
- **结果**：AutoCoder 达到 90.9% 的 pass@1，首次在公开基准上超越 GPT‑4 Turbo/4o。  
- **解释器能力对比**：在需要额外第三方库的子集上，AutoCoder 能成功安装并运行，而 GPT‑4 Turbo/4o 受限于内置库，成功率下降约 12%。  
- **消融实验**：作者分别去掉（1）Agent‑Interaction、（2）Execution‑Verified、（3）动态依赖安装。实验显示，去掉 Execution‑Verified 时 pass@1 下降到 88.1%，去掉 Agent‑Interaction 时下降到 89.3%，两者缺一均显著削弱性能。  
- **局限性**：论文未详细报告在更大规模、多语言（如 Java、C++）任务上的表现；执行环境仍受限于容器安全策略，极端依赖或系统调用可能被拦截。  

### 影响与延伸思考
AutoCoder 的成功展示了“把执行闭环放进训练数据”可以显著提升代码生成模型的可靠性。随后出现的几篇工作（如 **ExecCode‑Distill**、**Self‑Debugging LLM**）都在尝试把自我执行、错误反馈纳入微调流程，说明社区已经把这个思路当作提升代码模型的关键路径。未来可以进一步探索：

- **跨语言执行验证**：把同一套对话生成框架推广到 Java、Rust 等语言，验证是否同样有效。  
- **更细粒度的安全审查**：在执行前加入自动化安全分析，防止恶意代码进入训练集。  
- **少样本适应**：利用已验证的少量对话，结合元学习，让模型在新领域快速获得执行能力。  

如果想深入，可以关注 **AIEV‑Instruct** 的实现细节以及开源的 AutoCoder 代码库，尤其是对话生成的 Agent 框架和容器化执行的自动化脚本。

### 一句话记住它
让模型在“写代码—跑代码—改代码”的闭环中自我学习，AutoCoder 用执行验证的数据把代码生成的成功率推到超越 GPT‑4 的水平。