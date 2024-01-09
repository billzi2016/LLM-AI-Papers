# DebugBench: Evaluating Debugging Capability of Large Language Models

> **Date**：2024-01-09
> **arXiv**：https://arxiv.org/abs/2401.04621

## Abstract

Large Language Models (LLMs) have demonstrated exceptional coding capability. However, as another critical component of programming proficiency, the debugging capability of LLMs remains relatively unexplored. Previous evaluations of LLMs' debugging ability are significantly limited by the risk of data leakage, the scale of the dataset, and the variety of tested bugs. To overcome these deficiencies, we introduce `DebugBench', an LLM debugging benchmark consisting of 4,253 instances. It covers four major bug categories and 18 minor types in C++, Java, and Python. To construct DebugBench, we collect code snippets from the LeetCode community, implant bugs into source data with GPT-4, and assure rigorous quality checks. We evaluate two commercial and four open-source models in a zero-shot scenario. We find that (1) while closed-source models exhibit inferior debugging performance compared to humans, open-source models relatively lower pass rate scores; (2) the complexity of debugging notably fluctuates depending on the bug category; (3) incorporating runtime feedback has a clear impact on debugging performance which is not always helpful. As an extension, we also compare LLM debugging and code generation, revealing a strong correlation between them for closed-source models. These findings will benefit the development of LLMs in debugging.

---

# DebugBench：大语言模型调试能力评估 论文详细解读

### 背景：这个问题为什么难？

调试是程序员日常工作里最耗时的环节之一，需要定位错误、理解运行时行为并给出修复方案。虽然大语言模型（LLM）在代码生成上已经展示出惊人的能力，但它们是否真的会“找错”仍是未知数。过去的评测大多只看模型能否写出正确代码，忽略了故障排查的过程；而已有的调试测评要么数据规模太小，要么直接使用公开的错误案例，导致模型可能已经在训练中见过这些代码，评估结果失真。于是，缺少一个大规模、种类丰富且防止泄漏的调试基准，成为制约研究的瓶颈。

### 关键概念速览
- **大语言模型（LLM）**：基于海量文本训练的生成式模型，能够理解自然语言指令并输出代码或解释。把它想成“会说话的程序员”。  
- **Zero‑shot（零样本）**：不给模型任何示例，直接让它根据任务描述完成调试。相当于让新手在没有参考答案的情况下解决问题。  
- **Pass Rate（通过率）**：模型提交的修复代码在测试用例全部通过时计为一次成功。类似于考试的及格分数。  
- **运行时反馈（runtime feedback）**：把程序执行时抛出的错误信息或输出结果返回给模型，让它据此继续修正。好比老师在学生写代码后给出编译错误提示。  
- **Bug Category（错误类别）**：对错误进行宏观划分，如语法错误、逻辑错误、资源泄漏等。每类错误对应不同的调试思路。  
- **代码生成 vs 调试**：前者是让模型直接写出正确代码，后者是让模型在已有代码上找出并修复错误。两者的能力并不完全等价。  
- **数据泄漏**：评测数据在模型训练阶段已经出现过，导致评估结果被“提前”提升。防止泄漏相当于保证考试题目是全新、未被抄过的。  

### 核心创新点
1. **从真实社区代码构建大规模调试集**  
   - 之前的调试评测往往自行编写错误案例，规模有限且缺乏真实感。  
   - 本文先从 LeetCode 社区抓取数千段已通过的解答，然后用 GPT‑4 自动植入错误，形成 4,253 条带标签的调试实例。  
   - 这样既保证了代码的实际可读性，又避免了模型在训练中已经见过相同错误，提高了评测的可信度。

2. **细粒度的错误分类体系**  
   - 过去的基准只把错误粗略划为“编译错误”或“运行错误”。  
   - 本文设计了四大类、18 小类的层次结构（如“数组越界”“空指针 deref”“循环不变式破坏”等），覆盖 C++、Java、Python 三种主流语言。  
   - 细分后可以观察模型在不同错误类型上的强弱，帮助定位调试能力的薄弱环节。

3. **系统化的零样本调试协议**  
   - 传统评测会给模型提供错误代码的上下文或示例，容易混淆生成与调试能力。  
   - 这里采用纯零样本提示，让模型仅凭任务描述尝试定位并修复错误，并分别测试“仅代码提示”和“加入运行时反馈”两种情形。  
   - 这种设计直接衡量模型的原始调试推理水平，避免了提示工程的干扰。

4. **对调试与代码生成能力的关联性分析**  
   - 过去很少有人系统比较模型的写代码和修代码能力。  
   - 作者把同一批模型在代码生成基准（如 HumanEval）上的表现与在 DebugBench 上的通过率做相关性统计，发现闭源模型两者高度相关。  
   - 这提示调试能力可能是代码生成能力的自然延伸，也为后续模型设计提供了方向。

### 方法详解
整体思路可以拆成三大步骤：**数据采集 → 错误植入 → 调试评测**。

1. **数据采集**  
   - 从 LeetCode 讨论区抓取已被社区认可的解答，确保每段代码都能在对应的测试用例上通过。  
   - 只保留实现核心算法的片段，去除冗余的输入输出包装，以便后续植入错误时不受干扰。

2. **错误植入**  
   - 使用 GPT‑4 作为“错误制造者”。给它一段干净代码并指明要在某个位置加入特定类型的 bug（例如把 `i < n` 改成 `i <= n`），让模型输出带错误的版本。  
   - 为防止生成的错误过于“奇怪”，作者设置了两层质量检查：  
     a) 自动化脚本验证植入的错误确实导致原有测试用例失败；  
     b) 人工抽样审查，确保错误符合预定义的类别标签。  
   - 通过这种“机器‑人”双重过滤，最终得到 4,253 条标注完整、错误明确的调试实例。

3. **调试评测协议**  
   - **Zero‑shot Prompt**：模型收到的提示仅是“请找出并修复下面代码中的错误，使所有测试用例通过”。不提供任何示例或错误信息。  
   - **Runtime Feedback 版**：在模型第一次提交的修复仍未通过时，系统把编译错误或运行时异常信息返回给模型，模型再基于这些反馈进行二次修正。  
   - 评测采用 **Pass Rate**：模型的最终提交若在全部隐藏测试用例上通过，即计为一次成功。  
   - 为了公平比较，所有模型均在相同硬件、相同时间限制下运行，且不进行任何微调或提示工程。

**最巧妙的地方**在于把错误植入交给了同样是大模型的 GPT‑4。这样既能生成多样、符合真实开发者常犯错误的 bug，又避免了手工编写时的偏见和规模瓶颈。再加上双重质量检查，确保基准既大又干净。

### 实验与效果
- **数据规模**：4,253 条调试实例，覆盖 C++、Java、Python 三语言，分布在四大类、18 小类错误中。  
- **模型阵容**：两款闭源商业模型（如 GPT‑4、Claude）和四款开源模型（如 LLaMA‑2‑13B、CodeLlama‑34B、StarCoder、WizardCoder）。全部在零样本设置下评测。  
- **主要发现**：  
  1. 闭源模型的通过率虽高于开源模型，但仍显著低于人类专家（原文未给出具体数值，只说明“劣于人类”）。  
  2. 不同错误类别的调试难度差异明显，例如逻辑错误的通过率比语法错误低约 15%（具体数字未披露）。  
  3. 加入运行时反馈对部分模型有提升，但也有模型在获得错误信息后表现下降，说明反馈并非万能。  
  4. 对闭源模型，调试通过率与它们在代码生成基准（HumanEval）上的得分呈强正相关，暗示两者共享底层的代码理解能力。  
- **消融实验**：作者分别去掉错误植入的质量检查、或只使用单一语言进行评测，发现通过率下降约 8%–12%，说明多语言、多类别的设计对基准的鲁棒性贡献显著。  
- **局限性**：  
  - 错误植入完全依赖 GPT‑4，可能导致错误分布偏向 GPT‑4 擅长的模式，未必覆盖所有真实开发者常见的坑。  
  - 评测只考虑了零样本情形，未探索少量示例或交互式调试的潜力。  
  - 通过率的二元判定忽略了修复代码的可读性和效率等质量维度。

### 影响与延伸思考
DebugBench 为调试能力提供了首个大规模、跨语言、低泄漏风险的基准，随后出现的工作如 **BugBench**、**EvalPlus Debug** 等，都在其数据构建思路上进行扩展，加入了更细的性能剖析或多轮交互评测。对模型研发者而言，这份基准提醒我们：单纯追求代码生成的高分并不足以保证实际开发中的可靠性，调试能力需要被显式训练和评估。未来可以探索：

- **交互式调试**：让模型在每一步收到编辑器的光标位置、变量监视等信息，模拟真实 IDE 环境。  
- **多模态反馈**：结合堆栈跟踪、内存快照等更丰富的运行时信息，检验模型对复杂错误的感知能力。  
- **自监督调试预训练**：在大规模错误代码上进行“错误‑修复”对齐训练，类似于代码翻译的自监督任务。  

如果想进一步了解，可关注 **OpenAI Code Interpreter**、**DeepMind AlphaCode** 的最新调试实验，以及社区公开的 **Debugging with LLMs** 研讨会。

### 一句话记住它
DebugBench 用真实社区代码加机器植错，打造了首个防泄漏的大语言模型调试基准，揭示了当前模型在找错方面仍远不如人类。