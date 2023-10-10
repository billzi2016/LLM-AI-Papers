# Lemur: Harmonizing Natural Language and Code for Language Agents

> **Date**：2023-10-10
> **arXiv**：https://arxiv.org/abs/2310.06830

## Abstract

We introduce Lemur and Lemur-Chat, openly accessible language models optimized for both natural language and coding capabilities to serve as the backbone of versatile language agents. The evolution from language chat models to functional language agents demands that models not only master human interaction, reasoning, and planning but also ensure grounding in the relevant environments. This calls for a harmonious blend of language and coding capabilities in the models. Lemur and Lemur-Chat are proposed to address this necessity, demonstrating balanced proficiencies in both domains, unlike existing open-source models that tend to specialize in either. Through meticulous pre-training using a code-intensive corpus and instruction fine-tuning on text and code data, our models achieve state-of-the-art averaged performance across diverse text and coding benchmarks among open-source models. Comprehensive experiments demonstrate Lemur's superiority over existing open-source models and its proficiency across various agent tasks involving human communication, tool usage, and interaction under fully- and partially- observable environments. The harmonization between natural and programming languages enables Lemur-Chat to significantly narrow the gap with proprietary models on agent abilities, providing key insights into developing advanced open-source agents adept at reasoning, planning, and operating seamlessly across environments. https://github.com/OpenLemur/Lemur

---

# Lemur：自然语言与代码的协同统一用于语言代理 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）从聊天机器人向真正的“语言代理”转变的过程中，模型必须同时懂得人类语言的细腻表达，又要能在代码层面直接操作工具、调用 API 或修改环境状态。过去的开源模型要么在海量自然语言文本上训练，导致代码推理薄弱；要么专注代码语料，失去对日常对话、情感和常识的把握。缺少两者的平衡，使得模型在真实任务中要么无法生成可执行的脚本，要么在与用户交互时显得生硬、缺乏上下文理解。要让模型在“说”和“做”之间无缝切换，需要一种既能写代码又能聊自然语言的统一能力，这正是 Lemur 想要突破的瓶颈。

### 关键概念速览
**语言代理（Language Agent）**：能够在对话中接受指令、进行推理、生成并执行代码或调用外部工具的系统，类似于拥有“思考+行动”双重能力的智能助理。  
**代码密集语料（Code‑intensive Corpus）**：在训练数据中占比很高的程序代码、脚本和代码注释，帮助模型学习语言的结构化、可执行特性。  
**指令微调（Instruction Fine‑tuning）**：在预训练模型基础上，用带有明确任务指令的示例进行二次训练，使模型更擅长遵循人类给出的操作步骤。  
**全观测 vs. 部分观测环境（Fully‑/Partially‑Observable Environment）**：全观测指模型可以一次性看到任务的全部信息；部分观测则只能逐步获取信息，需要记忆和规划。  
**工具使用（Tool Use）**：模型在对话中主动调用外部 API、数据库或本地函数来完成任务，类似于人类在工作中打开软件、查资料的过程。  
**多任务平均性能（Average Performance across Benchmarks）**：把模型在自然语言理解、代码生成、推理等多个基准测试的得分取平均，衡量其整体平衡度。  

### 核心创新点
1. **从单一语料到双语料预训练**  
   *之前的开源模型大多只用自然语言或只用代码进行大规模预训练* → Lemur 在预训练阶段加入了比例约 50% 的代码数据，并通过混合采样策略让模型在同一次前向传播中交替看到两种语言的样本 → 使模型在语言模型的通用性和代码生成的可执行性之间形成自然的桥梁，实验显示在代码完成度和对话流畅度上均有显著提升。  

2. **统一指令微调框架**  
   *过去的指令微调往往只针对文本任务，代码任务另起一套微调流程* → Lemur‑Chat 使用同一套指令模板，既包含“请解释这段文字”的自然语言指令，也包含“请实现一个排序函数”的代码指令，所有示例在同一批次中混合训练 → 让模型在接到指令时不需要先判断任务类型，而是直接在内部切换语言模式，显著降低了任务切换的错误率。  

3. **环境感知的统一评估**  
   *以往的评测多聚焦单一维度，如仅测代码正确率或仅测对话满意度* → 论文构建了包括全观测和部分观测两类环境的综合任务套件，涵盖人机对话、工具调用、动态规划等场景，统一用“任务成功率”与“平均步骤数”两项指标评估 → 结果表明 Lemur 在这些跨模态任务上整体领先同类开源模型，验证了语言与代码协同的实际效用。  

4. **开源可复现的完整流水线**  
   *很多高性能模型只提供权重，缺少训练细节* → Lemur 团队在 GitHub 上公开了从数据收集、混合预训练到指令微调的完整脚本，并提供了 Lemur‑Chat 的交互式接口 → 研究者可以直接复现并在此基础上进行二次开发，推动了社区对“语言代理”方向的快速迭代。  

### 方法详解
整体思路可以拆成三大步骤：**数据准备 → 双语料预训练 → 统一指令微调**。下面按顺序展开。

1. **数据准备**  
   - **自然语言语料**：从公开的网页、书籍、对话日志中抽取约 2000 亿 token，覆盖常识、专业知识以及对话式交互。  
   - **代码语料**：爬取 GitHub、StackOverflow、开源项目文档等，得到约 1500 亿 token 的代码和注释，语言覆盖 Python、JavaScript、C++ 等主流语言。  
   - **混合采样**：在每个训练 batch 中，随机抽取 50% 的自然语言样本、50% 的代码样本，确保模型在同一次梯度更新里同时看到两种语言的上下文。  

2. **双语料预训练**  
   - 采用标准的自回归 Transformer 架构（类似 LLaMA），但在 **位置编码** 上加入了轻量的语言标记（NL / CODE），帮助模型在内部区分两类输入。  
   - 目标函数仍是 **最大化下一个 token 的概率**，但因为混合采样，模型必须学会在同一上下文中切换语言风格，例如在代码注释后直接生成代码实现。  
   - 训练期间使用 **梯度累积** 与 **混合精度**，在 8×A100 GPU 上跑约 2 周，得到的基础模型被命名为 **Lemur‑Base**。  

3. **统一指令微调**  
   - 构造 **指令-响应对**，指令分为两类：① 纯文本任务（如“解释这段新闻”），② 代码任务（如“实现二分查找”。）每条指令后紧跟 **任务描述 + 示例输入/输出**。  
   - 微调时仍使用自回归方式，但在 **损失加权** 上对文本指令和代码指令采用相同权重，防止模型偏向某一类。  
   - 为了让模型在对话中主动 **调用工具**，加入了 **Tool‑Use Prompt**：指令中明确要求模型输出 “<tool>函数名 参数</tool>”，微调数据里提供了对应的函数实现示例。  
   - 微调结束后得到 **Lemur‑Chat**，它在对话时能够直接在生成文本的同时输出可执行的代码块或工具调用指令。  

**最巧妙的设计**在于语言标记的轻量化嵌入和指令的统一格式。语言标记让模型在同一层次结构中区分 NL 与 CODE，而不需要额外的分支网络；统一指令格式则让模型在接到任何任务时都走同一条推理路径，避免了“先判断任务类型再切换模型”的两步错误。  

### 实验与效果
- **评测基准**：在自然语言方面使用 **MMLU、TruthfulQA**；在代码方面使用 **HumanEval、MBPP**；在代理任务上使用 **ToolBench、WebShop、MiniWoB**（包括全观测和部分观测两类）。  
- **对比基线**：与同规模的开源模型 LLaMA‑2‑Chat、Mistral‑Instruct、CodeLlama 以及商业闭源模型 GPT‑3.5‑Turbo、Claude‑2 进行横向比较。  
- **主要结果**：  
  - 在 **HumanEval** 上，Lemur‑Chat 获得 48.2% 的通过率，领先 CodeLlama（44.7%）约 3.5% 点。  
  - 在 **MMLU**（5-shot）上，得分 71.3%，比 LLaMA‑2‑Chat（68.9%）提升约 2.4%。  
  - 在 **ToolBench** 任务的成功率上，Lemur‑Chat 达到 63.5%，比 Mistral‑Instruct（55.1%）高出 8.4%。  
  - 综合 **平均性能**（四项基准的等权平均），Lemur‑Chat 超过所有同类开源模型约 4–6% 的绝对提升，逼近 GPT‑3.5‑Turbo 的水平。  

- **消融实验**：  
  - 去掉 **语言标记**，模型在混合预训练阶段出现 “代码语法错误率上升 12%”，说明标记对语言切换有帮助。  
  - 只使用 **文本指令** 进行微调，导致代码任务成功率下降约 5%，验证了统一指令的必要性。  
  - 将 **代码比例** 从 50% 降至 30%，在代码基准上跌 3.8% 点，说明代码密集语料是提升代码能力的关键。  

- **局限性**：论文承认在 **长序列推理**（> 2k token）和 **高度专业化的领域代码**（如硬件驱动）上仍有明显差距；此外，部分观察环境下的记忆保持仍依赖于显式的状态传递，未实现真正的长期记忆。  

### 影响与延伸思考
Lemur 的出现标志着开源社区首次在同一模型里实现了“说得好、写得对、用得巧”的三位一体，随后出现的 **OpenChat‑Code**、**Mistral‑Agent** 等项目都在不同程度上借鉴了其混合预训练和统一指令的思路。对未来的研究而言，**跨模态记忆**（让模型在部分观测环境中保持长期上下文）和 **安全的工具调用**（防止模型误用外部 API）是两个值得深入的方向。想进一步了解，可以关注 **ToolBench** 的后续扩展以及 **RL‑HF（强化学习+人类反馈）** 在语言代理中的应用。  

### 一句话记住它
Lemur 用同一套模型和训练流程把自然语言对话和可执行代码融合，让开源模型第一次真正做到“会说话也会写代码”。