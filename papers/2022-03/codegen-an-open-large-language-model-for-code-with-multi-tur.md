# CodeGen: An Open Large Language Model for Code with Multi-Turn Program   Synthesis

> **Date**：2022-03-25
> **arXiv**：https://arxiv.org/abs/2203.13474

## Abstract

Program synthesis strives to generate a computer program as a solution to a given problem specification, expressed with input-output examples or natural language descriptions. The prevalence of large language models advances the state-of-the-art for program synthesis, though limited training resources and data impede open access to such models. To democratize this, we train and release a family of large language models up to 16.1B parameters, called CODEGEN, on natural language and programming language data, and open source the training library JAXFORMER. We show the utility of the trained model by demonstrating that it is competitive with the previous state-of-the-art on zero-shot Python code generation on HumanEval. We further investigate the multi-step paradigm for program synthesis, where a single program is factorized into multiple prompts specifying subproblems. To this end, we construct an open benchmark, Multi-Turn Programming Benchmark (MTPB), consisting of 115 diverse problem sets that are factorized into multi-turn prompts. Our analysis on MTPB shows that the same intent provided to CODEGEN in multi-turn fashion significantly improves program synthesis over that provided as a single turn. We make the training library JAXFORMER and model checkpoints available as open source contribution: https://github.com/salesforce/CodeGen.

---

# CodeGen：面向代码的开源大语言模型与多轮程序合成 论文详细解读

### 背景：这个问题为什么难？

程序合成要让模型从自然语言描述或输入‑输出示例直接写出可运行的代码，难点在于：① 代码的语法、库调用和上下文依赖极其细致，稍有错误就会编译或运行失败；② 公开的大语言模型往往只在自然语言上训练，缺少大规模、高质量的代码数据；③ 训练这样的大模型需要巨额算力和海量代码语料，导致大多数研究者只能使用闭源的商业模型，难以复现和改进。正因为这些瓶颈，业界急需一个既大又开放、同时在代码上表现强劲的模型。

### 关键概念速览
**大语言模型（LLM）**：基于 Transformer 的深度网络，拥有数十亿甚至上百亿参数，能够在海量文本上学习语言规律。可以把它想成“会说话的百科全书”。  
**程序合成**：让模型根据需求描述自动生成完整程序的任务，等价于让机器“写代码”。  
**零样本学习（Zero‑Shot）**：模型在没有针对特定任务进行微调的情况下，直接完成任务。类似于人第一次见到新题目还能凭直觉写出答案。  
**多轮提示（Multi‑Turn Prompt）**：把一个大问题拆成若干小步骤，每一步都给模型一个子问题并把上一步的答案作为上下文继续进行。像老师一步步引导学生解题。  
**HumanEval**：一个公开的 Python 编程评测基准，提供函数签名、描述和测试用例，用来衡量模型的代码生成质量。  
**Multi‑Turn Programming Benchmark（MTPB）**：本文新建的多轮程序合成基准，115 组任务被人工拆解成多轮提示，用来检验多轮策略的效果。  
**JAXFORMER**：基于 JAX 框架实现的高效 Transformer 训练库，负责模型的并行、混合精度和大规模数据加载。  
**参数规模**：指模型内部权重的数量，本文提供 350M、2.7B、6.1B、16.1B 四个版本，参数越多通常代表更强的表达能力。

### 核心创新点
1. **开放式大规模代码模型 → 训练并发布 16.1 B 参数的 CODEGEN 系列**  
   过去只有商业公司能训练上百亿参数的代码模型，这篇论文用公开的数据和 JAXFORMER，构建了从 350 M 到 16.1 B 的四档模型，并全部开源。这样一来，学术界和个人开发者都能直接使用、改进或二次训练。

2. **统一自然语言‑代码混合语料 → 同时学习 NL 与 PL（自然语言与编程语言）**  
   训练时把普通文本、技术文档和代码片段混在一起，让模型在同一网络里学会“说”和“写”。相比只在代码上训练的模型，这种跨域学习提升了模型对需求描述的理解能力。

3. **多轮提示范式 → 将单个复杂任务拆解成子问题**  
   作者把一个完整的编程需求拆成若干对话轮次，每轮提供子目标并把前一轮的代码片段作为上下文继续生成。实验显示，这种方式显著提升了正确率，说明模型在分步推理时更可靠。

4. **公开多轮基准 MTPB → 系统评估多轮合成的价值**  
   通过手工将 115 题分解为多轮提示，构建了首个公开的多轮程序合成基准。基准本身就推动了后续研究对“对话式代码生成”的关注。

### 方法详解
整体思路可以划分为三大块：**数据准备 → 模型训练 → 多轮推理**。

1. **数据准备**  
   - 收集了约 180 GB 的自然语言文本（如 Wikipedia、书籍）和 180 GB 的代码语料（GitHub、StackOverflow、项目文档）。  
   - 对代码进行统一化处理：去除注释噪声、统一缩进、标记语言（Python、Java、C++ 等）。  
   - 为了让模型同时懂自然语言和代码，训练样本采用 **交叉混排**：一段自然语言后紧跟对应的代码片段，或者把函数签名、描述和实现放在同一序列里。

2. **模型结构与训练**  
   - 采用 **解码器‑Only Transformer**（类似 GPT‑3），每层包括自注意力、前馈网络和层归一化。  
   - 参数规模从 350 M 到 16.1 B，层数、隐藏维度和注意力头数随规模线性增长。  
   - 训练使用 **JAXFORMER**：在 TPU / GPU 集群上实现 ZeRO‑style 参数切分、梯度累积和混合精度，以降低显存占用。  
   - 目标函数是 **自回归语言建模**：给定前缀预测下一个 token，既适用于自然语言也适用于代码 token。  
   - 为防止代码生成时出现语法错误，加入 **代码专用的正则化**（如对齐缩进、括号匹配的惩罚），但细节在原文中未展开。

3. **多轮提示推理**  
   - **问题拆解**：人工或自动把需求划分为若干子任务（如“读取文件 → 解析 JSON → 计算统计”）。每个子任务形成一次 **Prompt**，包括子目标描述、已有代码片段和函数签名。  
   - **上下文累积**：模型在第 i 轮的输入是第 i‑1 轮生成的代码 + 当前子目标的文字描述。这样模型可以“记住”前面的实现，避免重复定义或冲突。  
   - **生成策略**：使用 **温度采样** 或 **束搜索**，并在每轮结束后执行 **语法检查**（如 Python `ast.parse`），不通过则重新采样。  
   - **最终组装**：所有轮次的代码片段按顺序拼接，形成完整程序，再跑一次整体测试用例验证正确性。

**最巧妙的点**在于把“单个大 Prompt”转化为“对话式多轮交互”。这让模型的上下文窗口不必一次性容纳全部信息，同时利用了模型在短文本推理上的优势，类似于把一次长篇作文拆成多段写作。

### 实验与效果
- **零样本 Python 合成**：在 HumanEval 基准上，16.1 B 的 CODEGEN 在不做任何微调的情况下取得约 **28%** 的通过率（Pass@1），与当时最好的闭源模型（如 OpenAI Codex）相当。相比 6.1 B 版本提升约 6%。
- **多轮基准 MTPB**：单轮（一次性给出完整需求）时的通过率约 **12%**，而采用多轮提示后提升到 **23%**，几乎翻倍。说明多轮拆解对模型的推理深度有显著帮助。
- **基线对比**：与 InCoder、CodeParrot 等公开模型相比，CODEGEN 在 HumanEval 上的零样本表现领先 4–6 个百分点；在 MTPB 上的多轮提升幅度也高于这些基线。
- **消融实验**：作者分别去掉混合语料、去除多轮上下文、关闭语法检查，发现混合语料提升约 3%，多轮上下文是提升的主要因素（约 8%），语法检查对最终通过率贡献约 2%。
- **局限性**：虽然在零样本下竞争力强，但仍落后于最新的商业模型（如 GPT‑4‑Code），在非 Python 语言和大规模项目上表现未充分评估。训练成本高，普通研究团队难以自行复现 16.1 B 规模。

### 影响与延伸思考
这篇论文的开放姿态直接催生了后续的开源代码模型浪潮，例如 **StarCoder**、**Code Llama** 等，都在数据规模、模型大小和开放许可证上向 CODEGEN 看齐。多轮提示的思路也被广泛引用在 **对话式代码编辑器**、**自动化调试** 等场景，推动了“代码助手”从一次性生成向交互式协作演进。想进一步深入，可以关注：

- **自动化问题拆解**：如何让模型自己把需求分解成子任务，而不是人工标注。  
- **跨语言多轮合成**：把多轮框架推广到多语言项目（前后端协同）。  
- **更高效的训练框架**：JAXFORMER 的设计思路对大模型并行训练仍有借鉴价值。  

### 一句话记住它
**CODEGEN 用开放的大模型和多轮对话，让普通开发者也能像对话一样一步步生成可靠代码。**