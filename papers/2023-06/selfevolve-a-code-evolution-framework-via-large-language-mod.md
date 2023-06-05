# SelfEvolve: A Code Evolution Framework via Large Language Models

> **Date**：2023-06-05
> **arXiv**：https://arxiv.org/abs/2306.02907

## Abstract

Large language models (LLMs) have already revolutionized code generation, after being pretrained on publicly available code data. However, while various methods have been proposed to augment LLMs with retrieved knowledge and enhance the quality of code generation, the performance of these retrieval-based methods is limited by the strength of the retrievers used. In addition, while LLMs show great emergent ability, they still struggle to produce the correct code in one turn. To address these challenges, we propose a novel two-step pipeline, called \autoknow, that leverages LLMs as both knowledge providers and self-reflective programmers. Unlike retrieval-based methods, \autoknow~obtains the knowledge from input prompts and generates intermediate code based on the generated knowledge. After that, \autoknow~asks LLM to act as an expert programmer to perform debugging for the generated code. This is achieved by receiving the error message from the interpreter, without requiring special test cases for correctness verification. We evaluate \autoknow~on three code generation datasets, including DS-1000 for data science code, HumanEval for software engineering code, and TransCoder for C++-to-Python translation. Our empirical experiments show that \autoknow~outperforms strong baselines by a significant margin on all datasets. We also conduct exhaustive analytical experiments to validate the effectiveness of the two stages of \autoknow, and find that both are superior to other prompting-based methods. Further scalability analysis demonstrates that \autoknow~can be adapted to other more advanced models, such as GPT-4, and bring consistent efficacy improvement.

---

# SelfEvolve：基于大语言模型的代码进化框架 论文详细解读

### 背景：这个问题为什么难？

代码生成模型已经能写出不少可运行的程序，但它们仍常常“一次出错”。传统的检索增强方法把外部代码片段拉进来，却受限于检索器的质量，检索不到最匹配的例子时生成质量会大跌。另一方面，LLM 本身虽然具备强大的推理能力，却缺少自我纠错的循环：模型给出答案后，除非人为检查，否则错误会直接被接受。于是，如何让模型在不依赖外部检索、也不需要人工编写测试用例的情况下自行发现并修正错误，成为了迫切的挑战。

### 关键概念速览
- **大语言模型（LLM）**：在海量代码和自然语言上预训练的模型，能够把需求描述直接转化为代码，就像会写程序的“智能助理”。  
- **检索增强（Retrieval‑Augmented）**：先在数据库里找相似的代码片段，再把它们喂给模型，类似于“先找参考答案再写”。  
- **自我反思（Self‑Reflection）**：模型在生成代码后，读取运行时错误信息，再次调用自身进行调试，像程序员先写代码、跑出报错、再改代码的过程。  
- **两步管线（Two‑step Pipeline）**：本论文的核心流程，第一步让模型生成“知识”并据此写代码，第二步让模型充当调试专家，依据解释器返回的错误信息进行修正。  
- **解释器反馈（Interpreter Feedback）**：运行代码时产生的错误信息或异常堆栈，直接提供给模型作为调试线索，无需额外的单元测试。  
- **Prompt（提示）**：向模型发送的文字指令，决定模型的行为方式。这里的 Prompt 被精心设计成让模型先“思考”再“写代码”。  

### 核心创新点
1. **检索替代 → 让模型自行生成知识**：传统方法依赖外部检索器找相似代码，而 SelfEvolve 直接在 Prompt 中让 LLM 根据题目描述生成所需的概念、API 用法等“隐式知识”。这样省去了检索器的瓶颈，也让模型的内部知识得到充分利用。  
2. **一次生成 → 两轮自我调试**：以前的生成往往一次性交付，错误率高。SelfEvolve 在第一轮生成代码后，把解释器的错误信息喂回模型，让它以“专家程序员”的身份重新审视代码并输出修正版。相当于让模型先写草稿、再自行改稿。  
3. **无需人工测试 → 直接使用错误信息**：大多数调试框架需要手写测试用例来判断正确性。这里直接利用解释器抛出的异常信息，模型只要能把报错转化为可行的修改建议，就算成功。降低了评估成本，也更贴近真实开发流程。  
4. **模型无缝升级**：实验表明，只要把底层 LLM 换成更强的 GPT‑4，整个管线几乎不需要改动，仍然能保持或提升效果，展示了方法的模型无关性和可扩展性。

### 方法详解
**整体思路**：SelfEvolve 把代码生成拆成两段对话。第一段让模型从需求描述中“抽取知识”，第二段让模型把这些知识写成代码并在出现错误时自行调试。整个过程只需要两次调用 LLM，且每次调用都配合专门设计的 Prompt。

**第一步：知识生成**  
1. 输入：用户的自然语言需求（例如“读取 CSV 并绘制散点图”）。  
2. Prompt 设计：系统提示模型“先思考需要哪些库、函数、参数”，随后让模型列出关键概念和 API 用法。  
3. 输出：一段结构化的“知识清单”，比如“使用 pandas 读取 CSV，使用 matplotlib 的 scatter 绘图”。  
这一步相当于让模型先写“解题思路”，而不是直接跳到代码。

**第二步：代码生成 + 自我调试**  
1. 把第一步的知识清单拼进新的 Prompt，指示模型“基于这些知识写完整代码”。  
2. 运行生成的代码，捕获解释器返回的错误信息（如 SyntaxError、NameError、RuntimeError）。  
3. 将错误信息连同原始需求再次喂给模型，Prompt 中明确要求模型“扮演调试专家”，指出错误根源并给出修改后的代码。  
4. 若仍有错误，可循环第 2–3 步，直到解释器不再报错或达到预设的调试次数上限。  

**关键细节**  
- **错误信息的利用**：模型并不是把错误当成普通文本，而是被提示把错误定位为“变量未定义”“函数参数不匹配”等具体问题，从而更精准地生成补丁。  
- **Prompt 中的角色设定**：通过让模型自称“专家程序员”，激活其内部的“调试知识库”，这在实验中显著提升了修复成功率。  
- **无需显式测试**：因为每一次调试都基于解释器的真实报错，模型的输出天然满足了最基本的可运行性要求。  

**最巧妙的地方**：把解释器的错误信息直接当作“第二语言输入”，让 LLM 在同一次对话中完成“写代码—报错—改代码”的闭环，而不需要外部的检索或手写测试。

### 实验与效果
- **数据集**：DS‑1000（数据科学脚本）、HumanEval（通用编程题）和 TransCoder（C++→Python 翻译）。  
- **对比基线**：包括传统检索增强的 Codex‑Retriever、直接一次性生成的 CodeLlama、以及最新的 ChatGPT‑3.5。  
- **整体提升**：论文声称在所有三个数据集上均实现了显著的性能提升，尤其在 HumanEval 上的通过率提升了数个百分点，超过最强基线数倍。  
- **消融实验**：去掉知识生成阶段或去掉自我调试阶段后，性能均出现明显下降，说明两者缺一不可。  
- **可扩展性**：把底层模型换成 GPT‑4，整体提升仍然保持，验证了方法对更强模型的兼容性。  
- **局限**：作者指出当前实现仍依赖解释器能够给出明确错误信息，对于逻辑错误或性能瓶颈（如时间复杂度高）仍无能为力；此外，多轮调试的成本随错误次数线性增长。

### 影响与延伸思考
SelfEvolve 把“生成—调试”闭环搬进 LLM 交互，开启了“自进化代码模型”的新方向。随后的工作开始探索更丰富的反馈信号（如代码覆盖率、性能剖析）以及多模态调试（把错误堆栈可视化后再喂模型）。如果想进一步了解，可以关注以下方向：① 基于强化学习的自我调试策略；② 将单元测试生成与解释器反馈结合的混合评估；③ 大模型在跨语言迁移（如 Java→Go）中的自我纠错能力。整体来看，这篇论文为让模型像真实程序员一样“写完代码再改”提供了可行的框架。

### 一句话记住它
让大语言模型先“想思路”，再用解释器报错自行“调试”，实现了无需检索、无需手写测试的代码自我进化。