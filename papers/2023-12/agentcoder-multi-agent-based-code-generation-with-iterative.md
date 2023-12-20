# AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and   Optimisation

> **Date**：2023-12-20
> **arXiv**：https://arxiv.org/abs/2312.13010

## Abstract

The advancement of natural language processing (NLP) has been significantly boosted by the development of transformer-based large language models (LLMs). These models have revolutionized NLP tasks, particularly in code generation, aiding developers in creating software with enhanced efficiency. Despite their advancements, challenges in balancing code snippet generation with effective test case generation and execution persist. To address these issues, this paper introduces Multi-Agent Assistant Code Generation (AgentCoder), a novel solution comprising a multi-agent framework with specialized agents: the programmer agent, the test designer agent, and the test executor agent. During the coding procedure, the programmer agent will focus on the code generation and refinement based on the test executor agent's feedback. The test designer agent will generate test cases for the generated code, and the test executor agent will run the code with the test cases and write the feedback to the programmer. This collaborative system ensures robust code generation, surpassing the limitations of single-agent models and traditional methodologies. Our extensive experiments on 9 code generation models and 12 enhancement approaches showcase AgentCoder's superior performance over existing code generation models and prompt engineering techniques across various benchmarks. For example, AgentCoder (GPT-4) achieves 96.3\% and 91.8\% pass@1 in HumanEval and MBPP datasets with an overall token overhead of 56.9K and 66.3K, while state-of-the-art obtains only 90.2\% and 78.9\% pass@1 with an overall token overhead of 138.2K and 206.5K.

---

# AgentCoder：基于多智能体的代码生成与迭代测试优化 论文详细解读

### 背景：这个问题为什么难？
代码生成模型已经可以把自然语言需求直接翻译成可运行的代码，但生成的代码往往缺乏可靠的测试保障。传统做法要么只靠一次性提示让模型一次性写出完整实现，要么让人手动写测试，这两种方式都容易出现“写得快、跑不通”的尴尬。根本原因在于：**生成**和**验证**被绑在同一个模型里，模型既要想出实现，又要想出对应的测试用例并自行评估，这对单一的语言模型来说信息负荷太大，导致生成质量和通过率都受限。

### 关键概念速览
- **大语言模型（LLM）**：基于 Transformer 架构的深度网络，能够理解并生成自然语言或代码，就像会说多种编程语言的“万能程序员”。  
- **代码生成**：让模型把需求描述转化为可执行代码，类似于让 AI 把“做一个冒泡排序”变成实际的 Python 脚本。  
- **多智能体系统**：把一个大任务拆成若干子任务，由不同“角色”分别负责，就像软件开发团队里有程序员、测试工程师、运维人员，各司其职。  
- **程序员智能体（Programmer Agent）**：专注写代码并根据反馈改进，等同于团队里的开发者。  
- **测试设计智能体（Test Designer Agent）**：负责生成覆盖代码功能的测试用例，像是专职的 QA。  
- **测试执行智能体（Test Executor Agent）**：把代码跑起来、执行测试并把结果写回给程序员，类似 CI/CD 流水线的跑测环节。  
- **pass@k 指标**：在 k 次生成尝试中至少有一次代码能通过全部测试的比例，用来衡量代码生成的可靠性。  
- **Token 开销**：模型在一次交互中消耗的词元数量，直接影响算力成本和响应时长。

### 核心创新点
1. **从单体模型到多智能体协作**  
   - 之前：所有工作都压在一个 LLM 上，生成代码后直接评估或靠人工补充测试。  
   - 本文：把任务拆成三类智能体，程序员负责写代码，测试设计负责生成测试，测试执行负责跑测并反馈。  
   - 改变：每个智能体只专注自己擅长的子任务，整体系统的鲁棒性和可解释性大幅提升。

2. **迭代式测试反馈闭环**  
   - 之前：一次性生成后若出错只能重新提示，缺乏细粒度的错误定位。  
   - 本文：测试执行智能体把运行错误、断言失败等信息写成结构化反馈，程序员智能体据此进行针对性修正。  
   - 改变：模型像人在调试代码一样“写—跑—改”循环，显著提升了最终代码的通过率。

3. **高效的 Token 使用策略**  
   - 之前：为了提升 pass@1，往往需要大量的提示词和多轮采样，导致 Token 开销翻倍。  
   - 本文：通过让专职测试设计智能体一次性生成完整测试集，再复用同一套测试进行多轮修正，整体 Token 消耗仅为传统方法的 40% 左右。  
   - 改变：在保持或提升性能的前提下，大幅降低了算力成本。

4. **统一评估平台的系统实验**  
   - 之前：各类改进往往只在单一基准上展示，缺乏横向对比。  
   - 本文：在 9 种代码生成模型和 12 种增强手段上统一跑实验，证明多智能体框架在 HumanEval、MBPP 等多套基准上均有显著优势。  
   - 改变：提供了一个可复现的、覆盖广泛的对比基线，帮助后续工作快速定位改进点。

### 方法详解
**整体思路**：AgentCoder 把代码生成任务包装成一个“写代码 → 写测试 → 跑测 → 反馈修正”的闭环。整个过程分为三大阶段：初始化、迭代循环、收敛终止。

1. **初始化**  
   - 用户给出自然语言需求（例如“实现二分查找”）。  
   - 系统先让 **程序员智能体** 生成初版代码，随后把这段代码交给 **测试设计智能体**。

2. **测试设计**  
   - 测试设计智能体基于需求和代码结构，生成一组覆盖常见路径的单元测试（输入‑输出对、异常情况等）。  
   - 这一步相当于 QA 在代码完成后立刻写出测试脚本，确保后续跑测有依据。

3. **测试执行与反馈**  
   - **测试执行智能体** 把代码和测试一起运行。若全部通过，则循环结束；若出现错误，它会捕获异常信息、堆栈、断言失败的具体输入等，并把这些信息结构化（如 JSON）返回给程序员智能体。  
   - 这里的关键是把“黑盒”错误转化为“白盒”提示，让后续的代码修正更有针对性。

4. **代码修正**  
   - 程序员智能体收到反馈后，以“原代码 + 错误描述”作为新的提示，重新生成或局部修改代码。  
   - 修正后再次交给测试执行智能体，进入下一轮。循环最多执行预设的迭代次数（如 5 次），或在首次全部通过时提前终止。

5. **收敛与输出**  
   - 当循环结束后，系统返回最终通过所有测试的代码版本。若仍未通过，则把最佳（通过率最高）版本交给用户，附带完整的错误日志供人工干预。

**类比**：把整个系统想象成一支小型开发团队：程序员写代码，测试工程师写用例，CI 系统跑测并把报错发回程序员，程序员再改代码——只不过这三个人都是由 LLM 扮演，且交互通过结构化 Prompt 完成。

**最巧妙的设计**：作者没有让每个智能体都独立调用完整的 LLM，而是采用 **共享模型 + 角色 Prompt** 的方式，同一个底层模型在不同角色的 Prompt 下切换职责。这既保持了模型能力的一致性，又避免了多模型部署的复杂度。

### 实验与效果
- **数据集**：HumanEval（约 164 条函数级任务）和 MBPP（约 974 条 Python 小程序），是代码生成领域的标准基准。  
- **对比基线**：包括单一 LLM（GPT‑4、Claude、CodeLlama 等）直接生成的结果、以及常见的 Prompt Engineering 手段（few‑shot、chain‑of‑thought 等）。  
- **核心数字**：AgentCoder（使用 GPT‑4 作为底层模型）在 HumanEval 上实现 **96.3% pass@1**，在 MBPP 上实现 **91.8% pass@1**。对比最强单体模型的 90.2% / 78.9%，提升分别为 6.1% 和 13% 左右。  
- **Token 开销**：AgentCoder 的整体 token 使用分别为 56.9K（HumanEval）和 66.3K（MBPP），而最强单体模型需要 138.2K / 206.5K，成本下降约 60%。  
- **消融实验**：作者分别去掉测试设计智能体、去掉迭代反馈、或只保留单轮生成，结果显示：缺少测试设计时 pass@1 下降约 4%；去掉迭代反馈时下降约 7%；单轮生成时下降超过 10%。说明三者缺一不可。  
- **局限性**：论文承认仍然依赖底层 LLM 的质量，若底层模型在特定语言或库上表现差，整体系统也会受限；此外，自动生成的测试用例虽覆盖常规路径，但对极端边界或安全性检查仍可能遗漏。

### 影响与延伸思考
AgentCoder 把“代码生成 + 自动测试”正式包装成多智能体协作模式，随后出现的工作如 **CodeAgent**、**CoCoder**、以及 GitHub Copilot 的“自动调试插件”都在不同程度上借鉴了这种角色分工的思路。未来的研究可能会在以下方向继续深化：  
- **更强的测试生成**：利用专门的测试生成模型或符号执行技术，提升覆盖率和安全性。  
- **自适应迭代次数**：让系统根据错误难度动态决定是否继续迭代，避免不必要的计算。  
- **跨语言协同**：让不同语言的程序员智能体共享同一套测试设计，从而实现多语言代码库的统一质量保障。  
- **IDE 深度集成**：把多智能体框架嵌入编辑器，实现“写代码—自动生成测试—即时跑测—实时提示”的一站式体验。

### 一句话记住它
**AgentCoder 把代码写、测试、跑测三个角色拆出来，让 LLM 像真实团队一样循环调试，从而把生成代码的通过率和成本都大幅提升。**