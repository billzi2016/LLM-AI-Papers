# SWE-bench: Can Language Models Resolve Real-World GitHub Issues?

> **Date**：2023-10-10
> **arXiv**：https://arxiv.org/abs/2310.06770

## Abstract

Language models have outpaced our ability to evaluate them effectively, but for their future development it is essential to study the frontier of their capabilities. We find real-world software engineering to be a rich, sustainable, and challenging testbed for evaluating the next generation of language models. To this end, we introduce SWE-bench, an evaluation framework consisting of $2,294$ software engineering problems drawn from real GitHub issues and corresponding pull requests across $12$ popular Python repositories. Given a codebase along with a description of an issue to be resolved, a language model is tasked with editing the codebase to address the issue. Resolving issues in SWE-bench frequently requires understanding and coordinating changes across multiple functions, classes, and even files simultaneously, calling for models to interact with execution environments, process extremely long contexts and perform complex reasoning that goes far beyond traditional code generation tasks. Our evaluations show that both state-of-the-art proprietary models and our fine-tuned model SWE-Llama can resolve only the simplest issues. The best-performing model, Claude 2, is able to solve a mere $1.96$% of the issues. Advances on SWE-bench represent steps towards LMs that are more practical, intelligent, and autonomous.

---

# SWE-bench：语言模型能解决真实 GitHub 问题吗？ 论文详细解读

### 背景：这个问题为什么难？

在过去，代码生成 benchmark 大多只给模型一个函数或一个小片段，让它补全或写出实现。真实的项目往往跨文件、跨类，甚至需要先运行测试、观察错误再决定改动。传统评测忽略了这些“全局”因素，导致模型在实验室里表现很好，却在实际项目中束手无策。要让模型真正像人类工程师那样阅读整个代码库、定位根因、编辑多个文件并确保不破坏已有功能，这种能力在当时几乎没有系统化的测评手段，也没有公开的数据集来推动研究。

### 关键概念速览
- **SWE-bench**：一个评估框架，收集了 2,294 条真实的 GitHub Issue 与对应的 Pull Request，覆盖 12 个流行的 Python 项目。它把“把代码库改对”当作任务，让模型在完整项目上下文中进行编辑。  
- **Issue‑to‑PR 任务**：给模型一个 Issue 描述和整个代码库，要求模型输出能够合并的代码改动（即 PR），类似人类开发者从需求到提交的完整流程。  
- **长上下文**：项目往往包含数千行代码，模型需要一次性读取或分块处理这些信息，远超常规代码补全的几百字符限制。  
- **执行环境交互**：解决很多 Issue 需要先运行单元测试、观察错误信息或执行脚本，模型必须能够调用外部工具并把结果反馈到推理中。  
- **多文件编辑**：有的 Issue 需要在多个模块之间同步修改，模型必须保持跨文件的一致性，类似在多个文档中同步替换关键词。  
- **SWE‑Llama**：作者基于 LLaMA 系列模型进行微调得到的专用模型，旨在提升在 SWE‑bench 上的表现。  
- **Claude 2**：Anthropic 推出的商业大模型，在本评测中是表现最好的模型，但仍只能解决约 2% 的问题。  

### 核心创新点
1. **从代码片段到完整项目的评测转变**  
   - 之前的评测只让模型处理单函数或单文件的补全任务 → SWE‑bench 把整个项目当作输入，要求模型在全局视角下定位并修复缺陷 → 这让评测更贴近真实软件工程的复杂度，暴露出模型在长上下文和跨文件协同上的短板。  

2. **真实 Issue‑PR 对齐的标注方式**  
   - 传统数据集往往人工编写或合成错误 → SWE‑bench 直接采集开发者在 GitHub 上提交的 Issue 与最终合并的 PR，保证每条样例都是“真实需求 → 真实解决方案” → 评测结果更具可信度，也为后续的监督学习提供了高质量的对齐数据。  

3. **统一的评估协议与自动化判分**  
   - 过去不同工作使用各自的脚本，难以直接比较 → 作者设计了一套自动化的评测流水线：先在干净的代码库上应用模型生成的补丁，再运行项目自带的测试套件，只有全部通过且行为与官方 PR 一致才算成功 → 这种严格的判分方式让成绩更具可重复性，也推动了社区对“可执行代码生成”标准的共识。  

4. **公开基准与可复现的基线**  
   - 之前缺少大规模、可公开获取的真实项目基准 → SWE‑bench 完全开源，提供数据、评测脚本以及几种主流模型的基线结果 → 研究者可以直接在此基准上进行微调、提示工程或新模型的实验，降低了进入门槛。  

### 方法详解
整体思路可以拆成三大步骤：**数据准备 → 任务建模 → 自动评测**。

1. **数据准备**  
   - 从 12 个活跃的 Python 仓库中抽取所有已关闭的 Issue，筛选出那些已经对应了 Pull Request 并成功合并的案例。  
   - 每条样例保留：Issue 标题、描述、相关的讨论（如果有）、对应的代码提交前的快照、以及 PR 中的实际代码改动（diff）。  
   - 为了让模型能够“看到”完整项目，作者把每个仓库的最新主分支作为基准代码库，并在评测时把 Issue 所在的历史版本回滚到对应的提交点。  

2. **任务建模**  
   - 输入：`<repo_root>`（整个项目文件树的文本化表示）+ `<<ISSUE>>`（Issue 标题+描述+关键评论）。  
   - 输出：模型需要生成一个合法的 Git diff，覆盖所有被修改的文件。这里没有采用逐文件的交互，而是让模型一次性输出完整的补丁，模拟真实开发者一次性提交 PR 的情形。  
   - 为了让模型能够利用执行信息，评测框架在模型生成前后分别运行项目的测试套件。若测试失败，框架会把错误日志作为额外的“观察”信息喂回模型，形成一次简易的“思考—行动—反馈”循环。  

3. **自动评测流水线**  
   - **Patch 应用**：把模型输出的 diff 应用到基准代码库，若出现冲突直接判为失败。  
   - **测试执行**：运行项目自带的 `pytest`（或其他 CI 脚本），记录是否全部通过。  
   - **行为对齐**：对比模型生成的改动与官方 PR 的 diff，要求两者在功能上等价（即相同的函数/类被修改，且改动的语义相近）。只有同时满足测试通过和行为对齐两条才算成功。  

**最巧妙的地方**在于把“运行测试并把错误信息反馈给模型”这一环节嵌入评测，而不是让模型自行决定是否需要调试。这样既保持了评测的自动化，又让模型有机会展示“调试思维”。  

### 实验与效果
- **数据规模**：2,294 条 Issue‑PR 对，覆盖 12 个流行的 Python 项目（如 `pandas`, `requests` 等），每个项目的代码行数从几千到上万不等。  
- **基线模型**：包括 OpenAI 的 GPT‑4、Claude 2、Claude 1.3、Google Gemini、Meta LLaMA‑2 系列，以及作者自行微调的 SWE‑Llama。  
- **主要结果**：Claude 2 在全部基线中表现最佳，但仅解决了 1.96%（约 45 条）的问题；GPT‑4、Gemini 等商业模型的成功率更低，均在 1% 以下。SWE‑Llama 虽然经过专门微调，但仍只能解决约 0.8% 的 Issue。  
- **消融实验**：作者分别去掉了“执行环境反馈”和“长上下文截断”两项功能，发现去掉反馈后成功率下降约 30%，去掉长上下文处理则几乎所有跨文件的 Issue 都无法解决，验证了这两块是当前模型瓶颈的关键因素。  
- **局限性**：评测仅限于 Python 项目，且所有 Issue 都已经有官方解决方案，可能偏向于已有的最佳实践；此外，模型只能输出一次性 diff，缺少迭代式的交互调试，这在真实开发中是常见的工作方式。  

### 影响与延伸思考
SWE‑bench 的发布让业界第一次拥有了“大规模、真实、可执行”的软件工程评测平台。随后出现的工作如 **CodeAct**、**EvalPlus** 等，都在不同程度上借鉴了其“Issue‑to‑PR”思路，尝试加入多轮对话或更细粒度的错误定位。对想进一步探索的读者，可以关注以下方向：  
- **长上下文模型**：如 Transformer‑XL、Retriever‑augmented Generation（RAG）在代码库检索中的应用。  
- **自动调试循环**：让模型在执行错误后主动生成修复补丁，而不是一次性输出完整 diff。  
- **跨语言扩展**：把基准搬到 Java、JavaScript 等生态，检验模型的语言通用性。  
- **人机协同**：研究如何让模型在编辑代码时接受开发者的即时指令，实现“AI 助手”而非全自动。  

### 一句话记住它
SWE‑bench 把真实的 GitHub Issue 当成考场，让语言模型必须在完整项目、长上下文和实际运行结果的三重约束下“写代码”，结果显示即便是最强商业模型也只能解决不到 2% 的问题。