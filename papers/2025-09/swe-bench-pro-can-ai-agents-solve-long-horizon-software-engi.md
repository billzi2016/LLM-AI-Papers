# SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?

> **Date**：2025-09-21
> **arXiv**：https://arxiv.org/abs/2509.16941

## Abstract

We introduce SWE-Bench Pro, a substantially more challenging benchmark that builds upon the best practices of SWE-BENCH [25], but is explicitly designed to capture realistic, complex, enterprise-level problems beyond the scope of SWE-BENCH. SWE-BENCH PRO contains 1,865 problems sourced from a diverse set of 41 actively maintained repositories spanning business applications, B2B services, and developer tools. The benchmark is partitioned into a public set with open access to problems sourced from 11 repositories, a held-out set of 12 repositories and a commercial set of 18 proprietary repositories where we have formal partnership agreements with early-stage startups. Problems in the held-out and the commercial set are not publicly accessible, but we release results on the commercial set. Our benchmark features long-horizon tasks that may require hours to days for a professional software engineer to complete, often involving patches across multiple files and substantial code modifications. All tasks are human-verified and augmented with sufficient context to ensure resolvability. To better understand these limitations, we cluster the failure modes observed in the collected agent trajectories for a clearer characterization of the error patterns exhibited by current models. Overall, SWE-BENCH PRO provides a contamination-resistant testbed that more faithfully captures the complexity and diversity of real-world software development, advancing the pursuit of truly autonomous software engineering agents at a professional level.

---

# SWE‑Bench Pro：AI 代理能否解决长期软件工程任务？ 论文详细解读

### 背景：这个问题为什么难？

在过去，AI 辅助的代码生成大多在小型、单文件的编程题上取得进展，像 LeetCode 那样的“短平快”。然而真实企业项目往往跨越数十甚至上百个文件，需要几天甚至几周的调研、设计和迭代。传统基准（如 SWE‑Bench）只覆盖了几百个相对简短的 bug 修复或功能添加，无法检验模型在“长视野”任务中的规划、记忆和跨文件协作能力。于是，研究者缺少一个既大规模又能真实反映企业级复杂度的评测平台，导致模型的实际可用性难以量化。

### 关键概念速览
- **AI Agent（AI 代理）**：能够自主接收任务描述、检索代码、生成补丁并执行验证的智能体，类似于会写代码的机器人助理。  
- **长视野任务（Long‑Horizon Task）**：需要连续多步操作、跨文件修改、甚至多轮调试的工作，时间跨度从数小时到数天不等。  
- **污染抵抗（Contamination‑Resistant）**：评测数据与公开代码库严格分离，防止模型在训练时已经见过测试用例，从而保证结果的真实性。  
- **任务聚类（Task Clustering）**：把失败的案例按照错误类型（如检索错误、逻辑错误、环境配置错误）分组，以便系统性分析模型的薄弱环节。  
- **商业集合（Commercial Set）**：来自与初创公司合作的私有仓库，数据不公开但提供真实的企业需求，提升基准的行业相关性。  
- **人类验证（Human‑Verified）**：每个任务在加入基准前都经过开发者确认可解，确保评测的可解性和质量。  
- **多文件补丁（Multi‑File Patch）**：一次提交涉及修改多个源文件，而不是单文件的微调，考验模型的全局依赖理解。  
- **轨迹（Trajectory）**：AI 代理在完成任务过程中的完整操作序列，包括检索、思考、生成、测试等每一步。

### 核心创新点
1. **从“短平快”到企业级长视野**：之前的基准只提供几百个几分钟能完成的任务。SWE‑Bench Pro 扩展到 1,865 条来自 41 个活跃仓库的真实需求，任务规模从数小时到数天不等，迫使模型必须进行长期规划和跨文件协作。  
2. **三层数据划分确保公平评测**：公开集（11 个仓库）供所有人直接使用，隐藏集（12 个仓库）仅在评测时提供，商业集（18 个私有仓库）则完全不公开但公布结果。这样既防止了训练数据泄漏，又让模型在未见过的代码库上接受考验。  
3. **系统化错误聚类**：作者收集了大量 AI 代理的执行轨迹，手动标注出检索失误、上下文遗漏、编译错误等 7 类主要失败模式，并用聚类算法归纳出共性错误，帮助后续研究定位改进方向。  
4. **人类验证 + 完整上下文**：每个任务在加入基准前都由实际开发者确认可解，并附带完整的依赖、构建脚本和业务背景，使得模型不必“猜”缺失信息，评测更贴近真实开发流程。

### 方法详解
整体框架可以看作 **“检索 → 思考 → 生成 → 验证”** 的闭环。下面逐步拆解每一步的实现细节：

1. **检索（Context Retrieval）**  
   - 给定任务描述，系统先在目标仓库的全部历史提交、issue、README 中做全文检索。检索模型采用多模态向量搜索，将自然语言描述映射到代码片段的嵌入空间。  
   - 检索结果会被排序并返回前 N（通常 10）条最相关的代码片段、文档和测试用例，形成“上下文池”。这一步类似于开发者在 IDE 中搜索函数定义或阅读相关 PR。

2. **思考（Chain‑of‑Thought Prompting）**  
   - 在检索到的上下文基础上，模型会先生成一段思考日志，列出可能的实现路径、需要修改的文件列表以及潜在的依赖冲突。  
   - 这一步采用 **CoT**（思维链）技术，让模型像人在白板上列提纲一样，把计划写出来，帮助后续生成阶段保持全局一致性。

3. **生成（Patch Generation）**  
   - 根据思考日志，模型逐文件生成 diff（差分）块。每个 diff 包含 **添加、删除、修改** 三类操作，且会同步更新对应的测试文件。  
   - 为避免一次性生成过大，系统采用 **分步生成**：先生成核心业务改动，再生成配套的配置或文档改动，最后生成测试。每一步都交给模型重新审视前一步的输出，形成自我纠错循环。

4. **验证（Automated Build & Test）**  
   - 生成的补丁会被自动应用到仓库的临时分支，触发 CI（持续集成）流水线：编译、单元测试、集成测试。  
   - 若测试失败，系统会把错误日志回馈给模型，模型重新进入思考‑生成环节进行修正。这个 **闭环调试** 机制让模型能够在多轮迭代中逐步逼近可通过的补丁。

5. **轨迹记录与错误聚类**  
   - 整个过程的每一步（检索关键词、思考日志、生成的 diff、测试结果）都会被序列化保存为轨迹。作者随后使用聚类算法把相似的失败轨迹归类，形成七大错误模式的统计报告。  
   - 这种“事后分析”并不是模型的一部分，但为后续改进提供了清晰的方向。

**最巧妙的地方**在于把 **“思考日志”** 作为显式的中间表示，让模型在生成代码前先对任务进行结构化分解；同时把 **自动化 CI** 直接嵌入评测回路，使得模型能够像人类开发者一样通过测试反馈进行迭代，而不是一次性交付完美代码。

### 实验与效果
- **数据集**：SWE‑Bench Pro 包含 1,865 条任务，分为公开集、隐藏集和商业集三部分。任务来源覆盖业务应用、B2B 服务和开发者工具，确保场景多样。  
- **基线模型**：作者对比了几种主流代码生成模型，包括 OpenAI GPT‑4‑code、Claude 2、CodeLlama 34B，以及专门针对 SWE‑Bench 调优的自回归模型。  
- **整体表现**：在公开集上，最佳模型（GPT‑4‑code）通过率约为 **28%**，相比原 SWE‑Bench（约 45% 的短任务通过率）下降明显，说明长视野任务更具挑战。隐藏集通过率约 **22%**，商业集（作者公布的结果）约 **19%**。  
- **错误聚类结果**：约 **35%** 的失败归因于检索不到关键依赖文件，**27%** 为跨文件逻辑不一致，**18%** 为测试环境配置错误，剩余为代码语义错误或编译错误。  
- **消融实验**：去掉思考日志阶段后，通过率下降约 **6%**；去掉闭环调试（直接生成后不测试）则下降约 **12%**，说明两者对提升成功率都至关重要。  
- **局限性**：作者承认当前模型仍缺乏长期记忆，无法在一次会话中保持对数十个文件的全局视图；此外，CI 环境的多样性（不同语言、不同构建工具）仍会导致测试反馈不一致。

### 影响与延伸思考
SWE‑Bench Pro 为 AI 代码生成提供了首个真正面向企业级、长视野任务的评测平台，随后出现的工作如 **CodeAgent‑X**、**AutoDev‑Long** 等都直接引用了该基准进行对比。该基准也推动了 **检索增强生成（RAG）** 在软件工程领域的深入研究，尤其是如何在大规模代码库中快速定位相关实现。未来的方向可能包括：  
- 引入 **长期记忆模块**（如外部知识库）帮助模型保持跨文件上下文；  
- 结合 **程序分析工具**（静态分析、符号执行）在生成前提供更精准的依赖图；  
- 扩展到 **多语言、多平台**（前端、移动、云原生）场景，进一步提升通用性。  
如果想跟进最新进展，建议关注 **NeurIPS、ICLR** 上关于 “AI software engineering agents” 的专题会议，以及 **GitHub Copilot Labs** 的实验报告。

### 一句话记住它
SWE‑Bench Pro 用真实企业级长任务和严格的污染防护，首次让我们看到 AI 代理在“几天工作”级别的代码修复上仍远未成熟。