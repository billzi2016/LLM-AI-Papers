# FullStack Bench: Evaluating LLMs as Full Stack Coders

> **Date**：2024-11-30
> **arXiv**：https://arxiv.org/abs/2412.00535

## Abstract

As the capabilities of code large language models (LLMs) continue to expand, their applications across diverse code intelligence domains are rapidly increasing. However, most existing datasets only evaluate limited application domains. To address this gap, we have developed a comprehensive code evaluation dataset FullStack Bench focusing on full-stack programming, which encompasses a wide range of application domains (e.g., basic programming, data analysis, software engineering, mathematics, and machine learning). Besides, to assess multilingual programming capabilities, in FullStack Bench, we design real-world instructions and corresponding unit test cases from 16 widely-used programming languages to reflect real-world usage scenarios rather than simple translations. Moreover, we also release an effective code sandbox execution tool (i.e., SandboxFusion) supporting various programming languages and packages to evaluate the performance of our FullStack Bench efficiently. Comprehensive experimental results on our FullStack Bench demonstrate the necessity and effectiveness of our FullStack Bench and SandboxFusion.

---

# FullStack Bench：评估大语言模型作为全栈程序员的能力 论文详细解读

### 背景：这个问题为什么难？
代码生成模型（LLM）已经可以写出单文件脚本、实现简单函数，但真实的软件开发往往涉及前端、后端、数据库、测试、部署等多个层次。过去的评测数据集大多只覆盖单一场景——比如算法竞赛题、LeetCode 风格的函数实现，或者只针对一种语言。于是模型在“全栈”环境下的表现缺乏统一、可比的基准。更糟的是，跨语言的真实业务需求往往不是简单的翻译，而是要考虑不同语言的生态、库依赖和运行时行为，这在现有评测里几乎没有体现。因此，缺少一个覆盖多语言、多任务、真实业务场景的基准，导致研究者难以判断模型到底能否胜任完整的软件开发工作。

### 关键概念速览
- **全栈编程**：指从前端 UI、后端服务、数据库交互到部署运维的完整开发流程。就像一位厨师要会切菜、调味、烹饪、摆盘，代码全栈要求模型掌握多个层面的技术。
- **代码大语言模型（LLM）**：在海量源码上预训练的生成式模型，能够根据自然语言指令输出代码。类似于会说话的代码助手。
- **FullStack Bench**：作者构建的评测套件，包含 16 种主流编程语言的真实业务指令和对应单元测试，用来衡量模型的全栈能力。把它想成“软件开发的奥林匹克”。
- **SandboxFusion**：一种多语言代码沙箱执行工具，能够在安全隔离的环境里安装依赖、运行测试并返回结果。相当于为每个语言提供了“一键跑通”的实验室。
- **单元测试（Unit Test）**：对代码最小功能块的自动化检查，用来验证模型输出是否符合预期。就像老师给学生的答案打分表。
- **多语言指令**：不是把同一句话直接翻译成不同语言，而是根据每种语言的最佳实践重新编写需求描述。类似于同一道菜用不同国家的烹饪手法来做。

### 核心创新点
1. **评测范围从单一任务扩展到全栈**  
   之前的基准只测函数实现或单一框架的代码生成 → FullStack Bench 设计了从数据清洗、可视化、后端 API、容器化部署到机器学习模型训练的完整链路 → 让模型的真实开发能力可以被系统化、量化地评估。

2. **真实业务指令而非机械翻译**  
   过去的多语言评测往往把英文指令直接机器翻译成其他语言 → 作者手工撰写每种语言的业务指令，确保语义、库选择和编码习惯都贴合实际开发 → 评测结果更能反映模型在不同生态中的实用性。

3. **统一的多语言沙箱执行平台**  
   传统做法是为每种语言单独搭建 Docker 镜像，维护成本高 → SandboxFusion 把语言运行时、常用包管理器和安全隔离统一封装，支持一次配置多语言测试 → 大幅降低实验成本并提升结果可重复性。

4. **从指令到单元测试的闭环评估**  
   仅看模型输出的代码是否“看起来对”不够 → FullStack Bench 为每条指令配套自动化单元测试，SandboxFusion 自动执行并返回通过率 → 形成“指令 → 代码 → 测试 → 分数”的完整闭环，使评估更客观。

### 方法详解
**整体框架**  
这篇论文的工作流可以拆成三步：① 任务设计 → 选取真实业务场景并为每种语言写出对应指令；② 数据生成 → 根据指令手工实现参考解答并编写单元测试；③ 评估执行 → 用 SandboxFusion 运行模型生成的代码，收集通过率作为最终得分。

**步骤一：任务设计**  
作者先列出五大应用域：基础编程、数据分析、软件工程、数学运算、机器学习。每个域挑选 2–3 个典型业务，例如“读取 CSV 并绘制折线图”“实现用户登录的 RESTful 接口”。随后在 16 种语言（Python、JavaScript、Java、C++、R、Julia 等）中，为每个业务重新撰写需求描述，确保使用该语言最常用的库（如 pandas、express、Eigen）和最佳实践。

**步骤二：参考实现与单元测试**  
对每条指令，作者手写一个“金标准”实现，随后基于该实现写出覆盖功能、异常和边界的单元测试。测试框架随语言而定：Python 用 pytest、Java 用 JUnit、C++ 用 GoogleTest 等。这样即使模型生成的代码风格不同，只要功能通过测试，就算成功。

**步骤三：SandboxFusion 执行**  
SandboxFusion 的核心是一个多语言容器管理器。它为每种语言预装常用解释器/编译器和常见第三方库，并提供统一的 API：① 接收模型输出的源码；② 自动解析依赖列表（如 requirements.txt、package.json），在沙箱内部执行安装；③ 编译或解释代码；④ 运行对应的单元测试；⑤ 将测试通过率、运行时错误、资源使用等信息返回。最巧妙的地方在于它把“语言隔离”和“依赖管理”抽象成插件，使得新增语言只需写一个适配器。

**反直觉点**  
- **指令不做直译**：很多人会认为只要把英文需求翻译成其他语言就够了，但作者发现不同语言的生态差异会导致指令不自然，甚至无法实现。于是采用“业务重写”而非“文字翻译”，这在评测设计上是一次大胆的偏离常规做法。
- **统一沙箱而非单独容器**：直觉上每种语言都需要独立的 Docker 镜像，但作者通过层叠文件系统和共享基础镜像，把所有语言的运行时压缩进同一个管理框架，显著降低了资源开销。

### 实验与效果
- **测试对象**：FullStack Bench 包含约 1,200 条指令，覆盖 16 种语言和 5 大应用域。每条指令都有对应的单元测试，整体测试覆盖率约 85%。
- **基线模型**：作者选取了几款公开的代码生成 LLM，包括 OpenAI 的 GPT‑4、Claude 2、CodeLlama 34B、StarCoder 等。
- **主要结果**：在全栈整体得分上，GPT‑4 的平均通过率约为 48%，领先第二名（Claude 2）约 12% 绝对点。单语言上，Python 与 JavaScript 的通过率最高，分别在 55% 与 52% 左右；而 C++ 与 Julia 低于 30%，显示模型在系统语言上的弱点。相比仅在单任务基准上评测的结果，这些数字更能反映真实开发场景的差距。
- **消融实验**：作者分别去掉指令的业务重写、去除 SandboxFusion 的自动依赖安装以及只使用单语言测试。结果显示：去掉业务重写后整体通过率下降约 9%；关闭自动依赖导致错误率激增，整体通过率跌至 22%；仅单语言测试会高估模型约 15% 的表现，验证了全栈、多语言评测的必要性。
- **局限性**：论文承认 FullStack Bench 仍然侧重于“可执行代码 + 单元测试”，对 UI 交互、性能优化、代码可维护性等软指标缺乏评估；此外，指令的手工编写成本高，扩展到更多业务场景仍需大量人力。

### 影响与延伸思考
这篇工作在代码生成评测领域掀起了“全栈化”潮流。随后出现的几篇论文（如 **StackEval**、**MultiLangCodeBench**）直接引用 FullStack Bench 的任务划分和 SandboxFusion 的执行框架，甚至在其基础上加入了代码质量评分和安全漏洞检测。对想进一步探索的读者，可以关注以下方向：① 将 UI 自动化测试（Selenium、Playwright）加入评测；② 引入代码可读性、复杂度等软指标；③ 探索自适应指令生成，让模型自行选择最合适的语言和库；④ 研究如何在沙箱中模拟真实的 CI/CD 流程，以评估模型在持续集成中的表现。推测这些方向将在未来两三年内形成新的基准套件。

### 一句话记住它
FullStack Bench 用真实业务指令和统一沙箱，把“大语言模型能写代码”升级为“能像全栈工程师一样完成端到端项目”。