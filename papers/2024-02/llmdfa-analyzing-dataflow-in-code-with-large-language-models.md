# LLMDFA: Analyzing Dataflow in Code with Large Language Models

> **Date**：2024-02-16
> **arXiv**：https://arxiv.org/abs/2402.10754

## Abstract

Dataflow analysis is a fundamental code analysis technique that identifies dependencies between program values. Traditional approaches typically necessitate successful compilation and expert customization, hindering their applicability and usability for analyzing uncompilable programs with evolving analysis needs in real-world scenarios. This paper presents LLMDFA, an LLM-powered compilation-free and customizable dataflow analysis framework. To address hallucinations for reliable results, we decompose the problem into several subtasks and introduce a series of novel strategies. Specifically, we leverage LLMs to synthesize code that outsources delicate reasoning to external expert tools, such as using a parsing library to extract program values of interest and invoking an automated theorem prover to validate path feasibility. Additionally, we adopt a few-shot chain-of-thought prompting to summarize dataflow facts in individual functions, aligning the LLMs with the program semantics of small code snippets to mitigate hallucinations. We evaluate LLMDFA on synthetic programs to detect three representative types of bugs and on real-world Android applications for customized bug detection. On average, LLMDFA achieves 87.10% precision and 80.77% recall, surpassing existing techniques with F1 score improvements of up to 0.35. We have open-sourced LLMDFA at https://github.com/chengpeng-wang/LLMDFA.

---

# LLMDFA：利用大语言模型分析代码数据流 论文详细解读

### 背景：这个问题为什么难？
传统的数据流分析依赖编译器把代码编译成中间表示，然后在此基础上跑固定的分析算法。要想分析未通过编译的代码、脚本语言或经常改动的项目，就必须先让编译器成功，这在实际工程中常常碰壁。即使编译成功，分析器也需要专家手工配置规则，才能捕捉特定的安全或性能问题。于是，面对快速迭代的代码库和多样化的语言特性，传统方法的“必须编译+手工定制”两大门槛让它们的可用性大打折扣。

### 关键概念速览
- **数据流分析**：追踪程序中值的产生、传播和使用，类似于在代码里放置“水流感应器”，帮助发现未初始化使用、资源泄漏等缺陷。  
- **大语言模型（LLM）**：像 ChatGPT 那样的深度学习模型，能够理解自然语言和代码，并生成符合语义的文本或代码片段。  
- **幻觉（Hallucination）**：LLM 在缺乏足够约束时会捏造不存在的事实，导致分析结果不可靠。  
- **链式思考（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先写出推理步骤，类似于解题时把思路写在草稿纸上，能显著降低幻觉。  
- **Few‑shot Prompting**：在提示中提供少量示例，让模型学习任务模式，类似于老师给学生几道例题后再让他们独立作答。  
- **外部专家工具**：专门的解析库、自动定理证明器等成熟软件，负责完成模型不擅长的精确计算或符号推理。  
- **编译‑自由分析**：不依赖编译器生成中间表示，而直接在源码层面进行分析，像是用放大镜直接检查原始代码。  
- **可定制化**：用户可以自行定义想要追踪的变量或路径，而不必受限于固定的分析规则。

### 核心创新点
1. **编译‑自由的数据流框架 → 让 LLM 直接在源码上工作**  
   传统方法必须先编译，LLMDFA 把这一环节省掉，直接把源码喂给 LLM。这样即使代码缺少依赖、语法错误或使用了非常规语言特性，仍然可以进行分析，极大提升了适用范围。

2. **任务分解 + 外部工具外包 → 把“细致推理”交给专业软件**  
   为了抑制幻觉，作者把整体分析拆成若干子任务：提取变量、判断路径可行性、汇总数据流事实等。每个子任务要么交给 LLM 生成调用外部工具的代码（比如用 `pycparser` 抽取 AST），要么直接让专用工具完成。相当于让模型只负责“指挥”，把“精确计算”交给“专家”，从而提升可靠性。

3. **Few‑shot 链式思考提示 → 把模型对齐到小函数的语义**  
   在每个函数内部，LLMDFA 用 few‑shot CoT 提示让模型先列出变量的定义、使用、传播路径，再输出总结。这样模型的注意力被限制在局部语义上，幻觉概率显著下降，尤其在处理短代码片段时效果尤佳。

4. **统一的结果聚合层 → 自动把子任务的输出合成为完整的数据流图**  
   所有子任务的输出（变量列表、可行路径集合、定理证明结果）被统一送入一个聚合模块，生成跨函数的数据流依赖图。相比传统工具需要手动编写规则或后处理脚本，LLMDFA 的聚合是自动化的、可插拔的。

### 方法详解
**整体框架**  
LLMDFA 的工作流可以概括为四大步骤：  
1) **源码预处理**：把待分析的文件读入，按函数划分。  
2) **子任务生成**：对每个函数，使用 few‑shot CoT 提示让 LLM 生成三段代码：① 用解析库抽取感兴趣的变量；② 调用自动定理证明器检查特定控制流路径是否可达；③ 汇总本函数的 dataflow 事实。  
3) **外部工具执行**：把 LLM 生成的代码实际运行，得到精准的变量列表和路径可行性证明。  
4) **全局聚合与报告**：把所有函数的局部结果合并，构建跨函数的数据流图，并根据用户自定义的检测规则输出潜在 bug。

**关键模块拆解**  
- **提示设计**：提示中先给出两三个手写的函数示例，展示如何从源码到变量抽取再到路径验证的完整链条。随后加入 “请先列出所有变量的定义和使用顺序，再判断以下路径是否可行” 的指令，强制模型走 CoT 流程。  
- **代码合成**：LLM 输出的代码实际上是 Python 脚本，内部使用 `ast`、`pycparser` 等库把源码转成抽象语法树（AST），再遍历 AST 收集变量信息。  
- **路径可行性验证**：对于条件分支，LLMDFA 把分支条件转成布尔公式，交给 Z3（一个开源定理证明器）求解。如果 Z3 给出 SAT（可满足）则该路径被视为真实可达。  
- **结果聚合**：每个函数返回的结构类似 `{var: [def_line, use_lines], reachable_paths: [...]}`，聚合层把这些结构映射到全局的依赖图中，并根据用户提供的 “bug 模式” （如未初始化使用、资源泄漏）进行匹配。

**最巧妙的地方**  
- **把 LLM 当指挥官**：模型不直接给出最终分析结果，而是生成可执行的“指令脚本”。这让模型的创造力与外部工具的严谨性相结合，极大降低幻觉。  
- **局部 CoT 对齐**：把思考范围限制在单函数内部，使得模型只需要记住少量局部信息，避免了跨函数记忆负担导致的错误。  

### 实验与效果
- **测试对象**：作者在两类数据集上评估：① 合成的微型程序，分别植入三种常见 bug（未初始化使用、空指针解引用、资源未释放）；② 真正的 Android 应用，针对开发者自定义的安全检测规则。  
- **对比基线**：包括传统的编译后静态分析工具（如 FlowDroid、Infer）以及最近的 LLM‑驱动分析方案。  
- **核心指标**：LLMDFA 在所有实验中平均 **精度 87.10%**、**召回 80.77%**，相较于最强基线的 F1 提升 **0.35**（约 10% 相对增幅）。  
- **消融实验**：去掉外部工具外包后，召回跌至约 65%；不使用 few‑shot CoT 提示则精度下降约 8%。这些结果表明两大技巧对整体性能贡献显著。  
- **局限性**：作者承认 LLMDFA 仍然依赖 LLM 的生成质量，极端长函数或高度动态语言（如 JavaScript）会导致提示失效；此外，调用外部工具会带来额外的运行时开销，分析大型项目时仍需优化。

### 影响与延伸思考
LLMDFA 的开源实现让社区可以直接在自己的代码库上尝试“编译‑自由”数据流分析，已经催生了几篇后续工作：  
- **LLM‑augmented taint analysis**：把污点追踪交给 LLM 生成的抽象解释器。  
- **IDE 插件原型**：实时在编辑器中给出数据流警告，利用 LLMDFA 的轻量化特性。  
- **跨语言统一分析**：研究者尝试把同一套提示迁移到 Python、Kotlin、Rust 等语言，验证提示的通用性。  
如果想进一步深入，可以关注两条主线：① 更高效的提示工程（如何在更少示例下让模型保持低幻觉），② 将 LLM 与符号执行、抽象解释等传统技术深度融合的理论框架。

### 一句话记住它
LLMDFA 把大语言模型当指挥官，生成调用外部工具的代码，再用 few‑shot 链式思考把局部数据流事实拼成全局图，实现了无需编译、可定制的高精度代码数据流分析。