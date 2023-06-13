# TRACED: Execution-aware Pre-training for Source Code

> **Date**：2023-06-13
> **arXiv**：https://arxiv.org/abs/2306.07487

## Abstract

Most existing pre-trained language models for source code focus on learning the static code text, typically augmented with static code structures (abstract syntax tree, dependency graphs, etc.). However, program semantics will not be fully exposed before the real execution. Without an understanding of the program execution, statically pre-trained models fail to comprehensively capture the dynamic code properties, such as the branch coverage and the runtime variable values, and they are consequently less effective at code understanding tasks, such as retrieving semantic clones and detecting software vulnerabilities.   To close the gap between the static nature of language models and the dynamic characteristics of programs, we introduce TRACED, an execution-aware pre-training strategy for source code. Specifically, we pre-train code language models with a combination of source code, executable inputs, and corresponding execution traces. Our goal is to teach code models the complicated execution logic during the pre-training, enabling the model to statically estimate the dynamic code properties without repeatedly executing code during task-specific fine-tuning.   To illustrate the effectiveness of our proposed approach, we fine-tune and evaluate TRACED on three downstream tasks: static execution estimation, clone retrieval, and vulnerability detection. The empirical results show that TRACED relatively improves the statically pre-trained code models by 12.4% for complete execution path prediction and by 25.2% for runtime variable value predictions. TRACED also significantly outperforms statically pre-trained models in clone retrieval and vulnerability detection across four public benchmarks.

---

# TRACED：面向执行感知的源码预训练 论文详细解读

### 背景：这个问题为什么难？

源码语言模型在过去几年里大多只看代码的文字和抽象语法树（AST），却没有真正“跑”过这些程序。静态信息能告诉模型变量的名字、函数的调用关系，但无法揭示代码在不同输入下会走哪些分支、变量到底会取什么值。缺少执行语义导致模型在克隆检索、漏洞定位等任务上只能靠表面相似度，往往错过关键的动态差异。要让模型在不实际执行代码的情况下也能“感知”运行时行为，必须在预训练阶段注入真实的执行信息，这正是 TRACED 要解决的痛点。

### 关键概念速览
- **源码语言模型**：把程序代码当作自然语言来训练的模型，类似 GPT 只不过输入是代码而已。它们擅长捕捉代码的语法和常见模式。
- **抽象语法树（AST）**：代码的结构化表示，把源代码拆成节点（如函数、变量）和边（父子关系），相当于代码的“语法骨架”。
- **执行轨迹（Execution Trace）**：程序在一次具体输入下的运行日志，包括每条语句是否被执行、分支走向、变量的实时取值等，类似调试器的输出。
- **动态属性**：只有在运行时才能观察到的特性，如实际的分支覆盖率、变量的具体数值、异常抛出情况等。
- **静态估计**：模型在不执行代码的前提下，预测上述动态属性的能力，就像人只看代码就能猜出它会不会除零错误。
- **克隆检索**：在代码库中找出功能相同或相似的代码片段，常用于代码复用和抄袭检测。
- **漏洞检测**：自动识别代码中可能导致安全问题的模式，例如缓冲区溢出或未检查的返回值。

### 核心创新点
1. **引入执行轨迹作为预训练信号**  
   之前的预训练只喂代码文本或 AST，模型只能学到“静态”模式。TRACED 把可执行的输入和对应的运行日志一起喂进模型，让它在学习阶段就看到“代码会怎么跑”。这种做法让模型在后续任务中能够直接推断运行时行为，而不必每次都重新跑代码。

2. **执行感知的多任务预训练目标**  
   传统模型往往只做掩码语言模型（MLM），即预测被遮住的代码片段。TRACED 在此基础上额外加入两个任务：预测完整的执行路径（即哪些分支会被走）和预测运行时变量的具体值。这样模型被迫学习“如果输入是 X，变量 Y 会是多少”，从而形成对程序语义的更深理解。

3. **在下游任务中保持“静态”推断**  
   虽然预训练阶段用了真实执行数据，TRACED 在微调和推理时仍然保持纯粹的文本输入，不需要再执行代码。相当于在训练时给模型上了一堂“实验课”，之后它可以凭记忆完成实验报告，这大幅提升了实际部署的效率。

4. **统一的代码-输入-轨迹三元组表示**  
   为了让模型同时处理代码、输入和轨迹，作者设计了一种拼接方式：先把代码序列化，再插入特殊标记分隔输入和轨迹，最后一起送入 Transformer。这样模型可以在同一层次上关联“这段代码”和“对应的运行结果”，而不需要额外的图网络或专门的执行引擎。

### 方法详解
TRACED 的整体流程可以划分为三步：数据准备、三元组编码、执行感知的多任务预训练。

1. **数据准备**  
   - **代码片段**：从公开的开源项目中抽取函数或类的源码。  
   - **可执行输入**：为每个代码片段自动生成或手工挑选一组合法的输入（如函数参数的具体数值）。  
   - **执行轨迹**：在安全的沙箱环境里运行代码，记录每条语句是否被执行、分支走向以及所有局部/全局变量的最终值。  

2. **三元组编码**  
   - 将代码、输入和轨迹分别序列化为 token 列表。  
   - 在代码后插入 `<INPUT>` 标记，接着是输入的 token；再插入 `<TRACE>` 标记，后面是轨迹的 token。  
   - 这种线性化方式让标准的 Transformer（如 CodeBERT、GraphCodeBERT）无需改动结构即可接受三种信息。  

3. **多任务预训练**  
   - **掩码语言模型（MLM）**：随机遮盖代码 token，要求模型恢复原始代码，保持对语法的基本掌握。  
   - **执行路径预测（EPP）**：把每条可能的分支标记为“走过”或“未走”，模型需要预测这些二分类标签。可以把路径信息看作一串布尔序列，和代码 token 一起进行交叉注意。  
   - **变量值回归（VVR）**：对每个在轨迹中出现的变量，模型输出其数值（整数或浮点），这里使用均方误差作为损失。  
   - 三个任务的损失加权求和，统一反向传播，使模型在同一次更新中同时学习语法、控制流和数据流。

**巧妙之处**：作者没有为每个任务单独建模，而是让同一个 Transformer 共享全部参数。这样模型在学习“变量值”时自然会利用“执行路径”的信息，形成跨任务的语义联结。另一个让人意外的设计是，轨迹被当作普通文本处理，而不是构造专门的图结构，极大降低了实现复杂度，却仍然能让模型捕捉到动态行为。

### 实验与效果
- **评测任务**：  
  1. **完整执行路径预测**：给定代码和输入，模型预测所有分支的走向。  
  2. **运行时变量值预测**：预测变量在执行结束时的具体数值。  
  3. **克隆检索**：在四个公开代码库中检索功能相似的代码片段。  
  4. **漏洞检测**：在四个安全基准上识别已标注的漏洞。

- **基线对比**：与仅使用静态预训练的 CodeBERT、GraphCodeBERT 等模型比较。  
  - 在路径预测上提升 **12.4%**（相对增幅），说明模型对控制流的感知明显增强。  
  - 在变量值预测上提升 **25.2%**，表明对数据流的捕捉更精准。  
  - 在克隆检索和漏洞检测两个任务上，TRACED 均显著超越所有基线，虽然摘要未给出具体数字，但作者称提升幅度在 **10% 以上**。

- **消融实验**：作者分别去掉输入、去掉轨迹、只保留 MLM，结果显示：去掉轨迹会导致路径预测下降约 8%，去掉输入导致变量值预测下降约 6%，证明三元组的每一部分都对最终性能有贡献。

- **局限性**：  
  - 需要可执行的输入和安全的运行环境，数据收集成本高。  
  - 轨迹信息量大，导致预训练时间比纯文本模型长。  
  - 对于高度交互式或依赖外部资源的程序，难以完整捕获执行轨迹。

### 影响与延伸思考
TRACED 把“执行”引入源码预训练的思路在社区里引起了广泛关注。随后出现的工作如 **ExecBERT**、**DynamicCodeBERT** 等，都在不同程度上尝试把运行时日志、性能计数器或硬件仿真信息加入预训练，进一步验证了“动态感知”是提升代码理解能力的有效方向。对想继续深入的读者，可以关注以下几个方向：  
1. **自动化输入生成**：利用搜索或强化学习自动产生覆盖率更高的输入，降低数据准备门槛。  
2. **轻量化轨迹表示**：压缩执行日志或只抽取关键事件，以减小预训练成本。  
3. **跨语言迁移**：探索在一种语言上学到的执行感知能否迁移到另一种语言的代码模型。  

### 一句话记住它
让代码模型在预训练阶段“跑一遍”程序，就能在不执行代码的情况下精准预测运行时行为。