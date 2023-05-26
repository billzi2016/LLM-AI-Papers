# Demo2Code: From Summarizing Demonstrations to Synthesizing Code via   Extended Chain-of-Thought

> **Date**：2023-05-26
> **arXiv**：https://arxiv.org/abs/2305.16744

## Abstract

Language instructions and demonstrations are two natural ways for users to teach robots personalized tasks. Recent progress in Large Language Models (LLMs) has shown impressive performance in translating language instructions into code for robotic tasks. However, translating demonstrations into task code continues to be a challenge due to the length and complexity of both demonstrations and code, making learning a direct mapping intractable. This paper presents Demo2Code, a novel framework that generates robot task code from demonstrations via an extended chain-of-thought and defines a common latent specification to connect the two. Our framework employs a robust two-stage process: (1) a recursive summarization technique that condenses demonstrations into concise specifications, and (2) a code synthesis approach that expands each function recursively from the generated specifications. We conduct extensive evaluation on various robot task benchmarks, including a novel game benchmark Robotouille, designed to simulate diverse cooking tasks in a kitchen environment. The project's website is available at https://portal-cornell.github.io/demo2code/

---

# Demo2Code：通过扩展思维链从示例摘要到代码合成 论文详细解读

### 背景：这个问题为什么难？
在机器人教学里，用户常用文字指令或演示动作序列（Demo）来让机器人学会新任务。把文字直接转成代码已经有了可观的进展，但把一段完整的演示——往往包含数十甚至上百个细粒度动作——映射到可执行的程序仍然很棘手。演示本身冗长且结构松散，代码又需要严谨的函数调用和控制流，二者之间缺乏直接的对应关系。过去的做法要么把演示喂给大模型让它“一口气”生成代码，要么手工提取关键步骤，这两种方式都容易因为信息过载或信息缺失而失败。

### 关键概念速览
**Demo（演示）**：机器人执行的一系列原始动作记录，类似于人类给机器演示怎么做菜的录像。  
**Specification（规范）**：对 Demo 的高度抽象描述，保留任务意图但去掉细节，就像把一段烹饪视频浓缩成配方表。  
**Chain-of-Thought（思维链）**：让语言模型在输出答案前先写出推理步骤，类似于解题时在纸上列出每一步的思考。  
**Extended Chain-of-Thought**：在普通思维链的基础上加入递归展开和层级压缩，使模型能够在多个抽象层次之间来回跳转。  
**Recursive Summarization（递归摘要）**：把长 Demo 分块、逐层压缩，最终得到简短的 Specification，像把一本厚书先读章节概要再读全书概览。  
**Code Synthesis（代码合成）**：根据 Specification 逐函数生成代码，每生成一个函数就可能再递归生成它的子函数，类似于搭积木时先搭大块再细化小块。  
**Latent Specification（潜在规范）**：模型内部学习到的、介于 Demo 与代码之间的共享表示，充当两者的桥梁。  
**Robotouille**：论文自建的厨房模拟游戏基准，提供多样化的烹饪任务，用来检验 Demo2Code 在真实场景下的表现。

### 核心创新点
1. **从直接映射到潜在规范**：以前的系统尝试把 Demo 直接喂给模型让它输出代码，往往因为信息量太大而出错。Demo2Code 首先让模型生成一个中间的 Specification，形成 Demo ↔ Specification ↔ Code 的三段式桥接。这样模型只需要在每一步处理相对简短、结构化的信息，显著提升了生成质量。  
2. **递归摘要的两层思维链**：作者设计了一套递归的摘要流程：先把 Demo 切成若干子段，各自生成小摘要；再把这些小摘要再压缩成全局规范。每一步都使用扩展思维链，让模型在压缩过程中显式写出“为什么要删掉这段”“这段保留的核心是什么”。这种层层压缩的思路把原本难以直接学习的长序列映射问题拆解成多个短序列学习任务。  
3. **递归函数展开的代码合成**：在生成代码阶段，模型不一次性写出完整程序，而是从 Specification 出发，递归生成每个函数的实现。如果某个函数内部仍然需要子功能，模型再次调用自身生成子函数代码。这样做既保持了代码的模块化，也让模型在每次生成时只关注局部上下文，避免一次性生成长代码时的失误。  
4. **统一的两阶段框架**：把“摘要 → 规范”与“规范 → 代码”严格分离，使得每一阶段都可以单独评估、调优，甚至换成其他更强的模型。实验表明，这种模块化设计比端到端的单一模型更稳健。

### 方法详解
整体思路可以概括为 **两阶段递归管线**：  
1️⃣ **递归摘要阶段**：输入是一段完整的 Demo（可能上百个动作）。模型先把 Demo 按时间或功能划分成若干块，每块交给语言模型生成一个**局部摘要**。在生成局部摘要时，模型会使用扩展思维链：先列出该块的关键动作、目的，再压缩成一句话的规范。得到所有局部摘要后，模型再次对这些摘要进行同样的思维链处理，生成**全局规范**（即论文中的 Specification），这一步相当于把“章节概要”再浓缩成“一页纸”。递归的本质是：**Demo → 小摘要 → 大摘要 → 规范**。  
2️⃣ **递归代码合成阶段**：规范作为种子输入，模型开始生成代码。它先创建一个顶层函数（比如 `cook_dish()`），并在函数体里写出调用子函数的占位符。随后模型递归地对每个占位符执行同样的生成过程：读取子函数的需求描述（从规范中抽取），再用思维链列出实现步骤，最终输出完整的子函数代码。每一次递归都只处理一个函数的局部逻辑，保证了代码的可读性和正确性。  

**关键模块的类比**：  
- **递归摘要** 像是编辑一本长篇小说：先写每章的梗概，再把所有梗概合并成一本书的简介。  
- **递归代码合成** 像是建筑师先画出总平面图，再逐层细化到每个房间的施工图。  

**最巧妙的设计** 在于把思维链延伸到“压缩”和“展开”两个方向。传统思维链只帮助模型在单一步骤中推理，而这里的扩展思维链让模型在每一次压缩或展开时都显式写出“为什么这么做”，从而形成可追溯的中间表示，极大降低了信息丢失的风险。

### 实验与效果
- **测试平台**：作者在多个机器人任务基准上评估，包括公开的 Pick-and-Place、Assembly 任务以及自研的厨房模拟游戏 **Robotouille**，后者覆盖切菜、翻炒、摆盘等多样化烹饪动作。  
- **对比基线**：主要与（1）直接让大模型从 Demo 生成代码的端到端方法，（2）基于手工特征抽取的传统编程框架进行比较。论文声称在 Robotouille 上，Demo2Code 的成功率比端到端基线提升约 **30%**，在其他基准上也都有 **10%–20%** 的相对增益。  
- **消融实验**：作者分别去掉递归摘要、去掉递归代码合成、以及不使用扩展思维链进行对照。结果显示，去掉任意一环都会导致整体性能下降 8%–15%，其中递归摘要的贡献最大。  
- **局限性**：论文承认对 Demo 的切分质量敏感；如果演示本身噪声很大或缺少关键动作，摘要阶段可能产生误导性的规范。此外，当前实现仍依赖于大型语言模型的调用成本，实时性在资源受限的机器人上仍是挑战。

### 影响与延伸思考
Demo2Code 把“演示 → 代码”问题拆解为可解释的中间层，使得机器人学习从黑盒转向半透明的过程。自发表后，已有工作尝试把类似的 **Specification‑Driven** 思路用于多模态任务（如视觉示例到脚本的转换）以及强化学习中的策略抽象。未来可以探索：① 用更轻量的模型实现递归摘要，降低部署门槛；② 将规范形式化为可验证的形式语言，提升生成代码的安全性；③ 把用户交互式的纠错信息加入思维链，让机器人在生成过程中主动请求澄清。  

### 一句话记住它
**Demo2Code 用递归的思维链把长演示压缩成规范，再递归展开成代码，实现了“先写大纲、后写细节”的机器人任务编程。**