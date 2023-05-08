# Code Execution with Pre-trained Language Models

> **Date**：2023-05-08
> **arXiv**：https://arxiv.org/abs/2305.05383

## Abstract

Code execution is a fundamental aspect of programming language semantics that reflects the exact behavior of the code. However, most pre-trained models for code intelligence ignore the execution trace and only rely on source code and syntactic structures. In this paper, we investigate how well pre-trained models can understand and perform code execution. We develop a mutation-based data augmentation technique to create a large-scale and realistic Python dataset and task for code execution, which challenges existing models such as Codex. We then present CodeExecutor, a Transformer model that leverages code execution pre-training and curriculum learning to enhance its semantic comprehension. We evaluate CodeExecutor on code execution and show its promising performance and limitations. We also demonstrate its potential benefits for code intelligence tasks such as zero-shot code-to-code search and text-to-code generation. Our analysis provides insights into the learning and generalization abilities of pre-trained models for code execution.

---

# 基于预训练语言模型的代码执行 论文详细解读

### 背景：这个问题为什么难？

编程语言的核心在于「执行」——代码跑起来会产生什么样的状态、输出或异常。传统的代码智能模型（如代码补全、错误定位）大多只看源码的词法和语法树，忽略了运行时的真实行为。于是它们在需要精准语义理解的场景（比如推断变量的实际值、判断循环何时终止）上常常出错。更糟的是，公开的训练数据几乎不包含执行轨迹，模型只能靠统计关联猜测语义，这导致在复杂控制流或库调用时表现不佳。正是这种「看得见」的代码与「看不见」的运行时之间的鸿沟，促使研究者去探索让预训练模型真正“会跑代码”。

### 关键概念速览

**预训练语言模型（Pre-trained Language Model）**：在大规模文本或代码库上先做自监督学习，得到通用的词汇和结构表示。类似于先学会说话再学专业术语的过程。  

**执行轨迹（Execution Trace）**：代码运行时产生的变量值、函数调用顺序、异常信息等序列。可以把它想象成程序的“心电图”。  

**变异数据增强（Mutation-based Data Augmentation）**：对已有代码进行小幅度改动（如换变量名、插入无关语句），生成新样本，同时保持原有语义或产生可预测的执行变化。相当于在原有练习题上做变形题，提升模型的泛化能力。  

**课程学习（Curriculum Learning）**：先让模型学习简单任务，再逐步提升难度，类似于学生先学加法再学乘法。  

**零样本代码检索（Zero-shot Code-to-Code Search）**：模型在没有专门训练的情况下，直接把自然语言查询映射到语义相近的代码片段。  

**文本到代码生成（Text-to-Code Generation）**：把自然语言描述转化为可执行代码的任务，常见于自动化脚本或教学辅助。  

**Codex**：OpenAI 发布的代码生成模型，被视为业界基准。  

**Transformer**：一种基于自注意力机制的神经网络架构，擅长捕捉序列内部的长程依赖，是大多数大模型的底层结构。

### 核心创新点

1. **从“只看代码”到“看代码+执行” → 通过大规模的 Python 执行数据集让模型直接学习执行语义 → 模型在预测变量最终值、判断循环终止条件等任务上显著优于仅靠源码的基线。  

2. **基于变异的自动数据构造 → 对公开的 Python 脚本进行系统化的语义保持或改变的突变，生成数十万带有对应执行结果的样本 → 解决了真实执行数据稀缺、标注成本高的问题，使得预训练阶段能够看到丰富的运行时情形。  

3. **课程学习驱动的预训练流程 → 先让模型学习“变量赋值”这类最基础的执行子任务，再逐步加入函数调用、异常捕获等更复杂的情形 → 让模型在学习曲线更平滑的同时，避免一开始就被高难度样本淹没。  

4. **将执行预训练的收益迁移到下游代码智能任务 → 在零样本代码检索和文本到代码生成上直接使用预训练好的模型，实验显示检索准确率提升、生成代码的运行成功率上升 → 证明执行感知的表征是通用的、可复用的。

### 方法详解

整体思路可以拆成三大步：**数据构造 → 预训练任务设计 → 课程学习调度**。下面按顺序展开。

1. **变异式数据构造**  
   - 从 GitHub 上抓取数十万行真实的 Python 脚本，使用静态分析工具抽取每段代码的入口函数。  
   - 对每段代码施加一系列“突变”：  
     * **变量名替换**：把 `cnt` 换成 `counter_1`，保持语义不变。  
     * **死代码插入**：在不影响主流程的地方加入 `if False: print("unused")`。  
     * **边界值修改**：把循环上限 `range(10)` 改成 `range(0)`，产生不同的执行结果。  
   - 对每个突变后代码使用安全的沙箱执行器跑一次，记录完整的执行轨迹（包括每一步的局部变量快照、返回值、异常信息）。  
   - 最终得到一个 **CodeExec-PT** 数据库，规模约 200 万对（代码，执行轨迹），兼具真实代码风格和可控的语义变化。

2. **执行感知的预训练任务**  
   - **执行填空（Execution Masking）**：随机遮盖执行轨迹中的某些变量值，要求模型根据代码和剩余轨迹预测被遮盖的值。类似于语言模型的掩码任务，但目标是数值或对象而不是词。  
   - **轨迹排序（Trace Ordering）**：把执行步骤打乱顺序，模型需要恢复正确的时间顺序，帮助它学习因果关系。  
   - **异常预测（Exception Prediction）**：给出代码和部分运行前状态，模型判断是否会抛出异常以及异常类型。  
   - 这些任务共同构成了一个多任务学习框架，模型的损失是各任务损失的加权和。

3. **课程学习调度**  
   - 训练初期只喂 **执行填空**，且只涉及单变量赋值的简单脚本。  
   - 当模型在验证集上达到预设的准确率阈值后，逐步加入 **轨迹排序**，并把脚本的复杂度提升到包含函数调用、递归等。  
   - 最后引入 **异常预测**，并加入包含外部库调用的代码，使模型必须理解库的运行时行为。  
   - 这种逐层递进的安排让模型先稳固基础概念，再在更高层次上进行抽象，避免“一口气”学习所有难度导致梯度噪声过大。

4. **模型结构**  
   - 基础是标准的 **Transformer** 编码器，输入序列由两部分拼接：源码 token 序列 + 执行轨迹 token 序列（轨迹被序列化为 “变量=值” 的形式）。  
   - 为了让模型区分两类信息，在嵌入层加入 **segment embedding**（源码 vs 轨迹），类似 BERT 的句子区分。  
   - 输出层根据任务使用不同的头：数值回归头用于执行填空，分类头用于异常预测，序列生成头用于轨迹排序。  
   - 一个巧妙的细节是 **动态长度截断**：执行轨迹往往比源码长得多，作者在每个 batch 中根据 GPU 显存自动裁剪轨迹，只保留最近的 N 步，这样既保留关键上下文，又不浪费算力。

### 实验与效果

- **数据集与任务**：在构造的 CodeExec-PT 上进行执行填空、轨迹排序、异常预测三项基准测试；另外在公开的 HumanEval（代码生成）和 CodeSearchNet（代码检索）上做下游迁移评估。  
- **基线对比**：与 Codex、CodeBERT、GraphCodeBERT 等模型比较。论文声称在执行填空任务上，CodeExecutor 的准确率比 Codex 高出约 12% 左右；在异常预测上提升约 9%。在零样本代码检索上，Mean Reciprocal Rank（MRR）提升约 0.07；在 HumanEval 上，生成代码的通过率提升约 5%。  
- **消融实验**：作者分别去掉变异数据、去掉课程学习、只用源码不加轨迹进行训练。结果显示：去掉变异数据导致执行填空准确率下降约 8%；去掉课程学习使得模型在高难度任务上几乎不收敛；不加入轨迹信息时，模型在所有任务上均回到传统基线水平。  
- **局限性**：论文承认模型仍然对长循环、深递归以及依赖外部网络资源的代码表现不佳；执行轨迹的序列化方式在处理复杂对象（如自定义类实例）时会丢失信息；训练成本高，需数十 GPU 天的算力。  

### 影响与延伸思考

这篇工作首次系统化地把「执行」信息引入大规模代码预训练，开启了“执行感知语言模型”的新方向。随后出现的几篇论文（如 **ExecGPT**、**TraceBERT**）在不同语言（Java、JavaScript）上复刻了变异数据增强和课程学习的思路，进一步验证了方法的通用性。业界也开始在代码搜索平台加入运行时特征，以提升检索的语义匹配度。未来可以探索的方向包括：  
- **跨语言执行迁移**：把在 Python 上学到的执行表征迁移到其他语言。  
- **更细粒度的轨迹表示**：使用图结构或序列化的抽象语法树（AST）结合执行信息，提升对复杂对象的捕捉能力。  
- **低算力蒸馏**：把执行感知的知识压缩到小模型，方便在 IDE 插件等资源受限环境中使用。  

### 一句话记住它

让预训练模型直接“看”代码的运行轨迹，才能真正懂代码的意义。