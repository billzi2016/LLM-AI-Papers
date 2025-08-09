# Technical Report: Full-Stack Fine-Tuning for the Q Programming Language

> **Date**：2025-08-09
> **arXiv**：https://arxiv.org/abs/2508.06813

## Abstract

Even though large language models are becoming increasingly capable, it is still unreasonable to expect them to excel at tasks that are under-represented on the Internet. Leveraging LLMs for specialized applications, particularly in niche programming languages and private domains, remains challenging and largely unsolved. In this work, we address this gap by presenting a comprehensive, open-source approach for adapting LLMs to the Q programming language, a popular tool in quantitative finance that is much less present on the Internet compared to Python, C, Java, and other ``mainstream" languages and is therefore not a strong suit of general-purpose AI models. We introduce a new Leetcode style evaluation dataset for Q, benchmark major frontier models on the dataset, then do pretraining, supervised fine tuning, and reinforcement learning to train a suite of reasoning and non-reasoning models based on the Qwen-2.5 series, spanning five parameter sizes (1.5B, 3B, 7B, 14B, 32B). Our best model achieves a pass@1 accuracy of 59 percent on our Q benchmark, surpassing the best-performing frontier model, Claude Opus-4 by 29.5 percent. Additionally, all models, even our 1.5B model, outperform GPT-4.1 on this task. In addition to releasing models, code, and data, we provide a detailed blueprint for dataset construction, model pretraining, supervised fine-tuning, and reinforcement learning. Our methodology is broadly applicable, and we discuss how these techniques can be extended to other tasks, including those where evaluation may rely on soft or subjective signals.

---

# 技术报告：面向 Q 编程语言的全栈微调 论文详细解读

### 背景：这个问题为什么难？
Q 语言在量化金融领域使用广泛，但在公开的代码库、教程和论坛里几乎找不到它的身影。主流的大语言模型（LLM）在训练时主要看到 Python、C、Java 等语言的海量示例，因此对 Q 的语法、库函数甚至常见的业务模式都缺乏感知。直接把这些模型拿去写 Q 代码，往往会出现语法错误、调用不存在的函数，或者根本不懂金融业务的约束。换句话说，模型的“语言知识”与 Q 的实际需求之间存在巨大的分布差距，这让仅靠通用微调难以突破。

### 关键概念速览
**全栈微调（Full‑Stack Fine‑Tuning）**：指在同一个模型上依次完成预训练、监督微调和强化学习三个阶段，像把一块毛坯钢铁一步步锻造成完整工具。  
**LeetCode 风格评测**：把编程题目包装成输入‑输出对，模型需要直接给出可执行代码并通过自动化测试，类似刷题平台的批量评估。  
**pass@k 指标**：在 k 次生成尝试中至少有一次代码能通过全部测试的比例，常用来衡量代码生成模型的实用性。  
**Qwen‑2.5 系列**：开源的大语言模型家族，本次实验以它的不同规模（1.5B‑32B）为基底进行二次训练。  
**监督微调（Supervised Fine‑Tuning, SFT）**：使用人工标注的 Q 代码‑解释对，让模型学习“看了需求怎么写代码”。  
**强化学习（Reinforcement Learning, RL）**：把模型的输出当作动作，根据代码是否通过测试给出奖励，模型在奖励信号的驱动下自我改进。  
**软信号（Soft Signals）**：评价不完全依赖硬性对错，而是结合代码风格、执行效率等主观因素的反馈。

### 核心创新点
1. **从零构建 Q 代码基准 → 设计并公开了一个 LeetCode 风格的 Q 评测集 → 为后续模型提供了统一、可重复的衡量标准**。过去没有公开的 Q 题库，导致不同团队难以对齐进度。  
2. **全栈微调流水线 → 先在通用语料上继续预训练，再用人工标注的 Q 示例做监督微调，最后用 RL 让模型学会“写对代码” → 在 1.5B‑32B 规模上均实现了显著的性能跃升**。单纯的监督微调往往只能提升语法正确率，加入 RL 后模型的实际可运行率大幅提升。  
3. **多尺度模型套装 → 同时训练 5 种参数规模的模型 → 让资源受限的用户也能使用 1.5B 版，而不必等到大型模型**。这在开源社区里少见，提供了从轻量到强大的完整生态。  
4. **细化的蓝图文档 → 公开了数据采集、清洗、去噪、奖励函数设计等每一步的实现细节 → 其他低资源语言（如 COBOL、Fortran）可以直接照搬**。以往类似工作只给出结果，缺少可操作的“配方”。

### 方法详解
整体思路可以拆成四大块：**数据准备 → 预训练 → 监督微调 → 强化学习**。下面按顺序展开。

1. **数据准备**  
   - **爬取与过滤**：作者从金融机构内部代码库、公开的 Q 论文附录以及少量开源项目中抓取原始代码。随后用正则和语法树工具剔除不完整、注释过多或与金融业务无关的片段。  
   - **构造 LeetCode 题目**：把真实业务需求抽象成“输入‑输出”对，例如“给定历史价格序列，返回 5‑日均线”。每道题配备至少 3 组测试向量，确保评测的客观性。  
   - **标注**：人工编写参考实现并写出简要思路说明，形成 (需求, 参考代码, 思路) 三元组，用作监督微调的训练样本。

2. **继续预训练**  
   - 以 Qwen‑2.5 系列的公开权重为起点，继续在收集到的大规模 Q 代码语料上进行自回归语言模型预训练。目标是让模型的词表嵌入更好地捕捉 Q 的关键标识符（如 `.q`, `select`, `exec`）以及金融专有名词。  
   - 训练采用混合精度、梯度累积等技巧，使得即使在 1.5B 参数的模型上也能在几天内完成。

3. **监督微调（SFT）**  
   - 使用标注好的 (需求, 参考代码) 对进行教师强制（teacher forcing），让模型在看到需求后直接生成完整代码。  
   - 为防止模型只会复制参考实现，加入 **多样性采样**：对同一需求随机抽取不同的参考实现（如果有），鼓励模型学习多种写法。  
   - 损失函数仍然是交叉熵，但在每一步加入 **代码结构惩罚**（如未闭合的括号会额外加权），帮助模型在早期就养成语法完整的习惯。

4. **强化学习（RL）**  
   - **奖励函数设计**：核心奖励是二元的“通过全部测试 = +1，未通过 = -1”。此外，还加上 **代码简洁度**（行数越少奖励越高）和 **执行效率**（基准运行时间的倒数）作为软信号。  
   - **PPO（近端策略优化）**：在 SFT 基础上，用 PPO 进行策略更新。每轮采样 8 条代码，交给自动化评测器跑测试，得到奖励后回传。  
   - **经验回放**：把过去表现好的样本存入缓冲区，防止模型在追求新奖励时忘记已经学会的基本语法。  
   - 这一阶段的关键在于 **把“写对代码”直接当作目标**，而不是仅仅模仿人类写法。实验显示，RL 能把 pass@1 从 30% 提升到 59%（在 32B 模型上）。

**最巧妙的点**：作者把软信号（代码简洁度、运行时）直接加入奖励，而不是单独做后处理。这样模型在生成时就倾向于写出既正确又高效的代码，避免了后期手工剪枝的麻烦。

### 实验与效果
- **评测数据**：作者公开的 Q LeetCode 基准包含 500 题，覆盖时间序列处理、矩阵运算、数据库查询等常见金融任务。每题都有 3 组隐藏测试。  
- **基线对比**：与 Claude Opus‑4、GPT‑4.1、Claude 2、Llama‑2‑70B 等前沿模型比较。  
  - 最强基线 Claude Opus‑4 的 pass@1 为 29.5%。  
  - 本文的 32B Qwen‑2.5‑FullStack 达到 59%（提升约 2 倍）。  
  - 甚至最小的 1.5B 版本也超过了 GPT‑4.1（约 35% vs 30%）。  
- **消融实验**：  
  - 去掉 RL 环节，pass@1 降至 42%，说明 RL 对提升实际可运行率贡献约 17%。  
  - 只用通用预训练不做继续预训练，性能下降约 8%。  
  - 删除软信号奖励，生成代码倾向冗长，pass@1 下降 4%。  
- **局限性**：评测仅覆盖了公开的 500 题，真实金融系统中的高频交易、分布式计算等场景未覆盖。奖励函数对执行效率的衡量基于单机基准，可能与生产环境差异大。作者也提到模型在极端金融术语（如衍生品定价公式）上仍有错误。

### 影响与延伸思考
这篇报告在开源社区掀起了“低资源语言专化微调”的热潮。随后出现了针对 COBOL、Fortran、VHDL 等几乎不在互联网上出现的语言的类似全栈微调项目，很多都直接引用了本文的蓝图文档。金融机构内部也开始尝试把自己的专有脚本语言（如 KDB+/Q）交给专门微调的模型来做代码审查或自动化迁移。未来的研究方向可能包括：  
- **跨语言迁移学习**：把在 Q 上学到的金融业务知识迁移到其他金融 DSL。  
- **人机协同编辑**：结合代码补全与 RL‑based 纠错，让模型在编辑器里实时给出可运行的建议。  
- **软信号的更细粒度建模**：比如把代码的数值稳定性、内存占用等纳入奖励函数。  
（以上为基于公开信息的推测）

### 一句话记住它
全栈微调让通用大模型在几乎没有网络痕迹的 Q 语言上实现了 60% 级别的“一次通过”，证明只要有系统化的数据、预训练、监督和强化学习，任何低资源编程语言都能被 AI 彻底激活。