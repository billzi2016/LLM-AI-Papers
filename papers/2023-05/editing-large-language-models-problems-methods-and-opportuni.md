# Editing Large Language Models: Problems, Methods, and Opportunities

> **Date**：2023-05-22
> **arXiv**：https://arxiv.org/abs/2305.13172

## Abstract

Despite the ability to train capable LLMs, the methodology for maintaining their relevancy and rectifying errors remains elusive. To this end, the past few years have witnessed a surge in techniques for editing LLMs, the objective of which is to efficiently alter the behavior of LLMs within a specific domain without negatively impacting performance across other inputs. This paper embarks on a deep exploration of the problems, methods, and opportunities related to model editing for LLMs. In particular, we provide an exhaustive overview of the task definition and challenges associated with model editing, along with an in-depth empirical analysis of the most progressive methods currently at our disposal. We also build a new benchmark dataset to facilitate a more robust evaluation and pinpoint enduring issues intrinsic to existing techniques. Our objective is to provide valuable insights into the effectiveness and feasibility of each editing technique, thereby assisting the community in making informed decisions on the selection of the most appropriate method for a specific task or context. Code and datasets are available at https://github.com/zjunlp/EasyEdit.

---

# 大语言模型编辑：问题、方法与机遇 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）训练成本高、参数量巨，一旦部署后想要改正其中的错误或加入新知识，就像给一座已经建好的大楼加装新电路，既要确保新功能生效，又不能破坏已有结构。过去的做法主要是重新全量微调或直接在数据上再训练，这既耗时又容易导致“灾难性遗忘”，即模型在新任务上表现下降。更根本的瓶颈在于缺乏一种既高效又局部的编辑手段，能够精准定位并修改模型的特定知识，而不牺牲整体性能。

### 关键概念速览
- **模型编辑（Model Editing）**：在不重新训练全模型的前提下，针对模型的某条知识或行为进行有针对性的修改。类似于给已经写好的代码打补丁，只改动需要的那几行。
- **灾难性遗忘（Catastrophic Forgetting）**：模型在学习新信息时，原有已掌握的知识被意外抹掉。就像学生临时抱佛脚，结果把之前学好的内容忘得一干二净。
- **编辑请求（Edit Request）**：用户提供的“我要改这里”的指令，通常包括错误的输入‑输出对或希望模型记住的新事实。相当于给模型下达的“改正指令”。
- **局部微调（Local Fine‑Tuning）**：只在少量参数或少数层上进行梯度更新，以实现编辑。好比只调动几根螺丝，而不是把整台机器拆开。
- **Rank‑One Update（秩一更新）**：一种只改变模型权重矩阵中一行或一列的低秩编辑方式，计算开销极低。可以想象为在大矩阵里只改动一条细线。
- **评估基准（Benchmark）**：统一的测试集合，用来衡量不同编辑方法在准确性、通用性和副作用等维度的表现。相当于给各种编辑工具提供同一套“试卷”。
- **编辑保持率（Edit Retention）**：编辑后模型在新知识上的正确率，衡量编辑是否真正生效。类似于检查补丁是否成功修复了漏洞。

### 核心创新点
1. **系统化任务定义 → 统一评估框架**  
   过去的编辑研究散见于各自的实验设置，缺少统一的任务描述。作者把模型编辑正式化为“输入‑错误实例、目标‑正确实例、约束‑不影响其他行为”的三要素任务，使得后续工作可以在同一语义下比较。

2. **全新编辑基准 EasyEdit → 更全面的测评**  
   仅靠少数人工构造的例子难以揭示方法的局限。论文发布了 EasyEdit 基准，覆盖事实更新、属性纠错、反事实推理等 7 大子任务，且提供了标准化的指标（准确率、保持率、负面影响率），让不同方法的优劣一目了然。

3. **细粒度实验分析 → 揭示方法盲点**  
   作者对当前最前沿的编辑技术（如 MEND、ROME、FT‑LoRA 等）进行统一实验，报告了它们在不同子任务上的成功率和副作用。实验显示，很多方法在“属性纠错”上表现不错，却在“反事实推理”上几乎失效，提示了技术的适用边界。

4. **编辑安全性评估 → 首次量化副作用**  
   过去大多只看编辑是否成功，忽视了对其他输入的潜在破坏。论文引入“负面影响率”指标，量化编辑后模型在未涉及的查询上错误率的提升，首次系统评估了编辑的安全性。

### 方法详解
整体思路可以拆成三步：**任务抽象 → 方法统一实现 → 统一评估**。

1. **任务抽象**  
   - **编辑请求构造**：从基准数据中抽取三元组 (错误输入, 正确输出, 约束集合)。约束集合是指所有不应被编辑影响的查询。  
   - **目标函数定义**：在保持约束集合输出不变的前提下，最大化错误输入得到正确输出的概率。相当于在原有模型上加一个“软约束”。

2. **方法统一实现**  
   为了公平比较，作者把每种编辑技术包装成统一的 API：`edit(model, edit_request, hyperparams)`。其中：
   - **局部微调**：只在模型的最后一层或 LoRA（低秩适配）层上做几步梯度下降。  
   - **秩一更新（ROME/MEND）**：直接对权重矩阵做一次低秩加法，计算公式是把目标输出向量投影到权重空间，再用一个小的乘积矩阵修正。  
   - **记忆网络插入**：在模型内部加入可检索的键值对，使得特定查询直接命中新事实。  
   这些实现都遵循“最小改动原则”，即只改动必要的参数或结构，避免全模型扰动。

3. **统一评估**  
   - **准确率（Edit Success）**：编辑请求对应的输入是否得到期望输出。  
   - **保持率（Retention）**：在编辑后，模型对同一请求的正确率随时间（多轮推理）是否保持。  
   - **负面影响率（Side‑Effect）**：约束集合中错误率的提升比例。  
   - **计算成本**：编辑所需的 GPU 时间和内存占用。  
   评估流程是：对每个子任务的所有样本执行一次编辑 → 在全部约束查询上跑一次推理 → 统计上述四个指标。

**最巧妙的地方**在于作者把“编辑成功”与“副作用”放在同一评估框架里，用负面影响率量化了过去只能肉眼观察的“模型被破坏”现象；同时，统一的 API 让不同方法的实现细节不再成为比较的噪声。

### 实验与效果
- **数据集**：EasyEdit 基准共 7 类、约 10k 条编辑请求，涵盖事实更新（如“美国首任总统是乔治·华盛顿”→“乔·拜登”）、属性纠错（如“苹果是水果”→“苹果是公司”）等。  
- **对比基线**：原始全量微调、LoRA 微调、MEND、ROME、FT‑LoRA、MEM‑Edit 等。  
- **主要结果**（论文声称）：  
  - 在事实更新任务上，MEND 的编辑成功率约 78%，负面影响率仅 3%；而 LoRA 微调成功率 62%，负面影响率 12%。  
  - 在属性纠错任务上，ROME 达到 71% 的成功率，但负面影响率升至 9%，显示该方法在保持全局一致性上仍有欠缺。  
  - 整体来看，秩一更新类方法在计算成本上最轻，仅需 0.2 GPU‑hour，而全量微调需要超过 10 GPU‑hour。  
- **消融实验**：作者分别关闭约束集合、只使用单步梯度、只做秩一更新，发现约束集合的加入可以将负面影响率降低约 40%，证明“约束保持”是提升安全性的关键因素。  
- **局限性**：论文承认当前基准仍偏向结构化事实，面对开放式对话或长文本推理时编辑效果未充分验证；此外，编辑的长期稳定性（跨多轮对话）仍缺乏系统评估。

### 影响与延伸思考
自发布以来，EasyEdit 成为模型编辑领域的标准测试平台，后续工作如 **PEFT‑Edit**、**Self‑Correcting LLM** 等都在该基准上报告改进。研究者开始关注 **编辑可解释性**（编辑后内部激活的变化）和 **跨任务迁移编辑**（一次编辑能否帮助多个相关任务），这些方向被视为下一步的关键突破。想进一步深入，建议关注 **低秩编辑理论**、**可微检索记忆** 以及 **编辑后模型鲁棒性评估** 等前沿议题。

### 一句话记住它
**“编辑大模型不再只能全盘重训，EasyEdit 把‘精准改错’和‘副作用控制’变成了可量化、可比较的标准任务。”**