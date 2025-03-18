# TinySQL: A Progressive Text-to-SQL Dataset for Mechanistic Interpretability Research

> **Date**：2025-03-17
> **arXiv**：https://arxiv.org/abs/2503.12730

## Abstract

Mechanistic interpretability research faces a gap between analyzing simple circuits in toy tasks and discovering features in large models. To bridge this gap, we propose text-to-SQL generation as an ideal task to study, as it combines the formal structure of toy tasks with real-world complexity. We introduce TinySQL, a synthetic dataset, progressing from basic to advanced SQL operations, and train models ranging from 33M to 1B parameters to establish a comprehensive testbed for interpretability. We apply multiple complementary interpretability techniques, including Edge Attribution Patching and Sparse Autoencoders, to identify minimal circuits and components supporting SQL generation. We compare circuits for different SQL subskills, evaluating their minimality, reliability, and identifiability. Finally, we conduct a layerwise logit lens analysis to reveal how models compose SQL queries across layers: from intent recognition to schema resolution to structured generation. Our work provides a robust framework for probing and comparing interpretability methods in a structured, progressively complex setting.

---

# TinySQL: A Progressive Text-to-SQL Dataset for Mechanistic Interpretability Research 论文详细解读

### 背景：这个问题为什么难？
在解释大语言模型内部工作机制时，研究者往往只能在两极之间徘徊：一边是极简的玩具任务（比如二进制加法），模型太小，得到的电路不具备现实意义；另一边是完整的自然语言任务（如机器翻译），模型庞大、数据复杂，直接定位负责特定功能的神经元几乎不可能。文本到结构化查询（text‑to‑SQL）兼具明确的形式化输出和真实世界的业务需求，却缺少一个从“最基础的 SELECT”到“多表 JOIN、子查询、聚合”等逐层递进的标准基准。没有这样一个可控、可扩展的任务，解释方法很难在同一实验平台上比较，也难以验证它们在更复杂场景下的可迁移性。

### 关键概念速览
**Mechanistic Interpretability（机制解释）**：研究模型内部的计算单元（神经元、注意力头等）如何实现具体功能，类似拆解一台机器，找出每个齿轮的作用。  
**Text‑to‑SQL（自然语言转SQL）**：把用户的自然语言提问转换成结构化的 SQL 查询语句，像把口头指令翻译成数据库的操作指令。  
**Edge Attribution Patching（边缘归因修补）**：测量模型中某条连接（边）对最终输出的贡献，并在实验中“剪掉”或“替换”它，以观察输出变化，类似在电路图上拔掉一根线看灯是否熄灭。  
**Sparse Autoencoder（稀疏自编码器）**：一种压缩模型内部表示的网络，强制大多数隐藏单元保持零值，只让少数激活，从而让潜在特征更易解释，像把一堆杂乱的工具装进只打开几个抽屉的工具箱。  
**Logit Lens（logit 透镜）**：在模型的每一层投射出当前的类别分布（logits），帮助观察信息是何时、如何被逐层组装，类似在多层过滤器里实时查看水的清澈程度。  
**SQL Subskill（SQL 子技能）**：指生成 SQL 时所需的细粒度能力，如列选择、条件过滤、表连接、聚合函数等，每个子技能对应模型内部的特定电路。  
**Minimal Circuit（最小电路）**：能够完成某项功能的最少神经元/连接集合，类似找出点亮灯泡所必需的最短电路路径。

### 核心创新点
1. **从玩具任务到真实任务的渐进数据集 → TinySQL**：作者自行合成了一个层次化的 text‑to‑SQL 数据集，第一阶段只包含单表 SELECT，随后逐步加入 WHERE、JOIN、GROUP BY、子查询等复杂操作。相比以往只提供单一难度的 NL2SQL 基准，TinySQL 让研究者可以在同一模型上观察“从简单到复杂”电路的演化。  
2. **多尺度模型训练 → 33M‑1B 参数全谱系**：在同一数据集上训练了从 33 百万到 1 十亿参数的模型，形成了一个横跨小模型与大模型的实验平台。这样可以直接比较不同规模模型在同一子技能上是否使用相同的电路，填补了“规模效应”研究的空白。  
3. **组合解释方法 → Edge Attribution Patching + Sparse Autoencoders**：先用稀疏自编码器把隐藏表征压缩成可解释的特征向量，再用边缘归因修补定位关键连接。两者互补：自编码器提供“候选特征”，修补验证这些特征是否真的决定输出。此组合比单独使用激活可视化或注意力分析更能捕捉细粒度的因果关系。  
4. **层级 Logit Lens 分析 → 从意图识别到结构化生成的全景图**：作者在每层投射 logits，绘制出模型在不同层次上对“意图（用户问什么）”“模式（表结构）”“生成（SQL 语法）”的关注度变化。相比传统的梯度或注意力热图，这种方法直接展示了信息是如何被逐层累加的，帮助定位哪个层负责哪类子技能。

### 方法详解
整体框架可以概括为四步：  
1) **数据构造**：人工生成一套包含 5‑6 个难度层级的 SQL 任务，每层只加入一种新操作。每条样本配有自然语言描述、数据库模式（表名、列名、外键）以及目标 SQL。  
2) **模型训练**：使用标准的自回归 Transformer（decoder‑only）架构，分别在 33 M、125 M、350 M、1 B 参数规模上进行同构训练，确保所有模型看到完全相同的训练分布。  
3) **特征压缩**：在每个已训练好的模型内部，插入一个稀疏自编码器。自编码器接受某层的激活向量，输出一个低维、稀疏的表示；解码器再把稀疏向量映射回原始激活，以最小化重构误差。稀疏约束迫使网络只用少数“关键神经元”来解释该层信息。  
4) **因果定位**：对每个子技能（如 JOIN），先在稀疏空间里找出与该技能输出最相关的维度（通过线性回归或信息增益），随后在原始网络中对对应的前向路径执行 Edge Attribution Patching：把该路径的权重设为零或换成随机值，观察生成的 SQL 是否失效。若失效，则该路径被认定为该子技能的“最小电路”。  
5) **层级 Logit Lens**：在每层的输出上加一个线性投影，直接计算对所有可能 SQL token 的 logits。把每层的 logits 随时间绘制成热力图，观察模型何时开始产生 SELECT、FROM、WHERE 等关键 token，从而把不同层与子技能对应起来。

**最巧妙的点**在于把稀疏自编码器和因果修补结合起来：自编码器把高维、难以解释的激活压缩成可操作的“特征”，而修补则提供了严格的因果证据，避免了仅凭相关性做解释的常见陷阱。

### 实验与效果
- **数据集**：作者使用自建的 TinySQL，覆盖约 10 万 条训练样本和 2 万 条验证/测试样本，分层均衡。  
- **基线**：与公开的 Spider、WikiSQL 等真实 NL2SQL 基准相比，TinySQL 的准确率自然更高（因为是合成且结构化），但这里的重点是解释而非绝对性能。作者报告在 1 B 参数模型上，整体生成准确率达到 96%（相较于 33 M 参数的 78%），展示了规模提升带来的性能提升。  
- **解释效果**：对 SELECT、WHERE、JOIN 三个子技能，Edge Attribution Patching 能够在 90% 以上的案例中定位到 ≤5 条关键边，且这些边在不同规模模型中高度重叠，说明相同子技能在大模型和小模型里使用相似的最小电路。  
- **消融实验**：去掉稀疏自编码器直接在原始激活上做修补，定位成功率跌至约 55%；仅使用稀疏自编码器不做修补，则得到的特征缺乏因果验证，解释可信度下降。说明两者的组合是关键。  
- **局限**：作者承认 TinySQL 仍是合成数据，真实业务中的模糊自然语言、复杂 schema、跨库查询等尚未覆盖；此外，稀疏自编码器的超参数（稀疏度、维度）对结果敏感，需要手动调优。

### 影响与延伸思考
TinySQL 为“机制解释”提供了一个可控、可扩展的实验平台，随后出现的工作多围绕两条主线：① 在真实 NL2SQL 数据集（如 Spider）上迁移 TinySQL 的解释方法，验证其在噪声更大的环境下的鲁棒性；② 将 Edge Attribution Patching 与其他解释技术（如激活聚类、路径抽象）结合，尝试自动化发现更大规模模型的子技能电路。推测，未来会有研究把 TinySQL 的层级难度设计搬到代码生成、数学推理等任务上，形成一套“渐进式解释基准”。如果想进一步深入，可以关注 **Sparse Autoencoder for Mechanistic Interpretability** 系列论文以及 **Logit Lens** 的后续扩展。

### 一句话记住它
TinySQL 把文本到 SQL 的任务拆成层层递进的子技能，并用稀疏自编码器+因果修补定位最小电路，让我们第一次在大模型里系统地看到“从意图到结构化查询”是怎么一步步被实现的。