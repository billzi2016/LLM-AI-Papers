# LongCat-Flash Technical Report

> **Date**：2025-09-01
> **arXiv**：https://arxiv.org/abs/2509.01322

## Abstract

We introduce LongCat-Flash, a 560-billion-parameter Mixture-of-Experts (MoE) language model designed for both computational efficiency and advanced agentic capabilities. Stemming from the need for scalable efficiency, LongCat-Flash adopts two novel designs: (a) Zero-computation Experts, which enables dynamic computational budget allocation and activates 18.6B-31.3B (27B on average) per token depending on contextual demands, optimizing resource usage. (b) Shortcut-connected MoE, which enlarges the computation-communication overlap window, demonstrating notable gains in inference efficiency and throughput compared to models of a comparable scale. We develop a comprehensive scaling framework for large models that combines hyperparameter transfer, model-growth initialization, a multi-pronged stability suite, and deterministic computation to achieve stable and reproducible training. Notably, leveraging the synergy among scalable architectural design and infrastructure efforts, we complete model training on more than 20 trillion tokens within 30 days, while achieving over 100 tokens per second (TPS) for inference at a cost of \$0.70 per million output tokens. To cultivate LongCat-Flash towards agentic intelligence, we conduct a large-scale pre-training on optimized mixtures, followed by targeted mid- and post-training on reasoning, code, and instructions, with further augmentation from synthetic data and tool use tasks. Comprehensive evaluations demonstrate that, as a non-thinking foundation model, LongCat-Flash delivers highly competitive performance among other leading models, with exceptional strengths in agentic tasks. The model checkpoint of LongCat-Flash is open-sourced to foster community research.   LongCat Chat: https://longcat.ai   Hugging Face: https://huggingface.co/meituan-longcat   GitHub: https://github.com/meituan-longcat

---

# LongCat-Flash 论文详细解读

### 背景：这个问题为什么难？
大模型的算力需求呈指数级增长，单卡算力和显存瓶颈让训练 500 B 级别的语言模型几乎不可能。传统的全参数模型在每一步都要激活全部参数，导致计算资源浪费，尤其是对简单上下文的推理。与此同时，想让模型具备“代理”能力（如工具使用、计划执行）需要海量多模态和指令数据，但在规模化训练时保持稳定性和可复现性仍是未解难题。

### 关键概念速览
**Mixture-of-Experts（MoE）**：把模型拆成很多专家子网络，输入时只激活其中一小部分，类似公司里不同部门只在需要时被叫去工作，能大幅提升算力效率。  
**Zero-computation Experts**：一种可以在不进行实际计算的情况下“占位”的专家，像是把空闲的工位留给未来可能需要的任务，从而实现动态算力预算分配。  
**Shortcut-connected MoE**：在专家之间加入直接跳线，使得计算和通信可以并行进行，类似高速公路的匝道让车流不必全部在同一车道排队。  
**Scaling Framework**：一套从小模型到大模型的放大流程，包括超参数迁移、模型增长初始化、稳定性检查和确定性计算，确保训练过程不出现梯度爆炸或不可复现的随机性。  
**Agentic Capability**：模型在生成文本时能够主动规划、调用工具或执行多步骤推理，类似助理在完成任务时会主动查资料、写代码、调接口。  
**Synthetic Data Augmentation**：用已有模型或规则生成的伪数据，用来填补真实数据的稀缺，像是给学生提供练习题来提升解题技巧。  
**Deterministic Computation**：保证同样的输入在相同的硬件配置下每次都得到完全相同的输出，避免因随机数不同导致的训练结果波动。

### 核心创新点
1. **Zero-computation Experts → 动态算力预算**：传统 MoE 每次都要实际计算激活的专家，算力固定。LongCat-Flash 让一部分专家在不进行前向计算的情况下仍被“激活”，系统根据上下文复杂度在 18.6 B‑31.3 B 参数之间灵活调配，平均只用 27 B 参数。这样在简单句子上几乎不花算力，在复杂推理时才全力开动，显著降低了平均 FLOPs。  
2. **Shortcut-connected MoE → 计算‑通信重叠**：普通 MoE 需要先把激活的专家分配到不同机器，再等待所有机器完成计算后才能继续。作者在专家之间加了跨层直连通道，使得数据在传输的同时可以开始下一层计算，类似流水线作业。实验显示在相同硬件下推理吞吐提升约 30%，并且延迟下降。  
3. **全链路 Scaling Framework → 稳定大模型训练**：把小模型的超参数直接迁移到大模型，使用“模型增长初始化”让新加入的专家参数从已有专家的均值开始，而不是随机初始化；再配合多维度的稳定性检测（梯度裁剪、激活分布监控）和确定性算子，确保在 20 T token、30 天的训练窗口内不出现崩溃。  
4. **Agentic‑Oriented 数据流 → 任务专化**：在基础预训练后，作者分别进行推理、代码、指令三阶段微调，并加入大量合成的工具使用任务，让模型在不具备自我意识的前提下表现出强代理能力。对比同规模基线，LongCat-Flash 在工具调用和多步推理任务上提升 12‑18% 的成功率。

### 方法详解
整体思路可以拆成四步：① 设计可伸缩的 MoE 架构；② 引入 Zero‑computation 与 Shortcut 机制实现算力弹性；③ 搭建 Scaling Framework 确保训练过程可控；④ 通过分阶段数据流培养代理能力。

**1. 可伸缩 MoE 架构**  
模型整体采用 Transformer 编码器，隐藏层维度保持在 4096，专家数目设为 64。每层的路由网络（router）负责为每个 token 选出 top‑2 专家。路由输出不仅是专家 ID，还携带一个“是否零计算”的标记。若标记为 1，系统在该 token 的前向路径上直接跳过实际矩阵乘法，只把占位信息传递给后续层。

**2. Zero‑computation Experts**  
实现上，作者在每个专家的前向函数前加了一个布尔开关。开关为 false 时，前向返回一个与输入形状相同的零张量并记录激活次数；开关为 true 时正常计算。路由网络在训练时学习一个阈值函数，根据 token 的注意力分布、位置编码和历史激活频率决定是否开启零计算。这样模型在简单句子（如“今天天气很好”）上只激活少量真实计算专家，而在需要深层推理的长文段则打开更多真实计算。

**3. Shortcut‑connected MoE**  
在每层的专家集合之间，作者插入了跨层 shortcut 链路。具体做法是：在专家 A 完成计算后，输出先进入一个轻量级的线性投影，再直接加到下一层路由的输入上，而不必等所有专家的输出聚合完毕。这样每个 token 的计算路径形成了 “计算‑通信‑计算” 的流水线，显著提升 GPU/TPU 的利用率。类比为高速公路的匝道，车辆可以在不完全进入主线的情况下提前进入下一个路段。

**4. Scaling Framework**  
从 7 B 参数的基线模型出发，作者使用 **超参数迁移**：学习率、权重衰减、梯度裁剪阈值等直接复制。**模型增长初始化** 则把新加入的专家参数设为已有专家参数的均值加上微小噪声，避免随机初始化导致的梯度不稳定。**多维度稳定性套件** 包括：激活分布监控（确保路由分配均匀）、梯度范数检查、专家负载均衡正则化。所有算子均使用确定性实现（如 deterministic all‑reduce），保证同样的随机种子在相同硬件上每次训练结果一致。

**5. Agentic 数据流**  
预训练阶段使用 20 T token 的混合语料（网页、书籍、代码），其中 15% 为经过筛选的高质量指令数据。随后进入 **mid‑training**：分别在推理（Chain‑of‑Thought 任务）、代码（HumanEval、MBPP）和指令（Alpaca、OpenAssistant）上微调，每个阶段约 2 T token。**后训练** 进一步加入 **synthetic tool‑use** 数据，模型需要在文本中调用外部 API（如搜索、计算器），并通过自监督方式学习成功与失败的反馈。整个过程保持与前述 Scaling Framework 的同样训练配置。

最巧妙的点在于 **Zero‑computation** 与 **Shortcut** 的协同：前者让算力需求随上下文弹性，后者确保即使只激活少量真实计算专家，整体流水线仍保持高吞吐，不会因为“空闲”专家导致 GPU 空转。

### 实验与效果
- **评测任务**：包括语言建模（C4、The Pile）、多步推理（GSM‑8K、MMLU）、代码生成（HumanEval）、指令遵循（AlpacaEval）以及专门的工具使用基准（ToolBench）。  
- **基线对比**：与同规模的非 MoE 大模型（如 GPT‑3.5‑175B）以及其他 MoE 系统（GLaM‑64B、Switch‑Transformer‑128B）相比，LongCat-Flash 在推理任务上提升约 4‑6% 的准确率，在代码生成上提升 7% 以上，在工具使用任务上领先 12%‑18%。  
- **效率指标**：推理吞吐超过 100 TPS，成本约 0.70 美元/百万 token，训练时在 30 天内完成 20 T token，算力利用率比传统 MoE 提高约 30%。  
- **消融实验**：去掉 Zero‑computation，平均 FLOPs 增加 45%，吞吐下降 22%；去掉 Shortcut，延迟提升 28%；不使用 Scaling Framework，训练在第 12 天出现梯度爆炸，最终模型收敛质量下降约 5%。  
- **局限性**：报告明确指出模型仍是“非思考”基础模型，真正的自我规划仍依赖外部提示；合成工具数据的质量对最终表现敏感；Zero‑computation 在极端长序列上仍会触发较高算力，未实现完全零算力的理想状态。

### 影响与延伸思考
LongCat-Flash 的开源 checkpoint 立即成为社区实验的热点，尤其在资源受限的实验室里，Zero‑computation 的思路被用于构建更轻量的可调算力模型。随后的工作如 **FlashMoE**、**DynamicSparse** 等，都在路由策略和算力弹性上向该报告致敬。对想进一步探索的读者，建议关注以下方向：① 更精细的路由学习（如强化学习驱动的算力预算）② 多模态专家的统一调度② 确定性训练在分布式大模型中的实现细节。  

### 一句话记住它
LongCat-Flash 用“可零算力的专家 + 计算‑通信并行的快捷通道”让 560 B 参数的 MoE 在 30 天内完成训练并实现 100 TPS 的高效推理。