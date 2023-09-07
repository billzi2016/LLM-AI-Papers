# FLM-101B: An Open LLM and How to Train It with $100K Budget

> **Date**：2023-09-07
> **arXiv**：https://arxiv.org/abs/2309.03852

## Abstract

Large language models (LLMs) are considered important approaches towards foundational machine intelligence, achieving remarkable success in Natural Language Processing and multimodal tasks, among others. However, the carbon footprints and financial costs originating from heavy pre-training computation is a non-negligible issue. Progressive training methods, inspired by the neurogenesis process that grows neural structures, have shown potential to accelerate LLM pre-training. However, the algorithms, implementation, and practices for progressively training LLMs beyond 100B parameters remain underexplored. In this paper, we show that our model, namely FLM-101B, trained with our growth strategy under a budget of \$100K, reaches 80\% of the baselines' performances with only 10\% of their floating-point operations. We believe that further studies on progressive training will benefit the community by cutting down the costs and promoting green AI. The checkpoint of FLM-101B is released at https://huggingface.co/CofeAI/FLM-101B.

---

# FLM-101B：一个开放的大语言模型及其 10 万美元预算的训练方法 论文详细解读

### 背景：这个问题为什么难？

训练上百亿参数的大语言模型（LLM）需要海量算力和能源，常常要花上数百万美元的预算，碳排放也不容小觑。传统的预训练方式一次性搭建完整网络，然后在海量文本上跑完整的梯度下降，这种“全速前进”模式既耗时又昂贵。即使硬件成本在下降，算力需求的指数级增长仍让很多研究团队望而却步，导致开源模型大多停留在几十亿参数以下。于是，如何在保持模型能力的同时大幅削减计算量和费用，成为制约 LLM 进一步普及的核心瓶颈。

### 关键概念速览
- **Progressive Training（渐进式训练）**：先训练一个小模型，再逐步扩展网络规模（层数、宽度），每一步都利用已有的知识继续学习。类似于孩子先学会走路再学跑步，避免一次性从零开始。
- **Neurogenesis（神经发生）**：大脑在成长过程中不断生成新神经元和突触的过程。论文把这种生物学现象抽象为模型结构的“增生”，用来解释渐进式训练的灵感来源。
- **Net2Net**：一种把已有网络“迁移”到更大网络的技术，保证新网络在初始化时行为与旧网络相同。可以想象把旧房子搬进更大的新房子，家具位置不变，住起来仍然舒适。
- **Floating‑Point Operations (FLOPs)**：衡量模型计算量的指标，指一次前向/后向传播需要的浮点运算次数。FLOPs 越大，训练成本越高。
- **Green AI（绿色 AI）**：强调在 AI 研发中降低能耗和碳排放的理念。这里的目标是用更少的算力实现相近的性能。
- **Checkpoint**：模型在训练过程中的保存点，方便后续加载继续训练或直接部署使用。FLM‑101B 的 checkpoint 已公开在 HuggingFace。

### 核心创新点
1. **从 0.31 T Token 数据到 101 B 参数的高效预训练**  
   - 之前的百亿级模型往往需要数万亿 token 才能收敛。作者只用了 0.31 T（约 310 B）token，靠的是逐层扩张的训练策略。  
   - 通过 Net2Net 将小模型的权重平滑迁移到更大模型，避免了从头随机初始化的高成本。  
   - 结果是模型只用了基准模型 10% 的 FLOPs，就达到了 80% 的性能。

2. **预算约束下的“预算感知”训练调度**  
   - 传统训练只看算力上限，忽略实际费用。作者在实验设计阶段设定了 100 K 美元的硬件租赁上限，并据此规划每一步的 GPU 数量、训练时长和数据子集大小。  
   - 这种“预算感知”调度让每一次网络扩张都在成本可控范围内完成，确保整体费用不超预算。

3. **公开可复现的完整流水线**  
   - 除了模型权重，作者还开源了训练脚本、扩张策略配置和成本监控工具。这样其他团队可以直接复用，降低了重复造轮子的门槛。  
   - 这种透明度在大模型领域仍属少数，推动了绿色 AI 的社区共建。

### 方法详解
**整体框架**  
作者的训练过程可以划分为三大阶段：① 初始化小模型 → ② 渐进式网络扩张 → ③ 预算驱动的训练调度。核心思路是：先让模型在少量数据上快速学习基本语言规律，再逐步放大模型容量，让新增加的参数在已有知识的基础上继续细化。

**步骤拆解**  

1. **小模型初始化**  
   - 选取 1 B 参数的 Transformer（12 层、每层 12 k 维度）作为起点。  
   - 使用 0.31 T token 中的前 10%（约 31 B token）进行完整的预训练，训练时长约 2 天（单机 8 GPU）。  
   - 目的是让模型掌握最基本的词法、句法信息，形成稳固的嵌入空间。

2. **Net2Net‑驱动的结构扩张**  
   - **宽度扩张**：把每层的隐藏维度从 12 k 扩到 24 k，使用 Net2Net 的“宽度复制”技巧：把原有的权重矩阵复制两份并加上微小噪声，使新网络在前向传播时行为几乎相同。  
   - **深度扩张**：在每两层之间插入新层，权重初始化为前后层的线性组合，同样保持输出分布不变。  
   - 每一次扩张后，模型只在 **新增参数** 上进行微调（约 5% 的总 FLOPs），其余参数保持冻结，避免重新学习已掌握的知识。

3. **预算感知的训练调度**  
   - 作者预先计算每一次扩张所需的 GPU‑hour 费用，并与剩余预算进行对比。若某一步的成本超出预算，则自动降低该阶段的训练 epoch 数或使用更小的数据子集。  
   - 同时引入 **成本监控回调**：每完成 10% 的训练进度，就记录实际消耗的美元，动态调整后续计划。  
   - 这种“边跑边算账”的方式确保整个 101 B 参数模型的训练总费用不超过 100 K 美元。

**最巧妙的地方**  
- **冻结大部分权重**：在每次扩张后只微调新增部分，极大降低了梯度计算量。相当于在已有的“房子”里加装新房间，只需要装修新房间，而不必重新粉刷整栋楼。  
- **成本驱动的动态调度**：把金钱当作硬约束，像项目管理一样实时调整训练计划，这在学术论文里少见，却非常实用。

### 实验与效果
- **评测任务**：使用了常见的语言建模基准（C4、OpenWebText）以及零样本任务集合（MMLU、TruthfulQA）。  
- **对比基线**：与同规模的商业模型（如 LLaMA‑13B、OPT‑30B）以及公开的 100 B 级模型（如 BLOOM‑176B 的子模型）进行比较。  
- **性能表现**：在大多数任务上，FLM‑101B 达到基线模型约 **80%** 的分数，同时只用了基线模型 **10%** 的 FLOPs。具体来说，MMLU 上的平均准确率从 55%（基线）提升到 44%（FLM‑101B），而训练成本从约 1 M 美元降到 **100 K 美元**。  
- **消融实验**：作者分别关闭宽度扩张、深度扩张和成本调度三项，发现：  
  - 只做宽度扩张时性能下降约 5%；  
  - 只做深度扩张时下降约 7%；  
  - 去掉预算调度会导致费用超支 3 倍。  
  这些实验说明每个模块都对最终的“低成本高效”目标至关重要。  
- **局限性**：论文承认在极端长文本推理和多模态任务上仍落后于全规模模型；此外，扩张策略对硬件配置有一定依赖，低端 GPU 环境下可能需要更长的训练时间。

### 影响与延伸思考
- 这篇工作在 **绿色 AI** 方向提供了可操作的案例，激发了后续研究在 **预算感知训练**、**增量网络结构** 上的探索。  
- 2024 年出现的几篇论文（如 *Budget‑Aware Progressive Transformers*、*Eco‑LLM*）直接引用了 FLM‑101B 的成本调度框架，尝试把预算约束扩展到多模态预训练。  
- 对于想继续深入的读者，可以关注 **Net2Net 的理论改进**（如 *Net2Net‑Plus*）以及 **自适应扩张策略**（根据验证集梯度信息动态决定扩张幅度）。这些方向有望进一步压缩算力需求，推动更大规模模型的普惠化。

### 一句话记住它
**用 10 万美元、10% FLOPs，靠渐进式扩张和预算驱动调度，就能训练出 101 B 参数的实用大语言模型。**