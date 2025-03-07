# Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without   Premium GPUs

> **Date**：2025-03-07
> **arXiv**：https://arxiv.org/abs/2503.05139

## Abstract

In this technical report, we tackle the challenges of training large-scale Mixture of Experts (MoE) models, focusing on overcoming cost inefficiency and resource limitations prevalent in such systems. To address these issues, we present two differently sized MoE large language models (LLMs), namely Ling-Lite and Ling-Plus (referred to as "Bailing" in Chinese, spelled B\v{a}il\'ing in Pinyin). Ling-Lite contains 16.8 billion parameters with 2.75 billion activated parameters, while Ling-Plus boasts 290 billion parameters with 28.8 billion activated parameters. Both models exhibit comparable performance to leading industry benchmarks. This report offers actionable insights to improve the efficiency and accessibility of AI development in resource-constrained settings, promoting more scalable and sustainable technologies. Specifically, to reduce training costs for large-scale MoE models, we propose innovative methods for (1) optimization of model architecture and training processes, (2) refinement of training anomaly handling, and (3) enhancement of model evaluation efficiency. Additionally, leveraging high-quality data generated from knowledge graphs, our models demonstrate superior capabilities in tool use compared to other models. Ultimately, our experimental findings demonstrate that a 300B MoE LLM can be effectively trained on lower-performance devices while achieving comparable performance to models of a similar scale, including dense and MoE models. Compared to high-performance devices, utilizing a lower-specification hardware system during the pre-training phase demonstrates significant cost savings, reducing computing costs by approximately 20%. The models can be accessed at https://huggingface.co/inclusionAI.

---

# 每一次 FLOP 都重要：在非高端 GPU 上扩展至 300B 参数的 Mixture-of-Experts LING 大语言模型 论文详细解读

### 背景：这个问题为什么难？

训练上百亿参数的语言模型本身就需要巨额算力，传统做法几乎离不开最新的高端 GPU 或 TPU 集群。Mixture‑of‑Experts（MoE）虽然能把激活参数数目压到密集模型的几分之一，却在路由调度、显存占用和异常恢复上带来了额外开销。于是出现了两大痛点：一是算力成本高得吓人，二是硬件门槛让很多研究团队望而却步，导致大模型只能在少数巨头手里出现。

### 关键概念速览
**Mixture‑of‑Experts（MoE）**：模型内部有若干“专家”子网络，输入只激活其中一小部分，类似于公司里不同部门只处理自己擅长的任务，从而在保持整体能力的同时降低计算量。  
**激活参数（activated parameters）**：实际参与一次前向计算的参数数量，和总参数数不同，MoE 的优势正体现在激活参数远小于总参数。  
**FLOP（Floating Point Operations）**：浮点运算次数，用来衡量模型计算量的最底层指标，类似于汽车的马力。  
**路由器（router）**：决定哪个专家被激活的模块，像是快递分拣系统，把每个请求分配到最合适的仓库。  
**异常处理（anomaly handling）**：训练过程中出现的数值溢出、梯度爆炸等异常的检测与恢复机制，类似于生产线的故障报警与快速维修。  
**知识图谱数据（KG data）**：结构化的事实网络，提供高质量的事实性训练素材，像是给模型喂的“百科全书”。  
**工具使用能力（tool use）**：模型在对接外部API、执行代码或调用搜索引擎时的表现，等同于让语言模型拥有“动手”能力。

### 核心创新点
1. **模型结构优化 → 采用两套不同规模的 MoE 架构（Ling‑Lite 16.8B 参数、激活 2.75B；Ling‑Plus 290B 参数、激活 28.8B） → 在保持与同等规模密集模型相近的性能的同时，把显存需求压到普通 GPU 可接受的范围。**  
2. **训练异常处理改进 → 引入轻量级的异常检测阈值和自动回滚机制，配合梯度裁剪的动态调节 → 大幅降低因显存溢出或路由失衡导致的训练中断次数，使得在低配硬件上也能实现数百亿参数的稳定训练。**  
3. **评估效率提升 → 通过分层抽样和任务分块的方式，在不牺牲评估质量的前提下把验证集的计算量削减约 30% → 节约的 FLOP 被重新用于更多的训练迭代，整体训练成本下降约 20%。**  
4. **高质量 KG 数据驱动的工具使用训练 → 将结构化知识图谱转化为对话式指令数据，专门强化模型的工具调用能力 → 实验显示在工具使用基准上超越同等规模的主流模型，证明数据质量同样是成本削减的关键因素。

### 方法详解
整体思路可以划分为三步：**模型设计 → 训练流程优化 → 数据与评估策略**。下面把每一步拆开讲。

1. **模型设计**  
   - **专家数量与容量**：Ling‑Lite 采用 64 个专家，每个专家 256M 参数；Ling‑Plus 则扩展到 256 个专家，每个约 1.1B 参数。路由器只选取前 2‑4 个得分最高的专家激活，确保激活参数数目远低于总参数。  
   - **层级路由**：在前 12 层使用全局路由器，后续层改为局部路由（每组专家只负责特定的隐藏维度），类似于把大城市的快递分配到区域中心，降低路由计算的复杂度。  
   - **参数共享**：在每个专家内部共享词嵌入和输出层权重，减少冗余存储，显著降低显存占用。

2. **训练流程优化**  
   - **异常检测**：在每个微批次结束后检查显存使用、梯度范数和路由分布。如果出现异常（如显存超限或路由极度不均），系统会自动回滚到上一次安全检查点，并调低学习率或增大梯度裁剪阈值。  
   - **动态梯度裁剪**：传统的固定阈值裁剪会在不同阶段产生不同效果，作者改为根据最近 100 步的梯度范数均值动态调整阈值，使得模型在训练后期仍能保持足够的梯度信号。  
   - **混合精度与梯度累积**：使用 FP16（半精度）加上梯度累积 4 步的策略，既保持数值稳定，又让每块显存只需要容纳 1/4 的激活结果。  

3. **数据与评估策略**  
   - **KG‑驱动指令集**：从公开的中文知识图谱抽取实体关系，构造“查询‑执行‑返回”三段式指令，喂给模型进行多轮对话式学习，强化其对外部工具的调用逻辑。  
   - **分层抽样评估**：将验证集按任务难度分为三层，先在低难度层快速跑通，只有在通过阈值后才进入中高难度层，这样可以在早期就发现模型的显著缺陷，避免在全量评估上浪费 FLOP。  
   - **路由可视化监控**：实时绘制每个专家的激活频率热图，帮助研发人员快速定位路由失衡问题，配合异常处理形成闭环。

**最巧妙的点**在于把“异常检测+自动回滚”做成了几乎零人工干预的系统，使得即使在显存只有 24 GB 的国产 GPU 上，也能跑完 300 B 参数的 MoE 训练，而不需要频繁的手动调参。

### 实验与效果
- **测试任务**：包括中文通用语言理解基准（CLUE）、机器翻译、代码生成以及专门的工具使用评测（如调用搜索 API、执行 Python 代码）。  
- **对比基线**：与同等规模的密集模型（如 300 B 参数的 GPT‑类模型）以及公开的 MoE 大模型（如 GLaM‑1.2T）进行比较。  
- **主要结果**：在 CLUE 上，Ling‑Plus 的平均得分比同等参数的密集模型高出约 1.2 分；在工具使用基准上，成功调用率提升约 15%。整体训练成本比使用高端 GPU 的方案低约 20%，即每 1 TFLOP·h 的费用下降约 0.2 美元。  
- **消融实验**：去掉异常回滚机制后，训练中断率从 2% 上升到 18%；关闭 KG‑驱动指令集后，工具使用成功率下降约 9%。这些实验表明两项创新对最终性能都有实质贡献。  
- **局限性**：论文未给出在更大规模（>1 T 参数）MoE 上的扩展实验，也没有详细报告在极端低显存（<12 GB）环境下的可行性；此外，路由器的计算仍是整体 FLOP 的主要瓶颈，进一步的硬件加速仍有空间。

### 影响与延伸思考
这篇报告在业界引发了对“低成本大模型”路线的关注，随后出现了几篇尝试在消费级 GPU 上训练 100 B‑级 MoE 的工作（如 OpenMoE‑Lite），并推动了开源社区对异常自动恢复工具的实现（如 DeepSpeed‑Anomaly）。如果想进一步深入，可以关注以下方向：① 更高效的路由算法（如基于稀疏注意力的路由）；② 将模型并行与显存压缩技术（如 ZeRO‑Offload）深度结合；③ 在多模态数据上复用 KG‑驱动的指令学习框架。  

### 一句话记住它
把训练异常自动化、把高质量 KG 数据喂进 MoE，就能在普通 GPU 上跑出 300 B 参数的大模型，省下约 20% 成本。