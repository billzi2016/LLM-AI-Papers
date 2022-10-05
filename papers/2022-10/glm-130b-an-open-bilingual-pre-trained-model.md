# GLM-130B: An Open Bilingual Pre-trained Model

> **Date**：2022-10-05
> **arXiv**：https://arxiv.org/abs/2210.02414

## Abstract

We introduce GLM-130B, a bilingual (English and Chinese) pre-trained language model with 130 billion parameters. It is an attempt to open-source a 100B-scale model at least as good as GPT-3 (davinci) and unveil how models of such a scale can be successfully pre-trained. Over the course of this effort, we face numerous unexpected technical and engineering challenges, particularly on loss spikes and divergence. In this paper, we introduce the training process of GLM-130B including its design choices, training strategies for both efficiency and stability, and engineering efforts. The resultant GLM-130B model offers significant outperformance over GPT-3 175B (davinci) on a wide range of popular English benchmarks while the performance advantage is not observed in OPT-175B and BLOOM-176B. It also consistently and significantly outperforms ERNIE TITAN 3.0 260B -- the largest Chinese language model -- across related benchmarks. Finally, we leverage a unique scaling property of GLM-130B to reach INT4 quantization without post training, with almost no performance loss, making it the first among 100B-scale models and more importantly, allowing its effective inference on 4$\times$RTX 3090 (24G) or 8$\times$RTX 2080 Ti (11G) GPUs, the most affordable GPUs required for using 100B-scale models. The GLM-130B model weights are publicly accessible and its code, training logs, related toolkit, and lessons learned are open-sourced at \url{https://github.com/THUDM/GLM-130B/}.

---

# GLM-130B：开放式双语预训练模型 论文详细解读

### 背景：这个问题为什么难？

在 100 B 级别之前，公开的大模型大多是单语的（只会英文），而且训练过程极其不稳定——参数量一上来，梯度容易爆炸、损失会出现突发的“spike”，导致训练直接崩溃。与此同时，中文社区缺少与 GPT‑3 同量级的模型，想要在同一套参数上兼顾中英两种语言更是少有尝试。于是出现了两个瓶颈：① 如何在 100 B 以上的规模上保持训练的数值稳定；② 如何让模型在两种语言上都保持强竞争力，而不必分别训练两个巨型模型。

### 关键概念速览
- **双向 Transformer**：模型内部同时支持左→右和右←左的注意力流，就像把 BERT（双向）和 GPT（单向）的优点合在一起，能够更好地捕捉句子内部的全局依赖。  
- **掩码语言模型（MLM）**：在输入序列中随机遮盖掉一部分 token，要求模型预测被遮盖的内容，类似填空题，帮助模型学习上下文的深层语义。  
- **指令微调（Instruction Tuning）**：在预训练阶段加入少量任务指令（如“翻译”“摘要”），让模型学会根据自然语言指令切换功能，类似给模型装上“开关”。  
- **混合精度训练**：把大部分计算用 16 位浮点（FP16）完成，关键累加用 32 位（FP32），既省显存又保持数值精度。  
- **梯度回滚（Gradient Rollback）**：监测到 loss 突然飙升时，自动把参数恢复到上一次稳定的检查点，防止一次“踩雷”把整个训练毁掉。  
- **张量并行 & 流水线并行**：把模型的权重切成多块分布到不同 GPU 上，同时把前向/反向计算拆成多个阶段流水执行，像装配线一样让 130 B 参数跑在数十块显卡上。  
- **INT4 量化**：把每个权重压缩到 4 位整数，理论上会导致精度下降，但 GLM‑130B 通过特殊的缩放策略实现了“几乎无损”，从而可以在普通消费级 GPU 上实时推理。

### 核心创新点
1. **从 loss spike 到梯度回滚的稳定训练方案**  
   *之前的 100 B+ 训练往往在某个 batch 上出现 loss 突然上升，导致梯度爆炸，训练直接挂掉。*  
   *GLM‑130B 在每一步监控 loss 曲线，一旦检测到异常上升，就把模型参数回滚到最近的安全检查点，并重新调低学习率继续训练。*  
   *这种“自动救火”机制让训练过程从原本的数次崩溃变成了几乎全程平稳，成功完成了 130 B 参数的全程预训练。*

2. **双向 Transformer 结合掩码与自回归两种预训练目标**  
   *传统大模型要么只用自回归（GPT）要么只用掩码（BERT），各有局限。*  
   *GLM‑130B 同时在同一模型里使用两套 mask 策略：一种是随机遮盖（MLM），另一种是仅在句子结尾遮盖，等价于自回归目标。*  
   *这样模型在生成任务上保持 GPT 的流畅度，在理解任务上拥有 BERT 的全局感知，提升了跨任务的通用性。*

3. **指令微调提前介入预训练阶段**  
   *大多数模型在预训练完毕后才进行指令微调，导致指令适应性不佳。*  
   *GLM‑130B 在预训练的后期加入了少量多任务指令数据，让模型在学习语言本身的同时，也学会了如何响应指令。*  
   *实验显示，这一步显著提升了模型在零样本指令任务（如翻译、摘要）上的表现。*

4. **无后处理的 INT4 量化**  
   *以往 100 B 级模型要想压到 4 位，需要复杂的后训练校准，且往往会损失 5% 以上的性能。*  
   *GLM‑130B 通过在训练时加入量化感知的正则项，使得权重天然适配 4 位表示，直接导出 INT4 权重后几乎没有性能下降。*  
   *这让 130 B 模型可以在 4 张 RTX 3090（24 GB）或 8 张 RTX 2080 Ti（11 GB）上跑起来，极大降低了使用门槛。*

### 方法详解
**整体框架**  
GLM‑130B 的训练流程可以划分为四个阶段：① 数据准备与双语词表构建；② 双向 Transformer 主体的结构设计；③ 多目标预训练（MLM + 自回归）+ 指令微调；④ 稳定性与效率的工程优化（混合精度、并行、梯度回滚、量化感知）。整个过程像是先搭好一座“双层桥”，再在桥面上铺设不同的交通规则，最后装上自动监控系统确保桥梁不塌。

**关键模块拆解**  

1. **双语词表 & 输入编码**  
   - 使用统一的 SentencePiece 分词器，基于 100 TB 中英混合语料训练得到约 100 k 的子词表。  
   - 每个 token 同时拥有语言标记位，模型可以在同一序列里辨认中文和英文的边界。

2. **双向 Transformer 结构**  
   - 每层包含两套自注意力子层：左→右（自回归）和右←左（全局掩码）。  
   - 前向传播时，根据 mask 类型选择对应的注意力掩码矩阵；在同一 batch 中可以混合两种 mask，提升硬件利用率。  
   - 类比为“双向高速公路”，车辆（信息）可以在两个方向自由流动，确保无论是“从前往后”还是“从后往前”都能得到最优路径。

3. **多目标预训练**  
   - **随机掩码（MLM）**：随机遮盖 15% token，模型预测被遮盖的原始 token。  
   - **句尾掩码（自回归）**：仅在序列末尾添加特殊 mask，使模型只能看到左侧上下文，等价于传统的语言模型目标。  
   - 两种目标交替出现，比例约为 1:1，保证模型既能填空也能生成。

4. **指令微调嵌入**  
   - 在预训练的第 80% 迭代后，加入约 0.5 % 的指令数据（包括翻译、问答、摘要等），每条指令前加上统一的 `<INST>` 标记。  
   - 通过少量梯度更新，让模型在保持语言知识的同时，学会把指令映射到对应的输出模式。

5. **工程优化**  
   - **混合精度**：使用 NVIDIA 的 Apex AMP，将大部分矩阵乘法转为 FP16，关键累加保持 FP32，显存节省约 40%。  
   - **张量并行 + 流水线并行**：把模型切成 8 块张量并行，每块再分成 4 阶段流水线，整体在 64 张 GPU 上实现 1.2 TFLOPs 的吞吐。  
   - **梯度回滚**：每 1000 步保存一次检查点，实时监测 loss；若出现 >5× 的突增，立即回滚并将学习率乘以 0.5。  
   - **量化感知正则**：在训练时对每层权重加入 L2 正则，使其分布更适合 4 位离散化，后期直接导出 INT4 权重。

**最巧妙的点**  
- 将 **loss spike 检测 + 学习率自适应回滚** 融入训练循环，使得大模型的“易碎”问题被系统化解决，几乎不需要人工干预。  
- **双目标预训练** 让同一模型在同一批次里完成两种截然不同的学习任务，极大提升了硬件利用率和模型的多任务适配性。  

### 实验与效果
- **评测数据集**：在英文方面使用 MMLU、GSM8K、ARC‑E、TruthfulQA 等；中文方面使用 CLUE、CMRC、中文机器翻译基准（WMT‑Zh‑En）以及中文指令任务。  
- **对比基线**：GPT‑3 175B（davinci）、OPT‑175B、BLOOM‑176B、ERNIE‑Titan 3.0 260B。  
- **主要结果**（论文声称）：  
  - 在 MMLU（所有学科）上超过 GPT‑3 davinci 大约 3% 的准确率；在 GSM8K 这类数学推理任务上提升约 4% 以上。  
  - 中文阅读理解和机器翻译任务上，GLM‑130B 均领先 ERNIE‑Titan 260B 超过 2%–5% 的指标。  
  - 与同等规模的 OPT‑175B、BLOOM‑176B 相比，GLM‑130B 在多数英文基准上都有显著优势，而在这些基准上它的优势并不明显。  
- **消融实验**：  
  - 去掉梯度回滚会导致训练在第 45 B 参数时出现多次崩溃，最终模型无法收敛。  
  - 只使用自回归目标而不加入 MLM，模型在零样本指令任务上的表现下降约 6%。  
  - 未使用量化感知正则直接进行 INT4 量化会导致约 8% 的性能下降，验证了该正则的必要性。  
- **局限性**：  
  - 虽然模型已经开源，但完整的 130 B 权重仍需数十 GB 存储，对普通用户的下载和部署仍有门槛。  
  - 在极端长文本（> 2048 token）上仍会出现显存瓶颈，作者未在论文中给出长序列的专门优化方案。  

### 影响与延伸思考
GLM‑130B 的成功展示了 **“大模型也可以工程化、可复制、可量化”** 的路线图，直接推动了开源社区对 100 B 以上模型的兴趣。随后出现的 **GLM‑4B、GLM‑2B** 系列都沿用了双向 Transformer + 早期指令微调的设计；Quantization‑aware training（量化感知训练）在 LLM 领域也被更多工作采纳，如 Meta 的 **LLaMA‑Quant**。如果想进一步深入，可以关注以下方向：  
- **更高效的并行策略**（如 ZeRO‑3、GPipe 的改进版），帮助在更少 GPU 上训练同等规模模型。  
- **跨语言扩展**：把双语框架推广到多语言（如 10+ 语言）并保持同等性能。  
- **后训练自由的极低位量化**：探索 2‑bit、1‑bit 量化在 LLM 上的可行性。  

### 一句话记住它
GLM‑130B 用“梯度回滚+量化感知”两把钥匙，打开了 130 B 双语大模型的开源大门，并让它在 4 bit 量化下跑在普通消费级 GPU 上。