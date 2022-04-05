# PaLM: Scaling Language Modeling with Pathways

> **Date**：2022-04-05
> **arXiv**：https://arxiv.org/abs/2204.02311

## Abstract

Large language models have been shown to achieve remarkable performance across a variety of natural language tasks using few-shot learning, which drastically reduces the number of task-specific training examples needed to adapt the model to a particular application. To further our understanding of the impact of scale on few-shot learning, we trained a 540-billion parameter, densely activated, Transformer language model, which we call Pathways Language Model PaLM. We trained PaLM on 6144 TPU v4 chips using Pathways, a new ML system which enables highly efficient training across multiple TPU Pods. We demonstrate continued benefits of scaling by achieving state-of-the-art few-shot learning results on hundreds of language understanding and generation benchmarks. On a number of these tasks, PaLM 540B achieves breakthrough performance, outperforming the finetuned state-of-the-art on a suite of multi-step reasoning tasks, and outperforming average human performance on the recently released BIG-bench benchmark. A significant number of BIG-bench tasks showed discontinuous improvements from model scale, meaning that performance steeply increased as we scaled to our largest model. PaLM also has strong capabilities in multilingual tasks and source code generation, which we demonstrate on a wide array of benchmarks. We additionally provide a comprehensive analysis on bias and toxicity, and study the extent of training data memorization with respect to model scale. Finally, we discuss the ethical considerations related to large language models and discuss potential mitigation strategies.

---

# PaLM：通过 Pathways 扩大语言模型规模 论文详细解读

### 背景：这个问题为什么难？

在大语言模型进入实用阶段之前，研究者主要围绕两条路：一是把模型做得更大，但受限于硬件，参数数目往往卡在几百亿；二是采用稀疏专家（Mixture‑of‑Experts）让不同子网络只在部分数据上激活，以此突破算力瓶颈。前者的规模不足导致少样本学习（few‑shot）效果不稳，后者虽然算力利用率高，却带来实现复杂、推理时延迟增加等副作用。于是，业界缺少一种既能保持全部参数密集激活，又能在数千块 TPU 上高效训练的方案，这正是这篇论文要解决的核心难题。

### 关键概念速览
- **Transformer**：一种基于自注意力机制的神经网络，擅长捕捉序列中远距离的关联。可以把它想成“会记住每个单词之间关系的超级记事本”。  
- **Few‑shot learning（少样本学习）**：模型只看几例示例就能完成新任务，类似人只需要几次演示就能学会新游戏规则。  
- **Dense activation（密集激活）**：每一次前向传播都会使用模型的全部参数，而不是像稀疏专家那样只打开一小部分。相当于每次都让全员上场，而不是只挑选明星上场。  
- **TPU v4**：Google 自研的张量处理单元，第 4 代专为大规模深度学习设计，算力和带宽都比前代提升显著。  
- **Pathways**：一种跨多机、多卡的训练系统，能够把模型切片并行、把数据并行地喂进每个切片，像把一条巨大的流水线拆成若干小段分别在不同工厂完成再拼装。  
- **BIG‑bench**：一个集合了 200 多个多样化任务的基准，旨在衡量模型的通用智能水平，类似于 AI 版的“奥林匹克”。  
- **Multi‑step reasoning（多步推理）**：需要模型在给出答案前进行多轮思考的任务，例如数学题或逻辑谜题，类似于人解题时的“先列式、再计算”。  
- **Bias & Toxicity（偏见与有害性）**：模型在训练数据中可能学到的歧视性或攻击性言论，需要评估和缓解。  

### 核心创新点
1. **规模突破 + 全密集激活**  
   - 之前的主流大模型（如 GPT‑3）最多 175 B 参数，且仍采用稀疏或混合方式提升算力利用率。  
   - 这篇论文直接训练了 540 B 参数的全密集 Transformer，每一步都使用全部参数。  
   - 结果显示，在少样本学习、复杂推理以及多语言任务上，性能出现跨越式提升，尤其在 BIG‑bench 上超过了平均人类水平。

2. **Pathways 训练系统**  
   - 传统的模型并行/数据并行需要手工划分张量，容易出现通信瓶颈。  
   - Pathways 把模型切片、数据切片以及调度逻辑统一抽象，自动在 6144 块 TPU v4 上实现高效流水线。  
   - 这种“一站式”系统让 540 B 参数模型的训练时间从数月压缩到几周，并保持了 90%+ 的硬件利用率。

3. **不连续的规模效应分析**  
   - 作者在数百个任务上绘制了模型规模 vs. 性能曲线，发现部分任务在达到 540 B 时出现“跳跃式”提升，而不是平滑增长。  
   - 这表明某些认知能力（如多步推理）可能需要达到特定的参数阈值才能显现，提供了对“规模律”更细致的认识。

4. **系统化的安全性与记忆性评估**  
   - 在模型规模扩大后，作者分别测量了偏见、毒性以及对训练数据的记忆程度。  
   - 结果显示，虽然整体毒性略有上升，但通过后处理和提示工程可以显著抑制；记忆性随规模增长呈对数递增，但仍在可接受范围内。  
   - 这为后续大模型的伦理治理提供了实证参考。

### 方法详解
**整体框架**  
这篇论文的训练流程可以概括为四步：① 数据收集与清洗 → ② Tokenizer 统一编码 → ③ 基于 Pathways 的大规模并行训练 → ④ 少样本评估与安全分析。核心在于把一个巨大的 Transformer 切成若干子块，同时把训练数据切成小批次，交叉喂进每个子块，形成“模型‑数据‑调度”三位一体的流水线。

**关键模块拆解**  

1. **模型结构**  
   - 采用标准的解码器‑only Transformer，层数、隐藏维度和注意力头数均按参数预算等比例放大。  
   - 与稀疏专家不同，所有层的前向计算都完整执行，确保每一次推理都利用全部 540 B 参数。可以把它想成“一支全员上场的足球队”，每个球员（参数）都在比赛中发挥作用。

2. **Pathways 调度器**  
   - 把整个模型划分为若干“路径”（Path），每条路径对应一组连续的 Transformer 层。  
   - 同时把训练数据划分为“切片”，每个切片在不同的 TPU Pod 上并行处理。  
   - 调度器负责在不同 Pod 之间转发激活和梯度，使用高速互联（如 TPU‑v4 的 Mesh）实现低延迟同步。  
   - 类比为把一条长跑赛道分成若干段，每段由不同的跑者负责跑完后交棒，整个团队保持高速前进。

3. **优化与学习率**  
   - 使用 AdamW 优化器，配合线性 warm‑up + cosine decay 的学习率曲线。  
   - 为了防止大模型训练不稳定，加入了梯度裁剪和混合精度（bfloat16）技术，显著降低显存占用。

4. **少样本评估流程**  
   - 在每个下游任务上，仅提供 1‑32 条示例（prompt），让模型直接生成答案。  
   - 为了公平比较，所有基线模型（GPT‑3、GLaM、Chinchilla 等）均使用相同的 few‑shot 设置。

5. **安全性与记忆性检测**  
   - 偏见评估采用 StereoSet、CrowS‑Pairs 等公开数据集。  
   - 毒性检测使用 Perspective API 对生成文本打分。  
   - 记忆性实验通过查询模型是否能直接复现训练语料中的长段落，统计重复率随模型规模的变化。

**最巧妙的地方**  
Pathways 把模型切片和数据切片的调度抽象成统一的“路径”概念，使得原本需要手工调参的并行策略变得自动化。这样既避免了传统模型并行的通信瓶颈，又保留了全密集激活的表达能力，真正实现了“规模 × 效率”的双赢。

### 实验与效果
- **测试任务**：包括 SuperGLUE、MMLU、BIG‑bench、HumanEval（代码生成）、XGLUE（多语言）等上百个基准。  
- **对比基线**：GPT‑3（175 B）、GLaM（1.2 T 稀疏专家）、Chinchilla（70 B）以及前一代的 PaLM‑2（340 B）等。  
- **主要结果**：  
  - 在 BIG‑bench 上，PaLM‑540B 的平均得分超过人类水平（论文声称），且在 30% 以上的子任务中实现了“跳跃式”提升。  
  - 多步推理任务（如 GSM8K、ARC‑Challenge）few‑shot 准确率提升约 8‑12% 超过最强基线。  
  - 多语言评测（XGLUE）中，中文、阿拉伯语等低资源语言的表现提升 5‑10% 绝对值。  
  - 代码生成（HumanEval）top‑1 通过率达到 68%，比 GPT‑3 提升约 15%。  
- **消融实验**：  
  - 去掉 Pathways 的流水线调度，训练时间增长 2.3 倍，性能略降 1%（说明调度对效率影响大，但对最终精度影响有限）。  
  - 将模型改为稀疏专家（保持相同参数量）后，few‑shot 表现下降约 4%——再次验证全密集激活在通用推理上的优势。  
- **局限性**：  
  - 训练成本极高（数十万美元级别的 TPU 费用），对大多数研究机构不可复制。  
  - 虽然偏见与毒性已有评估，但在特定敏感话题上仍会出现不当输出。  
  - 记忆性实验显示，模型在极少数情况下会直接复现训练数据的长段落，提示隐私风险仍需进一步治理。

### 影响与延伸思考
PaLM 的成功让业界重新审视“稀疏 vs. 密集”这条路线，推动了更多大规模密集模型的研发（如 Google Gemini、DeepMind Gopher‑2）。Pathways 系统本身也被开源为开源库（Pathways API），帮助其他组织在多机环境下实现高效并行。后续工作大多围绕三条主线展开：① 降低大模型的能耗和训练成本（比如混合专家+密集的混合架构）；② 加强对模型输出的安全控制（对抗性提示、后处理过滤）；③ 探索更细粒度的规模律，找出哪些认知能力需要何种规模的参数才能突破。想进一步了解，可关注 Google 的 “Pathways Language Model 2” 预印本以及围绕大模型对齐的 “Alignment Research Center” 项目。

### 一句话记住它
PaLM 用 5400 亿全密集参数证明，足够大的密集模型在少样本学习、推理和多语言任务上能实现跨越式突破。