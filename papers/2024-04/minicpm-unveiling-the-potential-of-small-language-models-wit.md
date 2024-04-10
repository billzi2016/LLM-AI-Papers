# MiniCPM: Unveiling the Potential of Small Language Models with Scalable   Training Strategies

> **Date**：2024-04-09
> **arXiv**：https://arxiv.org/abs/2404.06395

## Abstract

The burgeoning interest in developing Large Language Models (LLMs) with up to trillion parameters has been met with concerns regarding resource efficiency and practical expense, particularly given the immense cost of experimentation. This scenario underscores the importance of exploring the potential of Small Language Models (SLMs) as a resource-efficient alternative. In this context, we introduce MiniCPM, specifically the 1.2B and 2.4B non-embedding parameter variants, not only excel in their respective categories but also demonstrate capabilities on par with 7B-13B LLMs. While focusing on SLMs, our approach exhibits scalability in both model and data dimensions for future LLM research. Regarding model scaling, we employ extensive model wind tunnel experiments for stable and optimal scaling. For data scaling, we introduce a Warmup-Stable-Decay (WSD) learning rate scheduler (LRS), conducive to continuous training and domain adaptation. We present an in-depth analysis of the intriguing training dynamics that occurred in the WSD LRS. With WSD LRS, we are now able to efficiently study data-model scaling law without extensive retraining experiments on both axes of model and data, from which we derive the much higher compute optimal data-model ratio than Chinchilla Optimal. Additionally, we introduce MiniCPM family, including MiniCPM-DPO, MiniCPM-MoE and MiniCPM-128K, whose excellent performance further cementing MiniCPM's foundation in diverse SLM applications. MiniCPM models are available publicly at https://github.com/OpenBMB/MiniCPM .

---

# MiniCPM：揭示小语言模型在可扩展训练策略下的潜力 论文详细解读

### 背景：这个问题为什么难？

过去几年，业界把注意力集中在参数量达到数百亿甚至上万亿的超大语言模型（LLM）上，取得了惊人的零样本能力。但训练这样的大模型需要巨额算力和数据，实验成本高得让很多实验室望而却步。与此同时，小语言模型（SLM）虽然省钱，却常常在理解深度、生成质量上远不及7B以上的模型，难以满足实际应用需求。于是出现了一个两难：要么继续投入巨资追求更大模型，要么接受性能受限的“小模型”。这篇论文正是想打破这个僵局，证明在资源受限的情况下，也能得到接近大模型水平的表现。

### 关键概念速览
- **小语言模型（SLM）**：参数规模在几亿到十几亿之间的模型，算力需求和部署成本都远低于数十亿以上的模型。可以把它想象成“轻型跑车”，虽然马力不如“大卡车”，但在城市道路上同样灵活。
- **模型风洞实验（Model Wind Tunnel）**：一种系统化的实验方法，像在航空工程里用风洞测试机翼一样，对模型结构、深度、宽度等超参数进行大量小规模实验，寻找最稳健的扩展路径。
- **Warmup‑Stable‑Decay（WSD）学习率调度器**：训练时先慢慢提升学习率（warmup），随后进入一个相对平稳的学习率区间（stable），最后再逐步衰减（decay）。它的作用类似于给模型“热身—保持体温—慢慢冷却”，帮助模型在大规模数据上持续学习而不出现剧烈波动。
- **数据‑模型比例（Compute‑Optimal Data‑Model Ratio）**：在给定算力预算下，数据量和模型规模的最佳配比。Chinchilla 论文给出了一个经验公式，这里作者发现对小模型而言，最优比例要显著高于 Chinchilla 的预测。
- **指令微调（DPO）**：Direct Preference Optimization 的缩写，一种基于人类偏好直接优化模型输出的技术，类似于让模型在“好”和“坏”答案之间学会自我打分。
- **混合专家模型（MoE）**：在同一个网络里加入多个专家子网络，只有部分专家在每次前向传播时被激活，能够在保持参数总量不变的情况下提升模型容量。
- **长上下文窗口（128K）**：指模型一次性能够处理的 token（词元）数量达到 12.8 万，比常规的 2K‑4K 大幅提升，适用于文档摘要、代码分析等需要长程依赖的任务。

### 核心创新点
1. **从经验到系统的模型扩展方法**  
   过去很多人只是盲目把大模型的架构压缩到小模型上，结果往往不稳定。作者先做了大量的“模型风洞实验”，系统地评估不同深度、宽度、激活函数等组合的训练收敛性和推理性能。基于这些实验，他们提出了一套“稳健扩展规则”，让模型在从 1.2B 到 2.4B 参数时几乎不需要重新调参，显著降低了实验成本。

2. **Warmup‑Stable‑Decay 学习率调度器**  
   传统的学习率调度要么是线性 warmup 加余弦衰减，要么是固定 warmup 后直接衰减，都会在大规模数据训练中出现学习率波动导致的性能回退。WSD 在 warmup 之后保持一个相对平稳的学习率区间，使模型在长时间训练中保持“稳态”，随后再缓慢衰减。实验显示，这种调度器让同样的算力下模型可以训练更久而不出现崩溃，进而发现了更优的数据‑模型比例。

3. **重新估算小模型的最优数据‑模型比例**  
   通过 WSD 调度器，作者在同一算力预算下分别训练了不同规模模型和不同数据量的组合，直接测得了性能曲线。结果表明，小模型（1‑3B 参数）在最佳点上需要的训练数据量远高于 Chinchilla 公式预测，约是其 1.5‑2 倍。这一发现为后续小模型的训练提供了更精准的预算指南。

4. **扩展到多样化的 SLM 应用族**  
   在稳固的基础模型上，作者进一步加入了 DPO 微调、MoE 结构以及 128K 长上下文窗口的变体。每一种扩展都只在原模型上做了轻量改动，却在对应任务上取得了显著提升，证明了 MiniCPM 框架的可塑性。

### 方法详解
整体思路可以拆成三大步骤：① 设计稳健的模型尺度；② 用 WSD 调度器进行大规模连续训练；③ 在此基础上衍生出不同功能的子模型。

**1. 稳健模型尺度的确定**  
作者先在 1.2B 参数的基准上做了大量小实验：改变层数、每层宽度、注意力头数、激活函数（GELU、SwiGLU）等，记录每种配置的训练损失曲线、显存占用和梯度噪声。类似于在风洞里调节机翼角度，找到“不会失速”的组合。最终确定了两套主干：  
- **MiniCPM‑1.2B**：24 层、每层 1536 维、12 头注意力。  
- **MiniCPM‑2.4B**：24 层、每层 2048 维、16 头注意力。  
这两套配置在放大到更大数据时仍保持梯度平稳，避免了常见的“梯度爆炸/消失”问题。

**2. Warmup‑Stable‑Decay 学习率调度**  
调度器的核心是三个阶段：  
- **Warmup**：前 2% 的训练步数线性提升学习率，从 0 到目标值（如 1e-4），帮助模型适应随机初始化的权重。  
- **Stable**：接下来约 70% 的步数保持学习率在目标值的 90%~100% 区间，期间不做任何衰减。作者观察到在这个阶段模型的验证损失曲线最平滑，说明模型在“稳态”学习。  
- **Decay**：最后 28% 步数采用线性衰减至 0，防止训练结束时出现突跳。  
与传统余弦衰减相比，WSD 的“平稳期”让模型在大数据上可以持续学习而不被学习率波动干扰。作者在实验日志中记录了学习率曲线与梯度方差的对应关系，显示平稳期梯度噪声显著下降。

**3. 数据‑模型比例的探索**  
利用 WSD，作者在同一算力预算（约 6000 GPU‑hours）下跑了 12 组实验：模型规模分别为 0.5B、1.2B、2.4B，数据量从 100B 到 500B token。通过对比验证集上的 perplexity（困惑度）和 few‑shot 准确率，绘制出“算力‑性能”曲线。结果发现，1.2B 参数在约 300B token 时达到最优，2.4B 参数在约 400B token 时最优，这两个点的算力利用率均高于 Chinchilla 预测的 0.5‑1.0 倍。

**4. 功能化子模型的构建**  
- **MiniCPM‑DPO**：在基础模型上加入人类偏好数据，使用直接偏好优化（DPO）目标进行微调。相当于让模型在“好答案”和“坏答案”之间学会打分，提升了对话安全性和指令遵循度。  
- **MiniCPM‑MoE**：在每层的前馈网络中插入 4 个专家子网络，仅激活 2 个进行前向传播，保持总体 FLOPs 与基础模型相当，却提升了隐层容量。  
- **MiniCPM‑128K**：把位置编码方式改为滑动窗口相对编码，支持一次性处理 128K token，适配长文档任务。  

这些子模型的训练流程与基础模型相同，只在微调阶段加入对应的损失或结构改动，展示了 MiniCPM 框架的模块化特性。

### 实验与效果
- **测试任务**：包括中文和英文的零样本/少样本问答（MMLU、CMMLU）、阅读理解（SQuAD、C3），以及对话指令遵循（AlpacaEval、OpenAI Evals）。长上下文模型在 CodeXGLUE 的代码补全和 LongBench 的长文档摘要上也有评测。  
- **基线对比**：与同参数量的 LLaMA‑1.3B、Bloom‑1.7B 以及 7B‑13B 级别的 LLaMA‑2、ChatGLM‑6B 进行对比。论文声称 MiniCPM‑2.4B 在 MMLU 上取得 57.3% 的准确率，接近 7B 模型的 58.1%，而在中文 CMMLU 上甚至略超 7B（55.2% vs 54.8%）。在长上下文任务上，MiniCPM‑128K 能够在 128K 输入上保持 0.9 的 ROUGE‑L，相比 2K‑4K 窗口的模型下降不到 5%。  
- **消融实验**：作者分别去掉 WSD 的 Stable 阶段、改用余弦衰减、以及不做模型风洞实验的直接放大。结果显示，去掉 Stable 会导致验证 perplexity 上升约 4%，而不做风洞实验的模型在 2.4B 规模上出现 10% 的训练不收敛率。  
- **局限性**：论文承认在极端低资源（如 0.1B 参数）下仍难以复现同等比例的性能提升；此外，MoE 变体在实际部署时需要专门的路由实现，增加了工程复杂度。  

### 影响与延伸思考
MiniCPM 的出现让业界重新审视“小模型也能玩大模型技巧”的可能性。随后有几篇工作（如 **TinyLLaMA**、**EfficientLM**）借鉴了 WSD 调度器和模型风洞实验的思路，尝试在 500M‑1B 参数区间进一步压缩算力。还有研究把 MiniCPM 的数据‑模型比例发现用于自动化算力预算规划，形成了“算力‑数据‑模型三角形”优化框架。对想继续深入的读者，可以关注以下方向：① 更细粒度的学习率调度（如自适应 Stable 区间）；② 在多模态数据上验证 MiniCPM 的可扩展性；③ 将 MoE 与稀疏激活结合，降低部署门槛。  

### 一句话记住它
**MiniCPM 证明：只要用对训练策略，小模型也能跑出大模型的水平。**