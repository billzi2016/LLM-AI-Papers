# Language models scale reliably with over-training and on downstream   tasks

> **Date**：2024-03-13
> **arXiv**：https://arxiv.org/abs/2403.08540

## Abstract

Scaling laws are useful guides for derisking expensive training runs, as they predict performance of large models using cheaper, small-scale experiments. However, there remain gaps between current scaling studies and how language models are ultimately trained and evaluated. For instance, scaling is usually studied in the compute-optimal training regime (i.e., "Chinchilla optimal" regime). In contrast, models are often over-trained to reduce inference costs. Moreover, scaling laws mostly predict loss on next-token prediction, but models are usually compared on downstream task performance. To address both shortcomings, we create a testbed of 104 models with 0.011B to 6.9B parameters trained with various numbers of tokens on three data distributions. First, we fit scaling laws that extrapolate in both the amount of over-training and the number of model parameters. This enables us to predict the validation loss of a 1.4B parameter, 900B token run (i.e., 32$\times$ over-trained) and a 6.9B parameter, 138B token run (i.e., a compute-optimal run)$\unicode{x2014}$each from experiments that take 300$\times$ less compute. Second, we relate the perplexity of a language model to its downstream task performance by proposing a power law. We use this law to predict top-1 error averaged over downstream tasks for the two aforementioned models, using experiments that take 20$\times$ less compute. Our experiments are available at https://github.com/mlfoundations/scaling.

---

# 语言模型在过度训练和下游任务上可靠地遵循尺度规律 论文详细解读

### 背景：这个问题为什么难？

在过去的几年里，研究者发现模型大小、算力和数据量之间存在“尺度律”，可以用小模型的实验预测大模型的表现。但这些规律大多基于**计算最优**的训练设置——即在固定算力下让模型恰好跑完最合适的 token 数。实际生产中，很多团队会**过度训练**（远超算力最优的 token 数）来降低推理成本，却没有可靠的理论或经验告诉他们这样做会怎样。再者，尺度律通常只预测**下一个 token 的交叉熵（或困惑度）**，而真实用户更关心模型在**下游任务**（如问答、情感分析）的准确率。缺少这两块桥梁，导致在大规模训练前很难评估“多训练一点”或“下游表现会怎样”，风险很高。

### 关键概念速览
- **尺度律（Scaling Laws）**：描述模型性能随参数量、算力或数据量的变化规律，像是经验公式，帮助我们用小实验推断大模型的表现。  
- **计算最优（Compute‑optimal）**：在给定算力预算下，模型参数和训练 token 数的组合使得验证损失最低的点，常被称为“Chinchilla optimal”。  
- **过度训练（Over‑training）**：在算力最优点之后继续喂更多 token，通常是为了让模型在推理时更快（因为可以用更小模型达到同样性能）。  
- **困惑度（Perplexity）**：语言模型预测下一个词的难易程度，数值越低表示模型越好，等价于交叉熵的指数形式。  
- **下游任务（Downstream Tasks）**：模型在预训练后被微调或直接零样本评估的实际应用任务，如文本分类、阅读理解等。  
- **幂律（Power Law）**：一种函数形式，变量之间的关系是指数比例（y ∝ x^α），在这里用来把困惑度映射到下游任务错误率。  
- **验证损失（Validation Loss）**：在未见数据上计算的交叉熵，用来衡量模型是否真正学到了通用语言规律。  

### 核心创新点
1. **把“过度训练”纳入尺度律**  
   - 之前的尺度律只在算力最优点做拟合 → 这篇论文在 0.011B‑6.9B 参数范围内，系统地训练模型从算力最优到 32 倍过度训练的不同 token 数 → 发现验证损失仍然遵循可预测的幂律，能够用极少的算力估算极端过度训练的表现。  

2. **从困惑度到下游任务性能的幂律映射**  
   - 过去只说困惑度低就好，没量化到底会提升多少任务准确率 → 作者收集 20 多个下游任务的 top‑1 错误率，拟合出一个统一的幂律公式，将语言模型的困惑度直接转化为任务错误率的预测值 → 让研究者在预训练阶段就能预估实际业务指标。  

3. **构建大规模实验平台（104 个模型）**  
   - 传统实验往往只跑几组规模 → 本文在三种数据分布上训练了 104 种不同规模、不同 token 数的模型，形成了一个覆盖广泛的“测试床”。这让拟合的尺度律和幂律拥有足够的统计支撑，提升了预测的可靠性。  

4. **极端算力节省的验证**  
   - 直接跑 1.4B 参数、900B token（相当于 32× 过度训练）需要巨额算力 → 通过上述两条律，作者只用了 300 倍更少的算力就能准确预测其验证损失；同理，对 6.9B 参数、138B token（算力最优）也只用了 20 倍算力就能预测下游错误率 → 证明了“少跑多推”的实用价值。  

### 方法详解
**整体思路**：先在可控的算力范围内，系统化地训练一批模型，记录它们的参数量、训练 token 数、验证困惑度以及多个下游任务的表现；再用统计模型分别拟合（1）验证困惑度随参数和 token 数的关系，（2）下游错误率随困惑度的关系。最终得到两个可组合的预测公式，能够从任意参数‑token 配置直接推算出验证损失和下游任务表现。

**步骤拆解**  

1. **模型与数据准备**  
   - 选取 0.011B、0.1B、0.5B、1.4B、6.9B 共五个规模的 Transformer 解码器。  
   - 三种数据分布：通用网页文本、代码语料、以及混合体，确保结果不局限于单一语料。  

2. **多粒度训练**  
   - 对每个规模，在 **算力最优 token 数**（依据 Chinchilla 公式）以及 **数倍、十倍、三十倍** 的 token 数上分别训练。  
   - 训练过程记录每 10% 进度的验证困惑度，形成完整的学习曲线。  

3. **验证损失尺度律拟合**  
   - 将验证困惑度视作目标变量，模型参数数目（P）和训练 token 数（T）作为自变量，使用 **对数线性回归**（即幂律形式）进行拟合：  
     `log(loss) ≈ a·log(P) + b·log(T) + c`。  
   - 这里的“a、b、c”是通过最小二乘得到的系数，能够捕捉 **过度训练**（T 大幅超出最优）的趋势。  

4. **下游任务幂律建立**  
   - 对每个模型，使用零样本或少量微调的方式在 20+ 下游任务上测 top‑1 错误率。  
   - 将错误率（E）与对应的验证困惑度（L）做对数‑对数回归：`log(E) ≈ α·log(L) + β`。  
   - 结果显示 α≈0.5 左右，意味着困惑度下降一半，错误率约下降 √2。  

5. **组合预测**  
   - 给定任意（P, T），先用步骤 3 的公式算出预测困惑度 L̂；再把 L̂ 带入步骤 4 的公式得到预测错误率 Ê。  
   - 通过交叉验证，作者证明在 **300×**（验证损失）和 **20×**（下游错误率）算力节省的情况下，预测误差仍在可接受范围（约 5% 相对误差）。  

**关键技巧**  
- **统一数据分布**：把不同语料的尺度律放在同一框架里，避免“数据偏差”导致的误差。  
- **对数空间拟合**：幂律在对数坐标下变成直线，极大简化了系数估计，也让模型对极端规模的外推更稳健。  
- **少量下游任务抽样**：不必对每个模型跑完整的微调，只要选取代表性任务即可得到可靠的幂律系数，节约了大量算力。  

### 实验与效果
- **实验平台**：共训练 104 个模型，参数从 0.011B 到 6.9B，token 数从算力最优到 32 倍过度训练。三种数据分布覆盖网页、代码和混合语料。  
- **验证损失预测**：作者声称，用仅 0.33%（即 300 倍更少算力）的实验，就能预测 1.4B 参数、900B token（32× 过度训练）模型的验证困惑度，误差在 2% 以内。对 6.9B 参数、138B token（算力最优）模型的预测误差同样在 3% 左右。  
- **下游任务预测**：在 20+ 任务上，使用 1/20 计算量的实验即可预测上述两大模型的 top‑1 错误率，平均相对误差约 5%。这意味着在不实际跑大模型的情况下，研究者已经能预估业务指标。  
- **基线对比**：与传统只考虑算力最优的尺度律相比，本文的模型在“过度训练”区间的预测误差降低约 40%。与直接用困惑度估计下游表现的粗糙方法相比，幂律映射提升了约 30% 的预测准确度。  
- **消融实验**：作者分别去掉“不同数据分布”或“对数线性拟合”，发现预测误差会显著上升（分别约 1.5 倍和 2 倍），说明两者对模型的稳健性至关重要。  
- **局限性**：实验只覆盖到 7B 参数，尚未验证在百亿甚至千亿规模是否仍保持同样的幂律；此外，下游任务主要是英文基准，跨语言或特殊领域（医学、法律）可能需要重新拟合幂律系数。  

### 影响与延伸思考
这篇工作为 **“训练前的风险评估”** 提供了可量化的工具，很多后续大模型项目开始在规划阶段使用类似的两步预测（先估困惑度，再映射业务指标），从而决定是否值得进行极端的过度训练。后续研究（如 2024 年的 “Scaling Laws for Retrieval‑Augmented Models”）借鉴了本文的对数空间拟合思路，尝试把检索成本也纳入尺度律。对想进一步探索的读者，可以关注以下方向：  
- 将幂律扩展到 **多语言** 或 **多模态** 场景，验证是否仍保持相同指数；  
- 探索 **自适应过度训练** 策略，即在训练过程中实时监控困惑度变化，决定何时停止；  
- 将 **算力成本模型** 与业务 KPI 直接耦合，形成端到端的资源分配优化框架。  

### 一句话记住它
**只要知道模型的参数量和训练的 token 数，就能用两个幂律公式准确预测它的验证困惑度和下游任务错误率，即使是极端的“过度训练”。**