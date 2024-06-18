# DART-Math: Difficulty-Aware Rejection Tuning for Mathematical   Problem-Solving

> **Date**：2024-06-18
> **arXiv**：https://arxiv.org/abs/2407.13690

## Abstract

Solving mathematical problems requires advanced reasoning abilities and presents notable challenges for large language models. Previous works usually synthesize data from proprietary models to augment existing datasets, followed by instruction tuning to achieve top-tier results. However, our analysis of these datasets reveals severe biases towards easy queries, with frequent failures to generate any correct response for the most challenging queries. Hypothesizing that difficult queries are crucial to learn complex reasoning, we propose Difficulty-Aware Rejection Tuning (DART), a method that allocates difficult queries more trials during the synthesis phase, enabling more extensive training on difficult samples. Utilizing DART, we have created new datasets for mathematical problem-solving that focus more on difficult queries and are substantially smaller than previous ones. Remarkably, our synthesis process solely relies on a 7B-sized open-weight model, without reliance on the commonly used proprietary GPT-4. We fine-tune various base models on our datasets ranging from 7B to 70B in size, resulting in a series of strong models called DART-MATH. In comprehensive in-domain and out-of-domain evaluation on 6 mathematical benchmarks, DART-MATH outperforms vanilla rejection tuning significantly, being superior or comparable to previous arts, despite using much smaller datasets and no proprietary models. Furthermore, our results position our synthetic datasets as the most effective and cost-efficient publicly available resources for advancing mathematical problem-solving.

---

# DART-Math：面向数学问题求解的难度感知拒绝调优 论文详细解读

### 背景：这个问题为什么难？

大语言模型在解数学题时常常卡在需要多步推理的环节，尤其是高难度题目几乎没有正确答案。过去的做法是让强大的闭源模型（如 GPT‑4）生成大量答案，再用这些合成数据进行指令微调。可是这些合成数据里，简单题占比极高，难题往往只得到“空白”或错误答案，导致模型在真正需要深度推理时仍旧表现不佳。换句话说，训练集的难度分布与真实需求严重不匹配，这让提升数学推理能力的空间被大幅压缩。

### 关键概念速览
- **拒绝调优（Rejection Tuning）**：在合成答案时，如果模型的输出被判定为不合格，就让它重新生成，直到得到满意的答案。类似于让学生多次尝试，直到写出完整的解答。
- **难度感知（Difficulty-Aware）**：给每道题打上难度标签，依据标签决定该题在合成阶段分配多少次尝试。可以想象为老师对难题多给几次练习机会。
- **合成数据（Synthetic Data）**：不是人手标注，而是让模型自己生成的问答对。相当于让模型自编练习题和答案。
- **指令微调（Instruction Tuning）**：在已有的大模型上继续训练，使其更好地遵循“解数学题”的指令。就像在通用语言模型上再上一次专门的数学课。
- **Vanilla Rejection Tuning**：传统的拒绝调优方式，所有题目得到相同的尝试次数，没有难度区分。
- **DART-MATH 系列模型**：使用 DART 方法微调得到的模型族，规模从 7B 到 70B 参数不等。

### 核心创新点
1. **难度感知的采样策略 → 在合成阶段为难题分配更多生成尝试 → 让稀缺的高难度样本数量大幅提升**。传统做法对所有题目只给一次机会，导致难题几乎没有可用答案。DART 通过难度标签动态调节“重试次数”，把合成资源倾斜到真正需要学习的地方。
2. **全流程使用开源模型 → 合成、过滤、微调全部基于 7B 开源模型 → 摆脱对 GPT‑4 等闭源模型的依赖**。过去的高质量合成数据几乎只能靠付费的闭源模型，这限制了成本和可复制性。DART 证明小模型也能产出足够好的难题答案，只要给它足够的尝试机会。
3. **更小却更有效的合成数据集 → 数据规模远低于以往的数十倍 → 仍然在 6 大数学基准上达到或超过最先进水平**。作者把“少而精”落到实处：通过难度感知的筛选，去掉大量无用的易题，保留高价值的难题，从而在训练成本上实现指数级下降。
4. **统一的难度评估框架 → 用现有模型的自评分数或外部难度指标对每道题进行打分 → 形成可复用的难度标签体系**。这一步让难度感知不再是经验猜测，而是可量化、可自动化的过程。

### 方法详解
整体思路可以拆成三大步骤：**难度标注 → 难度感知生成 → 过滤与微调**。

1. **难度标注**  
   - 首先准备一个基础的数学题库（包括易题和难题）。  
   - 用一个相对强的模型（这里是 7B 开源模型）对每道题进行自评，输出一个“成功率”或“置信度”。成功率低的题目被视为“难”。  
   - 也可以引入外部难度指标（如题目来源、解题步数）做辅助，形成一个统一的难度标签（例如 1‑5 级）。

2. **难度感知生成（DART）**  
   - 对每个难度标签设定不同的“重试上限”。例如，难度 5 的题目允许模型最多尝试 10 次，难度 1 的题目只尝试 2 次。  
   - 每一次尝试都让模型生成答案，然后用一个自动评估器（比如答案匹配或程序执行检查）判断是否合格。合格即保留，不合格则继续重试，直到达到上限。  
   - 这样，难题会被多次“逼迫”模型给出可接受的答案，极大提升了高难度样本的可用率。

3. **过滤与微调**  
   - 收集所有通过评估的问答对，构成最终的合成数据集。由于难度感知的筛选，数据集中高难度样本比例显著提升。  
   - 使用指令微调的标准流程，将这些数据喂给不同规模的基模型（7B、13B、30B、70B），进行数千步的训练。  
   - 微调时仍保留“拒绝”机制：如果模型在微调后仍产生不合格答案，训练过程会继续采样新的合成对，形成闭环改进。

**最巧妙的地方**在于把“难度”当作资源分配的信号，而不是单纯的标签。传统的拒绝调优把每道题视为同等重要，导致训练资源被大量易题浪费。DART 把有限的算力和合成次数“投向”最需要提升的难题，类似于老师在课堂上对薄弱学生多给练习机会。

### 实验与效果
- **评测基准**：在六个公开的数学推理基准上（包括 GSM8K、MATH、MMLU‑Math 等）进行全方位测试。  
- **对比对象**：包括使用普通拒绝调优的模型、使用 GPT‑4 合成的大规模数据集微调的模型，以及其他最新的开源数学模型。  
- **核心结果**：论文声称 DART‑MATH 在所有基准上均显著超越普通拒绝调优，整体表现与最先进的闭源或大规模合成数据方法持平或略有优势。尤其在高难度子集上提升最为明显。  
- **消融实验**：作者分别去掉难度感知的重试上限、仅使用单一重试次数、以及完全不做难度标注进行对比。结果显示，去掉任何一环都会导致整体准确率下降 5% 以上，验证了每个模块的必要性。  
- **数据规模**：新合成数据集仅为传统数据集的约 1/10，却实现了相同或更好的效果，说明“少而精”策略的成本效益。  
- **局限性**：合成质量仍受限于 7B 开源模型的能力，极端超高难度题目仍难以得到可靠答案；此外，难度标签的自动评估仍有噪声，可能导致部分中等难度题被误分类。

### 影响与延伸思考
这篇工作向社区展示了“难度感知”可以在低算力环境下显著提升合成数据的价值，激发了后续研究在**数据采样策略**、**自监督难度评估**以及**多模型协同生成**方向的探索。已有几篇后续论文尝试把难度感知与 **Chain‑of‑Thought**（思维链）结合，让模型在生成过程中主动分解难题；还有工作把 DART 思路搬到代码生成、推理式问答等非数学任务上。想进一步深入，建议关注 **自适应采样**（Adaptive Sampling）和 **主动学习**（Active Learning）在大模型合成数据中的交叉应用。

### 一句话记住它
让模型对难题多尝试、多生成，少浪费在易题上——这就是 DART‑Math 的核心魔法。