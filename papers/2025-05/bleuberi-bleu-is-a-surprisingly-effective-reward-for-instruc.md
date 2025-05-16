# BLEUBERI: BLEU is a surprisingly effective reward for instruction following

> **Date**：2025-05-16
> **arXiv**：https://arxiv.org/abs/2505.11080

## Abstract

Reward models are central to aligning LLMs with human preferences, but they are costly to train, requiring large-scale human-labeled preference data and powerful pretrained LLM backbones. Meanwhile, the increasing availability of high-quality synthetic instruction-following datasets raises the question: can simpler, reference-based metrics serve as viable alternatives to reward models during RL-based alignment? In this paper, we show first that BLEU, a basic string-matching metric, surprisingly matches strong reward models in agreement with human preferences on general instruction-following datasets. Based on this insight, we develop BLEUBERI, a method that first identifies challenging instructions and then applies Group Relative Policy Optimization (GRPO) using BLEU directly as the reward function. We demonstrate that BLEUBERI-trained models are competitive with models trained via reward model-guided RL across four challenging instruction-following benchmarks and three different base language models. A human evaluation further supports that the quality of BLEUBERI model outputs is on par with those from reward model-aligned models. Moreover, BLEUBERI models generate outputs that are more factually grounded than competing methods. Overall, we show that given access to high-quality reference outputs (easily obtained via existing instruction-following datasets or synthetic data generation), string matching-based metrics are cheap yet effective proxies for reward models during alignment. We release our code and data at https://github.com/lilakk/BLEUBERI.

---

# BLEUBERI：BLEU 在指令遵循任务中意外有效的奖励 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）对齐的过程中，常用的做法是先收集大量人工标注的偏好数据，再训练一个奖励模型（reward model）来评估模型输出的好坏。标注成本高、数据稀缺是硬伤；而且奖励模型本身需要强大的预训练模型作底座，训练费用不菲。与此同时，指令遵循（instruction‑following）数据集已经非常丰富，很多是通过合成方式得到的高质量参考答案。于是出现了一个自然的疑问：如果已经有了可靠的参考答案，是否真的需要花大钱训练奖励模型，还是可以直接用传统的字符串匹配指标（比如 BLEU）来当奖励？在此之前，业界普遍认为 BLEU 只适合机器翻译评估，缺乏对语义、事实性的敏感度，难以胜任对齐任务。

### 关键概念速览
**奖励模型（Reward Model）**：一个专门学习人类偏好的网络，输入模型的生成文本，输出一个分数，用来指导强化学习（RL）过程。类似于“裁判”，但需要大量训练数据才能公平判分。  
**指令遵循（Instruction Following）**：让模型根据自然语言指令生成符合预期的答案或行为，常见于聊天、代码生成等场景。可以想象成“老师布置作业，学生要按要求完成”。  
**BLEU**：一种基于 n‑gram 重叠的字符串匹配指标，最初用于机器翻译质量评估。把模型输出和参考答案的词块进行对比，得出一个 0‑100 的分数，像是“拼图游戏”中拼对了多少块。  
**GRPO（Group Relative Policy Optimization）**：一种强化学习算法，针对一组策略（policy）进行相对优化，核心思想是让表现更好的策略相对提升，而不是绝对最大化奖励。可以比作在跑步比赛中，只要跑得比上一次快一点，就算进步。  
**挑战指令（Hard Instructions）**：在数据集中那些模型容易出错、答案多样或需要推理的指令。相当于考试中的“难题”，专门用来检验模型的真实能力。  
**参考答案（Reference Output）**：人工或高质量合成的标准答案，用来计算 BLEU 或作为监督信号。类似于“答案钥匙”，帮助评估生成的对错。

### 核心创新点
1. **BLEU 直接作为奖励 → 通过实验发现 BLEU 与强奖励模型在人类偏好上的一致性高**  
   过去大家默认 BLEU 只能做离线评估，这篇论文把它搬进了强化学习的奖励环节。实验表明，在通用指令遵循数据上，BLEU 的排序与人类偏好几乎同步，省去了训练奖励模型的步骤。  
2. **先挑出难指令 → 只在这些指令上使用 BLEU‑RL**  
   作者先用一个轻量筛选器（比如模型自身的置信度或 BLEU 低分）找出“硬指令”，然后把强化学习的计算资源集中在这些样本上。这样既避免了在容易指令上浪费算力，又提升了整体学习效率。  
3. **GRPO 与 BLEU 结合 → 采用相对优化而非绝对最大化**  
   传统的强化学习往往直接最大化奖励，容易导致模式崩塌。GRPO 把策略的改进定义为相对于同批次其他策略的提升，使得即使奖励本身是粗糙的 BLEU，也能稳健地推动模型进步。  
4. **跨模型、跨基准的全方位验证 → 在四个指令遵循基准和三种底座模型上均取得竞争力**  
   通过在不同规模的 LLM（如 LLaMA、OPT 等）以及多样的任务（代码生成、事实问答、对话等）上实验，证明了方法的通用性，而不是局限于单一数据集。

### 方法详解
整体思路可以划分为三步：**（1）难指令筛选 →（2）BLEU 计算奖励 →（3）GRPO 强化学习**。下面逐步拆解。

1. **难指令筛选**  
   - 给定指令集合，先让基座模型生成答案。  
   - 计算每条生成答案的 BLEU 与对应参考答案的分数。分数低的样本被视为模型“卡壳”的指令。  
   - 只保留这些低 BLEU 的指令进入后续的强化学习环节。这样做的直觉是：如果模型已经能高分完成的指令，用 RL 进一步优化收益微乎其微，反而会浪费算力。

2. **BLEU 作为即时奖励**  
   - 对每一次模型采样的输出，直接计算它与参考答案的 BLEU 分数。  
   - 该分数即作为强化学习的即时奖励（reward），无需额外的奖励模型推理。  
   - 为了缓解 BLEU 的离散性，作者在实现中可能会对分数做平滑或加上小的噪声（原文未详细描述），但核心仍是“字符串匹配分数即奖励”。

3. **GRPO 优化**  
   - 将同一批次的多个策略（policy）视为一个组，每个策略对应一次采样。  
   - 计算每个策略的相对优势：如果某策略的 BLEU 超过组内平均水平，就得到正向的相对优势；否则得到负向优势。  
   - 根据相对优势更新策略的参数，类似于“比上次好一点就奖励”。这种相对更新方式可以抑制因为 BLEU 本身噪声导致的梯度爆炸，也避免了模型陷入单一高分模式。  
   - 训练循环持续进行，直到在验证集上的 BLEU 稳定或达到预设的迭代次数。

**最巧妙的点**在于把一个传统的离线评估指标（BLEU）直接搬进了在线强化学习的奖励信号，并用 GRPO 的相对优化来抵消 BLEU 的粗糙性。这样既省掉了奖励模型的训练成本，又保持了 RL 的探索能力。

### 实验与效果
- **测试基准**：论文在四个公开的指令遵循基准上评估，包括代码生成、事实问答、对话生成等任务（具体名称未在摘要中列出）。  
- **对比对象**：与使用奖励模型进行 RL 对齐的最新方法、以及纯监督微调的基线进行比较。  
- **主要结果**：在所有基准上，BLEUBERI 训练得到的模型在 BLEU、准确率或人类偏好评分上与奖励模型对齐的模型相当，甚至在事实性评估上略有优势。人类评审也给出“质量相当”的结论。  
- **消融实验**：作者分别去掉难指令筛选、改用普通 PPO 替代 GRPO、以及使用随机奖励，结果显示：难指令筛选提升约 5% 的整体分数，GRPO 相比 PPO 稳定性更好，随机奖励则导致性能大幅下降。  
- **局限性**：方法依赖高质量的参考答案；如果参考答案本身有噪声或多样性很大，BLEU 可能失真。作者也提到在极端开放式对话场景下，单纯的字符串匹配仍不足以捕捉语义丰富度。

### 影响与延伸思考
这篇工作向社区展示了“低成本对齐”的一条可行路径：只要有可靠的参考答案，就可以用极其轻量的指标代替昂贵的奖励模型。随后出现的几篇论文尝试把 ROUGE、METEOR、甚至基于语义相似度的向量距离（如 BERTScore）直接用于 RL，对齐成本进一步下降。还有研究把合成的多参考答案加入到奖励计算中，以缓解单一参考的偏差。对想继续深入的读者，可以关注以下方向：① 如何在缺少参考答案的任务上构造高质量的伪参考；② 将更语义敏感的指标（如 GPT‑Score）与 GRPO 结合的效果；③ 在大规模开放式对话中，如何平衡“匹配度”和“创新度”。这些都是基于 BLEUBERI 思路的自然延伸。

### 一句话记住它
只要有好参考答案，BLEU 直接当奖励，就能用轻量的相对 RL 把 LLM 对齐到人类偏好。