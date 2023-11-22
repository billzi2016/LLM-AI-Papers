# LM-Cocktail: Resilient Tuning of Language Models via Model Merging

> **Date**：2023-11-22
> **arXiv**：https://arxiv.org/abs/2311.13534

## Abstract

The pre-trained language models are continually fine-tuned to better support downstream applications. However, this operation may result in significant performance degeneration on general tasks beyond the targeted domain. To overcome this problem, we propose LM-Cocktail which enables the fine-tuned model to stay resilient in general perspectives. Our method is conducted in the form of model merging, where the fine-tuned language model is merged with the pre-trained base model or the peer models from other domains through weighted average. Despite simplicity, LM-Cocktail is surprisingly effective: the resulted model is able to achieve a strong empirical performance in the whole scope of general tasks while preserving a superior capacity in its targeted domain. We conduct comprehensive experiments with LLama and BGE model on popular benchmarks, including FLAN, MMLU, MTEB, whose results validate the efficacy of our proposed method. The code and checkpoints are available at https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail.

---

# LM‑Cocktail：通过模型合并实现语言模型的弹性微调 论文详细解读

### 背景：这个问题为什么难？
预训练语言模型（LLM）在大规模语料上学到丰富的通用知识，但在实际应用时往往需要在特定任务上进一步微调。传统的微调会把模型参数全部拉向目标任务的最优点，结果是模型在该任务上表现提升，却常常在其他通用任务上出现明显退化。之前的解决思路要么是保持原模型不动，要么是使用额外的适配层（如 LoRA）增加参数开销，却仍然难以在不牺牲通用能力的前提下获得专属性能。因此，如何在一次微调后让模型既保持专注任务的优势，又不失通用能力，成为一个亟待突破的难点。

### 关键概念速览
**预训练语言模型**：在海量通用文本上训练得到的模型，拥有广泛的语言理解和生成能力。类似于一个“全能学生”，可以应付各种学科的基础题目。  
**微调**：在特定任务的数据上继续训练预训练模型，使其在该任务上表现更好。相当于让全能学生专攻某门课程的强化训练。  
**性能退化**：微调后模型在原本擅长的通用任务上表现下降。就像学生只顾专攻数学，导致语文成绩下滑。  
**模型合并**：把两个或多个模型的参数按照一定比例进行线性组合，得到一个新的模型。可以想象把两位老师的教学经验混合成一本新教材。  
**加权平均**：在模型合并时为每个模型分配一个权重，权重越大该模型的参数贡献越多。类似于在团队讨论中给不同成员的发言分配不同的影响力。  
**同域模型 / 跨域模型**：同域模型指在相似任务或数据上微调得到的模型，跨域模型则来自完全不同的任务领域。前者像是同学的笔记，后者像是别专业的参考书。  
**鲁棒性**：模型在面对未见任务或分布变化时仍能保持稳定性能。相当于学生的“应变能力”。  
**基准测试（Benchmark）**：公开的评测集合，用来统一衡量模型在不同任务上的表现，如 FLAN、MMLU、MTEB。  

### 核心创新点
1. **直接在参数空间做加权平均 → 通过对微调模型和基模型（或其他领域模型）进行线性组合 → 在保持专属任务优势的同时，显著恢复或提升通用任务的表现**。这一步不需要额外的适配层或再训练，操作极其简洁。  
2. **使用小规模目标验证集来决定合并权重 → 在该验证集上评估每个候选模型的得分，然后把得分转化为权重比例 → 权重自然倾向于在目标任务上表现更好的模型**。这样避免了人工调参，权重由数据驱动。  
3. **把模型合并视为“一杯调和鸡尾酒”，而不是“硬性拼接” → 通过加权平均实现平滑的参数折中，而不是简单的参数覆盖 → 使得模型在参数空间中找到一个兼顾多任务的中间点**。这种思路在语言模型微调领域尚属首次系统化提出。  
4. **在多个主流大模型（LLama、BGE）和多样化基准上做全方位验证 → 结果显示合并后模型在 FLAN、MMLU、MTEB 等通用任务上均显著优于单纯微调模型，同时在目标任务上仍保持领先 → 证明方法的普适性和实用性**。  

### 方法详解
整体思路可以拆成四个步骤：

1. **准备候选模型**  
   - 先对预训练基模型进行目标任务的微调，得到 **微调模型**。  
   - 再挑选一个或多个 **参考模型**，可以是原始基模型，也可以是其他领域已经微调好的模型（同域或跨域）。

2. **在目标验证集上评估**  
   - 用一个规模不大的验证集（通常是目标任务的开发集）分别跑通每个候选模型，记录它们的任务得分（如准确率、F1 等）。  
   - 这些得分直接反映了模型在目标任务上的相对实力。

3. **计算合并权重**  
   - 将每个模型的得分归一化为概率分布，得分高的模型对应更大的权重。  
   - 若有 N 个模型，权重 w₁,…,w_N 满足 Σw_i = 1。  
   - 这一步相当于让验证集“投票”，决定每个模型在最终混合体中的发言力度。

4. **参数加权平均**  
   - 对每一层的参数张量（权重矩阵、偏置向量等），按照对应的权重做线性组合：  
     `θ_final = Σ w_i * θ_i`，其中 θ_i 是第 i 个模型的参数。  
   - 合并过程不改变模型的结构，也不需要再进行梯度更新。  
   - 最终得到的 **LM‑Cocktail** 即为融合后的模型，可直接用于推理或进一步微调。

**最巧妙的地方**在于：传统上人们认为微调会把模型“拉偏”，导致通用知识被抹掉，而这里通过一次线性插值就能把通用知识“拉回”。权重的自动化决定让整个过程无需人工调参，几乎可以“一键”完成。

### 实验与效果
- **实验对象**：作者选用了两类主流模型——开源的 LLaMA 系列和 BGE（基于嵌入的检索模型），分别在它们的微调版本上进行合并。  
- **评测任务**：覆盖了指令微调基准 FLAN、通用知识测评 MMLU（Multi‑Task Language Understanding）以及检索/嵌入评测套件 MTEB。  
- **对比基线**：包括（1）纯微调模型、（2）仅使用基模型（未微调）、（3）常见的参数适配方法（如 LoRA）等。  
- **结果概述**：论文声称 LM‑Cocktail 在所有通用基准上均显著超过纯微调模型，且在目标任务上仍保持或略有提升。例如在 FLAN 上的整体得分提升了数个百分点，在 MMLU 上的准确率几乎恢复到基模型水平。  
- **消融实验**：作者分别去掉参考模型、只使用基模型或只使用同域模型进行合并，发现加入跨域模型可以进一步提升通用任务的鲁棒性，验证了多模型融合的价值。  
- **局限性**：合并权重依赖于目标验证集的质量；若验证集过小或分布偏差，得到的权重可能不够稳健；此外，极大规模模型（数百亿参数）上直接加权平均的内存开销仍是一个实际挑战。  

### 影响与延伸思考
LM‑Cocktail 把“模型合并”从实验室玩具提升为一种实用的微调后恢复通用能力的工具。随后出现的工作如 **Model Soups**、**Weight Interpolation for Multi‑Task Learning** 等，都在不同场景下借鉴了加权平均的思路，进一步探索如何在参数空间中寻找多任务的折中点。对想深入的读者，可以关注以下方向：  
- **自适应权重学习**：让模型在训练过程中自动学习最优合并比例，而不是依赖验证集。  
- **跨模态合并**：把语言模型与视觉、音频模型的参数进行融合，探索多模态“一杯鸡尾酒”。  
- **大模型高效合并**：研发分块或低秩近似的合并技术，降低内存占用。  

### 一句话记住它
只要把微调模型和原始（或其他领域）模型的参数按目标任务表现加权平均，就能让语言模型在专注任务的同时保持通用能力。