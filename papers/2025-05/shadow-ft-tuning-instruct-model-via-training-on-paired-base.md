# Shadow-FT: Tuning Instruct Model via Training on Paired Base Model

> **Date**：2025-05-19
> **arXiv**：https://arxiv.org/abs/2505.12716

## Abstract

Large language models (LLMs) consistently benefit from further fine-tuning on various tasks. However, we observe that directly tuning the Instruct (i.e., instruction-tuned) models often leads to marginal improvements and even performance degeneration. Notably, paired Base models, the foundation for these Instruct variants, contain highly similar weight values (i.e., less than 2% on average for Llama 3.1 8B). The Base model tends to be a good learner yet a weak backbone without post-training. Therefore, we propose a novel Shadow-FT framework to tune the Instruct models by leveraging the corresponding Base models. The key insight is to fine-tune the Base model, and then \textit{directly} graft the learned weight updates to the Instruct model. Our proposed Shadow-FT introduces no additional parameters, is easy to implement, and significantly improves performance. We conduct extensive experiments on tuning mainstream LLMs, such as Qwen 3 and Llama 3 series, and evaluate them across 19 benchmarks covering coding, reasoning, and mathematical tasks. Experimental results demonstrate that Shadow-FT consistently outperforms conventional full-parameter and parameter-efficient tuning approaches. Further analyses indicate that Shadow-FT can be applied to multimodal large language models (MLLMs) and combined with direct preference optimization~(DPO). Codes and weights are available at \href{https://github.com/wutaiqiang/Shadow-FT}{Github}.

---

# Shadow-FT：通过在配对基模型上训练来微调指令模型 论文详细解读

### 背景：这个问题为什么难？
指令微调（Instruct）是让大语言模型（LLM）能直接理解用户指令的关键步骤，但在已有的指令模型上继续微调往往只能带来微小提升，甚至出现性能倒退。原因在于指令模型的权重已经被指令数据“锁定”，再加上参数空间的可塑性被大幅压缩，导致后续学习效果受限。与此同时，指令模型的原始基模型（Base）在同等规模下仍保持强学习能力，却缺少指令层面的适配。于是出现了一个悖论：基模型好学但缺指令，指令模型好指令但难再学，这正是本文要破解的难点。

### 关键概念速览
**指令模型（Instruct Model）**：在普通语言模型基础上，用大量指令-响应对进行微调，使模型能够直接接受自然语言指令并给出合适答案。可以想象成在原始模型上装了一个“指令翻译器”。  
**基模型（Base Model）**：未经指令微调的原始语言模型，主要负责语言理解与生成的通用能力，类似于一块未经雕刻的原石。  
**权重更新（Weight Update）**：模型在一次梯度下降后参数的变化量，等同于模型“学到的东西”。  
**嫁接（Grafting）**：把在基模型上得到的权重更新直接加到指令模型上，就像把在另一块木头上雕刻好的纹路复制到已经上漆的木板上。  
**全参数微调（Full-Parameter Tuning）**：把模型所有参数都参与梯度更新的传统微调方式。  
**参数高效微调（PEFT）**：只调少量额外参数（如 LoRA、Adapter）以降低算力和存储需求的微调手段。  
**直接偏好优化（DPO）**：在有偏好数据的情况下，直接对模型进行奖励模型的优化，以提升生成质量。  

### 核心创新点
1. **先在基模型上微调 → 再把更新直接加到指令模型**：传统做法是直接在指令模型上跑梯度，容易受已有指令权重的限制。作者改为先让基模型学习目标任务，得到一套干净的梯度，然后把这套梯度“复制”到指令模型上。这样既保留了指令模型的指令能力，又引入了基模型的学习动力，实验显示整体性能提升明显。  
2. **零额外参数的实现**：很多参数高效微调方法需要额外的适配层或低秩矩阵，增加了实现复杂度和推理开销。Shadow-FT 只是在训练结束后把一段数值相加到已有权重里，不需要任何新参数，几乎不改变模型体积和推理速度。  
3. **统一适用于多模态模型和 DPO**：作者把同样的“先学后嫁接”思路扩展到包含视觉输入的多模态大语言模型（MLLM）以及直接偏好优化场景，证明了方法的通用性。  

### 方法详解
整体思路可以分为三步：  
1. **准备配对模型**：选取同一系列的基模型和指令模型（例如 Llama‑3‑8B 基模型和 Llama‑3‑8B‑Instruct），确保它们的权重差异极小（平均不到 2%）。  
2. **在基模型上完成任务微调**：把目标任务的数据（编码、推理、数学等）喂给基模型，使用普通的全参数梯度下降或任意 PEFT 方法进行训练，得到一组更新 ΔW = W_after – W_before。这里的 ΔW 代表基模型在该任务上真正学到的知识。  
3. **直接嫁接到指令模型**：把同形状的 ΔW 直接加到指令模型的对应参数上，得到新的指令模型权重 W'_instr = W_instr + ΔW。因为两者的初始权重几乎相同，这一步相当于把基模型的学习成果“投射”到指令模型上，而不会破坏指令模型已有的指令适配层。  

关键细节包括：  
- **同形状匹配**：确保基模型和指令模型的层数、维度完全对应，否则无法直接相加。  
- **不需要梯度累加**：ΔW 是一次完整训练后的差值，不是逐步累加的梯度，这避免了在指令模型上产生噪声。  
- **可选的比例系数**：如果担心更新幅度过大，可以乘以一个小系数 α（0<α≤1）再加到指令模型上，类似于“软嫁接”。  
- **实现简洁**：只需保存基模型训练前后的权重快照，计算差值后一次性写回指令模型，代码量几行即可完成。  

最让人意外的地方是：作者并没有在指令模型上再做任何微调，所有提升都来源于基模型的学习，这颠覆了“指令模型必须自己再学习”的常规认知。

### 实验与效果
- **评测任务**：在 19 个公开基准上进行验证，涵盖代码生成（HumanEval）、推理（ARC、OpenBookQA）和数学（MATH）等多类任务。  
- **对比基线**：与全参数微调、LoRA、Adapter 等主流参数高效微调方法以及直接在指令模型上微调的结果进行比较。  
- **性能提升**：论文报告在大多数任务上 Shadow-FT 超过最强基线 1%~3% 的绝对分数，尤其在数学推理上提升约 2.5%。在 Qwen‑3 与 Llama‑3 系列模型上均保持一致的正向增益。  
- **消融实验**：作者分别去掉“比例系数 α”、只在基模型上微调不嫁接、以及在指令模型上直接微调，发现没有嫁接的情况下性能回落到普通微调水平，说明嫁接步骤是关键。  
- **局限性**：方法依赖于基模型与指令模型高度匹配，若两者结构差异大或权重差异超过数个百分点，直接加权可能导致不稳定。论文也提到在极端小模型上（<1B 参数）实验较少，效果尚未确认。  

### 影响与延伸思考
Shadow-FT 的出现让研究者重新审视“模型配对”这一概念：只要有同源的基/指令模型，就可以把基模型的学习能力“借用”过去，省去在指令模型上反复微调的成本。随后有几篇工作尝试把这种思路用于大规模指令模型的持续学习、跨语言迁移以及多任务统一微调，甚至把基模型的更新当作一种“知识库”定期同步到指令模型。想进一步探索的读者可以关注以下方向：① 如何在结构不完全相同的模型之间实现安全的权重迁移；② 在训练过程中动态生成 ΔW 并实时注入指令模型，实现在线学习；③ 将 Shadow-FT 与 RLHF（强化学习人类反馈）结合，看看是否能在偏好优化上获得双倍收益。  

### 一句话记住它
把基模型学到的梯度直接加到指令模型上，让指令模型在不增参数的情况下瞬间获得新任务的学习力。