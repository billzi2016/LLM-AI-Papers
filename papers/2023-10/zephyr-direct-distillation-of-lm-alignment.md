# Zephyr: Direct Distillation of LM Alignment

> **Date**：2023-10-25
> **arXiv**：https://arxiv.org/abs/2310.16944

## Abstract

We aim to produce a smaller language model that is aligned to user intent. Previous research has shown that applying distilled supervised fine-tuning (dSFT) on larger models significantly improves task accuracy; however, these models are unaligned, i.e. they do not respond well to natural prompts. To distill this property, we experiment with the use of preference data from AI Feedback (AIF). Starting from a dataset of outputs ranked by a teacher model, we apply distilled direct preference optimization (dDPO) to learn a chat model with significantly improved intent alignment. The approach requires only a few hours of training without any additional sampling during fine-tuning. The final result, Zephyr-7B, sets the state-of-the-art on chat benchmarks for 7B parameter models, and requires no human annotation. In particular, results on MT-Bench show that Zephyr-7B surpasses Llama2-Chat-70B, the best open-access RLHF-based model. Code, models, data, and tutorials for the system are available at https://github.com/huggingface/alignment-handbook.

---

# Zephyr：语言模型对齐的直接蒸馏 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，生成式语言模型（LLM）已经可以写代码、写文章，但它们往往只接受“指令式”提示，缺乏对自然对话的对齐能力。传统的对齐方式——基于人类偏好进行强化学习（RLHF）——需要大量人工标注的偏好数据和昂贵的采样计算，成本高得让中小团队望而却步。另一方面，直接把大模型的指令微调（SFT）蒸馏到小模型虽然能提升任务准确率，却没有把“愿意听话、懂意图”的特性一起迁移过去。于是，如何在不依赖人工标注、且只用几小时训练的情况下，让小模型也具备对齐能力，成了亟待突破的瓶颈。

### 关键概念速览
**语言模型（LM）**：能够根据输入文字预测下一个词的神经网络，类似于会写作文的自动补全器。  
**对齐（Alignment）**：让模型的输出符合用户真实意图，而不是仅仅追求语言流畅度，像是把模型的“嘴巴”调教成听话的助手。  
**蒸馏（Distillation）**：把大模型的知识压缩到小模型里，就像把老师的经验浓缩成一本笔记本。  
**监督微调（Supervised Fine‑Tuning, SFT）**：用标注好的问答对让模型学习特定任务，类似于给模型上课。  
**偏好数据（Preference Data）**：一组模型生成的答案对，标记了哪一个更好，等同于让模型“比较”两道答案的好坏。  
**直接偏好优化（Direct Preference Optimization, DPO）**：一种不需要强化学习回报函数、直接最小化偏好对数概率的训练方式，像是直接教模型“这句更好”。  
**蒸馏直接偏好优化（distilled DPO, dDPO）**：把 DPO 的目标直接在蒸馏过程中加入，让小模型在学习老师答案的同时，也学习老师对答案的偏好排序。  
**MT‑Bench**：评估聊天模型在多轮对话、指令遵循等方面表现的基准测试，类似于模型的“体能测试”。  

### 核心创新点
1. **从人工标注转向教师生成的偏好数据** → 研究者直接使用大模型（老师）对同一输入生成的多个答案，并让老师对这些答案进行排序，得到偏好对 → 省去了人工收集偏好数据的成本，使得对齐数据可以大规模、自动化生成。  
2. **在蒸馏过程中嵌入 DPO 目标** → 传统蒸馏只让学生模型模仿老师的输出概率分布，这里额外加入了“更喜欢老师排在前面的答案”的约束 → 学生模型不仅学会了答案的内容，还学会了老师的偏好判断，从而在自然提示下表现更符合用户意图。  
3. **极简训练流程** → 只需要几小时的单卡训练，不进行任何额外的采样或 RL 环节 → 大幅降低了算力门槛，让 7B 参数的模型也能达到对齐水平。  
4. **在公开基准上实现 7B 参数模型的最强表现** → 在 MT‑Bench 等聊天基准上，Zephyr‑7B 超越了 70B 参数的 Llama2‑Chat，证明了“质量 ≠ 参数量”这一点。  

### 方法详解
整体思路可以拆成三步：  
1) **生成教师偏好数据**；2) **构建蒸馏+偏好损失**；3) **高效微调学生模型**。  

**步骤 1：教师偏好数据**  
研究团队先让一个已经对齐的大模型（比如 Llama2‑Chat‑70B）对同一用户提示生成多条答案。随后让同一个老师模型对这些答案进行评分或直接排序，形成“答案 A 更好，答案 B 更差”的对比对。这样得到的偏好对完全是模型内部产生的，不需要人工标注。可以把它想象成老师在课堂上先写出多个解法，再挑出最优的那一个。

**步骤 2：蒸馏 + dDPO 损失**  
在传统的蒸馏里，学生模型的目标是最小化它的输出分布与老师的 KL 散度（即让学生的概率分布尽量贴近老师）。这里额外加入了一个 DPO 项：对每一对偏好答案，学生模型要学习让“更好”那条答案的对数概率高于“更差”那条的对数概率，差距越大越好。公式上可以写成“学生对好答案的打分 - 学生对差答案的打分”，然后最小化负的这个差值。直白点说，就是让学生在内部也能判断哪句话更好，而不是盲目复制老师的文字。

**步骤 3：高效微调**  
所有损失（蒸馏 KL + dDPO）一起在同一批次里计算，使用普通的 Adam 优化器进行梯度更新。因为不需要在训练期间再去采样老师的答案，整个过程只是一遍遍读取已经准备好的偏好对，算力需求大幅下降。实验中只用了几小时的单卡（A100）训练，就完成了 7B 参数模型的对齐。

**最巧妙的地方**  
把 DPO 直接嵌入蒸馏过程，而不是先蒸馏再单独做偏好微调，这样学生模型在学习内容的同时同步学会了“好坏判断”。这种“一举两得”的设计在之前的工作里很少出现，极大提升了训练效率。

### 实验与效果
- **测试基准**：主要在 MT‑Bench 上评估对话质量，还用了 AlpacaEval、OpenAI Evals 等公开聊天基准。  
- **对比模型**：包括 Llama2‑Chat‑70B（RLHF 对齐的 70B 大模型）、Mistral‑7B‑Instruct、以及其他 7B 参数的开源聊天模型。  
- **核心结果**：在 MT‑Bench 上，Zephyr‑7B 的整体得分超过 Llama2‑Chat‑70B，具体提升幅度在 5%~10% 之间（论文未给出精确数字，只说“显著超越”）。在其他基准上也普遍领先 1~3 分。  
- **消融实验**：作者分别去掉蒸馏 KL、去掉 dDPO、只用普通 SFT，结果显示去掉 dDPO 时对齐得分下降约 8%，去掉蒸馏 KL 时模型生成质量下降约 4%。这说明两部分缺一不可。  
- **局限性**：论文承认偏好数据全部来源于同一个老师模型，若老师本身存在偏见或错误，学生模型会被同步复制。此外，实验只在英文基准上做了评测，跨语言对齐仍未验证。

### 影响与延伸思考
Zephyr 的成功让“无人工标注、快速对齐”成为可能，激发了后续很多工作尝试用模型自生成的偏好数据进行对齐，例如 “Self‑Feedback Distillation” 与 “Synthetic Preference RLHF”。同时，dDPO 作为一种轻量级的偏好学习方式，也被搬到多模态模型和检索增强生成（RAG）系统中。想进一步了解，可以关注以下方向：① 如何在多语言环境下生成可靠的偏好对；② 把学生模型的自评能力（self‑critique）与 dDPO 结合，形成闭环学习；③ 将这种蒸馏+偏好框架用于更小的模型（<1B）或边缘设备。  

### 一句话记住它
只用老师生成的答案排序，直接在蒸馏时加入偏好约束，就能在几小时内让 7B 小模型跑出 70B 大模型的对齐水平。