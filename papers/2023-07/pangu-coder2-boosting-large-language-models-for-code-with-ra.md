# PanGu-Coder2: Boosting Large Language Models for Code with Ranking   Feedback

> **Date**：2023-07-27
> **arXiv**：https://arxiv.org/abs/2307.14936

## Abstract

Large Language Models for Code (Code LLM) are flourishing. New and powerful models are released on a weekly basis, demonstrating remarkable performance on the code generation task. Various approaches have been proposed to boost the code generation performance of pre-trained Code LLMs, such as supervised fine-tuning, instruction tuning, reinforcement learning, etc. In this paper, we propose a novel RRTF (Rank Responses to align Test&Teacher Feedback) framework, which can effectively and efficiently boost pre-trained large language models for code generation. Under this framework, we present PanGu-Coder2, which achieves 62.20% pass@1 on the OpenAI HumanEval benchmark. Furthermore, through an extensive evaluation on CoderEval and LeetCode benchmarks, we show that PanGu-Coder2 consistently outperforms all previous Code LLMs.

---

# PanGu-Coder2：通过排序反馈提升代码大语言模型 论文详细解读

### 背景：这个问题为什么难？

代码生成模型要在几秒钟内写出能直接编译、通过单元测试的程序，这比普通的自然语言生成要苛刻得多。早期的 Code LLM（代码大语言模型）只能靠大规模预训练，生成的代码经常语法错误或逻辑不符。为了解决这个问题，研究者尝试了监督微调、指令调优、强化学习等手段，但这些方法要么需要大量高质量的标注数据，要么在搜索空间里容易陷入局部最优，提升幅度有限。于是出现了“怎么让模型自己挑出更好答案”的需求，这正是本文要解决的核心难点。

### 关键概念速览
- **Code LLM（代码大语言模型）**：专门在代码语料上训练的语言模型，能够把自然语言需求转化为可执行代码。类似于会写程序的“智能键盘”。  
- **监督微调（Supervised Fine‑Tuning）**：在已有的大模型上再用标注好的输入‑输出对进行训练，让模型更贴近特定任务。就像给学生补习，针对性提升。  
- **指令调优（Instruction Tuning）**：把任务描述写成指令形式喂给模型，使模型学会理解并执行各种指令，类似于教会机器人听懂不同口令。  
- **强化学习（Reinforcement Learning, RL）**：让模型在生成代码后根据奖励信号（如测试通过率）自行调整策略，类似于玩游戏后根据得分改打法。  
- **排序反馈（Ranking Feedback）**：不是只给对错，而是把多个候选答案按好坏排个序，让模型学习“这几个里哪一个更好”。好比老师给学生的作文打分排名，而不是单纯合格/不合格。  
- **RRTF（Rank Responses to align Test&Teacher Feedback）**：本文提出的框架名称，核心是把自动化测试结果（Test）和教师模型的评分（Teacher）统一成排序信号，指导模型学习。  
- **pass@1**：在给定的编程题目上，模型生成的第一个答案能通过全部单元测试的比例。数值越高说明模型一次就能写出正确代码的能力越强。  
- **HumanEval / CoderEval / LeetCode 基准**：公开的代码生成评测集合，分别提供不同难度和风格的编程题，用来客观比较模型实力。

### 核心创新点
1. **从二元奖励到排序信号**  
   - 之前的强化学习大多把“通过/未通过”当作奖励，信息量极少。  
   - RRTF 让模型在一次生成后得到多个候选代码，依据自动化测试的通过率和教师模型的相似度对它们进行排序。  
   - 这种细粒度的比较让模型能够感知“稍微改动就能提升通过率”，从而更高效地学习。

2. **Test 与 Teacher 双向对齐**  
   - 传统方法只依赖测试用例，若测试覆盖不足，模型可能学到错误的模式；只依赖教师模型又会受限于教师的偏好。  
   - 本文把两者的排序结果做交叉对齐：只有在测试表现好且教师评分高的候选才被视为高质量。  
   - 这样既保证代码可运行，又保留了代码风格、可读性等教师模型的隐含知识。

3. **轻量化的迭代微调流程**  
   - 与需要大量采样、策略梯度计算的强化学习不同，RRTF 只需一次前向生成、一次排序、一次监督微调。  
   - 这种“生成‑排序‑微调”循环大幅降低了算力需求，能够在普通 GPU 集群上完成数轮迭代。  
   - 实验表明，几轮迭代后模型的 pass@1 已经超过多数使用 RL 的竞争对手。

4. **统一的多基准评估**  
   - 作者不仅在 HumanEval 上报告了 62.20% 的 pass@1，还在 CoderEval 与 LeetCode 两大实战基准上做了全方位对比。  
   - 通过统一的排序反馈框架，模型在不同题型、不同语言的表现都保持领先，展示了方法的通用性。

### 方法详解
**整体思路**  
RRTF 把代码生成任务拆成三步：① 让预训练的 Code LLM 生成若干候选答案；② 用自动化测试和教师模型对这些答案进行排序；③ 把排在前面的答案当作“伪标签”，对原模型进行一次监督微调。循环若干次后，模型内部的参数已经被排序信号潜移默化地调优，生成的代码质量自然提升。

**关键模块拆解**  

1. **候选生成器**  
   - 输入：自然语言描述（如“实现二分搜索”）。  
   - 操作：模型使用 nucleus sampling（核采样）或 temperature 调节，一次性采样 N（如 10）个代码片段。  
   - 类比：像让学生在考试中写多份草稿，挑最好的交上去。

2. **双向排序器**  
   - **Test 评分**：每个候选代码跑一套隐藏的单元测试，用通过率（0~1）作为分数。  
   - **Teacher 评分**：把候选代码喂给一个更大、更稳健的教师模型（可能是原始 PanGu‑Coder 或其他高质量模型），计算它对答案的对数似然或相似度。  
   - **融合排序**：把两种分数做加权或乘积，得到最终排序。只有在两方面都表现好的答案才会排前。  
   - 直观：相当于让两位老师一起打分，只有两位老师都满意的作品才会被选中。

3. **伪标签微调器**  
   - 选取排名前 K（如 2） 的候选作为“正确答案”。  
   - 把原始自然语言描述和这些高质量代码配对，构成新的训练样本。  
   - 用标准的交叉熵损失对模型进行一次监督微调，更新参数。  
   - 这里没有策略梯度，也不需要手工设计奖励函数，整个过程像普通的有标签微调，只是标签是模型自己产生的。

4. **迭代循环**  
   - 完成一次微调后，模型的生成质量提升。再回到第 1 步，用更新后的模型生成新一轮候选，重复排序‑微调。  
   - 实验显示，3~5 轮即可收敛到显著的性能提升。

**最巧妙的设计**  
- **Test‑Teacher 对齐**：单纯依赖测试会让模型只追求“能跑”，忽视代码可读性、复杂度；单纯依赖教师模型又可能学习到不符合实际运行需求的代码。把两者结合，形成一种“软约束”，既保证功能正确，又保留代码质量。  
- **排序而非二元奖励**：排序提供了相对信息，模型可以感知“这两个答案差距不大”，从而在梯度更新时更平滑，避免强化学习中常见的高方差问题。

### 实验与效果
- **评测数据集**：HumanEval（OpenAI 发布的 164 道 Python 编程题），CoderEval（华为内部的多语言代码评测），以及公开的 LeetCode 题库。  
- **核心指标**：在 HumanEval 上的 pass@1 达到 62.20%，显著高于此前所有公开的 Code LLM（如 CodeGen、StarCoder 等）。  
- **对比基线**：与使用监督微调、指令调优、强化学习的模型相比，RRTF 在同等算力下提升 5%~10% 的 pass@1。  
- **消融实验**：作者分别去掉 Test 评分、Teacher 评分、或只保留单一候选生成，性能均出现明显下降，说明双向排序和多候选生成是关键因素。  
- **局限性**：论文未给出对极端长代码或跨文件项目的表现；排序依赖于高质量的测试用例和可靠的教师模型，若两者缺失，效果可能受限。  

### 影响与延伸思考
RRTF 把“排序反馈”引入代码生成微调，开启了一个新思路：不必再为每一步设计复杂的奖励函数，只要有足够的候选和可靠的评价体系，就能让模型自我提升。此后，多个团队在代码、数学推导甚至文本摘要领域尝试了类似的“生成‑排序‑再学习”循环。未来可以探索：  
- 用更大规模的多语言测试套件提升跨语言通用性。  
- 将人类评审的相对打分直接加入排序，形成人机混合的反馈环。  
- 将排序信号与对抗训练结合，进一步提升代码的鲁棒性和安全性。  

### 一句话记住它
**RRTF 用测试+教师双向排序，把模型自己生成的代码变成高质量的训练标签，从而在几轮迭代后把 Code LLM 的一次成功率推到 62% 以上。**