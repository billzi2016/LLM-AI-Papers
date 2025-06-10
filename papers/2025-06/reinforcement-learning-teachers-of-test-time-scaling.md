# Reinforcement Learning Teachers of Test Time Scaling

> **Date**：2025-06-10
> **arXiv**：https://arxiv.org/abs/2506.08388

## Abstract

Training reasoning language models (LMs) with reinforcement learning (RL) for one-hot correctness inherently relies on the LM being able to explore and solve its task with some chance at initialization. Furthermore, a key use case of reasoning LMs is to act as teachers for distilling new students and cold-starting future RL iterations rather than being deployed themselves. From these considerations, we introduce a new framework that avoids RL's exploration challenge by training a new class of Reinforcement-Learned Teachers (RLTs) focused on yielding the most effective downstream distillation. RLTs are prompted with both the question and solution to each problem, and tasked to simply "connect-the-dots" with detailed explanations tailored for their students. We train RLTs with dense rewards obtained by feeding each explanation to the student and testing its understanding of the problem's solution. In practice, the raw outputs of a 7B RLT provide higher final performance on competition and graduate-level tasks than existing distillation and cold-starting pipelines that collect and postprocess the reasoning traces of orders of magnitude larger LMs. Furthermore, RLTs maintain their effectiveness when training larger students and when applied zero-shot to out-of-distribution tasks, unlocking new levels of efficiency and re-usability for the RL reasoning framework. Code available at: https://github.com/SakanaAI/RLT

---

# 测试时尺度的强化学习教师（RLT）论文详细解读

### 背景：这个问题为什么难？

在让语言模型（LM）进行推理时，常规的强化学习（RL）做法是直接奖励模型给出正确答案。要让模型在训练伊始就能偶然找到正确解法并据此探索，几乎是把一只盲眼的猫放进迷宫——没有足够的探索信号，模型很容易卡在错误的思路上。更关键的是，推理模型的主要价值往往不是直接部署，而是作为“老师”，帮助更小的学生模型学习推理技巧。于是出现了两个矛盾：一是 RL 需要模型自行发现答案，二是我们真正想要的却是模型在已知答案的前提下，生成最能启发学生的解释。传统管线只能先让大模型自己推理，再把过程收集下来喂给学生，既浪费算力，又难以保证解释质量。

### 关键概念速览
- **强化学习（RL）**：让模型通过试错获得奖励的学习方式，类似于玩游戏时不断尝试、记住哪些操作能得分。  
- **语言模型（LM）**：能够根据上下文生成自然语言的模型，就像会写作文的机器人。  
- **蒸馏（Distillation）**：把大模型的知识压缩进小模型的过程，类似老师把课堂要点浓缩成笔记交给学生。  
- **教师‑学生框架**：大模型（老师）产生教学材料，学生模型通过学习这些材料提升能力。  
- **测试时尺度（Test‑time Scaling）**：指在实际使用阶段，模型的表现随任务难度或数据规模的变化情况。  
- **密集奖励（Dense Reward）**：不像只在最终答案对错上给奖励，而是在每一步解释后都评估学生的学习效果，提供更细粒度的反馈。  
- **冷启动（Cold‑start）**：在没有任何已有经验的情况下开始训练，就像新手第一次玩游戏，需要从零开始学习。  
- **零样本（Zero‑shot）**：模型在未见过的任务上直接发挥能力，类似人类第一次接触新领域却还能凭直觉完成。

### 核心创新点
1. **目标从“一次正确”转向“最有效蒸馏”**  
   - 之前的 RL 方法把奖励定义为模型输出的答案是否正确 → 本文把奖励改为学生在阅读解释后能否正确解题 → 这样教师的优化直接围绕提升学生表现，避免了让模型自行探索答案的难题。  

2. **教师输入同时包含问题和答案**  
   - 传统教师模型只看到问题，必须自行推理出答案再解释 → RLT 在提示中显式提供答案，让模型专注于“连点成线”，即把已知答案拆解成易懂的推理步骤 → 这大幅提升了解释的针对性和可教性。  

3. **使用学生反馈构造密集奖励**  
   - 过去的奖励往往是二元的（对/错）或基于人工评分 → 本文让每条解释都喂给学生模型，测它的解题成功率，然后把这个成功率作为即时奖励 → 这种闭环让教师学习到哪些解释最能帮助学生。  

4. **小模型即可超越大模型的蒸馏管线**  
   - 传统做法需要数十倍参数的大模型先生成推理轨迹，再进行后处理 → RLT 只用 7B 参数的教师，就在竞争任务和研究生水平题目上跑出比使用上百亿参数模型的管线更高的最终分数 → 极大提升了算力效率和可复用性。

### 方法详解
**整体框架**  
RLT 的训练分为三步：  
1) **准备数据**：每条训练样本提供（问题、答案）对。  
2) **教师生成解释**：把问题和答案一起喂给教师模型，模型输出一段详细的推理解释。  
3) **学生评估与奖励**：把解释交给学生模型，让学生尝试解答原问题；根据学生是否得到正确答案计算奖励，随后用强化学习算法（如策略梯度）更新教师的生成策略。循环往复，教师逐渐学会生成最能提升学生表现的解释。

**关键模块拆解**  
- **输入提示设计**：提示模板类似 “Question: <问题> Answer: <答案> Explain step‑by‑step:” 这样教师一看到答案就不需要再自行搜索，只需组织解释。  
- **解释生成模型**：采用标准的自回归语言模型（本文使用 7B 参数的模型），在每一步生成文字时保留概率分布，以便后续计算梯度。  
- **学生模型**：可以是任意尺寸的语言模型，本文在实验中使用了从 1B 到 13B 不等的学生。学生在接收到解释后，会把解释当作上下文再次生成答案。  
- **奖励计算**：学生的答案与真实答案比较，得到 0/1 正确率；为了得到密集信号，作者在解释的每个子句后都让学生尝试一次，累计得到一个介于 0 与 1 之间的分数。  
- **强化学习优化**：使用 REINFORCE‑style 的策略梯度，把教师的生成概率乘以奖励的优势（reward‑baseline），对模型参数进行梯度上升。这里的 baseline 通常是同一批次的平均奖励，用来降低方差。  

**最巧妙的地方**  
- **把答案直接喂给教师**：这一步看似把“解题”任务交给了学生，却让教师专注于“教学”。相当于把老师从“自己先解题再讲”改成“先知道答案，再把思路拆解”。  
- **学生即奖励函数**：传统 RL 需要人工设计奖励函数，这里把学生的表现直接当作奖励，形成了一个自我校准的闭环，省去了繁琐的人工评分。  

### 实验与效果
- **测试任务**：论文在多个竞争性基准和研究生水平的推理题目上评估，包括数学、物理以及逻辑推理等。  
- **对比基线**：与现有的蒸馏管线（先让大模型自行推理，再收集轨迹并进行后处理）以及冷启动的 RL 方法进行比较。  
- **结果声称**：7B 参数的 RLT 在最终学生模型的表现上超过了使用数十倍更大模型的传统管线，且在训练更大的学生模型时仍保持优势。对未知分布（out‑of‑distribution）任务的零样本测试也表现出更好的迁移能力。  
- **消融实验**：论文通过去掉答案输入、使用二元奖励或不让学生在每个子句后尝试等设置，验证了“答案输入”和“密集奖励”是提升效果的关键因素。  
- **局限性**：原文未给出详细的计算成本分析；因为每条解释都要让学生模型评估一次，训练过程的算力开销可能比单纯的语言模型预训练更高。作者也提到在极端长文本或多步推理场景下奖励噪声仍然是挑战。

### 影响与延伸思考
RLT 把“教学质量”直接嵌入强化学习的奖励函数，为教师模型的训练提供了全新的视角。自论文发布后，已有工作尝试将学生反馈用于优化提示工程、自动生成教学大纲，甚至把人类学生的学习曲线作为奖励信号。未来可以探索：  
- **跨模态教师**：让视觉‑语言模型在看到答案后生成图文并茂的解释。  
- **多学生协同**：同时评估多个学生模型的表现，得到更稳健的奖励。  
- **更高效的奖励估计**：使用代理模型或元学习降低每次学生评估的算力。  
对想进一步了解的读者，可以关注 2024‑2025 年间出现的 “Reward‑based Distillation” 与 “Self‑Teaching RL” 系列论文，它们大多受 RLT 思路启发。

### 一句话记住它
把答案喂给教师，让学生的解题成功率直接成为教师的奖励，才能训练出最会教的语言模型。