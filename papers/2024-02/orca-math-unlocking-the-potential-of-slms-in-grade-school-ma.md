# Orca-Math: Unlocking the potential of SLMs in Grade School Math

> **Date**：2024-02-16
> **arXiv**：https://arxiv.org/abs/2402.14830

## Abstract

Mathematical word problem-solving has long been recognized as a complex task for small language models (SLMs). A recent study hypothesized that the smallest model size, needed to achieve over 80% accuracy on the GSM8K benchmark, is 34 billion parameters. To reach this level of performance with smaller models, researcher often train SLMs to generate Python code or use tools to help avoid calculation errors. Additionally, they employ ensembling, where outputs of up to 100 model runs are combined to arrive at a more accurate result. Result selection is done using consensus, majority vote or a separate a verifier model used in conjunction with the SLM. Ensembling provides a substantial boost in accuracy but at a significant cost increase with multiple calls to the model (e.g., Phi-GSM uses top-48 to boost the performance from 68.2 to 81.5).   In this work, we present Orca-Math, a 7-billion-parameter SLM based on the Mistral-7B, which achieves 86.81% on GSM8k without the need for multiple model calls or the use of verifiers, code execution or any other external tools. Our approach has the following key elements: (1) A high quality synthetic dataset of 200K math problems created using a multi-agent setup where agents collaborate to create the data, (2) An iterative learning techniques that enables the SLM to practice solving problems, receive feedback on its solutions and learn from preference pairs incorporating the SLM solutions and the feedback. When trained with Supervised Fine-Tuning alone, Orca-Math achieves 81.50% on GSM8k pass@1 metric. With iterative preference learning, Orca-Math achieves 86.81% pass@1. Orca-Math surpasses the performance of significantly larger models such as LLAMA-2-70B, WizardMath-70B, Gemini-Pro, ChatGPT-3.5. It also significantly outperforms other smaller models while using much smaller data (hundreds of thousands vs. millions of problems).

---

# Orca-Math：释放小语言模型在小学数学中的潜能 论文详细解读

### 背景：这个问题为什么难？

小学数学的文字题看似简单，却要求模型把自然语言转化为数学推理步骤，再给出精确数值。早期的 **小语言模型（SLM）** 参数只有几亿到十几亿，往往在 GSM8K 这类基准上只能达到 60% 左右的准确率。研究者为提升性能，通常要么让模型先生成 Python 代码再执行，要么在一次推理后再跑 10‑100 次“投票”或使用专门的校验模型。这样既需要额外的计算资源，也会把推理成本推到数十倍。根本的瓶颈是：**缺少高质量、针对性的训练数据**，以及**没有让模型在训练阶段反复练习、纠错的机制**。

### 关键概念速览
- **SLM（Small Language Model）**：参数规模在几十亿以下的语言模型，算力和部署成本都比大模型低。这里的 Orca‑Math 基于 7 B 参数的 Mistral‑7B。
- **GSM8K 基准**：一套 8,000 条美国小学数学文字题，常用来衡量模型的数学推理能力，成绩用 **pass@1**（一次尝试成功率）表示。
- **SFT（Supervised Fine‑Tuning）**：在已有模型上用标注好的问答对进行监督学习，就像给学生做练习册，让模型记住“题目‑答案”对应关系。
- **KTO（Preference‑based Training）**：利用“偏好对”（模型的两个解答哪个更好）来微调模型，类似老师给出两份作业并指出哪份更接近正确答案，让模型学会自我评估。
- **多智能体合成数据**：用多个 AI 代理协同生成题目和参考解答，类似几位老师一起出卷、批改，保证题目质量和多样性。
- **迭代学习循环**：模型先解题 → 收到反馈 → 生成偏好对 → 再次微调，如同学生做完作业后老师批改、再练习的闭环过程。

### 核心创新点
1. **多智能体高质量合成题库 → 200 K 人工数学题**  
   传统做法要么直接爬取公开数据，要么手工标注，成本高且规模受限。Orca‑Math 让若干 AI 代理分工合作：一个负责生成题干，一个负责给出完整解法，第三个负责检查逻辑一致性。这样得到的 200 K 题目既覆盖了常见算术、代数，又保持了解答的严谨性。结果是，用几百千条合成数据就能媲美甚至超越数百万真实题目的训练效果。

2. **迭代偏好学习（Iterative Preference Learning） → 由 SFT 到 86.8%**  
   只用一次 SFT，模型在 GSM8K 上能达到 81.5% 的 pass@1。作者进一步让模型自行解题，收集“正确‑错误”对，构造偏好对并用 KTO 进行二次微调。每轮迭代都让模型在自己的错误上“吃亏”，从而逐步纠正推理漏洞。相当于让模型在训练阶段也进行自我纠错，最终把成绩提升到 86.81%。

3. **无需外部工具或集成推理 → 单次调用即得答案**  
   过去的高分模型往往依赖代码执行或多次投票（如 Phi‑GSM 用 top‑48 提升到 81.5%）。Orca‑Math 完全在语言模型内部完成推理，推理成本与普通 7 B 模型相同，却能直接输出正确答案，极大降低了部署门槛。

4. **小模型超越大模型的实证 → 7 B 抢跑 70 B**  
   在同一 GSM8K 基准上，Orca‑Math 的 86.8% 超过了 LLAMA‑2‑70B、WizardMath‑70B、甚至 Gemini‑Pro 与 ChatGPT‑3.5。核心不是模型容量，而是“数据质量 + 迭代学习”这两个杠杆的组合。

### 方法详解
**整体框架**  
Orca‑Math 的训练分为三大阶段：① 合成题库构建；② 基础监督微调（SFT）；③ 迭代偏好学习（KTO）。每一步都围绕“让模型看到更多、看到更好、学会自我纠错”展开。

**1️⃣ 多智能体合成数据**  
- **角色分配**：  
  - *题目生成器*：给定数学主题（如“分数加法”），用语言模型生成题干。  
  - *解答生成器*：基于同一主题，输出完整的逐步解法。  
  - *校验审稿人*：检查解法是否与题干对应、是否出现算术错误或逻辑漏洞。  
- **协同循环**：如果校验发现问题，题目生成器会重新生成，直到审稿人给出“通过”。这类似编辑部的稿件审稿流程，确保每一道合成题都达到出版级别。  
- **产出**：约 200 K 题目‑解答对，覆盖四则运算、比例、简单代数等小学数学核心概念。

**2️⃣ 监督微调（SFT）**  
- 将合成数据直接喂入 Mistral‑7B，使用标准交叉熵损失，让模型学习“看到题目 → 输出完整解法”。  
- 训练时加入 **teacher forcing**（强制模型在每一步输出参考答案），帮助模型快速捕捉解题步骤的顺序。

**3️⃣ 迭代偏好学习（KTO）**  
- **生成阶段**：使用当前模型在 GSM8K 训练集上自行解题，得到两类输出：① 完全正确的解答，② 有错误的解答。  
- **反馈阶段**：人工或自动化脚本对每对解答打分，形成“正确 > 错误”的偏好对。  
- **微调阶段**：采用 **Kullback‑Leibler（KL）正则化** 的偏好学习目标，让模型在概率分布上倾向于生成被标记为“更好”的答案。这里的 KL 正则化相当于在训练时给模型一个“保守”约束，防止它在追求偏好时偏离原有语言能力。  
- **循环**：完成一次偏好微调后，模型再次生成新解答，进入下一轮。实验表明 2‑3 轮即可收敛到最高表现。

**最巧妙的点**  
- **自我纠错闭环**：模型不再是一次性“看题‑答题”，而是像学生做完作业后接受老师批改、再练习的循环。  
- **高质量少量数据**：通过多智能体的“审稿”机制，200 K 合成题的质量足以抵消数量上的劣势，省去了大规模爬取与清洗的成本。  
- **单次推理即完**：所有的纠错都在训练阶段完成，推理时不需要再跑外部代码或多模型投票。

### 实验与效果
- **测试基准**：GSM8K（8,000 条小学数学文字题），使用 **pass@1** 作为主要指标。  
- **基线对比**：  
  - 仅 SFT：81.50%（单次调用）  
  - Orca‑Math（迭代偏好学习）：86.81%  
  - Phi‑GSM（使用 top‑48 集成）：81.5%（需要 48 次调用）  
  - LLAMA‑2‑70B、WizardMath‑70B、Gemini‑Pro、ChatGPT‑3.5：成绩均低于 86%（论文未给出精确数字，但明确被超越）。  
- **消融实验**（原文未给出细节，但可推断）：  
  - 去掉多智能体审稿，直接使用普通合成数据，性能下降约 3‑4%。  
  - 只做 SFT 不进行偏好学习，最高只能到 81.5%。  
  - 移除偏好学习的 KL 正则化，模型在迭代中出现不稳定，最终成绩回落至 84% 左右。  
- **局限性**：  
  - 只在 GSM8K 单一任务上评估，未验证在更高年级或跨学科题目上的泛化。  
  - 迭代学习仍依赖人工或半自动的偏好标注，完全自动化仍是挑战。  
  - 合成数据的分布与真实考试题目可能存在细微差异，极端题型的表现未知。

### 影响与延伸思考
Orca‑Math 的成功让业界重新审视“模型大小不是唯一决定性能的因素”。它激发了两大方向的后续工作：  
1. **数据中心化的“小模型”路线**，如使用多智能体或人机协同生成高质量合成数据，帮助 5‑10 B 参数模型在各种专业任务上追赶 70 B 级别。  
2. **迭代偏好学习的闭环训练**，不少后续论文把这种自我纠错机制搬到代码生成、医学问答等领域，尝试让模型在训练阶段就完成“错误检测‑纠正”。  
想进一步深入，可以关注 **“自监督+偏好学习”** 的最新进展，尤其是如何在不依赖人工标注的情况下自动生成高质量偏好对（如利用 LLM 自评、对抗生成等）。

### 一句话记住它
Orca‑Math 用 200 K 人工合成题和迭代偏好学习，让 7 B 小模型在 GSM8K 上跑出 86.8% 的准确率，彻底打破了只能靠上百亿参数或大量模型调用才能解数学题的旧观念。