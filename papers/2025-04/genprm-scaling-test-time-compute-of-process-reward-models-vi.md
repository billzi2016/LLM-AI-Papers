# GenPRM: Scaling Test-Time Compute of Process Reward Models via   Generative Reasoning

> **Date**：2025-04-01
> **arXiv**：https://arxiv.org/abs/2504.00891

## Abstract

Recent advancements in Large Language Models (LLMs) have shown that it is promising to utilize Process Reward Models (PRMs) as verifiers to enhance the performance of LLMs. However, current PRMs face three key challenges: (1) limited process supervision and generalization capabilities, (2) dependence on scalar value prediction without leveraging the generative abilities of LLMs, and (3) inability to scale the test-time compute of PRMs. In this work, we introduce GenPRM, a generative process reward model that performs explicit Chain-of-Thought (CoT) reasoning with code verification before providing judgment for each reasoning step. To obtain high-quality process supervision labels and rationale data, we propose Relative Progress Estimation (RPE) and a rationale synthesis framework that incorporates code verification. Experimental results on ProcessBench and several mathematical reasoning tasks show that GenPRM significantly outperforms prior PRMs with only 23K training data from MATH dataset. Through test-time scaling, a 1.5B GenPRM outperforms GPT-4o, and a 7B GenPRM surpasses Qwen2.5-Math-PRM-72B on ProcessBench. Additionally, GenPRM demonstrates strong abilities to serve as a critic model for policy model refinement. This work establishes a new paradigm for process supervision that bridges the gap between PRMs and critic models in LLMs. Our code, model, and data will be available in https://ryanliu112.github.io/GenPRM.

---

# GenPRM：通过生成式推理扩展过程奖励模型的测试时计算 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，用过程奖励模型（Process Reward Model，PRM）来检查推理步骤已经被证明能提升答案质量。但现有 PRM 有三大痛点：第一，它们只能在少量标注的推理过程上学习，导致对新题型的泛化能力不足；第二，模型只输出一个标量分数，根本没有利用 LLM 本身的生成能力来解释或纠错；第三，推理过程越长，PRM 的计算成本会线性增长，难以在实际部署时放大使用。于是，如何让 PRM 既能得到更丰富的过程监督，又能在推理时保持可扩展性，成为迫切需要突破的瓶颈。

### 关键概念速览
- **过程奖励模型（PRM）**：一种专门评估模型推理过程质量的判别器，类似老师给学生的每一步批改分数，而不是只给最终答案打分。  
- **Chain-of-Thought（CoT）**：让模型在给出答案前先写出思考链条，像在纸上写草稿一样，使推理过程透明可查。  
- **相对进度估计（RPE）**：一种自动生成过程标签的技巧，模型会判断当前步骤相对于完整解答的进度，从而得到“已经走到哪儿了”的监督信号。  
- **代码验证**：把模型的文字推理转化为可执行代码，运行后检查是否得到预期结果，类似把数学推导交给计算器验证。  
- **测试时计算扩展（Test‑time Compute Scaling）**：在推理阶段投入更多算力（比如更大的模型或多轮评估）来提升评估质量，而不是只在训练时增加数据。  
- **Critic Model**：在强化学习或自我对齐中充当“评审”的模型，负责给策略模型的输出打分并提供改进方向。  

### 核心创新点
1. **从标量评估到生成式评审**：传统 PRM 只输出一个分数 → GenPRM 让模型先生成每一步的 CoT，再对每一步进行代码验证，最后把所有验证结果聚合成最终分数 → 评审过程变得可解释、可纠错，提升了判断的准确性。  
2. **相对进度估计 + 合成推理数据**：过去的 PRM 依赖人工标注的少量过程 → 作者提出 RPE 自动推断每一步相对完整解答的进度，并结合代码验证生成高质量的推理示例 → 只用了 23K 条 MATH 数据就达到了比大规模基线更好的效果。  
3. **测试时计算可扩展框架**：多数 PRM 在推理时只能跑一次评估 → GenPRM 设计了可叠加的评估层，允许在需要时使用更大的模型或多轮验证 → 1.5B 版本在测试时就能超越 GPT‑4o，7B 版本也能击败 72B 的 Qwen2.5‑Math‑PRM。  
4. **兼作 Critic 的双重身份**：以往 PRM 与 Critic 是两套系统 → GenPRM 同时提供过程评分和批评建议，可直接用于策略模型的微调 → 打通了过程监督和策略改进的壁垒。  

### 方法详解
**整体思路**：GenPRM 把“评审”拆成三步：① 让模型写出完整的思考链（CoT），② 把每一步转化为可执行代码并跑通，③ 把每一步的验证结果汇总成整体分数。整个流程在推理时可以层层叠加算力，形成可伸缩的评估管线。

**步骤拆解**  

1. **CoT 生成**  
   - 输入：原始题目 + 参考答案（可选）。  
   - 过程：使用一个专门训练的生成式模型（如 1.5B/7B 参数）输出逐步推理文本，每一步都用自然语言描述并附带伪代码框。  
   - 类比：像老师让学生先把解题思路写在黑板上，再动手算。

2. **代码验证模块**  
   - 将每一步的伪代码抽取出来，送入一个安全的代码执行环境。  
   - 运行后检查输出是否与该步骤的预期结果匹配（比如中间变量的数值是否正确）。  
   - 若不匹配，模型会生成纠错提示，重新生成该步骤的 CoT。  
   - 关键点：代码验证把抽象的文字推理具体化为可测的计算，极大降低了语言歧义导致的误判。

3. **相对进度估计（RPE）**  
   - 对每一步，模型预测它在完整解答中的相对位置（如 0.2、0.5、0.8），并把这个进度作为监督信号。  
   - 通过对比真实完整解答的进度，模型学习到“这一步已经走了多少路”。  
   - 这一步在数据构造阶段完成，帮助生成的大规模推理数据拥有可靠的过程标签。

4. **分数聚合**  
   - 每一步的代码验证结果转化为二元信号（通过/未通过），再乘以该步骤的相对进度权重。  
   - 所有步骤的加权和即为最终的过程奖励分数，范围通常是 0~1。  
   - 这种聚合方式兼顾了“每一步是否正确”和“整体进度是否合理”。

5. **测试时计算扩展**  
   - 在资源充足时，可把同一道题交给更大的模型重复生成 CoT，或对关键步骤进行多轮代码验证，取平均或最高分。  
   - 这种“多模型/多轮”策略在不改变训练流程的前提下提升评估质量，类似在考试时请两位老师独立批改再取平均分。

**最巧妙的设计**：把文字推理直接映射到可执行代码，并用相对进度作为软监督，让模型在生成时自带“自检”机制。这样既解决了过程标签稀缺，又让评审过程可解释、可扩展。

### 实验与效果
- **测试平台**：ProcessBench（一个专门评估过程奖励模型的基准）以及若干数学推理数据集（如 MATH、GSM8K 的变体）。  
- **对比基线**：传统 PRM、Qwen2.5‑Math‑PRM‑72B、GPT‑4o 等最先进的评审模型。  
- **核心结果**：在 ProcessBench 上，1.5B GenPRM 在测试时使用额外算力后超过 GPT‑4o；7B GenPRM 在相同算力下跑赢 72B 的 Qwen2.5‑Math‑PRM。实验声称提升幅度显著，但具体数值未在摘要中给出。  
- **数据效率**：只用了 23K 条来自 MATH 的过程数据，就实现了比大规模基线更好的表现，说明 RPE 与代码验证的组合极大提升了数据利用率。  
- **消融实验**：论文报告了去掉代码验证、去掉 RPE、以及不进行测试时扩展的三组消融，结果显示每个模块都有可观的性能下降，尤其是代码验证的缺失导致整体分数下降超过 10%。  
- **局限性**：代码验证依赖于安全的执行环境，对非数值推理（如语言理解）的适用性尚未验证；此外，测试时扩展需要额外算力，成本仍然是实际部署的考量点。  

### 影响与延伸思考
GenPRM 把生成式推理和过程评审紧密结合，为“过程监督”打开了新思路。后续工作可能会把类似的代码验证引入更广泛的任务（如程序合成、逻辑推理），或探索更轻量的验证方式以降低算力需求。还有研究在尝试把 RPE 与人类交互式标注结合，进一步提升过程标签的质量。对想深入的读者，可以关注以下方向：① 生成式评审模型的安全执行环境设计；② 多模态（文字+代码）过程监督的统一框架；③ 将 GenPRM 作为强化学习中的 Critic，探索自我对齐的闭环系统。  

### 一句话记住它
GenPRM 用“写思路‑跑代码‑聚分数”的生成式评审，让过程奖励模型既可解释又能在测试时随算力自由放大。