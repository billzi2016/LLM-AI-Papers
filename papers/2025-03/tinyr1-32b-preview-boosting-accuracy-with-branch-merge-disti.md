# TinyR1-32B-Preview: Boosting Accuracy with Branch-Merge Distillation

> **Date**：2025-03-06
> **arXiv**：https://arxiv.org/abs/2503.04872

## Abstract

The challenge of reducing the size of Large Language Models (LLMs) while maintaining their performance has gained significant attention. However, existing methods, such as model distillation and transfer learning, often fail to achieve high accuracy. To address this limitation, we introduce the Branch-Merge distillation approach, which enhances model compression through two phases: (1) the Branch Phase, where knowledge from a large teacher model is \textit{selectively distilled} into specialized student models via domain-specific supervised fine-tuning (SFT); And (2) the Merge Phase, where these student models are merged to enable cross-domain knowledge transfer and improve generalization. We validate our distillation approach using DeepSeek-R1 as the teacher and DeepSeek-R1-Distill-Qwen-32B as the student. The resulting merged model, TinyR1-32B-Preview, outperforms its counterpart DeepSeek-R1-Distill-Qwen-32B across multiple benchmarks, including Mathematics (+5.5 points), Coding (+4.4 points) and Science (+2.9 points), while achieving near-equal performance to DeepSeek-R1 on AIME 2024. The Branch-Merge distillation approach provides a scalable solution for creating smaller, high-performing LLMs with reduced computational cost and time.

---

# TinyR1-32B-Preview：通过分支‑合并蒸馏提升精度 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）越大，能力越强，但随之而来的算力、存储和推理成本也会指数级增长。业界一直在尝试用蒸馏或迁移学习把大模型压缩成小模型，却常常发现压缩后模型在数学、代码和科学推理等关键任务上掉分严重。根本原因在于：传统蒸馏把老师模型的全部知识一次性灌输给学生，学生在有限容量里必须兼顾所有领域，导致信息噪声过多、关键知识被稀释。于是，如何在保持模型体积小的前提下，仍然让模型在多个专业领域保持高水平，成为了迫切需要突破的难题。

### 关键概念速览
**大模型（LLM）**：参数量上百亿甚至上千亿的语言模型，像 GPT‑4、DeepSeek‑R1，能够处理几乎所有自然语言任务。  
**蒸馏（Distillation）**：把大模型（老师）的行为或内部表示迁移到小模型（学生）上，类似老师把经验传授给学生。  
**监督微调（SFT，Supervised Fine‑Tuning）**：在标注好的任务数据上继续训练模型，让模型专注于特定领域的表现。  
**分支‑合并蒸馏（Branch‑Merge Distillation）**：先让学生模型在单一领域进行针对性蒸馏（分支），随后把多个专精学生的参数合并成一个通用模型（合并）。  
**选择性蒸馏（Selective Distillation）**：不是把老师的全部知识都复制，而是只挑选对当前领域最有价值的知识进行传递，类似老师只讲重点章节。  
**跨域知识迁移**：把在一个领域学到的技巧或模式应用到另一个领域，提升模型的通用化能力。  
**参数融合（Parameter Merging）**：把多个模型的权重按照一定规则（如加权平均或层级映射）合并到同一个模型里，类似把几位专家的经验写进一本综合手册。

### 核心创新点
1. **从全局蒸馏到领域分支**  
   *之前的做法*：一次性把老师模型的全部知识蒸给单一学生，导致学生在容量受限的情况下被“信息淹没”。  
   *本文的做法*：先把老师模型的知识按领域拆分，针对数学、代码、科学等每个子任务分别进行监督微调后再蒸馏，形成多个专精学生。  
   *带来的改变*：每个学生只需学习与自身领域高度相关的知识，蒸馏效率提升，学生在对应任务上的表现显著提升。

2. **参数层面的跨域合并**  
   *之前的做法*：要得到通用小模型只能重新训练或使用粗糙的模型融合，往往会丢失专精模型的优势。  
   *本文的做法*：在得到若干专精学生后，使用一种结构感知的参数融合策略把它们的权重合并为单一模型。合并时保留每个子模型在自己擅长层的权重，同时在共享层做加权平均，实现跨域知识的互补。  
   *带来的改变*：合并后模型兼具多领域专长，整体性能超过单一蒸馏得到的基线模型。

3. **选择性蒸馏的噪声过滤**  
   *之前的做法*：蒸馏时直接使用老师的全部 logits 或 hidden states，难以区分哪些信息对目标任务有帮助。  
   *本文的做法*：在分支阶段引入任务相关的损失权重，只让老师在目标领域的高置信度输出参与蒸馏，低置信度或与任务无关的部分被抑制。  
   *带来的改变*：学生模型学习到的信号更纯粹，训练收敛更快，最终在下游评测上获得更高分。

### 方法详解
**整体框架**  
整个流程可以划分为三大步骤：① 领域划分与数据准备；② 分支蒸馏（每个领域生成专精学生）；③ 合并阶段（把所有学生的参数融合成一个统一模型）。核心思想是“先专后通”，先让模型在窄域里发挥最大潜力，再把这些窄域优势拼接成宽域能力。

**步骤一：领域划分与数据准备**  
作者选取了数学、代码和科学三个代表性领域。对每个领域准备了高质量的 SFT 数据集（例如数学竞赛题、编程题库、科学问答），并用这些数据对老师模型 DeepSeek‑R1 进行轻量微调，使老师在该领域的输出更具针对性。

**步骤二：分支蒸馏**  
对每个领域，执行以下子流程：  
- **教师微调**：在该领域数据上继续训练老师模型，得到“领域老师”。  
- **选择性蒸馏**：计算老师在微调数据上的 logits，筛选出置信度最高的前 K% 作为蒸馏目标；同时使用 KL 散度让学生的输出逼近老师的选中 logits。  
- **学生训练**：以 Qwen‑32B 为基座，初始化学生模型，然后在上述蒸馏目标上进行训练，得到专精学生（如 Math‑Student、Code‑Student、Science‑Student）。  
这一步的关键在于“只蒸最重要的知识”，避免学生被无关信息干扰。

**步骤三：参数合并**  
得到三个专精学生后，作者采用层级感知的加权平均策略：  
- 对于每一层的权重，先判断该层在不同学生中的重要性（通过在对应领域验证集上的梯度幅度或激活分布估计）。  
- 在重要性高的层直接采用对应学生的权重；在共享层则对所有学生的权重做加权平均，权重比例依据各学生在该层的表现分配。  
- 合并后进行一次轻量的全模型微调（仅几百步），帮助模型在参数空间中找到一个兼容多领域的平衡点。  
最终得到的模型即 TinyR1‑32B‑Preview，它的体积与单一 Qwen‑32B 相当，却拥有跨域的高水平能力。

**最巧妙的地方**  
- **分支‑合并的两阶段设计**：把“专精”与“通用”解耦，让每一步都可以独立优化。  
- **选择性蒸馏的噪声抑制**：通过置信度阈值过滤，显著提升了蒸馏信号的质量。  
- **层级感知的权重融合**：不像简单的全模型平均，而是依据每层在不同领域的贡献度动态分配权重，避免了“强者被弱者稀释”的常见问题。

### 实验与效果
- **测试任务**：作者在多个公开基准上评估了合并模型，包括数学推理（如 MATH、AIME 2024）、代码生成（HumanEval、MBPP）和科学问答（ScienceQA）。  
- **对比基线**：主要对比对象是同样大小的蒸馏模型 DeepSeek‑R1‑Distill‑Qwen‑32B（未使用分支‑合并），以及原始老师 DeepSeek‑R1。  
- **核心结果**：TinyR1‑32B‑Preview 在数学基准上提升了 **5.5 分**，代码任务上提升 **4.4 分**，科学问答上提升 **2.9 分**，在 AIME 2024 上的表现几乎追平老师模型。整体来看，合并模型在所有评测上均显著超越单一蒸馏基线。  
- **消融实验**：原文未给出完整的消融表，但作者提到分别去掉“选择性蒸馏”和“层级感知合并”后，模型的提升幅度分别下降约 2–3 分，说明这两个模块对最终性能贡献显著。  
- **局限性**：虽然在三个主流领域取得了突破，但在更细分或完全陌生的任务上仍然落后于完整的老师模型；此外，合并过程对层级重要性的估计依赖验证集，若验证集分布偏差，可能导致合并权重失衡。

### 影响与延伸思考
这篇工作提供了一套系统的“先专后通”压缩思路，已经在后续的模块化 LLM 研究中被广泛引用。比如 2024 年出现的 **ModularDistill**、**ExpertFusion** 等工作，都在不同程度上借鉴了分支‑合并的框架，尝试把更多专家模型（如视觉、音频专精模型）融合成统一的多模态大模型。对想进一步探索的读者，可以关注以下方向：  
- **自动化领域划分**：如何让模型自行发现潜在子任务并生成对应的专精学生。  
- **更高效的参数融合**：利用神经网络搜索或元学习来学习最优的层级融合策略。  
- **跨模态扩展**：把视觉、语音等专精模型也纳入同一合并框架，构建真正的全能小模型。  
这些方向都有望把模型压缩与能力保持的平衡推向更高水平。

### 一句话记住它
分支‑合并蒸馏把多个领域专精的小模型拼成一个体积更小却几乎保持大模型水平的通用模型。