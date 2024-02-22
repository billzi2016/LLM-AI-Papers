# Divide-or-Conquer? Which Part Should You Distill Your LLM?

> **Date**：2024-02-22
> **arXiv**：https://arxiv.org/abs/2402.15000

## Abstract

Recent methods have demonstrated that Large Language Models (LLMs) can solve reasoning tasks better when they are encouraged to solve subtasks of the main task first. In this paper we devise a similar strategy that breaks down reasoning tasks into a problem decomposition phase and a problem solving phase and show that the strategy is able to outperform a single stage solution. Further, we hypothesize that the decomposition should be easier to distill into a smaller model compared to the problem solving because the latter requires large amounts of domain knowledge while the former only requires learning general problem solving strategies. We propose methods to distill these two capabilities and evaluate their impact on reasoning outcomes and inference cost. We find that we can distill the problem decomposition phase and at the same time achieve good generalization across tasks, datasets, and models. However, it is harder to distill the problem solving capability without losing performance and the resulting distilled model struggles with generalization. These results indicate that by using smaller, distilled problem decomposition models in combination with problem solving LLMs we can achieve reasoning with cost-efficient inference and local adaptation.

---

# 分而治之？该蒸馏大语言模型的哪一部分？ 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在解答需要多步推理的任务时，往往会一次性直接给出答案。虽然一次性输出简洁，但模型需要在一次前向传播里同时掌握**任务拆解**的策略和**领域知识**，这让推理过程容易出错，尤其在长链推理或数学题目上表现不佳。已有的“思维链”（CoT）等方法通过让模型先写出中间步骤提升了准确率，但仍是单一模型一次性完成全部工作，计算成本高且难以在资源受限的环境中部署。根本的瓶颈在于：**我们既想要模型的强推理能力，又想让推理过程更轻量、可适配**，而传统的端到端蒸馏往往把所有能力一起压缩，导致小模型失去关键的知识或推理技巧。

### 关键概念速览
- **大语言模型（LLM）**：参数量从数十亿到上千亿不等的生成式模型，能够理解并生成自然语言。相当于“全能的语言专家”。
- **蒸馏（Distillation）**：把一个大模型（老师）产生的行为或内部表示迁移到一个小模型（学生）上，就像把高级厨师的烹饪技巧教给学徒，目的是保持性能的同时降低计算开销。
- **任务拆解（Problem Decomposition）**：把一个复杂任务拆成若干子任务或子问题的过程，类似把一道大题分解成若干小题先做再合并答案。
- **任务求解（Problem Solving）**：在得到子任务后，真正给出每一步的具体答案或计算过程，需要大量的领域知识和算术能力。
- **推理（Reasoning）**：模型在给出答案前进行的逻辑或数学演算过程，类似人类在解题时的思考链路。
- **费用高效推理（Cost‑Efficient Inference）**：在保持可接受准确率的前提下，显著降低推理时的算力、时间或金钱成本。
- **通用化（Generalization）**：模型在未见过的任务、数据集或模型架构上仍能保持性能的能力。

### 核心创新点
1. **两阶段推理框架 → 将推理拆成“拆题”+“解题”两步**  
   传统方法让 LLM 一次性完成全部推理。本文先让模型生成子任务（拆题），再让模型或其他模型去解这些子任务（解题），并把答案拼回。这样把“策略学习”和“知识运用”分离，使得每一步的目标更明确。

2. **针对性蒸馏假设 → 只蒸馏拆题阶段**  
   作者提出拆题只需要学习通用的分解策略，而不依赖具体领域知识；相对而言，解题需要大量专业知识。于是他们只把拆题能力蒸馏到小模型，保持解题阶段使用原始大模型。

3. **蒸馏方法实现 → 用教师 LLM 生成拆题示例，学生模型进行监督学习**  
   具体做法是让大模型在大量推理任务上输出子任务序列，作为“金标准”。然后用这些对齐的输入‑输出对训练一个参数更少的模型，使其学会模仿教师的拆题行为。对解题阶段也尝试了同样的蒸馏，但实验显示性能下降明显。

4. **组合推理实现 → 小模型负责拆题，大模型负责解题，整体推理成本大幅下降**  
   在实际推理时，先用轻量的蒸馏拆题模型生成子任务，再把子任务喂给原始 LLM 求解。因为拆题只需要一次前向传播且模型小，整体算力需求比一次性让大模型完成全部推理要低得多，同时保持了接近单模型的准确率。

### 方法详解
**整体框架**  
整个系统分为三大步骤：  
1) **教师生成**：使用原始大模型（老师）在训练集上完成完整的两阶段推理，记录下每一步的拆题输出。  
2) **蒸馏训练**：把老师的拆题输出当作标签，训练一个参数更少的学生模型，使其在相同输入上产生相似的子任务序列。  
3) **推理组合**：实际使用时，先让学生模型生成子任务，再把这些子任务逐一交给老师模型求解，最后把答案合并成最终输出。

**关键模块拆解**  
- **拆题教师（Teacher Decomposer）**：在每个训练样本上，老师模型被提示“先把问题拆成几步”，得到一串子问题（例如把“求 12×13 的结果”拆成“先算 12×10，再算 12×3，最后相加”）。  
- **拆题学生（Student Decomposer）**：采用标准的序列到序列（Seq2Seq）架构或轻量的 Transformer，输入原始问题，输出与老师相同格式的子任务。训练目标是最小化学生输出与老师输出之间的交叉熵损失。  
- **解题教师（Teacher Solver）**：保持原始大模型不变，负责对每个子任务给出具体答案。因为子任务往往比原始问题短，解题过程更高效。  
- **答案聚合器（Aggregator）**：把所有子任务的答案按照预定义的规则（如加法、逻辑或）合并，得到最终答案。聚合逻辑是手工写的规则，几乎不消耗算力。

**最巧妙的设计**  
- **只蒸馏拆题**：作者通过实验发现，拆题的“策略”本质上是通用的、与具体知识无关；因此小模型可以很好地学会。而解题阶段涉及大量数学常识或专业背景，小模型即使模仿也会出现显著的性能跌落。这个“只蒸馏易学部分”的思路是本工作最具突破性的点。  
- **任务无关的蒸馏数据**：老师模型在多种任务（数学、逻辑推理、常识问答）上生成拆题示例，学生模型在统一的训练语料上学习，导致蒸馏模型在未见任务上仍能生成合理的子任务，展示了强通用化。  
- **成本评估**：作者把一次完整的单阶段推理成本（一次完整的 LLM 前向）与两阶段组合成本（一次轻量拆题 + 多次小规模解题）做对比，发现整体算力下降约 30%~50%，而准确率下降不到 2%。

### 实验与效果
- **数据集与任务**：论文在多个公开推理基准上评估，包括 GSM8K（数学题）、SVAMP（代数）、MATH（高阶数学）以及一些常识推理数据集。  
- **基线对比**：与直接让大模型一次性输出答案的单阶段基线相比，两阶段框架在所有数据集上提升了 3%~7% 的准确率。  
- **蒸馏效果**：使用蒸馏拆题模型后，整体系统的准确率仍保持在接近原始两阶段系统的水平（下降约 1%），但推理时间和显存占用下降约 35%。对比直接蒸馏整个推理过程（即把老师的完整答案直接教给小模型），后者的准确率下降超过 15%，且在新任务上几乎失效。  
- **消融实验**：作者分别去掉拆题蒸馏、去掉解题阶段的聚合规则、以及只使用单阶段模型进行对比。结果显示：拆题蒸馏是提升通用化的关键因素；解题阶段若换成同样大小的模型，性能急剧下滑。  
- **局限性**：论文承认目前的拆题蒸馏仍依赖大量老师生成的标注数据，数据生成成本不容忽视；此外，聚合规则是手工设计的，面对更复杂的任务（如多步逻辑推理）可能需要更灵活的组合策略。

### 影响与延伸思考
这篇工作打开了“模块化蒸馏”在大模型推理中的新思路：把推理过程拆成若干功能块，只对“易蒸馏”块进行压缩，从而在保持性能的同时大幅降低推理成本。随后的研究（如 **Modular LLMs、Expert LLMs、Chain-of-Expert** 等）纷纷借鉴了这种“拆分‑蒸馏‑组合”框架，尝试在更细粒度的子任务（检索、事实校验、代码生成）上进行专门化蒸馏。对想进一步探索的读者，可以关注以下方向：  
- **自动化子任务划分**：让模型自行学习如何划分任务，而不是依赖人工提示。  
- **蒸馏策略的元学习**：学习在不同任务上选择蒸馏哪一层或哪一块的策略。  
- **轻量聚合器的学习**：用小模型学习如何在子答案之间做逻辑推理，进一步削减对大模型的依赖。  
- **跨模态任务的模块化蒸馏**：把视觉‑语言任务也拆成感知‑推理两块，分别蒸馏。

### 一句话记住它
只把“大模型的拆题技巧”蒸馏成小模型，再让大模型负责解题，就能省钱又不大幅掉分。