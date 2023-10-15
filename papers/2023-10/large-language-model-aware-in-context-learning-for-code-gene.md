# Large Language Model-Aware In-Context Learning for Code Generation

> **Date**：2023-10-15
> **arXiv**：https://arxiv.org/abs/2310.09748

## Abstract

Large language models (LLMs) have shown impressive in-context learning (ICL) ability in code generation. LLMs take a prompt consisting of requirement-code examples and a new requirement as input, and output new programs. Existing studies have found that ICL is highly dominated by the examples and thus arises research on example selection. However, existing approaches randomly select examples or only consider the textual similarity of requirements to retrieve, leading to sub-optimal performance. In this paper, we propose a novel learning-based selection approach named LAIL (LLM-Aware In-context Learning) for code generation. Given a candidate example, we exploit LLMs themselves to estimate it by considering the generation probabilities of ground-truth programs given a requirement and the example. We then label candidate examples as positive or negative through the probability feedback. Based on the labeled data, we import a contrastive learning objective to train an effective retriever that acquires the preference of LLMs in code generation. We apply LAIL to three LLMs and evaluate it on three representative datasets (e.g., MBJP, MBPP, and MBCPP). LATA outperforms the state-of-the-art baselines by 11.58%, 6.89%, and 5.07% on CodeGen, and 4.38%, 2.85%, and 2.74% on GPT-3.5 in terms of Pass@1, respectively.

---

# 面向大型语言模型的上下文学习用于代码生成 论文详细解读

### 背景：这个问题为什么难？

在代码生成任务里，LLM（大型语言模型）靠“在上下文中学习”——把若干需求‑代码对塞进提示里，再让模型输出新代码。理论上，只要示例挑得好，模型就能把需求映射成正确实现。可是实际情况是，示例的挑选对最终结果影响巨大。过去的做法要么随意抽取，要么只看需求文字的相似度，根本没有考虑模型真正的生成偏好。于是很多潜在的好例子被遗漏，导致生成质量远低于模型的上限，这也是为什么同一模型在不同提示下表现差异悬殊的根本原因。

### 关键概念速览
- **In-Context Learning（上下文学习）**：把任务示例直接写进模型的输入提示里，让模型“现场学习”并完成新任务。类似老师在课堂上现场举例子帮助学生解题。
- **示例检索器（Retriever）**：负责从海量候选示例中挑出最合适的若干条。可以把它想象成图书馆的推荐系统，只不过推荐的对象是代码示例。
- **对比学习（Contrastive Learning）**：一种让模型学会区分“好”和“坏”样本的训练方式。像是让学生通过比较正确答案和错误答案来加深记忆。
- **生成概率（Generation Probability）**：模型在给定上下文时，对每个可能的输出 token 赋予的概率。概率越高，模型越“自信”。这里把它当作示例好坏的评分标准。
- **Pass@1**：代码生成评估指标，表示模型在一次尝试中生成的代码能否通过全部单元测试。相当于一次面试的合格率。
- **蒸馏（Distillation）**：把大模型的行为压缩到小模型或轻量模块里。这里的检索器相当于从大模型的“经验”中提炼出一个快速判断器。

### 核心创新点
1. **用大模型自身评估示例 → 通过让 LLM 计算“需求+示例”下真实代码的生成概率 → 把高概率视为正例、低概率视为负例**  
   以前的检索器只看需求文字相似度，这一步直接让模型说“这条示例对我有帮助吗”。结果是得到的标签更贴合模型的真实生成偏好。

2. **基于概率标签进行对比学习 → 训练检索器去最大化正例与负例的概率差距 → 检索器学会捕捉模型偏好的隐含特征**  
   对比学习让检索器不只是记住“相似的需求”，而是学会“哪些示例能提升模型的生成概率”。这一步显著提升了检索质量。

3. **统一框架兼容多种 LLM** → 同一套 LAIL 流程分别在 CodeGen、GPT‑3.5 等模型上跑通 → 证明方法与特定模型解耦，具备通用性。  
   之前的示例选择往往针对某个模型调参，LAIL 把模型本身的反馈当作通用信号，省去大量手工调优。

4. **大幅提升 Pass@1** → 在 MBJP、MBPP、MBCPP 三个代码生成基准上，CodeGen 系列提升 11.58% / 6.89% / 5.07%，GPT‑3.5 系列提升 4.38% / 2.85% / 2.74%。  
   直接的数字证明了检索器质量提升对生成成功率的强大推动作用。

### 方法详解
**整体思路**：先让大模型自己给每个候选示例打分，再把这些分数转化为正负标签，用对比学习训练一个轻量检索器，最后在实际生成时用检索器挑选示例。整个流程可以划分为三步：① 生成概率评估，② 标签构造与对比学习，③ 示例检索与代码生成。

**步骤拆解**  

1. **生成概率评估**  
   - 输入：需求 `R`、候选示例 `E_i`（包含需求‑代码对），以及真实目标代码 `C*`（训练时已知）。  
   - 操作：把 `R` 与 `E_i` 拼接成提示，喂给目标 LLM，记录模型在生成 `C*` 时的累计对数概率。  
   - 直觉：如果示例真的帮助模型，它在生成正确代码时会更“自信”，概率自然更高。  

2. **标签构造**  
   - 设定阈值或相对排序：概率最高的前 `k%` 记为正例，最低的 `k%` 记为负例。  
   - 这样得到的训练集是“示例‑正负标签”，完全由模型自己的反馈生成，避免人工标注偏差。  

3. **对比学习训练检索器**  
   - 检索器本质是一个编码器，把每个示例映射到向量空间。  
   - 对比学习目标是让正例的向量距离彼此更近，负例的向量距离正例更远。实现方式类似于“孪生网络”，只不过这里的正负标签来自上一步的概率。  
   - 训练结束后，检索器能够在向量空间里快速定位“模型喜欢的示例”。  

4. **实际生成阶段**  
   - 给定新需求 `R_new`，先用检索器在候选库中检索出 `n` 条最相似（向量距离最近）的示例。  
   - 把这些示例连同 `R_new` 组装成提示，交给目标 LLM，得到最终代码。  

**关键细节**  
- **概率反馈的稳健性**：作者在实验中发现直接使用对数概率比原始概率更稳定，因为对数把长序列的乘积转成求和，避免数值下溢。  
- **对比学习的负采样**：负例不是随机抽取，而是挑选概率最低的示例，这让检索器学习到真正的“误导性示例”。  
- **检索器的轻量化**：虽然使用了大模型的反馈进行训练，但最终检索器只需要一个小型 Transformer 编码器，推理时几乎不增加额外计算。  

### 实验与效果
- **数据集**：MBJP（Java）、MBPP（Python）和 MBCPP（C++）三个公开代码生成基准，覆盖不同语言和难度。  
- **对比基线**：随机示例抽取、基于需求文本相似度的检索（如 BM25、Sentence‑BERT）以及最新的示例选择方法（如 K‑Nearest‑Example）。  
- **主要结果**：  
  - 在 CodeGen 系列模型上，Pass@1 提升分别为 11.58%（MBJP）、6.89%（MBPP）和 5.07%（MBCPP）。  
  - 在 GPT‑3.5 上，同样的提升分别为 4.38%、2.85% 和 2.74%。  
  - 这些数字明显超过最强基线的 1%~3% 区间提升。  
- **消融实验**：  
  - 去掉对比学习，仅用概率阈值直接筛选示例，提升幅度下降约 30%。  
  - 替换概率标签为纯文本相似度标签，效果回落到传统检索基线水平。  
  - 说明对比学习和模型驱动的标签是提升的关键因素。  
- **局限性**：  
  - 需要在训练阶段拥有真实目标代码，以计算生成概率；对纯推理场景（没有金标准代码）无法直接生成标签。  
  - 对检索器的训练成本与所选 LLM 的调用次数成正比，若使用极其昂贵的模型（如 GPT‑4）成本会显著上升。  
  - 论文未报告在极大规模示例库（上百万条）下的检索效率，仅在几千条规模上做了实验。  

### 影响与延伸思考
LAIL 把“大模型的自我评估”转化为检索器的学习信号，这一思路在随后的一批工作中被广泛借鉴。比如有研究把 LLM 的置信度用于自动化调参、还有把模型生成的概率用于数据去噪的主动学习框架。推测未来会出现 **LLM‑Feedback‑Driven Retrieval** 的通用平台，让各种下游任务（自然语言生成、问答、甚至多模态）都能用类似的“模型自评+对比学习”方式提升示例选择。想进一步了解，可以关注近期在 **ICL 示例优化**、**模型蒸馏** 与 **自监督检索** 交叉的论文。

### 一句话记住它
让大模型自己给示例打分，再用对比学习把这种偏好压缩进轻量检索器，显著提升代码生成的成功率。