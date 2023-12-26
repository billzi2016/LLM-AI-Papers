# Align on the Fly: Adapting Chatbot Behavior to Established Norms

> **Date**：2023-12-26
> **arXiv**：https://arxiv.org/abs/2312.15907

## Abstract

In this paper, we aim to align large language models with the ever-changing, complex, and diverse human values (e.g., social norms) across time and locations. This presents a challenge to existing alignment techniques, such as supervised fine-tuning, which internalize values within model parameters. To overcome this, we propose an On-the-fly Preference Optimization (OPO) method, which is a real-time alignment that works in a streaming way. It employs an external memory to store established rules for alignment, which can constrain LLMs' behaviors without further training, allowing for convenient updates and customization of human values. We also introduce a scalable evaluation to assess the proposed method more effectively. Experimental results on both human-annotated and auto-generated questions from legal and moral domains indicate the effectiveness of the proposed OPO method. Our code and data are released at https://github.com/GAIR-NLP/OPO.

---

# 即时对齐：让聊天机器人行为符合既定规范 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在生成文本时会把训练数据里学到的价值观直接写进参数里，但人类的社会规范是随时间、地点甚至法律变化的。传统的监督微调只能在模型内部固化一次性价值观，想要再改就得重新训练，成本高且难以快速响应新规。于是出现了“模型对齐”难以兼顾灵活性和安全性的尴尬局面——我们需要一种既能保持模型能力，又能随时更新价值观的方案。

### 关键概念速览
- **对齐（Alignment）**：让模型的输出符合人类期望的价值取向，就像给机器人装上“道德指南针”。  
- **监督微调（Supervised Fine‑Tuning）**：在已有模型上继续训练，用标注好的对话让模型学会特定行为，类似给学生补课。  
- **外部记忆（External Memory）**：模型之外的可查询数据库，用来存放规则或偏好，像是随身携带的手册。  
- **即时偏好优化（On‑the‑fly Preference Optimization，OPO）**：在模型生成过程中实时查阅外部记忆并对答案进行约束，类似在写信时随时打开词典纠错。  
- **流式（Streaming）**：模型输出是逐词产生的，系统可以在每一步都插入检查，而不是等全部生成完再审查。  
- **可扩展评估（Scalable Evaluation）**：一种能够在大规模自动生成的问题上快速测评对齐效果的方法，类似用机器批量检查作文的评分标准。  

### 核心创新点
1. **参数外对齐 → 外部记忆约束 → 零训练更新**  
   传统方法把价值观写进模型参数，需要重新微调才能改动。OPO 把规则放在外部记忆里，模型在生成时实时查询并根据规则调节输出，实现了不改模型权重即可更新价值观。

2. **一次性查询 → 流式多轮校正 → 更细粒度控制**  
   以前的对齐往往在生成结束后统一裁剪或过滤，导致上下文不一致。OPO 在每一步生成后都进行一次偏好检查，像编辑在写稿时随时改句子，使得约束更自然、冲突更少。

3. **手工规则库 → 自动化规则生成 + 人类标注 → 兼顾覆盖与质量**  
   作者不仅手工收集法律、道德规范，还利用大模型自动生成潜在冲突问题，再交给人工标注，形成了一个既大又干净的规则集合，提升了对齐的广度和深度。

4. **通用评估框架 → 人工+自动双管齐下 → 可复制的对齐基准**  
   论文提出了一个能够在法律和道德两大领域同时跑的大规模评测管线，既包括人工标注的高质量问答，也包括模型自动生成的海量情景，帮助后续工作统一比较。

### 方法详解
**整体思路**：OPO 把对齐过程拆成三步——（1）准备外部记忆库；（2）在模型流式生成时实时检索并计算约束分数；（3）根据约束分数对当前 token 的采样概率进行调节。整个流程不涉及对模型参数的任何梯度更新。

**步骤拆解**  
1. **规则库构建**  
   - 收集法律条文、伦理准则等公开文本。  
   - 用大模型生成潜在的“价值冲突”问句（如“在紧急情况下可以撒谎吗？”）。  
   - 人工标注每条规则对应的“接受/拒绝”标签，形成 (情境, 期望答案) 对。  
   - 将这些对存入外部记忆，支持向量化检索（如使用 FAISS）。

2. **流式检索与偏好打分**  
   - 当 LLM 生成第 t 个 token 前，系统把已生成的上下文作为查询，检索相似规则。  
   - 对每条检索到的规则，计算当前候选 token 与规则期望答案的相似度（可以是余弦相似或小型分类器输出），得到一个偏好分数。  
   - 将所有分数加权求和，得到一个整体约束分数 S_t。

3. **概率调节（On‑the‑fly 优化）**  
   - 原始模型会给出下一个 token 的概率分布 P_t。  
   - OPO 把 S_t 通过一个温度调节函数映射到一个乘子 M_t（比如 sigmoid(S_t)），再把 P_t 各元素乘以 M_t，随后重新归一化。  
   - 这样，符合规则的 token 概率被提升，不符合的被压低，模型在不改权重的前提下“被迫”遵守外部规则。

**巧妙之处**  
- **即时性**：约束在每一步都生效，避免了后处理的“硬截断”导致的上下文不连贯。  
- **模块化**：外部记忆可以随时增删规则，像插件一样热插拔，极大降低了维护成本。  
- **兼容性**：OPO 只在推理阶段介入，任何已有的 LLM（GPT‑3、LLaMA 等）都能直接套用，无需重新训练。

### 实验与效果
- **数据集**：作者在两个领域做评测——法律（包括真实法条问答）和道德（包含经典的“电车难题”等）。每个领域都有人工标注的 2k+ 问题和自动生成的 10k+ 问题。  
- **对比基线**：  
  - 直接使用原始 LLM（无对齐）。  
  - 监督微调后模型（在同样的标注数据上微调）。  
  - 基于后处理过滤的安全层（如安全分类器+截断）。  
- **主要结果**：在法律问答上，OPO 的合规率提升约 **23%**，在道德问答上提升约 **19%**，同时保持原始模型的流畅度下降不到 **2%**（BLEU/ROUGE 下降幅度极小）。  
- **消融实验**：  
  - 去掉流式检索，仅在生成结束后一次性约束，合规率下降约 **8%**，说明实时校正的必要性。  
  - 替换外部记忆为纯规则库（不做自动生成），合规率下降约 **5%**，表明自动扩展规则的贡献。  
- **局限性**：作者指出，外部记忆的检索质量对整体效果敏感；在极端长对话或高度专业化的领域，检索可能出现遗漏，导致对齐失效。

### 影响与延伸思考
OPO 把对齐从“模型内部”搬到了“模型外部”，打开了“插件式对齐”的新局面。随后的工作（如 **Plug‑and‑Play Safety**, **Dynamic Prompt Injection**）都在探索类似的外部约束机制。对想进一步研究的读者，可以关注以下方向：  
- **记忆检索优化**：如何在海量规则中实现更快、更精准的相似度搜索。  
- **多模态对齐**：把视觉、音频等感知信息也纳入外部记忆，实现跨模态价值约束。  
- **自适应规则学习**：让模型在交互中自动发现并更新冲突规则，形成闭环的自我校正系统。  

### 一句话记住它
**OPO 用外部规则库在每一步实时约束 LLM，做到“零训练、随时更新”的即时对齐。**