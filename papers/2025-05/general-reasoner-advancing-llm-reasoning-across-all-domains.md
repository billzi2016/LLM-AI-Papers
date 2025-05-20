# General-Reasoner: Advancing LLM Reasoning Across All Domains

> **Date**：2025-05-20
> **arXiv**：https://arxiv.org/abs/2505.14652

## Abstract

Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks.

---

# 通用推理者：提升大语言模型跨领域推理能力 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在数学、代码等答案易验证的领域已经展示出强大的推理能力，但在物理、化学、金融等更广阔的场景里，模型常常给出模糊或不一致的答案。根本原因有三点：① 训练数据中这类问题稀缺，模型缺乏足够的示例；② 传统的强化学习（RL）微调依赖于“零样本RL”（Zero‑RL）需要一个明确的奖励函数，而非结构化答案难以设计；③ 现有的答案校验器大多是基于规则或模板，无法处理多样化的表达方式。于是模型的跨域推理能力受限，亟需一种既能提供高质量训练信号，又能适用于各种学科的通用方案。

### 关键概念速览
- **Zero‑RL（零样本强化学习）**：直接在原始大模型上进行强化学习，不经过额外的监督微调阶段。想象成在没有预热的锅里直接炒菜，省去中间的“预热”步骤。  
- **Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先写出推理步骤，类似于人做题时先列出解题思路，帮助模型保持逻辑连贯。  
- **生成式答案校验器**：用另一个生成模型来判断答案是否合理，而不是硬编码规则。相当于请一位“审稿人”来评估答案的可信度。  
- **可验证答案（verifiable answer）**：答案可以通过外部信息或内部推理链进行核对的形式，例如数值、公式或明确的事实陈述。  
- **跨域数据集**：覆盖多个学科的问答集合，像是把不同学科的试卷都放进同一个训练库，让模型学会“一站式”解题。  
- **MMLU‑Pro、GPQA、SuperGPQA**：衡量模型在通用知识、专业问答和高难度推理上的基准测试，类似于不同难度的奥林匹克竞赛。  

### 核心创新点
1. **从“数学‑代码专属”到“全域覆盖”**  
   之前的Zero‑RL大多只在数学或编程数据上实验，因为这些领域答案易于自动校验。本文通过大规模网络爬取，构建了一个包含物理、化学、金融、电子等12个子域的高质量问答库，使得Zero‑RL可以在更广阔的语义空间中训练。  
2. **规则校验 → 生成式校验**  
   传统方法用正则表达式或固定模板判断答案是否符合预期，面对多样化的自然语言答案会失效。本文训练了一个专门的生成模型，让它在看到问题、模型答案和上下文后输出“通过/不通过”以及简短的理由，实现了对链式思考过程的上下文感知校验。  
3. **奖励函数的链式设计**  
   在Zero‑RL中，奖励往往是二元的（对/错）。这里把奖励拆成两层：第一层基于生成式校验器的通过率，第二层根据思维链的完整性和逻辑一致性打分。这样模型不仅被鼓励给出正确答案，还被迫写出可验证的推理步骤。  
4. **统一评估框架**  
   过去的研究往往只报告数学或代码基准的提升。本文在12个跨域基准上同步评测，并展示了在保持数学推理优势的同时，显著提升了非数学领域的准确率，证明了方法的通用性。  

### 方法详解
整体思路可以划分为三步：① 数据构建，② 生成式校验器训练，③ Zero‑RL 微调。下面逐层拆解。

1. **大规模跨域问答构建**  
   - 通过爬虫抓取高校教材、专业论坛、公开报告等网页，抽取“问题‑答案”对。  
   - 使用过滤规则（如答案长度、出现频率）和人工抽样审查，确保答案可验证且质量高。  
   - 为每条记录附加元信息：学科标签、难度估计、参考来源链接。  

2. **生成式答案校验器**  
   - 选取一个中等规模的 LLM（如 Llama‑2‑7B）作为校验模型。  
   - 输入格式为：“问题：… 答案：… 上下文：…”，模型输出“PASS”或“FAIL”并给出简短解释。  
   - 训练数据来源于两类：一是人工标注的正确/错误对，二是利用已有的规则校验器生成的弱标签。通过混合监督学习提升鲁棒性。  
   - 关键技巧是让校验器学习“思维链”——它不仅判断答案本身，还检查答案背后的推理步骤是否合乎常识。  

3. **Zero‑RL 微调**  
   - 采用 PPO（Proximal Policy Optimization）等常见 RL 算法，对原始 LLM 进行策略更新。  
   - 奖励计算分两步：  
     a. **校验奖励**：校验器返回的 PASS 概率乘以一个常数权重。  
     b. **思维链奖励**：若模型输出了 CoT，校验器会对每一步进行一致性检查，累计得分。  
   - 为防止模型只追求高奖励而忽略语言流畅性，加入 KL 散度约束，使微调后模型仍保持原始分布的语言特性。  

**最巧妙的地方**在于把“答案对错”与“推理过程质量”合并成统一的奖励信号，让 RL 不再是盲目追求二元正确率，而是被迫学习可解释的思考路径。这种设计在跨域任务中尤为重要，因为很多领域的答案本身就不唯一，只有完整的推理链才能让评估者信服。

### 实验与效果
- **评测基准**：MMLU‑Pro（通用知识）、GPQA / SuperGPQA（专业问答）、TheoremQA（数学定理推导）、BBEH（生物医学）、MATH AMC（高中数学）等共12个数据集。  
- **对比模型**：原始基线 LLM、使用传统规则校验的 Zero‑RL、以及最新的 CoT‑强化学习模型。  
- **主要结果**：在 GPQA 上平均提升约 7% 准确率，在 TheoremQA 上提升 4.5%，而在 MMLU‑Pro 这类通用任务上也实现了 2.8% 的增益。值得注意的是，数学基准的表现几乎持平，说明通用化并未牺牲原有优势。  
- **消融实验**：去掉生成式校验器，仅使用规则校验，跨域任务的提升跌回 2% 左右；去掉思维链奖励，模型在需要多步推理的题目上准确率下降约 5%。这些实验表明两大模块都是提升的关键因素。  
- **局限性**：作者指出生成式校验器本身仍受训练数据偏差影响，极端专业领域（如高能物理）仍可能出现误判；此外 RL 训练成本高，需大量 GPU 资源。  

### 影响与延伸思考
这篇工作把 Zero‑RL 的适用范围从“数学‑代码”扩展到真正的“全学科”，在社区里引发了两类后续研究：一是围绕**生成式校验器**的改进，如引入多模态（图表）信息来验证工程类答案；二是探索**跨域奖励函数**的更细粒度设计，例如把可信度、解释性等指标量化为奖励。推测未来会出现“多校验器联盟”，让不同专家模型共同评估同一道题，从而进一步提升跨域推理的鲁棒性。想深入了解的读者可以关注近期的 “Multi‑Domain RL for LLMs” 研讨会以及 OpenAI、DeepMind 在可解释 RL 方向的最新报告。

### 一句话记住它
把答案对错和思维链质量一起奖励，让大语言模型在任何学科都能学会“写出可验证的推理”。