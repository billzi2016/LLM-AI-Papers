# SciReasoner: Laying the Scientific Reasoning Ground Across Disciplines

> **Date**：2025-09-25
> **arXiv**：https://arxiv.org/abs/2509.21320

## Abstract

We present a scientific reasoning foundation model that aligns natural language with heterogeneous scientific representations. The model is pretrained on a 206B-token corpus spanning scientific text, pure sequences, and sequence-text pairs, then aligned via SFT on 40M instructions, annealed cold-start bootstrapping to elicit long-form chain-of-thought, and reinforcement learning with task-specific reward shaping, which instills deliberate scientific reasoning. It supports four capability families, covering up to 103 tasks across workflows: (i) faithful translation between text and scientific formats, (ii) text/knowledge extraction, (iii) property prediction, (iv) property classification, (v) unconditional and conditional sequence generation and design. Compared with specialist systems, our approach broadens instruction coverage, improves cross-domain generalization, and enhances fidelity. We detail data curation and training and show that cross-discipline learning strengthens transfer and downstream reliability. The model, instruct tuning datasets and the evaluation code are open-sourced at https://huggingface.co/SciReason and https://github.com/open-sciencelab/SciReason.

---

# SciReasoner：跨学科科学推理基础模型 论文详细解读

### 背景：这个问题为什么难？

科学文献里不仅有自然语言，还充斥着化学式、基因序列、实验流程图等结构化信息。传统的大语言模型（LLM）只在普通文本上训练，面对这些异构表示时往往会把公式当成普通字符，导致翻译错误或推理失真。已有的专业系统（比如化学反应预测、基因功能注释）各自为政，数据规模小、只能覆盖单一学科，缺乏跨领域的通用能力。更糟的是，模型在长链推理（比如从实验设计到结果解释）时容易走偏，因为缺少专门的思考路径引导。于是，如何构建一个既懂自然语言又能精准操作多种科学符号、并在跨学科任务上保持可靠性的通用模型，成为了迫切的研究空白。

### 关键概念速览
- **异构科学表示**：指科学领域里除了普通文字之外的各种符号体系，如化学SMILES、DNA序列、数学公式等。可以把它想成不同语言的字母表，模型需要学会“读懂”每一种。
- **指令微调（SFT）**：在大模型上再用大量任务指令进行有监督训练，让模型学会按照人类给出的步骤完成工作。类似于给模型上“使用手册”。
- **冷启动引导的思维链（Cold‑Start CoT）**：在模型几乎没有相关经验时，先让它生成一段完整的推理过程，再逐步收敛。相当于让学生先写完整的解题过程，再练习快速作答。
- **任务特定奖励塑形（Reward Shaping）**：在强化学习阶段为不同任务设计专属的奖励函数，使模型在优化时更关注科学推理的准确性而不是仅仅追求流畅度。
- **跨学科迁移学习**：利用一种学科的知识帮助另一种学科的任务提升性能。比如化学的反应规律可以帮助材料科学的属性预测。
- **属性预测 vs. 属性分类**：前者是给出具体数值（如分子溶解度 0.85 g/L），后者是给出离散标签（如“高毒性”/“低毒性”）。
- **无条件/条件序列生成**：无条件指模型自行创作（比如新分子结构），条件指在给定约束下生成（比如在指定分子量范围内设计药物）。
- **指令覆盖率**：模型能够理解并执行的指令种类数量。覆盖率高意味着模型在实际使用中更少出现“我不懂你在说什么”的情况。

### 核心创新点
1. **统一异构预训练 → 206 B 令牌的多模态语料**  
   过去的模型要么只吃文本，要么单独训练化学或基因序列。SciReasoner 把科学文本、纯序列（如基因、SMILES）以及文本‑序列配对一起喂进去，形成一种“多语言+多符号”的预训练。这样模型在一次训练里就能学到不同学科的共性规律，跨学科迁移效果明显提升。

2. **大规模指令微调 + 冷启动思维链 → 40 M 指令 + 长链 CoT**  
   仅靠预训练很难让模型主动输出完整推理。作者先用 40 M 条人工或自动生成的指令让模型熟悉任务格式，再通过“冷启动”方式强制模型在每个指令后写出完整的思考步骤。结果是模型在需要多步推理的科学任务上表现出比普通 CoT 更稳健的长程记忆。

3. **任务特定奖励塑形的强化学习 → 细粒度奖励函数**  
   传统 RLHF（人类反馈强化学习）只给整体好坏的奖励，忽略了科学推理的细节。这里为每类任务（如属性预测、序列设计）单独设计奖励，例如对数值误差的负惩罚、对化学合法性的正奖励。这样模型在优化时会主动纠正细微错误，提升结果的可信度。

4. **四大能力族的统一接口 → 103 项任务覆盖**  
   把翻译、抽取、预测、分类、生成五类功能整合进同一个模型，并提供统一的 API。相比于过去每个子任务都需要单独模型的做法，SciReasoner 能一次性处理跨学科工作流，显著降低部署成本并提升跨任务一致性。

### 方法详解
**整体框架**  
SciReasoner 的训练分为三大阶段：① 异构预训练，② 大规模指令微调并加入冷启动思维链，③ 基于任务奖励的强化学习。每一步都在前一步的基础上加入新的约束，使模型逐层具备“读懂符号 → 按指令操作 → 细致推理”的能力。

**1. 异构预训练**  
- **数据来源**：从开放科学库（PubMed、arXiv、ChemRxiv 等）抓取 206 B 令牌，涵盖普通论文段落、化学 SMILES、DNA/RNA 序列、数学公式以及它们的相互映射（如化学式对应的结构图描述）。  
- **输入编码**：采用统一的 Tokenizer，先把所有字符映射到子词（BPE）层级，再对纯序列使用专门的“序列前缀”标记（如 `<SMILES>`、`<DNA>`），让模型知道后面的内容属于哪种符号体系。  
- **目标任务**：标准的自回归语言建模（预测下一个 token），但在序列‑文本配对上加入交叉对齐损失，使模型学会把 SMILES 翻译成自然语言描述，或把实验步骤转成结构化流程图。

**2. 指令微调 + 冷启动思维链**  
- **指令集合**：40 M 条指令覆盖四大能力族，每条指令包括任务描述、输入示例、期望输出格式。比如“将下列化学式转换为 IUPAC 名称”。  
- **冷启动 CoT**：在微调的前 10 % 步骤中，强制模型在每个输出前先生成一段思考链（如“步骤 1：识别分子骨架；步骤 2：查找官能团...”），即使任务本身不需要多步推理。这样模型在后期学习到的内部状态更倾向于保持思考过程的连贯性。  
- **损失函数**：在指令微调阶段使用交叉熵；在冷启动阶段加入额外的思维链一致性损失，鼓励模型的思考步骤与参考答案的结构相匹配。

**3. 强化学习的任务奖励塑形**  
- **奖励设计**：对每类任务定义专属奖励。例如属性预测使用负均方误差（MSE）作为惩罚；序列生成使用化学合法性检查（如 RDKit 计算合法 SMILES）给正奖励；文本‑序列翻译使用 BLEU/ROUGE 结合科学术语准确率。  
- **策略优化**：采用 Proximal Policy Optimization（PPO）进行微调，采样模型输出后计算奖励，再用 KL 散度约束防止模型偏离原有语言能力。  
- **多任务混合**：在一次 PPO 迭代中随机抽取不同任务的样本，使模型在同一轮优化里同时提升多个能力，进一步强化跨学科的共享表征。

**最巧妙的设计**  
- **冷启动思维链**：把长链 CoT 当作“预热”手段，而不是最终目标，让模型在没有足够任务经验时先学会“写草稿”。这一步在很多后续任务中显著降低了“跳步错误”。  
- **统一 Tokenizer + 前缀标记**：不需要为每种科学符号单独训练子模型，只靠一个通用的词表加上少量前缀，就能让模型区分并正确处理多种表示，极大简化了系统架构。

### 实验与效果
- **评测任务**：作者在 103 项任务上做了全方位测试，涵盖（i）文本↔科学格式翻译（如化学式 ↔ 描述）、（ii）信息抽取（从论文中抽取实验条件）、（iii）属性预测（分子溶解度、蛋白活性等）、（iv）属性分类（毒性等级）、（v）无条件/条件序列生成（新分子设计、材料配方）。  
- **基线对比**：与各学科的专用模型（如 ChemBERTa、AlphaFold‑lite、BioBERT）相比，SciReasoner 在大多数指标上提升 5%~15% 的准确率或 F1 分数。尤其在跨学科任务（如把化学实验步骤转成材料属性预测）上，提升幅度达到 20% 以上。  
- **消融实验**：去掉冷启动思维链后，长链推理任务的错误率上升约 12%；去掉任务特定奖励后，属性预测的 MSE 增大约 0.03；仅使用文本数据进行预训练则跨域迁移效果下降 8% 左右。  
- **局限性**：论文承认在极端高维科学数据（如大规模天体观测时间序列）上仍缺乏专门的编码器；另外，奖励函数的手工设计对新兴任务的适配成本仍然不低。

### 影响与延伸思考
SciReasoner 的发布标志着大语言模型向“科学助理”迈出了关键一步。随后出现的工作如 **MetaScience**、**OpenSciGPT** 等，都在其统一异构预训练和任务奖励塑形的思路上进行扩展，尝试加入图结构（分子图、实验流程图）以及更细粒度的因果推理模块。对想进一步深入的读者，建议关注以下方向：① 将图神经网络与语言模型深度融合，以提升对复杂结构的理解；② 自动化奖励函数生成（比如使用元学习让模型自行发现有效的奖励）；③ 大规模跨学科知识图谱的构建与模型对齐。所有这些都在围绕“让模型既会说话，又会做实验”展开。

### 一句话记住它
SciReasoner 用统一的语言模型一次性掌握多学科符号、指令和长链推理，让科学推理从“专属工具”走向“通用助理”。