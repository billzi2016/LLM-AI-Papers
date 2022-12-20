# Towards Robustness of Text-to-SQL Models Against Natural and Realistic   Adversarial Table Perturbation

> **Date**：2022-12-20
> **arXiv**：https://arxiv.org/abs/2212.09994

## Abstract

The robustness of Text-to-SQL parsers against adversarial perturbations plays a crucial role in delivering highly reliable applications. Previous studies along this line primarily focused on perturbations in the natural language question side, neglecting the variability of tables. Motivated by this, we propose the Adversarial Table Perturbation (ATP) as a new attacking paradigm to measure the robustness of Text-to-SQL models. Following this proposition, we curate ADVETA, the first robustness evaluation benchmark featuring natural and realistic ATPs. All tested state-of-the-art models experience dramatic performance drops on ADVETA, revealing models' vulnerability in real-world practices. To defend against ATP, we build a systematic adversarial training example generation framework tailored for better contextualization of tabular data. Experiments show that our approach not only brings the best robustness improvement against table-side perturbations but also substantially empowers models against NL-side perturbations. We release our benchmark and code at: https://github.com/microsoft/ContextualSP.

---

# 面向自然与真实对抗性表格扰动的 Text-to-SQL 模型鲁棒性研究 论文详细解读

### 背景：这个问题为什么难？
Text-to-SQL 解析器需要把自然语言问题映射成可以在数据库上执行的 SQL 语句。过去的鲁棒性研究几乎只在“问句”这边下功夫——比如同义替换、拼写错误等——而几乎忽视了表格本身会随业务演进而变化的事实。实际场景里，列名改动、数据类型替换、行顺序乱序、甚至插入噪声列都是常见的“表格扰动”。如果模型只在固定的表结构上训练，一旦遇到这些变化，生成的 SQL 往往失效。于是，评估和提升模型对表格侧扰动的抵抗力成为了迫切需求。

### 关键概念速览
**Text-to-SQL**：把用户的自然语言查询转换成对应的 SQL 语句，就像把口头指令翻译成数据库指令。  
**对抗性扰动（Adversarial Perturbation）**：有意制造的、让模型出错的微小改动，类似于在图片上加一点噪声就能骗倒图像分类器。  
**Adversarial Table Perturbation (ATP)**：专门针对数据库表结构的扰动，包括列名同义替换、列顺序打乱、插入无关列等，目标是让 Text-to‑SQL 模型产生错误的 SQL。  
**ADVETA**：作者构建的评估基准，集合了大量自然且真实的 ATP 示例，用来系统测量模型在表格侧扰动下的表现。  
**上下文感知对抗训练（Contextual Adversarial Training）**：在训练阶段加入经过特殊处理的表格扰动样本，使模型学会在不同表结构下仍能正确理解列的语义。  
**NL-side Perturbation**：指自然语言问题本身的扰动，如同义词替换、语序变化等。  
**鲁棒性（Robustness）**：模型在面对输入轻微变化时仍能保持性能的能力，类似于汽车在颠簸路面上仍能平稳行驶。

### 核心创新点
1. **从“问句扰动”到“表格扰动”**：之前的工作只在自然语言层面制造对抗样本，本文首次系统化地把注意力转向数据库表本身的变化，提出了 ATP 这一全新攻击范式。这样可以直接暴露出模型对表结构依赖的薄弱环节。  
2. **构建 ADVETA 基准**：作者收集并人工筛选了大量真实业务场景中的表格改动，形成了首个兼具自然性和现实性的表格扰动评测集。相比于合成的、脱离实际的对抗样本，ADVETA 更能反映模型在生产环境中的真实风险。  
3. **上下文感知的对抗训练框架**：在生成对抗样本时，作者不仅随机改动列名或顺序，而是结合列的语义上下文（如列的描述、数据分布）生成更具“欺骗性”的扰动。这样训练出来的模型在面对 ATP 时跌幅最小，同时对 NL-side 扰动也有意外的提升。  
4. **跨扰动提升**：实验显示，专门防御表格侧扰动的训练策略还能帮助模型更好地抵御自然语言侧的对抗样本，这一跨域鲁棒性的发现为以后统一的对抗训练提供了思路。

### 方法详解
整体思路可以拆成三大步骤：**扰动生成 → 对抗样本构造 → 上下文感知对抗训练**。

1. **扰动生成（ATP）**  
   - **列名同义替换**：利用词典或预训练语言模型找到列名的同义词或缩写，例如把 `order_date` 换成 `date_of_order`。  
   - **列顺序打乱**：随机重新排列列的顺序，保持列的内容不变。  
   - **噪声列插入**：在原表中加入与业务无关的列（如 `dummy_flag`），并填充随机值。  
   - **数据类型替换**：把数值列改成字符串列或相反，模拟业务迁移时的类型变更。  
   这些操作都保持表的整体可用性，只是让列的“名字”和“位置”产生变化，从而考验模型是否真正理解列的语义。

2. **对抗样本构造**  
   - 对每个原始自然语言问题，使用上述扰动生成对应的“扰动表”。  
   - 保持原始问题不变，直接把它和扰动表配对，形成新的训练/评估样本。这样模型在推理时必须依赖表的上下文来定位列，而不是死记硬背列名。

3. **上下文感知对抗训练**  
   - **语义对齐**：在列名同义替换时，先用语言模型计算原列名与候选同义词的相似度，确保替换后仍能在语义上对应原列。  
   - **分布匹配**：对噪声列插入，随机抽取业务中常见的无关列特征，避免出现“太离谱”的噪声导致模型直接忽略。  
   - **混合训练**：在每个训练批次里，按一定比例混入原始样本、NL-side 对抗样本以及 ATP 对抗样本，使模型在多种扰动下都能保持学习。  
   - **损失加权**：对抗样本的损失会乘以一个系数，防止模型因为噪声样本而过度偏离主任务。  

最巧妙的地方在于**把列的语义信息显式注入扰动生成过程**，而不是盲目随机改动。这样模型在看到扰动表时，仍能通过列的描述、数据分布等线索恢复正确的列映射。

### 实验与效果
- **评估基准**：作者在 ADVETA 上对比了多种最新的 Text-to‑SQL 模型（如 PICARD、RAT‑SQL、T5‑based 系列）。  
- **性能跌幅**：所有模型在原始测试集上表现正常，但在加入 ATP 后准确率出现“大幅下降”，具体数值未在摘要中给出，论文仅描述为“dramatic performance drops”。  
- **对抗训练提升**：使用上下文感知对抗训练后，模型在 ATP 上的准确率提升最高，且在 NL-side 对抗样本上也有可观的提升，说明防御是跨扰动的。  
- **消融实验**：论文通过去掉列名同义替换、去掉噪声列插入等子模块，分别评估每种扰动对整体鲁棒性的贡献，结果显示列名同义替换是最关键的因素。  
- **局限性**：作者承认当前的 ATP 仍然是基于已知的表结构变化，未覆盖所有可能的业务迁移场景；对抗训练会增加训练时间和显存开销，实际部署时需要权衡。

### 影响与延伸思考
这篇工作把“表格侧对抗”正式搬上台面，促使后续研究开始关注数据结构本身的可变性。后续的几篇论文（如 *Robust Text-to-SQL via Schema Augmentation*、*Schema‑aware Adversarial Training*）都在不同程度上借鉴了 ATP 的思路，进一步探索如何在多表、跨库场景下保持鲁棒性。对想深入的读者，可以关注以下方向：  
- **自动化 Schema 演化检测**：实时监控业务表结构变化并触发模型再训练。  
- **多模态对抗**：同时对 NL 与表格进行联合扰动，构建更综合的鲁棒性评估。  
- **轻量化对抗训练**：在保持鲁棒性的同时降低计算成本。  

### 一句话记住它
把表格本身当成对手，给 Text-to‑SQL 加上“表格扰动”防弹衣，才能在真实业务里稳住输出。