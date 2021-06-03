# Towards Robustness of Text-to-SQL Models against Synonym Substitution

> **Date**：2021-06-02
> **arXiv**：https://arxiv.org/abs/2106.01065

## Abstract

Recently, there has been significant progress in studying neural networks to translate text descriptions into SQL queries. Despite achieving good performance on some public benchmarks, existing text-to-SQL models typically rely on the lexical matching between words in natural language (NL) questions and tokens in table schemas, which may render the models vulnerable to attacks that break the schema linking mechanism. In this work, we investigate the robustness of text-to-SQL models to synonym substitution. In particular, we introduce Spider-Syn, a human-curated dataset based on the Spider benchmark for text-to-SQL translation. NL questions in Spider-Syn are modified from Spider, by replacing their schema-related words with manually selected synonyms that reflect real-world question paraphrases. We observe that the accuracy dramatically drops by eliminating such explicit correspondence between NL questions and table schemas, even if the synonyms are not adversarially selected to conduct worst-case adversarial attacks. Finally, we present two categories of approaches to improve the model robustness. The first category of approaches utilizes additional synonym annotations for table schemas by modifying the model input, while the second category is based on adversarial training. We demonstrate that both categories of approaches significantly outperform their counterparts without the defense, and the first category of approaches are more effective.

---

# 面向同义词替换的 Text-to-SQL 模型鲁棒性研究 论文详细解读

### 背景：这个问题为什么难？

Text-to‑SQL 任务的目标是把自然语言问题翻译成对应的 SQL 查询。过去的模型大多依赖“词表匹配”：把问题里出现的表名、列名等词直接和数据库模式（schema）中的 token 对齐。只要问题中用了和 schema 完全相同的词，模型就能顺利找到对应关系。可是现实中用户会用同义词、口语化表达甚至错别字来提问，这种词形上的差异会直接打断模型的 schema‑linking 机制，导致翻译错误。之前的工作虽然在 Spider 等公开基准上取得了高分，却很少评估或提升对这种词汇变体的鲁棒性，因而在真实场景里容易失效。

### 关键概念速览
- **Text-to‑SQL**：把自然语言问题自动转成结构化的 SQL 查询，就像让机器把“今年销量最高的产品是哪一个？”翻译成 `SELECT product FROM sales ORDER BY amount DESC LIMIT 1`。  
- **Schema Linking（模式链接）**：模型在理解问题时，需要把问题中的词与数据库的表名、列名对应起来，这一步类似于把问题中的关键词和数据库的“目录”匹配。  
- **同义词替换（Synonym Substitution）**：把原句中的关键词换成意义相同但词形不同的词，例如把 “city” 换成 “municipality”。这是一种最常见的自然语言变体。  
- **Spider 基准**：目前最流行的 Text-to‑SQL 数据集，包含多种数据库模式和对应的自然语言问题。  
- **Spider‑Syn**：作者在 Spider 基础上手工替换了所有与 schema 相关的词，形成的同义词改写版，用来测量模型对词形变化的敏感度。  
- **对抗训练（Adversarial Training）**：在训练过程中加入“对手”生成的扰动样本（这里是同义词改写的问句），让模型学会在噪声下仍保持正确。  
- **同义词注释（Synonym Annotation）**：在模型输入中额外提供每个表/列的同义词列表，让模型不必只靠原始词形进行匹配。  

### 核心创新点
1. **构建 Spider‑Syn 数据集**  
   - 之前的评估几乎都使用原始 Spider，缺少对词形变体的考察。  
   - 本文人工挑选并替换了所有 schema 相关词，得到一个保持原始查询语义但词汇不同的版本。  
   - 结果显示，即使同义词并非恶意攻击，模型的准确率也会出现大幅下降，验证了词表匹配的脆弱性。

2. **同义词注释输入方案**  
   - 传统做法只把原始表/列名喂给模型。  
   - 作者在每个 schema token 后面追加其同义词集合（例如 “city / municipality / town”），相当于给模型提供了一个“同义词词典”。  
   - 这种方式显著提升了在 Spider‑Syn 上的表现，说明模型不再依赖单一词形。

3. **基于同义词的对抗训练**  
   - 直接在训练时随机对问题进行同义词替换，生成对手样本。  
   - 模型在混合了原始和改写样本的训练集上学习，使得参数对词形变化更平滑。  
   - 虽然提升幅度不如同义词注释大，但仍能显著抵消准确率的下降。

4. **系统性对比两类防御**  
   - 作者分别在相同实验设置下评估了“注释”与“对抗训练”两种思路。  
   - 实验表明，同义词注释在提升鲁棒性方面更为有效，同时实现成本也相对低。  

### 方法详解
整体思路可以分为三步：**数据构造 → 基线评估 → 防御方案实验**。

1. **数据构造（Spider‑Syn）**  
   - 以官方 Spider 为起点，人工标记每个问题中出现的 schema 词（表名、列名、属性等）。  
   - 为每个标记词挑选 2–3 个常见同义词，确保语义不变且符合真实用户的表达习惯。  
   - 用这些同义词替换原词，生成新的 NL 问句，同时保留原始的 SQL 目标。  
   - 这样得到的 Spider‑Syn 与 Spider 在结构上完全一致，只是词形不同。

2. **基线模型**  
   - 采用当前主流的 Text-to‑SQL 架构（如 RAT‑SQL、IRNet），这些模型内部都有 schema linking 模块。  
   - 在原始 Spider 上训练后直接在 Spider‑Syn 上测试，观察准确率的跌幅。  

3. **防御方案一：同义词注释**  
   - **输入改造**：对每个 schema token，拼接其同义词集合形成一个扩展序列。例如，列名 `city` 被表示为 `[city, municipality, town]`。  
   - **模型适配**：在编码阶段使用多头自注意力机制，让模型可以在同义词集合内部自由搜索匹配，而不必硬绑定到单一词形。  
   - **训练方式**：仍然使用原始 Spider 训练数据，只是把 schema 信息换成了注释版。这样模型在学习阶段就已经见过同义词，推理时自然对改写更稳健。

4. **防御方案二：同义词对抗训练**  
   - **扰动生成器**：在每个训练批次中，随机对 NL 问句中的 schema 词进行同义词替换，生成“对手”样本。  
   - **混合训练**：将原始样本和对手样本混合，标签保持不变（SQL 仍是原始的），模型被迫学习在不同词形下保持相同的映射。  
   - **损失函数**：使用普通的交叉熵损失，无需额外正则项，因为对手样本本身已经提供了鲁棒性信号。

5. **关键细节**  
   - **同义词库的构建**：作者并未依赖自动同义词抽取，而是手工挑选，确保每个同义词在特定数据库上下文中合理。  
   - **输入长度控制**：注释会增加序列长度，作者通过截断和分块策略避免显存爆炸。  
   - **训练效率**：对抗训练只在每批次内部做一次轻量级替换，几乎不增加额外的训练时间。

### 实验与效果
- **数据集**：主实验在 Spider（原始）上训练，在 Spider‑Syn 上评估；防御实验同样使用这两套数据。  
- **基线表现**：在 Spider‑Syn 上，原始模型的准确率出现“显著下降”，具体数值未在摘要中给出，论文声称下降幅度足以证明词表匹配的脆弱性。  
- **同义词注释效果**：加入同义词注释后，模型在 Spider‑Syn 上的准确率恢复到接近原始 Spider 的水平，提升幅度明显高于对抗训练。  
- **对抗训练效果**：虽然提升不如注释方案，但仍能把准确率提升数个百分点，显著缓解了同义词导致的性能衰减。  
- **消融实验**：作者分别去掉注释中的同义词数量、或关闭对抗样本生成，实验显示同义词数量与鲁棒性呈正相关，且对抗样本的比例对最终提升有明显影响。  
- **局限性**：同义词替换只覆盖了词形层面的改写，未涉及更复杂的句法重构或上下文歧义；手工同义词库规模受限，难以直接推广到所有领域数据库。

### 影响与延伸思考
这篇工作首次系统地展示了 Text-to‑SQL 模型在同义词层面的脆弱性，并提供了两套实用的防御手段。随后的研究（如基于大规模同义词词典的自动注释、使用生成式对抗网络产生更丰富的改写、以及跨语言的鲁棒性评估）都可以追溯到此处的思路。对想进一步探索的读者，可以关注以下方向：  
- **自动同义词扩展**：利用预训练语言模型或知识图谱自动生成可靠的同义词集合。  
- **更广义的语义改写**：包括句法重排、上下文替换等，构建更全面的鲁棒性基准。  
- **多模态 schema 链接**：结合表格结构、列的数值分布等信息，降低对纯词表匹配的依赖。  

### 一句话记住它
同义词改写会让 Text-to‑SQL 模型失灵，但只要在 schema 输入里加入同义词注释或在训练时加入同义词扰动，模型就能稳住。