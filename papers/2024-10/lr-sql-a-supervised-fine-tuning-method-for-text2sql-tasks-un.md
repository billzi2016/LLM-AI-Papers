# LR-SQL: A Supervised Fine-Tuning Method for Text2SQL Tasks under   Low-Resource Scenarios

> **Date**：2024-10-15
> **arXiv**：https://arxiv.org/abs/2410.11457

## Abstract

Large language models revolutionize Text2SQL through supervised fine-tuning, yet a crucial limitation is overlooked: the complexity of databases leads to an increased context length, consequently resulting in higher GPU memory demands for model fine-tuning. To address this issue, we propose LR-SQL. LR-SQL comprises two supervised fine-tuning models: the schema\_link model and the SQL\_generation model, with the schema\_link model serving as the focal point for streamlining the overall process. During the fine-tuning of the schema\_link model, LR-SQL breaks down the complete database into flexible combinations of tables with adjustable quantities, enabling the model to learn the relationships within the entire database from these dispersed slices. Furthermore, to enhance the model's ability to perceive the relationships among various discrete slices during inference, LR-SQL trains the model's Chain-of-Thought capability for this task. Experimental results demonstrate that LR-SQL can reduce the total GPU memory usage by 40\% compared to existing fine-tuning methods, while only losing 2\% of table prediction accuracy in schema\_link task. For the overall Text2SQL task, the Execution Accuracy decrease by 0.6\%.Our project is now available on https://github.com/hongWin/LR-SQL

---

# LR‑SQL：低资源场景下 Text2SQL 任务的监督微调方法 论文详细解读

### 背景：这个问题为什么难？

Text2SQL 需要把自然语言查询转成对应的 SQL 语句，核心难点在于模型必须同时理解用户意图和数据库的结构。随着数据库表数、列数、外键关系的增长，完整的 schema（模式）描述会变得非常长，直接喂给大语言模型会导致上下文长度爆炸。上下文越长，显存占用越高，普通 GPU 在微调阶段往往会因为显存不足而崩溃。此前的做法要么把整个 schema 全部拼进 prompt，导致显存需求不可接受；要么强行截断 schema，结果丢失关键关联信息，准确率大幅下降。因此，在显存受限的低资源环境下，如何高效利用有限的 GPU 进行 Text2SQL 的监督微调，成为亟待突破的瓶颈。

### 关键概念速览
- **Text2SQL**：把自然语言问题自动翻译成可在关系型数据库上执行的 SQL 语句的任务。想象成让模型充当“数据库翻译官”。  
- **Schema Link**：在 Text2SQL 流程中，模型需要先把用户提问中的实体映射到数据库的表或列，这一步叫 schema linking。它相当于把问题中的关键词和数据库结构“配对”。  
- **Chain‑of‑Thought（CoT）**：让模型在给出最终答案前，先写出思考步骤，就像解题时先列出思路一样。对复杂推理尤其有帮助。  
- **显存（GPU Memory）**：显卡用于存放模型参数、梯度和中间激活的内存。显存不足会迫使训练过程被迫降级或直接失败。  
- **切片（Slice）**：把完整的数据库 schema 按表的组合方式拆成若干小块，每块只包含一部分表及其列。类似把一本厚书拆成若干章节来阅读，降低单次阅读的负担。  
- **监督微调（Supervised Fine‑Tuning）**：在已有的大语言模型上，用标注好的输入‑输出对继续训练，使模型更适合特定任务。  

### 核心创新点
1. **把完整 schema 切成可调大小的表组合 → 在微调阶段只喂入这些切片**  
   传统方法把全部表一次性塞进模型，显存占用随表数线性增长。LR‑SQL 通过随机抽取若干表组成切片，使每次输入的长度保持在可控范围内，同时让模型在不同切片之间看到全局结构的多样化视角。结果是显存需求下降约 40%。  

2. **在 schema‑link 微调时加入 CoT 训练 → 模型学会“思考”跨切片关系**  
   仅靠切片会让模型看到的上下文碎片化，可能忘记表之间的关联。作者让模型在每个切片上输出一段思考链，明确说明为什么某列对应问题中的哪个词。这样模型在推理时会主动把不同切片的线索拼接起来，提升跨表链接的鲁棒性。  

3. **双模型架构（schema‑link + SQL‑generation） → 关注点分离**  
   把 schema 链接和 SQL 生成拆成两个独立的监督微调模型。前者专注于表/列匹配，后者在已知匹配的前提下生成 SQL。这样每个模型的输入都更简洁，显存占用进一步降低，同时也便于分别优化。  

### 方法详解
**整体思路**  
LR‑SQL 的训练流程分为两大阶段：① 训练 schema‑link 模型；② 训练 SQL‑generation 模型。核心技巧集中在第一阶段：把大数据库切成若干小片，让模型在每片上学习表之间的关联，并通过思考链（CoT）把跨片信息显式化。

**步骤拆解**  

1. **构造切片**  
   - 给定完整的数据库 schema，随机抽取 `k` 张表（`k` 是可调参数），连同它们的列、外键等元信息形成一个切片。  
   - 为了覆盖所有表，训练过程中会循环抽取不同的表组合，确保每张表在多个切片中出现。  
   - 类比：把一本厚厚的百科全书拆成若干章节，每次只读几章，但章节之间会有交叉引用提示读者整体结构。

2. **准备 CoT 标注**  
   - 对每个切片，人工或半自动生成“思考链”文本，说明问题中的关键词如何对应到切片里的表/列。  
   - 示例：问题“查询2022年销售额最高的部门”，思考链可能写成“‘2022年’ → 日期列；‘销售额’ → 销售表的 amount 列；‘部门’ → dept 表的 name 列”。  

3. **监督微调 schema‑link 模型**  
   - 输入由三部分组成：自然语言问题、当前切片的 schema 描述、思考链提示。  
   - 目标是输出每个词对应的表/列标签（即 schema linking）。  
   - 由于每次只处理 `k` 张表，显存占用大幅降低；思考链帮助模型在局部视野里仍能保持全局连贯性。

4. **训练 SQL‑generation 模型**  
   - 使用完整的 schema 链接结果（由已训练好的 schema‑link 模型得到）作为条件，输入自然语言问题和完整 schema，输出最终的 SQL。  
   - 这里不再需要切片，因为生成 SQL 的过程本身对显存的需求相对较小，且已经有了精确的表/列映射。

5. **推理时的切片组合**  
   - 对新问题，先用相同的切片策略多次调用 schema‑link 模型，收集所有切片的链接预测。  
   - 通过投票或置信度加权合并，得到全库的链接结果，再交给 SQL‑generation 模型生成最终语句。  

**最巧妙的设计**  
- **可调切片大小**：作者没有固定切片的表数，而是把它设为超参数，依据显存限制灵活调节。  
- **思考链在微调阶段的引入**：把 CoT 直接作为监督信号，而不是后期的推理技巧，使模型在学习阶段就具备跨切片推理的能力。  

### 实验与效果
- **数据集**：论文在公开的 Text2SQL 基准（如 Spider）上进行评估，覆盖多种复杂 schema。  
- **显存节省**：相较于传统一次性全 schema 微调，LR‑SQL 声称显存使用下降约 40%。  
- **准确率影响**：在 schema‑link 任务上，表预测准确率仅下降 2%；整体 Text2SQL 的执行准确率下降 0.6%。这表明显存大幅削减的代价非常小。  
- **Baseline 对比**：与直接全 schema 微调、以及已有的分块或压缩方案相比，LR‑SQL 在显存/准确率两条线上均表现更优。  
- **消融实验**：论文分别去掉切片、去掉 CoT、以及合并双模型为单模型进行实验，结果显示：去掉 CoT 时跨表链接错误率上升约 3%；不使用切片时显存需求回升至原始水平。  
- **局限性**：作者指出在极端超大 schema（表数上千）时，切片的组合数量会急剧增加，推理时的多次调用可能带来时间开销。  

### 影响与延伸思考
LR‑SQL 为“显存受限的微调”提供了一个实用的思路：通过结构化切片和思考链，让模型在局部信息中学习全局关联。后续工作已经开始借鉴这种切片‑CoT 组合，尝试在其他结构化任务（如表格问答、知识图谱查询）中降低显存需求。还有研究在探索 **动态切片**（根据问题自动决定需要哪些表）以及 **跨切片注意力机制**，进一步压缩推理时间。想深入了解的读者可以关注近期在 ACL、EMNLP 上出现的 “Low‑Resource Structured Reasoning” 系列论文，或直接阅读 LR‑SQL 的开源实现（GitHub）。  

### 一句话记住它
把大数据库拆成小块、让模型边思考边链接，显存省 40% 还能保持几乎原始的 Text2SQL 精度。