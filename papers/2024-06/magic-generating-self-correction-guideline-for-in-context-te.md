# MAGIC: Generating Self-Correction Guideline for In-Context Text-to-SQL

> **Date**：2024-06-18
> **arXiv**：https://arxiv.org/abs/2406.12692

## Abstract

Self-correction in text-to-SQL is the process of prompting large language model (LLM) to revise its previously incorrectly generated SQL, and commonly relies on manually crafted self-correction guidelines by human experts that are not only labor-intensive to produce but also limited by the human ability in identifying all potential error patterns in LLM responses. We introduce MAGIC, a novel multi-agent method that automates the creation of the self-correction guideline. MAGIC uses three specialized agents: a manager, a correction, and a feedback agent. These agents collaborate on the failures of an LLM-based method on the training set to iteratively generate and refine a self-correction guideline tailored to LLM mistakes, mirroring human processes but without human involvement. Our extensive experiments show that MAGIC's guideline outperforms expert human's created ones. We empirically find out that the guideline produced by MAGIC enhances the interpretability of the corrections made, providing insights in analyzing the reason behind the failures and successes of LLMs in self-correction. All agent interactions are publicly available at https://huggingface.co/datasets/microsoft/MAGIC.

---

# MAGIC：为上下文文本到SQL生成自纠正指南 论文详细解读

### 背景：这个问题为什么难？
文本到SQL（Text‑to‑SQL）让模型把自然语言问题翻译成数据库查询语句，但大模型常会产生语法错误、列名不匹配或逻辑偏差。传统的自纠正方式依赖人工编写的纠错指南，这既费时又受限于人类对模型错误模式的洞察深度。换句话说，模型的错误种类繁多且随模型规模、提示方式变化，人工指南很难覆盖全部情形，导致实际使用时纠错效果不稳。

### 关键概念速览
**Text‑to‑SQL**：把自然语言提问转成SQL查询，类似把口头指令翻译成机器指令。  
**In‑Context Learning（上下文学习）**：在提示中直接给出示例，让模型在当前对话里“现场学习”，不需要额外微调。  
**Self‑Correction（自纠正）**：让模型在发现自己生成的SQL有误后，再次生成更正的版本。  
**Guideline（纠错指南）**：一套结构化的提示模板，告诉模型如何检查并改正错误。  
**Multi‑Agent System（多代理系统）**：由多个功能专一的模型（代理）协同工作，每个代理负责特定子任务。  
**Manager Agent（管理代理）**：负责调度、记录进度并决定何时结束迭代。  
**Correction Agent（纠错代理）**：实际生成SQL修正建议的模型。  
**Feedback Agent（反馈代理）**：评估纠错结果并给出改进意见，类似“审稿人”。  

### 核心创新点
1. **手工指南 → 自动生成指南**：过去需要专家手写纠错模板，MAGIC 通过让三个专职代理在训练集上反复碰撞错误，自动产出针对模型常犯错误的指南。这样省去人工成本，也能捕捉到模型独有的细微失误。  
2. **单一模型纠错 → 多代理协同**：传统自纠正往往让同一个模型先检查再改写，容易陷入同样的思维陷阱。MAGIC 把检查、改写、评估职责拆分，让每个代理专注于自己的角色，形成类似“编辑‑作者‑审稿人”的闭环，提高纠错质量。  
3. **一次性指南 → 迭代细化**：MAGIC 不是一次性生成固定模板，而是让管理代理监控纠错成功率，若仍有未覆盖的错误模式，就让纠错和反馈代理继续对这些案例进行分析并补充新规则，形成逐步精炼的指南。  
4. **解释性提升 → 可解释纠错**：生成的指南不仅能让模型改正错误，还保留了错误来源的文字说明，帮助研究者直观看到模型是因列名拼写、聚合函数误用还是子查询结构错误而失败。

### 方法详解
MAGIC 的整体流程可以概括为三步循环：**错误收集 → 规则生成 → 规则评估**，循环直到纠错成功率满足阈值。

1. **错误收集**  
   - 在训练集上运行基线 Text‑to‑SQL 模型，记录所有生成的错误 SQL（包括语法错误、执行错误和语义偏差）。  
   - 这些错误实例连同原始自然语言问题一起交给管理代理。

2. **规则生成（由纠错代理完成）**  
   - 管理代理挑选一批错误实例，构造提示让纠错代理“思考”如何改正。提示中会包含错误 SQL、对应的错误类型（由反馈代理标注）以及期望的纠正方向。  
   - 纠错代理输出一段结构化文本，形式类似：“如果出现 …（错误模式），请改为 …（修正模板）”。这段文本即为**自纠正指南的候选条目**。

3. **规则评估（反馈代理）**  
   - 管理代理把新生成的指南回塞进原始模型的提示中，重新让模型对同一错误实例进行自纠正。  
   - 反馈代理检查纠正后的 SQL 是否通过执行或符合预期语义，并给出“通过/未通过 + 说明”。如果通过，指南被正式加入最终指南库；如果未通过，反馈代理会指出指南的不足（比如条件不够具体），并把该案例重新送回纠错代理进行改写。

4. **迭代与收敛**  
   - 管理代理统计本轮新增通过的案例比例。若比例低于预设阈值，说明仍有未覆盖的错误模式，系统继续下一轮迭代。  
   - 迭代结束后，得到的指南集合即为 MAGIC 生成的 **Self‑Correction Guideline**，可直接嵌入任何基于 In‑Context Learning 的 Text‑to‑SQL 提示中。

**巧妙之处**：把错误分析和规则生成交给模型本身，而不是人工标注，使得指南能够捕捉到模型特有的“思维盲点”。另外，管理代理的“收敛判定”让整个过程自动停机，避免无限循环。

### 实验与效果
- **数据集**：论文在公开的 Spider、Spider‑Realistic 以及自建的训练子集上评估。  
- **基线**：与手工编写的专家指南、直接让模型自纠正（无指南）以及最新的自监督纠错方法对比。  
- **结果**：论文声称 MAGIC 生成的指南在 Spider 上提升了约 4% 的执行准确率，超过人工指南约 2% 的增幅。对比直接自纠正，提升更为显著（约 6%）。  
- **消融实验**：去掉反馈代理或管理代理会导致指南质量下降，准确率分别下降约 1.5% 和 2%。这说明三代理协同是关键。  
- **局限**：指南的生成依赖于训练集的错误分布，若在全新领域出现全新错误模式，仍需重新运行 MAGIC。作者也提到生成的指南在极端长查询上可读性会下降。

### 影响与延伸思考
MAGIC 把“人类专家”角色交给了 LLM，展示了多代理系统在提示工程中的潜力。随后的工作开始探索 **自动提示优化**、**跨任务自纠正指南生成**，甚至把类似框架用于代码生成、数学推理等需要自我检查的场景。对想进一步研究的读者，可以关注以下方向：  
- **跨模型通用指南**：让同一套指南适配不同规模的模型。  
- **少样本错误发现**：结合主动学习，让系统在少量错误案例上快速定位新错误模式。  
- **解释性分析**：深入挖掘指南中错误原因的语言描述，构建错误类型的可视化图谱。

### 一句话记住它
MAGIC 用三个专职 LLM 代理自动“写指南”，让模型自己发现并系统化纠错规则，省去人工编写，显著提升 Text‑to‑SQL 的自纠正能力。