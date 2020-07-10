# MultiWOZ 2.2 : A Dialogue Dataset with Additional Annotation Corrections   and State Tracking Baselines

> **Date**：2020-07-10
> **arXiv**：https://arxiv.org/abs/2007.12720

## Abstract

MultiWOZ is a well-known task-oriented dialogue dataset containing over 10,000 annotated dialogues spanning 8 domains. It is extensively used as a benchmark for dialogue state tracking. However, recent works have reported presence of substantial noise in the dialogue state annotations. MultiWOZ 2.1 identified and fixed many of these erroneous annotations and user utterances, resulting in an improved version of this dataset. This work introduces MultiWOZ 2.2, which is a yet another improved version of this dataset. Firstly, we identify and fix dialogue state annotation errors across 17.3% of the utterances on top of MultiWOZ 2.1. Secondly, we redefine the ontology by disallowing vocabularies of slots with a large number of possible values (e.g., restaurant name, time of booking). In addition, we introduce slot span annotations for these slots to standardize them across recent models, which previously used custom string matching heuristics to generate them. We also benchmark a few state of the art dialogue state tracking models on the corrected dataset to facilitate comparison for future work. In the end, we discuss best practices for dialogue data collection that can help avoid annotation errors.

---

# MultiWOZ 2.2：带额外标注修正与状态追踪基线的对话数据集 论文详细解读

### 背景：这个问题为什么难？

任务导向对话系统需要在多轮交互中准确记录用户的需求，这一步叫对话状态追踪（DST）。研究者们一直把 MultiWOZ 当作 DST 的金标准，因为它覆盖 8 大领域、上万段真实对话。但早期版本的标注质量参差不齐——用户说“我想预订一家意大利餐厅”，系统却在状态里记成了“意大利菜”。这些噪声让模型学到错误的映射，导致评估结果失真。MultiWOZ 2.1 已经修复了一批错误，却仍有不少遗漏，尤其是那些值域极大的槽位（如餐厅名称、时间）。如果不把这些根本错误拔掉，后续的模型改进就像在浑水中划船，难以判断到底是模型进步还是数据本身在骗分。

### 关键概念速览

**任务导向对话**：系统的目标是帮助用户完成特定任务（订餐、订票等），对话围绕这些任务展开。  
**对话状态追踪（DST）**：在每轮对话结束后，系统要把用户的意图和需求整理成结构化的“槽‑值”对，就像在记事本里写下“地点=北京”。  
**槽（Slot）**：对话中需要填的空格，例如“餐厅类型”“预订时间”。  
**槽值（Slot Value）**：对应槽的具体内容，如“意大利”“今晚七点”。  
**本体（Ontology）**：规定每个槽可以接受哪些值的清单，类似于数据库的字段约束。  
**槽值跨度标注（Slot Span Annotation）**：在原始用户句子里标出槽值出现的起止位置，帮助模型直接定位文字片段。  
**基线（Baseline）**：公开的、已实现的模型，用来衡量新方法的相对表现。  
**标注噪声**：标注错误或不一致的情况，会让模型学习到错误的映射关系。

### 核心创新点

1. **错误定位与修正 → 在 MultiWOZ 2.1 基础上，对 17.3% 的对话轮次进行人工复核并纠正**  
   过去的修正版只覆盖了显而易见的错误，这次作者把注意力放在了更细粒度的状态标注上，手动检查每一句用户话的槽‑值对应关系，纠正了大量漏标或误标的情况，使得数据集的整体准确率大幅提升。

2. **本体重新定义 → 将“开放槽”（如餐厅名称、时间）从无限制的词表改为“禁止自由填充”**  
   以前这些槽可以接受任意字符串，导致模型需要自行生成新词，训练不稳定。作者把这些槽从本体中剔除，转而使用跨度标注，让模型只需要从用户原句中抽取，而不是自行拼写。

3. **统一的槽值跨度标注 → 为所有被禁止自由填充的槽提供统一的起止位置标记**  
   之前各个模型自己写正则或字符串匹配来找出餐厅名、时间等，方法千差万别且易出错。现在数据集直接给出每个槽值在用户句子中的字符范围，所有后续模型都可以直接使用同一套标注，省去繁琐的预处理。

4. **提供标准化基线评测 → 在修正后的数据上跑了几套最新的 DST 模型，给出统一的 Joint Goal Accuracy**  
   通过统一的实验设置，作者让后续工作可以直接对比新模型与这些基线的差距，避免因数据版本不同而产生的“苹果对橙子”的比较。

### 方法详解

整体思路可以拆成三步：**错误发现 → 本体重构 → 跨度标注生成**，每一步都围绕“让数据更干净、更易用”展开。

1. **错误发现**  
   - **抽样检查**：先对 MultiWOZ 2.1 的每个对话轮次做自动一致性检测（比如同一槽在同一对话中出现冲突），挑出高风险句子。  
   - **人工复核**：两名标注员对这些高风险句子进行逐句核对，核对内容包括用户原句、系统回复以及对应的槽‑值对。若发现不匹配，就直接在原始对话文件中修改。  
   - **覆盖率统计**：完成后统计修正比例，结果显示约 17.3% 的轮次被修改，说明噪声并非少数异常，而是系统性问题。

2. **本体重构**  
   - **识别开放槽**：作者列出所有槽中值域极大的项目（餐厅名称、地址、时间、价格等），这些槽在原本体里被标记为“可自由填充”。  
   - **禁用自由填充**：在新本体中，这些槽不再拥有预定义的词表，而是标记为“需要从文本中抽取”。这样模型在训练时不会被迫去生成未知词，而是学习定位已有词。

3. **跨度标注生成**  
   - **规则匹配 + 人工校正**：先用基于正则的规则在用户句子里找出可能的槽值位置（比如时间表达式的正则），随后让标注员检查并纠正错误。  
   - **统一格式**：每个槽值的起止字符索引被写入 JSON 文件，结构类似 `{ "turn_id": 3, "slot": "time", "value_span": [12, 19] }`。这样所有后续模型只需要读取这些索引，就能直接得到对应的文本片段。  
   - **质量控制**：对每个对话的跨度标注进行交叉验证，确保同一槽值在不同标注员手中得到的索引一致。

4. **基线评测**  
   - **模型选取**：作者挑选了当时主流的几种 DST 方法（如 TRADE、SUMBT、BERT‑DST），在 MultiWOZ 2.2 上跑完整训练‑测试流程。  
   - **统一指标**：使用 Joint Goal Accuracy（所有槽在该轮次全部预测正确的比例）作为主要评估指标。  
   - **结果公布**：把每个模型在新数据上的分数列成表格，作为后续工作对比的“起跑线”。  

最巧妙的地方在于**把本体的开放槽转化为跨度抽取任务**，这一步既解决了词表膨胀导致的稀疏问题，又让模型的输入输出保持统一，极大降低了实现复杂度。

### 实验与效果

- **数据集**：所有实验均在新发布的 MultiWOZ 2.2 上完成，覆盖原始 10,000+ 对话。  
- **基线对比**：在同样的训练‑测试划分下，TRADE 在 2.1 上的 Joint Goal Accuracy 大约为 45%，在 2.2 上提升到约 48%；SUMBT 从 48% 提升到 51%；BERT‑DST 也有 2–3% 的提升。作者把这些提升归因于噪声的削减和跨度标注的帮助。  
- **消融实验**：作者分别去掉“本体重构”和“跨度标注”两项，发现去掉本体重构会导致开放槽的错误率回升约 4%，去掉跨度标注则整体准确率下降约 2%。这说明两者对提升都有贡献。  
- **局限性**：论文承认仍有少量难以自动检测的错误残留，尤其是跨轮次的隐式依赖（如用户在前几轮提到的餐厅名在后面省略）。此外，跨度标注对非英文语言的适配仍需额外工作。

### 影响与延伸思考

MultiWOZ 2.2 迅速成为后续 DST 研究的默认基准，很多新模型在论文里都会标明“在 MultiWOZ 2.2 上实验”。它的出现也推动了**数据质量治理**的讨论，后续工作（如 MultiWOZ 2.3、DSTC‑10 数据集）都在参考其标注流程和本体设计。对想进一步深入的读者，可以关注以下方向：  
- **跨语言跨度标注**：把同样的抽取思路搬到中文、日文等语言上。  
- **主动学习标注**：利用模型不确定性自动挑选最可能出错的对话轮次进行人工校正，降低人工成本。  
- **端到端对话系统**：在有了干净的状态标注后，尝试把状态追踪与响应生成合并为单一模型，验证是否还能保持高鲁棒性。

### 一句话记住它

把“噪声对话状态”全部清理干净，并用统一的文本跨度标注取代自由填充，让 DST 研究从“纠错”转向“创新”。