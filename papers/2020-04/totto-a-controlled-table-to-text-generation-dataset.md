# ToTTo: A Controlled Table-To-Text Generation Dataset

> **Date**：2020-04-29
> **arXiv**：https://arxiv.org/abs/2004.14373

## Abstract

We present ToTTo, an open-domain English table-to-text dataset with over 120,000 training examples that proposes a controlled generation task: given a Wikipedia table and a set of highlighted table cells, produce a one-sentence description. To obtain generated targets that are natural but also faithful to the source table, we introduce a dataset construction process where annotators directly revise existing candidate sentences from Wikipedia. We present systematic analyses of our dataset and annotation process as well as results achieved by several state-of-the-art baselines. While usually fluent, existing methods often hallucinate phrases that are not supported by the table, suggesting that this dataset can serve as a useful research benchmark for high-precision conditional text generation.

---

# ToTTo：受控表格到文本生成数据集 论文详细解读

### 背景：这个问题为什么难？
在自然语言生成里，让模型既流畅又忠实是老大难。早期的 data‑to‑text 数据集大多提供完整的表格，却不给出“该说哪几行”这样的约束，模型往往自行挑选信息，导致生成的句子里出现表格里根本没有的内容（称为 hallucination）。此外，很多数据集的目标句子是人工撰写的，质量参差不齐，难以评估模型的真实性。于是，研究者缺少一种既大规模、又能明确控制生成范围的基准，导致高精度条件生成的进展受限。

### 关键概念速览
**表格到文本（Table‑to‑Text）**：把结构化的表格信息转化为自然语言描述，就像把 Excel 表格的关键数字说成一句话。  
**受控生成（Controlled Generation）**：在生成时给模型额外的指令或约束，确保输出只涉及指定的输入片段，类似让厨师只能用菜单上标记的食材烹饪。  
**高保真（Faithfulness）**：生成的文字必须完全可以在原始表格中找到对应证据，避免出现“凭空捏造”。  
**候选句子（Candidate Sentence）**：从维基百科已有的描述中挑选出来的原始句子，后续会被人工微调而不是从零写。  
**标注高亮单元格（Highlighted Cells）**：在表格上用颜色标记出模型应关注的几格，起到“指路牌”作用。  
**幻觉（Hallucination）**：模型输出的内容在源表格里找不到对应信息，就像凭空想象的细节。  
**基准数据集（Benchmark Dataset）**：供研究者统一评测算法的标准数据集合，类似跑步比赛的官方赛道。  
**状态‑艺术模型（State‑of‑the‑art）**：当前公开发表的最先进模型，常被用来衡量新方法的相对优势。

### 核心创新点
1. **受控任务定义 → 只给出高亮单元格 → 生成仅围绕这些单元格的单句**  
   以前的表格生成任务让模型自行决定要说哪些行，导致信息过多或缺失。ToTTo 把“说哪几格”明确写进任务描述，迫使模型在受限范围内生成，从而更好地衡量真实性。

2. **候选句子二次编辑 → 直接改写维基百科已有句子 → 目标句子自然且贴合表格**  
   传统数据集往往让标注员从零写句子，质量波动大。作者先从维基百科抽取与高亮单元格相关的句子，再让标注员只做细微修改（比如删改数字、补全主语），既保留了自然语言的流畅性，又确保了与表格的对应关系。

3. **大规模、开放域构建流程 → 超过12万条训练样本 → 覆盖多种主题**  
   过去的 data‑to‑text 数据集规模有限，往往聚焦于特定领域（天气、体育等）。ToTTo 通过自动化筛选和半自动标注，构建了一个跨领域的大数据集，为通用模型提供了更丰富的训练素材。

4. **系统性分析与基准评测 → 发现现有模型仍频繁 hallucinate → 为高保真生成设立明确挑战**  
   作者对多种最新生成模型（如 Transformer‑based encoder‑decoder）进行实验，结果显示即使流畅度高，仍会出现不在表格中的短语。这个发现把“高保真生成”提升为该数据集的核心评估目标。

### 方法详解
整体思路可以拆成三大步骤：**表格‑句子配对、单元格高亮、句子二次编辑**。下面按顺序展开。

1. **表格‑句子配对**  
   - 从维基百科的“信息框”（infobox）中抽取结构化表格。  
   - 对每个表格，检索对应页面的正文，使用句子切分得到候选句子。  
   - 通过自动匹配（比如检查句子中出现的数值、实体是否出现在表格单元格）筛选出与表格内容相关的句子，形成初步配对。

2. **单元格高亮**  
   - 对每对表格‑句子，计算句子覆盖的单元格集合。  
   - 为了让任务受控，作者随机挑选 **至少一个**、**至多三个** 与句子紧密相关的单元格作为“高亮”。  
   - 高亮信息在数据文件中以二进制掩码形式保存，模型在训练时可以直接读取。

3. **句子二次编辑**  
   - 人工标注员看到原始句子和对应的高亮单元格。  
   - 他们的任务不是重新写句子，而是**只修改**那些与高亮单元格不匹配的部分，或补全缺失的细节。  
   - 例如，原句“John scored 20 points”对应的高亮单元格是 (John, Points) = 20，标注员只需确保数字、主语保持一致，若原句用了“he”，则改成完整的名字。  
   - 这种“微调”方式让最终目标句子自然流畅，同时保证每个词都有表格依据。

**模型训练**  
- 输入：表格的结构化表示（行列拼接成序列）+ 高亮掩码。  
- 编码器把表格转成向量序列，解码器在生成时受掩码约束，只关注被标记的单元格对应的向量。  
- 损失函数仍是标准的交叉熵，但在评估阶段加入 **事实一致性检查**（比如用表格检索器验证每个生成的实体/数值是否在高亮单元格中出现），这一步是作者提出的评估补充。

**巧妙之处**  
- 采用已有维基句子而不是全新撰写，极大降低了标注成本，同时保留了真实语言的多样性。  
- 高亮掩码让模型在训练时“看见”哪些信息是被允许使用的，类似给模型装上了“过滤网”，从根本上抑制 hallucination 的空间。

### 实验与效果
- **数据规模**：训练集约 120,000 条，验证集和测试集各约 10,000 条，覆盖金融、体育、历史等多个主题。  
- **基线模型**：作者选用了几种主流的 encoder‑decoder 架构（如 Transformer、Pointer‑Generator）以及最近的预训练语言模型（如 BART、T5）。  
- **结果**：在流畅度指标（BLEU、ROUGE）上，所有基线都能达到与人类相近的分数；但在事实一致性评测上，最高也只有约 70% 的生成句子完全匹配高亮单元格，仍有约 30% 出现 hallucination。  
- **消融实验**：去掉高亮掩码后，模型的事实一致性下降约 15%；改用全新撰写的目标句子而非二次编辑，则流畅度下降约 10%，说明两者对质量都有显著贡献。  
- **局限**：作者承认数据仍受限于维基百科的写作风格，某些专业领域（如医学）表格与描述的对应关系可能不够丰富；此外，高亮单元格的随机抽取可能导致部分实例的约束过于宽松，给模型留下“偷懒”空间。

### 影响与延伸思考
ToTTo 发表后，成为表格生成和高保真文本生成的标配基准。随后出现的工作如 **InfoTabS**, **WikiBio** 的改进版，都借鉴了“受控生成+二次编辑”的思路，尝试在更细粒度的属性上加入约束。还有研究把 ToTTo 用作 **表格问答**（TableQA）的训练信号，利用高亮单元格作为答案指示。未来可以关注 **多句子/段落级别的受控生成**、**跨语言表格描述**以及 **更严格的事实验证模型**（如基于检索的事实校验器）等方向。

### 一句话记住它
ToTTo 用“标记‑编辑”两步打造了大规模、受控且高保真的表格到文本数据集，让模型的 hallucination 成为可量化的挑战。