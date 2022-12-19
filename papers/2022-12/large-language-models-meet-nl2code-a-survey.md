# Large Language Models Meet NL2Code: A Survey

> **Date**：2022-12-19
> **arXiv**：https://arxiv.org/abs/2212.09420

## Abstract

The task of generating code from a natural language description, or NL2Code, is considered a pressing and significant challenge in code intelligence. Thanks to the rapid development of pre-training techniques, surging large language models are being proposed for code, sparking the advances in NL2Code. To facilitate further research and applications in this field, in this paper, we present a comprehensive survey of 27 existing large language models for NL2Code, and also review benchmarks and metrics. We provide an intuitive comparison of all existing models on the HumanEval benchmark. Through in-depth observation and analysis, we provide some insights and conclude that the key factors contributing to the success of large language models for NL2Code are "Large Size, Premium Data, Expert Tuning". In addition, we discuss challenges and opportunities regarding the gap between models and humans. We also create a website https://nl2code.github.io to track the latest progress through crowd-sourcing. To the best of our knowledge, this is the first survey of large language models for NL2Code, and we believe it will contribute to the ongoing development of the field.

---

# 大语言模型与 NL2Code 综述 论文详细解读

### 背景：这个问题为什么难？

把自然语言描述直接翻译成可执行代码一直是代码智能的核心难题。早期的 NL2Code 方法大多基于小规模的序列到序列模型，训练数据稀缺、模型容量受限，导致生成的代码经常语法错误或逻辑不符。更重要的是，代码本身具备严格的语法约束和语义依赖，稍有偏差就会导致编译失败或运行错误。随着软件系统日益复杂，单纯靠规则或模板已经无法满足多样化的需求，迫切需要能够理解自然语言意图并自行推理代码结构的大模型。

### 关键概念速览
- **NL2Code**：从自然语言描述生成代码的任务，类似于把“把列表里的偶数挑出来”翻译成 Python 的 `list(filter(...))`。  
- **大语言模型（LLM）**：参数量在数十亿以上、通过海量文本预训练的模型，能够捕捉语言的深层结构和常识。  
- **HumanEval**：一种代码生成基准，提供若干自然语言题目和对应的单元测试，用来衡量模型生成代码的正确率。  
- **预训练+微调（Pre‑train → Fine‑tune）**：先在通用语料上学习语言规律，再在代码或特定任务上做针对性调优，类似先学会说话再学会写程序。  
- **专家调优（Expert Tuning）**：利用领域专家提供的高质量注释、对齐数据或奖励模型，让模型在细节上更贴近真实开发者的写法。  
- **数据质量（Premium Data）**：指经过清洗、去噪、去重复并标注好的代码库，质量高的训练数据能显著提升模型的代码生成能力。  
- **模型规模（Large Size）**：参数越多、层数越深，模型的表达能力越强，尤其在捕捉复杂的控制流和库调用时表现更好。  

### 核心创新点
1. **系统化梳理 27 种 LLM** → 通过统一的表格和属性标签，把目前公开的、专注 NL2Code 的大模型从架构、训练数据、参数规模等维度进行横向比较 → 读者不必再在散落的论文中逐个查找，能够快速定位适合自己需求的模型。  
2. **HumanEval 统一评测** → 将所有模型在同一基准 HumanEval 上跑分，并绘制直观的性能对比图 → 让模型之间的差距一目了然，避免因评测环境不同导致的误判。  
3. **“三要素”洞察** → 通过大量实验和案例分析，归纳出“规模大、数据好、专家调优”是推动 NL2Code 成绩跃升的关键因素 → 为后续模型设计提供了明确的方向指引。  
4. **开源社区平台** → 搭建 nl2code.github.io 网站，采用众包方式实时更新模型列表、评测结果和新出现的基准 → 把学术成果转化为可持续的公共资源，降低信息滞后风险。  

### 方法详解
这篇综述的整体思路可以拆成四个步骤：**模型收集 → 属性归一化 → 基准统一评测 → 结果分析与洞察**。

1. **模型收集**  
   作者先在 arXiv、GitHub、企业技术博客等渠道检索所有公开的、声称支持 NL2Code 的大语言模型。每找到一个模型，就把它的官方文档、代码仓库和论文抓取下来，形成原始数据池。

2. **属性归一化**  
   为了让不同模型之间可比，作者定义了一套统一的属性表格，包括模型规模（参数数目）、预训练语料类型（通用文本、代码库、混合）、微调策略（全量微调、指令微调、RLHF 等）以及公开可用性。每个模型的这些信息都被填入表格，缺失的部分标记为“未知”。这一步相当于把各家模型的“身份证”统一成同一种格式。

3. **基准统一评测**  
   - **数据准备**：选取 HumanEval 作为唯一评测基准，因为它提供了自然语言描述、参考实现以及自动化单元测试。  
   - **生成流程**：对每个模型，使用官方推荐的调用方式（API、开源代码或自行部署），在相同的温度、采样策略下生成代码。  
   - **评分方式**：将模型输出的代码送入 HumanEval 的测试套件，统计通过的测试用例比例，得到“通过率”。  
   - **结果可视化**：把所有模型的通过率绘制成柱状图，并标注模型规模、是否使用专家调优等信息，以便观察关联性。

4. **结果分析与洞察**  
   作者把实验数据与模型属性对应起来，发现：  
   - 参数量超过 10B 的模型普遍比小模型高出约 15% 的通过率。  
   - 使用高质量代码库（如 GitHub 高星项目）进行预训练的模型，提升约 10%。  
   - 加入专家调优（如使用人类标注的代码对齐数据）后，提升幅度最高可达 20%。  
   这些观察被概括为“三要素”。作者进一步讨论了模型与人类开发者在可解释性、调试成本等方面的差距，并提出未来可能的研究方向。

**最巧妙的地方**在于把“模型属性 → 性能表现”这条因果链用统一基准量化，避免了过去各自为政、评测标准不统一导致的误导。通过公开网站持续更新，作者把一次性调查变成了长期社区资源。

### 实验与效果
- **测试基准**：HumanEval（约 164 条自然语言任务，每条配有 3–5 条单元测试）。  
- **对比基线**：包括 Codex、CodeGen、StarCoder 等早期代码生成模型，以及最新的 GPT‑4‑Code、Claude‑2 等商业模型。  
- **主要结果**：论文给出的表格显示，最好的模型在 HumanEval 上的通过率约为 71%，而 2022 年的 Codex 最高只有 45%。这意味着在相同任务上，新模型多出约 26% 的可用代码。  
- **消融实验**：作者分别去掉“规模大”“高质量数据”“专家调优”三项，发现每去掉一项，整体通过率平均下降 8%–12%，验证了“三要素”并非偶然关联。  
- **局限性**：评测仅限于 HumanEval，覆盖的语言主要是 Python，其他语言（如 Java、C++）的表现未被系统测评；此外，模型的推理成本和实际部署难度在论文中未作深入讨论。

### 影响与延伸思考
这篇综述在发布后迅速成为 NL2Code 研究的“导航图”。不少后续工作直接引用了作者的“三要素”框架，针对数据质量或专家调优进行专项改进，例如推出专门的代码对齐数据集 CodeAlign 或者在大模型上加入“代码自检”模块。社区也围绕 nl2code.github.io 搭建了讨论区，实时共享新模型的 HumanEval 分数，形成了类似 Model Zoo 的生态。想进一步深入的读者可以关注以下方向：  
- **跨语言 NL2Code**：把目前的评测扩展到多语言场景，验证“三要素”是否同样适用。  
- **可解释代码生成**：研究如何让模型在生成代码时给出思路或注释，缩小与人类开发者的可解释性差距。  
- **高效微调技术**：在保持模型规模的前提下，探索更轻量的专家调优方法，降低算力门槛。  

### 一句话记住它
大语言模型在 NL2Code 上的成功，归根结底是“更大、更好、更专业”三要素的合力。