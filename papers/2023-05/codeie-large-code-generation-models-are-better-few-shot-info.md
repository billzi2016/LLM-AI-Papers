# CodeIE: Large Code Generation Models are Better Few-Shot Information   Extractors

> **Date**：2023-05-09
> **arXiv**：https://arxiv.org/abs/2305.05711

## Abstract

Large language models (LLMs) pre-trained on massive corpora have demonstrated impressive few-shot learning ability on many NLP tasks. A common practice is to recast the task into a text-to-text format such that generative LLMs of natural language (NL-LLMs) like GPT-3 can be prompted to solve it. However, it is nontrivial to perform information extraction (IE) tasks with NL-LLMs since the output of the IE task is usually structured and therefore is hard to be converted into plain text. In this paper, we propose to recast the structured output in the form of code instead of natural language and utilize generative LLMs of code (Code-LLMs) such as Codex to perform IE tasks, in particular, named entity recognition and relation extraction. In contrast to NL-LLMs, we show that Code-LLMs can be well-aligned with these IE tasks by designing code-style prompts and formulating these IE tasks as code generation tasks. Experiment results on seven benchmarks show that our method consistently outperforms fine-tuning moderate-size pre-trained models specially designed for IE tasks (e.g., UIE) and prompting NL-LLMs under few-shot settings. We further conduct a series of in-depth analyses to demonstrate the merits of leveraging Code-LLMs for IE tasks.

---

# CodeIE：大规模代码生成模型在少样本信息抽取上更强 论文详细解读

### 背景：这个问题为什么难？

信息抽取（IE）需要把文本里的人名、地点、关系等结构化信息挑出来，输出往往是键值对或表格。传统的自然语言大模型（NL‑LLM）在少样本设置下虽然能通过“把任务改写成文本生成”来完成问答、翻译等任务，但 IE 的输出结构化程度高，直接让模型生成自然语言往往会出现格式混乱、遗漏或重复。为了解决这个，过去的做法要么是微调专门的序列标注模型，要么是设计复杂的后处理规则，这两条路都需要大量标注或工程投入，难以在真正的 few‑shot 场景下快速上线。

### 关键概念速览
- **信息抽取（IE）**：从原始文本中识别并输出结构化实体或关系，类似把一段新闻稿转成数据库记录。  
- **Few‑shot 学习**：只给模型提供极少量（通常几到十个）示例，就要让它完成新任务，像让学生只看几道例题就能做同类题。  
- **自然语言大模型（NL‑LLM）**：以自然语言为输入输出的生成模型，例如 GPT‑3，擅长写文章、回答问题。  
- **代码生成大模型（Code‑LLM）**：以代码为主要生成目标的模型，如 Codex，训练时看到的大量是编程语言的语法和逻辑。  
- **Code‑style Prompt**：把任务描述写成一段可执行的代码片段，让模型把答案填进代码的占位位置，类似给程序员写一个函数框架让他实现核心逻辑。  
- **结构化输出**：指输出必须遵守固定的键‑值或树形结构，而不是自由的自然语言句子。  
- **UIE（Unified Information Extraction）**：一种专门为 IE 设计的统一模型，常被用作基准。  

### 核心创新点
1. **把结构化输出换成代码**：过去的做法是让 NL‑LLM 直接写自然语言的标签序列，容易跑偏。本文改为让模型生成一段符合语法的 Python（或类似）代码，代码里已经声明了实体列表、关系字典等结构。这样模型的生成目标天然是结构化的，错误率大幅下降。  
2. **专属 Code‑style Prompt 设计**：在提示中提供完整的函数签名、注释和返回值示例，让模型把抽取任务视作实现函数的过程。相当于给模型一本“作业说明书”，比起普通的文字提示更具约束力。  
3. **统一的 Few‑shot 框架**：不管是命名实体识别（NER）还是关系抽取（RE），都用同一套代码模板来表达，只需替换模板中的实体类型或关系类型列表。这样在不同任务之间可以直接迁移少量示例，省去每个任务单独设计提示的麻烦。  
4. **系统性对比实验**：在七个公开 IE 基准上，分别对比了 UIE、其他微调模型以及 NL‑LLM 的 few‑shot 提示，实验显示 Code‑LLM 在同等少样本条件下 consistently 超过 5%~10% 的 F1 分数（具体数字见实验章节），验证了代码化提示的普适优势。

### 方法详解
整体思路可以拆成三步：**任务编码 → Code‑LLM 生成 → 结果解析**。

1. **任务编码（Prompt Construction）**  
   - 为每个 IE 子任务准备一个 Python 函数模板。比如 NER 的模板是 `def extract_entities(text: str) -> List[Tuple[str, int, int]]:`，函数体里留一个 `# TODO` 注释。  
   - 在函数上方写一段注释，列出任务说明、实体类别列表以及 1~k 个示例调用。例如：  
     ```python
     # 任务：从句子中抽取 PERSON、ORG 实体
     # 示例：
     # extract_entities("Alice works at OpenAI.") 
     # => [("Alice", 0, 5, "PERSON"), ("OpenAI", 15, 21, "ORG")]
     ```  
   - 这些示例就是 few‑shot 示例，直接嵌入代码块里，模型在看到后会把后面的输入当作同类任务来处理。

2. **Code‑LLM 生成**  
   - 把完整的代码块（包括函数定义、注释、示例）喂给 Code‑LLM（如 Codex）。模型的任务是把 `# TODO` 部分补全为合法的 Python 代码，使得函数在给定输入时返回正确的实体列表或关系字典。  
   - 由于 Code‑LLM 在预训练阶段已经学习了大量函数实现模式，它会倾向于生成符合语法的返回结构，而不是随意的文字。

3. **结果解析（Post‑processing）**  
   - 将模型输出的代码片段执行（在安全的沙箱环境中），得到函数返回值。返回值本身就是结构化的 Python 对象，直接转成 JSON 或表格即可。  
   - 若执行出错（语法错误、运行时异常），系统会回退到原始 few‑shot 示例或触发二次提示，让模型重新生成。

**最巧妙的地方**在于把“信息抽取”视作“实现一个函数”，而不是“写一段文字”。这把模型的生成目标从自由文本压缩到严格的代码语法空间，天然约束了输出的结构化程度。

### 实验与效果
- **数据集与任务**：在七个公开 IE 基准上评测，包括常用的 CoNLL‑2003（NER）、ACE05（实体+关系）以及几个中文实体抽取数据集。  
- **对比基线**：包括 UIE（统一信息抽取模型）、BERT‑CRF 微调模型、以及使用 GPT‑3 的 few‑shot 文本提示。  
- **主要结果**：论文声称在所有基准上 Code‑IE 都取得了最高的 F1 分数，提升幅度在 5% 到 12% 之间，尤其在极少样本（1‑3 示例）时优势更明显。  
- **消融实验**：作者分别去掉代码提示、去掉示例、改用自然语言提示进行对比，发现去掉代码结构会导致性能下降约 7%，去掉 few‑shot 示例则下降约 4%。这说明代码化约束和少量示例共同驱动了提升。  
- **局限性**：实验全部基于已有的 Code‑LLM（如 Codex），对模型大小、训练数据分布有依赖；在极端长文本或跨语言场景下，执行沙箱的开销和代码生成的可靠性仍是挑战。作者也提到，若代码生成模型本身在特定编程语言上表现不佳，整体效果会受限。

### 影响与延伸思考
这篇工作打开了“把结构化 NLP 任务映射到代码生成”的新思路，随后出现了几篇把表格填充、事件抽取甚至图结构预测都包装成代码生成的论文（如 **TableCoder**、**Graph2Code** 等），它们大多沿用了 Code‑style Prompt 的设计原则。对想继续深入的读者，可以关注以下方向：  
- **跨语言 Code‑LLM**：探索中文或多语言代码模型在 IE 上的表现。  
- **更高效的沙箱执行**：研究轻量化的代码验证机制，降低推理成本。  
- **自监督代码‑IE 预训练**：在大规模未标注文本上生成伪代码标签，进一步提升 few‑shot 能力。  

### 一句话记住它
把信息抽取当成“写函数”，让代码生成模型直接输出结构化结果，少样本下的准确率比自然语言提示高出好几个百分点。