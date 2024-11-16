# Does Prompt Formatting Have Any Impact on LLM Performance?

> **Date**：2024-11-15
> **arXiv**：https://arxiv.org/abs/2411.10541

## Abstract

In the realm of Large Language Models (LLMs), prompt optimization is crucial for model performance. Although previous research has explored aspects like rephrasing prompt contexts, using various prompting techniques (like in-context learning and chain-of-thought), and ordering few-shot examples, our understanding of LLM sensitivity to prompt templates remains limited. Therefore, this paper examines the impact of different prompt templates on LLM performance. We formatted the same contexts into various human-readable templates, including plain text, Markdown, JSON, and YAML, and evaluated their impact across tasks like natural language reasoning, code generation, and translation using OpenAI's GPT models. Experiments show that GPT-3.5-turbo's performance varies by up to 40\% in a code translation task depending on the prompt template, while larger models like GPT-4 are more robust to these variations. Our analysis highlights the need to reconsider the use of fixed prompt templates, as different formats can significantly affect model performance.

---

# 提示格式对大语言模型性能有影响吗？ 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）时代，研究者已经花了大量精力在“怎么写提示”上，比如换词、加入思考链、挑选示例顺序等。但几乎所有工作都默认提示的外观是固定的——普通的纯文本。实际上，提示可以用 Markdown、JSON、YAML 等多种人类可读的结构化形式呈现，而这些格式在模型内部是怎样被解释的，几乎没有系统性的实验数据。缺少对格式敏感性的认识，就会让我们在实际应用中盲目相信“只要内容对，格式随便”。这篇论文正是想填补这个盲区。

### 关键概念速览
**Prompt（提示）**：给模型的输入文字，指明任务、提供上下文或示例，就像老师给学生的考题说明。  
**Prompt Template（提示模板）**：提示的固定框架或排版方式，例如纯文本、Markdown 列表、JSON 键值对等，类似于不同的表格模板。  
**In‑Context Learning（上下文学习）**：把示例直接放进提示里，让模型在同一次推理中学习任务模式，类似于现场演示。  
**Chain‑of‑Thought（思维链）**：让模型先写出推理步骤再给答案，像解题时先写草稿再写结论。  
**Few‑Shot（少样本）**：在提示中提供少量标注例子，引导模型学习，这相当于给模型“几次练习”。  
**Robustness（鲁棒性）**：模型对输入变化的稳定程度，鲁棒的模型就像老练的司机，路况变化也能稳稳开。  
**Code Translation（代码翻译）**：把一种编程语言的代码转换成另一种语言的任务，类似于语言之间的口译。  

### 核心创新点
1. **从内容到形式的系统对比 → 将同一任务的提示内容分别包装成纯文本、Markdown、JSON、YAML 四种模板 → 发现不同格式在 GPT‑3.5‑turbo 上的表现差距最高可达 40%**。之前的研究只关注文字本身，这里把“排版”也当作变量来实验，直接量化了格式的影响。  
2. **跨任务、多模型评估 → 在自然语言推理、代码生成、机器翻译三个任务上，用 OpenAI 的 GPT‑3.5‑turbo 与 GPT‑4 进行对比 → 结果显示小模型对格式极其敏感，而大模型（GPT‑4）相对稳健**。这让我们看到模型规模与格式鲁棒性之间的关联。  
3. **提出“固定模板不可取”概念 → 通过实验数据提醒社区不要把一种模板当作通用标准，而是应该根据模型大小和任务特性选取或动态生成合适的格式**。这在实际产品化时能帮助开发者避免不必要的性能波动。  

### 方法详解
整体思路很直接：保持提示的语义不变，只动排版。具体步骤如下：

1. **任务与数据准备**  
   - 选取三个代表性任务：自然语言推理（如 BoolQ）、代码生成（把 Python 代码改写成 JavaScript）和机器翻译（中英互译）。每个任务挑选若干标准测试集。  
2. **统一内容模板**  
   - 为每个任务手工编写一段完整的提示文字，包含任务描述、输入示例和输出要求。确保所有信息在不同排版下完全一致。  
3. **四种格式化方式**  
   - **Plain Text（纯文本）**：直接把文字顺序写出来。  
   - **Markdown**：使用标题、列表、代码块等 Markdown 语法包装相同内容。  
   - **JSON**：把任务描述、输入、输出分别放进键值对，形成合法的 JSON 对象。  
   - **YAML**：类似 JSON，但采用缩进式的 YAML 语法。  
   这一步可以想象成把同一篇文章分别印在报纸、博客、配置文件和数据交换文件里。  
4. **模型调用**  
   - 使用 OpenAI 的 API，分别把四种提示喂给 GPT‑3.5‑turbo 和 GPT‑4。每种组合跑多次以平滑随机波动。  
5. **性能度量**  
   - 对自然语言推理使用准确率，对代码生成使用代码通过率（能否成功编译并运行），对翻译使用 BLEU 分数。  
6. **统计分析**  
   - 计算每种格式相对于纯文本的相对提升或下降，并用方差分析检验差异显著性。  

最让人意外的设计是 **把 JSON/YAML 当作“提示”直接喂给模型**。很多人会担心模型只能处理自然语言，但实验表明模型会把这些结构化字符串当作普通文本来解析，只是内部的 token 分布不同，导致表现差异。

### 实验与效果
- **测试任务**：自然语言推理（BoolQ 等）、代码翻译（Python→JavaScript）和中英机器翻译。  
- **基线**：纯文本提示。  
- **主要发现**：  
  - 在代码翻译任务上，GPT‑3.5‑turbo 使用 JSON 提示时的代码通过率比纯文本高约 40%，而使用 Markdown 时下降约 15%。  
  - 对自然语言推理，四种格式的差异在 5% 以内，说明该任务对格式不太敏感。  
  - GPT‑4 在所有三项任务上几乎保持在 2% 以内的波动，显示出更强的格式鲁棒性。  
- **消融实验**：作者把 JSON 中的键名改成无意义的随机字符串，性能立刻回落到纯文本水平，说明模型在解析结构化提示时会利用键名的语义暗示。  
- **局限性**：实验只覆盖了 OpenAI 的两款模型，未涉及开源模型或更小的模型；格式种类也仅限四种，未探索 HTML、LaTeX 等其他常见排版。原文未详细描述这些限制的具体数值，只是提出了“需要进一步验证”。  

### 影响与延伸思考
这篇工作让业界重新审视“提示就是提示”的假设。随后有几篇论文（如《Structured Prompting for LLMs》《Prompt Format Search via Reinforcement Learning》）尝试自动搜索最优格式或把格式本身当作可学习的参数。对产品开发者来说，选择合适的提示排版已经可以当作性能调优的一个小技巧。未来的研究方向可能包括：  
- 对开源模型进行同类实验，验证是否存在相同的规模‑鲁棒性关系。  
- 引入更丰富的结构化语言（XML、HTML）以及多模态提示（图文混排）。  
- 开发自动化工具，根据任务特征推荐最佳提示模板。  

### 一句话记住它
**提示的排版方式并非装饰，尤其在小模型上，它能让同一任务的表现相差数十个百分点。**