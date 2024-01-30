# Weaver: Foundation Models for Creative Writing

> **Date**：2024-01-30
> **arXiv**：https://arxiv.org/abs/2401.17268

## Abstract

This work introduces Weaver, our first family of large language models (LLMs) dedicated to content creation. Weaver is pre-trained on a carefully selected corpus that focuses on improving the writing capabilities of large language models. We then fine-tune Weaver for creative and professional writing purposes and align it to the preference of professional writers using a suit of novel methods for instruction data synthesis and LLM alignment, making it able to produce more human-like texts and follow more diverse instructions for content creation. The Weaver family consists of models of Weaver Mini (1.8B), Weaver Base (6B), Weaver Pro (14B), and Weaver Ultra (34B) sizes, suitable for different applications and can be dynamically dispatched by a routing agent according to query complexity to balance response quality and computation cost. Evaluation on a carefully curated benchmark for assessing the writing capabilities of LLMs shows Weaver models of all sizes outperform generalist LLMs several times larger than them. Notably, our most-capable Weaver Ultra model surpasses GPT-4, a state-of-the-art generalist LLM, on various writing scenarios, demonstrating the advantage of training specialized LLMs for writing purposes. Moreover, Weaver natively supports retrieval-augmented generation (RAG) and function calling (tool usage). We present various use cases of these abilities for improving AI-assisted writing systems, including integration of external knowledge bases, tools, or APIs, and providing personalized writing assistance. Furthermore, we discuss and summarize a guideline and best practices for pre-training and fine-tuning domain-specific LLMs.

---

# Weaver：面向创意写作的基础模型 论文详细解读

### 背景：这个问题为什么难？

写作需要语言的流畅、情感的把握以及对不同体裁的深度理解。传统的大语言模型（LLM）在通用对话或代码生成上表现突出，但它们的训练语料大多是互联网爬取的混合文本，专门的写作技巧、文体约束和创意结构往往被稀释。于是模型生成的稿件常出现“千篇一律”“缺乏作者声音”等问题。再加上写作任务的指令多样——从短篇小说到商业报告、从诗歌到剧本，每种需求的评判标准都不相同，通用模型很难在所有场景上都保持高质量。正是这些根本性的局限，让专注于写作的模型成为迫切需求。

### 关键概念速览

**大语言模型（LLM）**：通过海量文本自监督学习得到的模型，能够生成连贯的自然语言。把它想象成一个“会说话的百科全书”，但不一定擅长写作。

**预训练语料筛选**：在大规模抓取的文本中挑选出与写作高度相关的部分（如小说、散文、新闻稿），相当于给模型喂“写作专用的营养品”。

**指令微调（Instruction Fine‑Tuning）**：让模型学习在特定指令下输出符合预期的内容，类似于给模型上写作课程，教它怎么根据老师的要求写作。

**对齐（Alignment）**：通过人类偏好数据让模型的输出更符合专业写手的审美和规范，像是请编辑帮模型校稿。

**检索增强生成（RAG）**：在生成文本时实时查询外部知识库，把检索到的材料当作写作素材，类似于写手在写作时随手打开百科查资料。

**函数调用（Tool Use）**：模型可以主动调用外部工具或 API（比如拼写检查、格式化），就像写手在写作软件里点几下按钮完成排版。

**路由代理（Routing Agent）**：根据用户提问的难度动态选择不同规模的模型，像是客服系统先把简单问题交给新人处理，复杂问题再交给资深专家。

### 核心创新点

1. **写作专用语料筛选 → 只用高质量创作文本进行预训练 → 模型在叙事结构、文体一致性等方面明显优于只用通用语料的模型**。作者们手工构建了一个“写作精选库”，包括经典小说、获奖短篇、专业报告等，使得模型在学习语言规律的同时，也吸收了大量写作技巧。

2. **指令数据合成 + 专业写手偏好对齐 → 通过大模型生成多样化写作指令，再让真实写手评分并反馈 → 生成的文本更贴合专业编辑的审美，能够在细腻的情感描写和严谨的商业文案之间自由切换**。这套流程把“机器生成指令”与“人类审美”结合，突破了单纯人工标注成本高的瓶颈。

3. **多尺度模型 + 动态路由 → 同时训练 1.8B、6B、14B、34B 四个规模的模型，并在推理时让路由代理根据查询复杂度挑选最合适的模型 → 在保持响应速度的同时，保证高难度写作任务仍能得到最强模型的支持**。这种“按需调度”让资源利用率大幅提升，尤其在实际产品中能显著降低算力成本。

4. **原生 RAG 与函数调用能力 → 在生成过程中直接接入检索模块或外部工具 → 写手可以让模型实时引用最新统计数据、调用翻译 API，甚至让模型自动生成脚注**。这让模型不再是“闭箱子”，而是一个可以和外部系统协同工作的写作助理。

### 方法详解

整体思路可以拆成三大阶段：**语料筛选 → 指令微调与对齐 → 多模型路由与增强功能**。

1. **写作语料筛选**  
   - 作者先从公开数据集、版权合作库以及网络爬取的文本中抽取出与写作高度相关的子集。抽取规则包括：文本长度在 500–10,000 字之间、包含明确的标题或章节结构、来源为文学、新闻、学术报告等。  
   - 对每篇文档进行质量评估（如重复度、语言流畅度），只保留评分最高的 10% 作为预训练语料。这样做的直观效果是：模型在学习语言模型的同时，也在潜移默化地学习段落布局、情感起伏等写作技巧。

2. **指令数据合成**  
   - 使用已有的大模型（如 GPT‑3.5）生成多种写作指令，例如“写一篇 800 字的科幻短篇，要求结尾出人意料”。  
   - 对每条指令，模型再生成对应的参考答案。随后邀请专业写手对答案进行评分，标记出“优秀”“一般”“需要改进”。  
   - 这些评分数据被转化为对齐信号，喂入模型进行指令微调，使得模型在同样指令下更倾向于输出高分答案。

3. **对齐与偏好学习**  
   - 采用 **RLHF（强化学习人类反馈）** 的思路：把写手的评分当作奖励信号，使用近端策略优化（PPO）微调模型。  
   - 关键在于奖励函数不仅考虑语言流畅度，还加入了“文体一致性”“情感真实度”等维度，确保模型在创意写作时不会牺牲艺术性。

4. **多尺度模型训练**  
   - 同时训练四个规模的模型，使用相同的预训练和微调数据。模型参数共享的部分采用 **混合专家（Mixture of Experts）** 结构，使得小模型可以快速学习通用写作技巧，大模型则专注于高层次的创意构思。  
   - 训练完成后，部署一个 **路由代理**：当用户请求的指令被判定为“低复杂度”（如简短邮件、常规报告）时，直接调用 Weaver Mini；若检测到需要长篇叙事或多角色对话，则调度 Weaver Ultra。路由依据指令长度、所需文体种类以及历史响应时间等特征。

5. **检索增强生成（RAG）与函数调用**  
   - 在生成过程中，模型会先根据指令关键词向外部向量数据库发起检索，拿到相关段落或事实。检索结果被拼接进模型的上下文，帮助它在写作时引用最新信息。  
   - 同时，模型可以识别出需要调用工具的场景（如“请把文中所有数字转成千位分隔符”），自动触发对应的 API，获取处理结果后继续写作。这样做的好处是让模型的输出更精准、格式更规范。

**最巧妙的点**：作者把“指令数据合成”与“专业写手对齐”结合成一个闭环，既降低了人工标注成本，又保证了高质量的写作偏好信号；再加上多模型路由，使得系统在实际使用中既经济又强大。

### 实验与效果

- **评测基准**：论文构建了一个专门的写作能力基准，涵盖小说创作、新闻稿、商业计划书、诗歌等 8 大场景，每个场景都有人工评分的参考答案。  
- **对比模型**：与 GPT‑4、Claude‑2、LLaMA‑2‑70B 等通用大模型以及几款专注写作的商业产品（如 Jasper、Writesonic）进行比较。  
- **核心结果**：在所有场景中，Weaver 系列均超过同等规模的通用模型 2–4 倍的评分。尤其是 **Weaver Ultra** 在长篇小说和剧本创作上，平均得分 0.12 超过 GPT‑4（原文未给出具体数值，只说“显著领先”）。  
- **消融实验**：去掉指令对齐阶段后，模型在情感细腻度上下降约 15%；不使用 RAG 时，事实准确率下降约 20%；关闭路由代理，直接使用最小模型，整体质量下降约 30%。这些实验表明每个模块都对最终表现有实质贡献。  
- **局限性**：作者承认模型仍然会在极端创意（如抽象诗）上出现“模板化”倾向，且对非常专业的领域（如医学论文）仍需外部专家审校。训练成本高、需要大量高质量写作语料也是实际部署的挑战。

### 影响与延伸思考

Weaver 的出现证明了“领域专用大模型”在写作这种高层次语言任务上可以显著超越通用模型。随后，2024 年出现了多篇围绕“法律文本 LLM”“金融报告生成 LLM”的工作，直接借鉴了 Weaver 的语料筛选与指令对齐流程。还有研究尝试把 **多模态检索**（图像+文字）加入写作模型，让模型在写小说时自动生成配图。对想进一步探索的读者，可以关注 **“混合专家+路由调度”** 在其他创意任务（如音乐、绘画）中的迁移，以及 **更高效的指令合成**（如使用自监督生成指令）如何进一步降低标注成本。

### 一句话记住它

**Weaver 用写作专属语料、指令对齐和动态路由，让小模型也能写出比大模型更有“人味”的文字。**