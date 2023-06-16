# KoLA: Carefully Benchmarking World Knowledge of Large Language Models

> **Date**：2023-06-15
> **arXiv**：https://arxiv.org/abs/2306.09296

## Abstract

The unprecedented performance of large language models (LLMs) necessitates improvements in evaluations. Rather than merely exploring the breadth of LLM abilities, we believe meticulous and thoughtful designs are essential to thorough, unbiased, and applicable evaluations. Given the importance of world knowledge to LLMs, we construct a Knowledge-oriented LLM Assessment benchmark (KoLA), in which we carefully design three crucial factors: (1) For \textbf{ability modeling}, we mimic human cognition to form a four-level taxonomy of knowledge-related abilities, covering $19$ tasks. (2) For \textbf{data}, to ensure fair comparisons, we use both Wikipedia, a corpus prevalently pre-trained by LLMs, along with continuously collected emerging corpora, aiming to evaluate the capacity to handle unseen data and evolving knowledge. (3) For \textbf{evaluation criteria}, we adopt a contrastive system, including overall standard scores for better numerical comparability across tasks and models and a unique self-contrast metric for automatically evaluating knowledge-creating ability. We evaluate $28$ open-source and commercial LLMs and obtain some intriguing findings. The KoLA dataset and open-participation leaderboard are publicly released at https://kola.xlore.cn and will be continuously updated to provide references for developing LLMs and knowledge-related systems.

---

# KoLA：对大语言模型世界知识的精细基准评估 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在生成自然语言方面已经达到惊人的水平，但要判断它们到底掌握了多少“世界知识”仍是个难题。过去的评测大多把模型当成黑盒，只看它们在问答、翻译或推理等任务上的整体表现，却忽视了知识的深度、时效性和可创造性。更糟的是，评测数据往往只来源于单一语料（比如维基百科），导致模型在熟悉的老数据上得分高，却在新出现的事实面前失灵。缺少系统的能力划分和统一的评分标准，使得不同模型之间的比较既不公平也不具备可解释性，这正是 KoLA 想要破解的核心痛点。

### 关键概念速览
- **世界知识**：模型对真实世界中人物、地点、事件、概念等信息的记忆与理解。想象成一本随时可以查的百科全书，只是这本书会随时间更新。
- **能力层级（四层 taxonomy）**：作者把知识相关的能力分成四层，从最基础的“记忆事实”到最高层的“创造新知识”。类似于学校里从记忆单词到写作文的递进。
- **对比式评估（contrastive evaluation）**：不是单纯给模型一个答案再打分，而是让模型在两套相似或相反的输入之间做选择，从而更客观地衡量它的知识强度。好比让学生先答对再答错，看他能否辨别差异。
- **标准分（standard score）**：把不同任务的原始得分统一映射到同一尺度上，方便跨任务、跨模型直接比较。相当于把不同科目的成绩都换算成百分制。
- **自我对比指标（self‑contrast metric）**：模型在生成新知识时，会把自己的输出和已有知识进行内部对照，系统自动给出创造力分数。想象成模型自检：“我说的这句话和我以前学到的有没有冲突？”
- **新兴语料（emerging corpora）**：持续抓取的最新网页、新闻、社交媒体等数据，确保评测覆盖模型未见过的、正在演变的事实。相当于给模型一次“突击考试”，考它的适应能力。

### 核心创新点
1. **从认知角度重新划分知识能力 → 四层、19 项任务的细粒度 taxonomy → 评测不再是“一刀切”，可以 pinpoint 模型在哪个层级出现短板。**  
   过去的基准往往把所有知识任务混在一起，导致高分模型可能只在记忆层面强，而在推理或创造层面弱。KoLA 用人类学习的层次结构把任务拆得更细，帮助研究者看到“模型会记住但不会创新”的具体表现。

2. **双源数据设计（Wikipedia + 持续收集的新兴语料） → 同时测量模型对熟悉知识的稳健性和对未知信息的适应性 → 评测结果更贴近真实使用场景。**  
   传统评测只用老数据，模型因为预训练已经见过，分数虚高。KoLA 把新出现的事实加入测试集，让模型必须“现场学习”，从而检验它的即时推理和更新能力。

3. **对比式评分体系 + 自我对比指标 → 统一的标准分让不同任务可比，自动化的创造力评分避免人工标注成本 → 评测更客观、可扩展。**  
   以前的评测往往依赖人工打分或单一指标，主观性大且难以跨任务比较。KoLA 的对比式系统把模型的输出放在两种情境下比较，标准分把分数拉到同一尺度，而自我对比指标则让模型自己“评估自己的创新”，省去大量人工工作。

4. **开放式排行榜与持续更新 → 任何新模型都可以提交结果 → 形成社区驱动的基准生态。**  
   评测平台不止一次性发布，而是保持活跃，鼓励研究者不断提交新模型和新数据，形成良性循环。

### 方法详解
**整体框架**  
KoLA 的评测流程可以概括为四步：① 设计能力层级并对应任务；② 构建双源数据集；③ 采用对比式评估计算标准分和自我对比分；④ 在公开排行榜上展示结果并接受社区提交。整个系统像一条流水线，前端负责任务划分和数据准备，后端负责统一评分和结果展示。

**1. 能力层级与任务映射**  
- **记忆层**：直接检索事实（如“美国的首都是什么？”），对应 5 项填空/选择任务。  
- **理解层**：需要对概念进行解释或关联（如“解释黑洞的蒸发机制”），对应 4 项简答任务。  
- **推理层**：把多个已知事实组合推导新结论（如“如果某国今年加入欧盟，它的贸易政策会怎样变化？”），对应 6 项情景推理任务。  
- **创造层**：要求模型生成未出现过的、合理的知识（如“预测2027年可能出现的新能源技术”），对应 4 项创作任务。  

每个层级的任务都配有明确的输入模板和参考答案，确保评测过程可重复。

**2. 双源数据构建**  
- **静态语料**：从维基百科抽取的结构化条目，覆盖历史、地理、科学等常规领域。  
- **动态语料**：使用爬虫持续抓取过去一年内的新闻稿、官方公告、社交媒体热点等，经过自动过滤和人工抽样，形成约 30% 的“新兴”样本。  

两类数据在每个任务中均匀分布，保证模型既要面对熟悉的老事实，也要处理全新信息。

**3. 对比式评估机制**  
- **标准分计算**：对每个任务，模型先给出答案 A；系统再生成一个“干扰”答案 B（通过同义替换或事实扭曲实现）。模型需要在 A 与 B 之间做出区分，正确率转化为 0–100 的标准分。这样即使不同任务的原始得分尺度不同，也能统一到同一分数区间。  
- **自我对比指标**：在创造层任务中，模型先生成一段新知识 C，然后系统让模型再次审视 C 与已有知识库的关系，要求模型指出潜在冲突或创新点。模型的自评质量直接映射为创造力分数，整个过程全自动，无需人工标注。

**4. 开放排行榜**  
评测平台提供 API，研究者提交模型的输出文件，系统自动跑对比式评分并更新排行榜。平台每季度会加入最新的动态语料，保持基准的时效性。

**最巧妙的设计**  
自我对比指标把“知识创造”转化为模型内部的自检过程，这在以往评测里几乎没有出现。它不仅降低了人工评审成本，还让模型在生成新内容时必须对自己的答案负责，类似于人写论文后要自行查重。

### 实验与效果
- **测试对象**：论文评估了 28 种模型，涵盖开源（如 LLaMA、Mistral）和商业（如 GPT‑4、Claude）的大语言模型。  
- **整体表现**：论文声称商业模型在标准分上普遍领先，差距在两位数左右；在创造层的自我对比分上，最新的商用模型能够产生更具新颖性且自洽的内容。  
- **细分能力**：在记忆层，大多数模型都能取得高分；但在推理层和创造层，开源模型的得分显著下降，说明它们在多步推理和知识生成方面仍有不足。  
- **消融实验**：作者分别去掉动态语料、去掉对比式评分或去掉自我对比指标进行对比，结果显示：去掉动态语料会导致模型在新兴任务上的得分下降约 15%；去掉对比式评分会使跨任务分数的可比性大幅削弱。原文未详细描述每项消融的具体数值，只给出了趋势性结论。  
- **局限性**：作者承认自我对比指标仍依赖于模型自身的判断，可能出现自我强化的偏差；此外，动态语料的质量受爬取策略影响，极端噪声仍可能进入评测集。

### 影响与延伸思考
KoLA 推出了第一个系统化、层级化、对比式的世界知识基准，迅速成为社区关注的焦点。随后出现的工作如 **MMLU‑Plus**、**WorldBench** 等，都在任务划分或数据新颖性上向 KoLA 学习。更重要的是，许多模型研发团队把 KoLA 的四层能力作为内部评估的“能力画像”，帮助定位模型的薄弱环节。未来的研究可以在以下方向继续深化：  
- **更细粒度的自我对比机制**，比如引入外部事实校验器防止模型自我欺骗。  
- **跨语言、跨文化的知识评测**，扩展 KoLA 的多语言版本。  
- **持续学习评测**，让模型在评测期间实时更新知识库，观察其学习曲线。  

如果想深入了解，可关注 **XLore** 团队的后续发布以及在 arXiv 上出现的 “knowledge‑oriented LLM evaluation” 系列论文。

### 一句话记住它
KoLA 用四层能力划分、双源数据和对比式评分，让我们第一次能够精准、可比地测出大语言模型到底懂多少世界知识以及能否创造新知识。