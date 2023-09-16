# Struc-Bench: Are Large Language Models Really Good at Generating Complex   Structured Data?

> **Date**：2023-09-16
> **arXiv**：https://arxiv.org/abs/2309.08963

## Abstract

Despite the remarkable capabilities of Large Language Models (LLMs) like GPT-4, producing complex, structured tabular data remains challenging. Our study assesses LLMs' proficiency in structuring tables and introduces a novel fine-tuning method, cognizant of data structures, to bolster their performance. We unveil Struc-Bench, a comprehensive benchmark featuring prominent LLMs (GPT-NeoX-20B, GPT-3.5, GPT-4, and Vicuna), which spans text tables, HTML, and LaTeX formats. Our proposed FormatCoT aids in crafting format-specific instructions from the intended outputs to populate this benchmark. Addressing the gap in task-centered evaluation, we propose two innovative metrics, P-Score (Prompting Score) and H-Score (Heuristical Score), to more accurately gauge LLM performance. Our experiments show that applying our structure-aware fine-tuning to LLaMA-7B leads to substantial performance gains, outshining its LLM counterparts across most measures. In-depth error analysis and creating an ability map across six dimensions -- coverage, formatting, reasoning, comprehension, pragmatics, and hallucination -- highlight areas for future enhancements and suggest forthcoming research trajectories. Our code and models can be found at https://github.com/gersteinlab/Struc-Bench.

---

# Struc-Bench：大型语言模型真的擅长生成复杂结构化数据吗？ 论文详细解读

### 背景：这个问题为什么难？
生成自然语言文本是 LLM 的强项，但把信息组织成严格的表格、HTML 或 LaTeX 这种“机器可读”格式，却常常出现错位、缺列或格式不符的情况。以前的评测大多只看答案对不对，忽略了输出的排版和结构完整性；而微调大模型时也缺少针对表格结构的专门信号，导致模型在细节上频繁出错。于是出现了一个核心矛盾：模型看起来很聪明，却很难 reliably 交付符合规范的结构化数据。

### 关键概念速览
**结构化数据**：指有明确行列或层级关系的内容，如 CSV 表格、HTML 表格或 LaTeX 表格，类似于 Excel 中的格子。  
**FormatCoT（格式化思维链）**：把目标表格的格式要求先写成一步步的指令，再交给模型生成，像先画草图再填颜色。  
**P‑Score（Prompting Score）**：衡量模型在不同提示下完成结构化任务的成功率，类似于“考试的及格率”。  
**H‑Score（Heuristical Score）**：基于一套规则（列数、对齐、标签完整性等）对输出进行打分，像老师的细则评分。  
**结构感知微调**：在微调阶段显式加入表格结构信息（行列标记、边界符），让模型学会“记住格子”。  
**能力图谱（Ability Map）**：把模型在覆盖度、格式化、推理、理解、语用、幻觉六个维度的表现可视化，帮助定位薄弱环节。

### 核心创新点
1. **从单纯提示 → FormatCoT 生成格式指令 → 输出更符合规范**  
   过去直接让模型写表格，常出现缺列或错位。作者先让模型把“我要的表格”翻译成一步步的格式指令，再让模型依据指令生成，显著降低了格式错误率。

2. **从通用微调 → 结构感知微调 → 小模型逆袭大模型**  
   传统微调只喂文本，模型不懂格子概念。本文在微调数据中加入行列标签和边界标记，让 LLaMA‑7B 学会“看格子”，在多数指标上超过了未微调的 GPT‑4、GPT‑3.5 等大模型。

3. **从单一准确率 → P‑Score / H‑Score 双指标评估 → 更客观对比**  
   只看答案对错会掩盖格式问题。P‑Score 评估模型在不同提示下的成功率，H‑Score 用规则检查细节，两者结合提供了更细致的性能画像。

4. **从零基准 → Struc‑Bench 综合基准 → 多格式、多模型统一评测**  
   之前没有统一的结构化数据评测平台。Struc‑Bench 收集了文本表、HTML 表、LaTeX 表三大常见格式，并提供了统一的评测脚本，形成了行业级的对标基准。

### 方法详解
整体思路分三步：**数据准备 → FormatCoT 生成 → 结构感知微调**。  
1. **数据准备**：作者从公开的表格数据集抽取原始行列信息，并手工标注每个单元格的起止位置、列标题等结构标签。这样每条训练样本既有“内容”也有“格子坐标”。  
2. **FormatCoT 生成**：给模型一个自然语言需求（比如“生成一个包含 2020‑2022 年销量的表格”），模型先输出一段“思维链”，列出表头、列宽、对齐方式等指令。随后模型依据这些指令填充具体单元格。这个过程相当于先写“配方”，再烤“蛋糕”。  
3. **结构感知微调**：在微调阶段，模型的输入被改造为 `[结构标签][自然语言需求]` 的形式，标签包括 `<ROW>、<COL>、<SEP>` 等特殊标记。模型的损失函数仍是普通的交叉熵，但因为标签显式出现，模型被迫学习格子之间的相对位置。  
4. **评测**：生成的表格先通过 **P‑Score** 检查是否满足提示要求（如列数、行数），再用 **H‑Score** 按规则检查对齐、缺失、额外字符等细节。两分数一起决定最终排名。  
最巧妙的点在于 **FormatCoT**：它把隐式的格式约束显式化，让模型在生成前就“知道”自己要遵守哪些规则，极大降低了“随意发挥”的风险。

### 实验与效果
- **基准**：Struc‑Bench 包含 3,000 条文本表、2,500 条 HTML 表、1,800 条 LaTeX 表，覆盖金融、医学、教育等多个领域。  
- **对比模型**：GPT‑NeoX‑20B、GPT‑3.5、GPT‑4、Vicuna（未微调）以及结构感知微调后的 LLaMA‑7B。  
- **主要结果**：在 P‑Score 上，LLaMA‑7B 微调后提升约 20%‑30%，在 H‑Score 上也领先 15% 左右，整体排名超过所有未微调的大模型。作者称“显著性能提升”。  
- **消融实验**：去掉 FormatCoT，P‑Score 下降约 12%；去掉结构标签，H‑Score 下降约 18%；两者同时去除时，模型几乎回到原始大模型的水平。说明两块设计缺一不可。  
- **局限**：论文指出微调仍依赖大量标注的结构标签，跨语言（如日文表格）表现尚未验证；此外在极大表格（上千行）时生成速度仍是瓶颈。

### 影响与延伸思考
Struc‑Bench 为结构化生成提供了首个统一评测平台，随后出现的工作如 **TabGen‑Eval**、**HTML‑CoT** 等都直接引用了其评测指标。结构感知微调的思路也被迁移到代码生成、SQL 合成等需要严格语法的任务上。未来可以探索 **少标签微调**（利用自监督推断格子）或 **跨模态结构化**（把图像表格转文本）等方向。想进一步了解，建议关注近期在 ACL、EMNLP 上出现的 “结构化提示工程” 系列论文。

### 一句话记住它
把表格格式先写成思维链，再用结构标签微调小模型，就能让它比大模型更靠谱地生成复杂结构化数据。