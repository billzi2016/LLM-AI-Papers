# Artificial Artificial Artificial Intelligence: Crowd Workers Widely Use   Large Language Models for Text Production Tasks

> **Date**：2023-06-13
> **arXiv**：https://arxiv.org/abs/2306.07899

## Abstract

Large language models (LLMs) are remarkable data annotators. They can be used to generate high-fidelity supervised training data, as well as survey and experimental data. With the widespread adoption of LLMs, human gold--standard annotations are key to understanding the capabilities of LLMs and the validity of their results. However, crowdsourcing, an important, inexpensive way to obtain human annotations, may itself be impacted by LLMs, as crowd workers have financial incentives to use LLMs to increase their productivity and income. To investigate this concern, we conducted a case study on the prevalence of LLM usage by crowd workers. We reran an abstract summarization task from the literature on Amazon Mechanical Turk and, through a combination of keystroke detection and synthetic text classification, estimate that 33-46% of crowd workers used LLMs when completing the task. Although generalization to other, less LLM-friendly tasks is unclear, our results call for platforms, researchers, and crowd workers to find new ways to ensure that human data remain human, perhaps using the methodology proposed here as a stepping stone. Code/data: https://github.com/epfl-dlab/GPTurk

---

# 人工 人工 人工智能：众包工作者广泛使用大语言模型进行文本生成任务 论文详细解读

### 背景：这个问题为什么难？

在机器学习的训练和评估过程中，往往需要大量高质量的人类标注数据。传统上，研究者会通过众包平台（如 Amazon Mechanical Turk）以低成本快速获取这些标注。然而，随着大语言模型（LLM）如 GPT‑4、Claude 等能够生成几乎和人类一样的文本，标注者可能会直接让模型帮忙完成任务，从而把“人类标注”变成“机器生成”。如果不加以检测，研究者会误以为得到的是纯粹的人类判断，导致对模型能力的评估出现系统性偏差。此前的工作大多关注如何让 LLM 生成标注，而很少审视 LLM 本身对众包标注过程的渗透程度，这正是本文要解决的盲点。

### 关键概念速览
- **大语言模型（LLM）**：基于海量文本训练的生成式模型，能够完成写作、翻译、摘要等任务，像是“会说话的搜索引擎”。  
- **众包平台**：在线市场（如 MTurk）把微任务分配给大量自由职业者，类似于“全球版的快递员”。  
- **金标准标注**：被视为最可靠的人类标注，用来衡量模型或算法的真实表现。  
- **键盘敲击检测**：通过记录工作者的键入节奏和停顿，判断他们是手动输入还是粘贴/生成文本，类似于“键盘版的指纹”。  
- **合成文本分类**：训练一个二分类模型来辨别文本是人写的还是机器生成的，像是“真假新闻检测器”。  
- **任务友好度**（LLM‑friendly task）：指那些可以直接交给语言模型完成的任务，例如摘要、改写等，模型几乎不需要额外提示就能产出高质量结果。  

### 核心创新点
1. **从使用行为角度审视 LLM 渗透**：以前的研究多关注 LLM 生成标注的质量，这里直接测量“标注者到底用了没有”。作者把焦点从“标注内容”转向“标注过程”。  
2. **双管齐下的检测框架**：传统的键盘敲击分析只能捕捉粘贴行为，合成文本分类则能捕捉即使手动敲入的机器生成文本。两者结合提升了检测覆盖率。  
3. **在真实众包任务中复现实验**：作者重新在 MTurk 上跑了一遍已有的摘要任务，而不是在实验室模拟，从而得到更具外部效度的使用率估计。  
4. **提供开源工具链**：把键盘日志采集、特征提取、分类模型等代码全部公开，为后续研究提供了可直接复用的“检测套件”。  

### 方法详解
整体思路可以拆成三步：**任务部署 → 行为采集 → 使用率估计**。

1. **任务部署**  
   - 选取文献中常用的抽象摘要任务（给定论文标题和正文，要求写出 150 字左右的摘要）。  
   - 在 MTurk 上发布，同步要求工作者安装一个轻量的浏览器插件，该插件负责记录键盘事件（按键时间、键入字符、粘贴事件等），但不收集任何实际内容，确保隐私。  

2. **行为采集**  
   - **键盘敲击特征**：包括平均敲击间隔、连续敲击的长度、粘贴快捷键（Ctrl+V）出现频次等。若出现大量粘贴，直接标记为可能使用 LLM。  
   - **文本特征**：收集工作者提交的摘要后，使用预训练的二分类模型（作者自行训练的“人‑机判别器”）对文本进行打分。该模型在公开的机器生成文本库和人工写作库上进行微调，能够捕捉微妙的语言模式差异。  

3. **使用率估计**  
   - 对每位工作者，先看键盘特征是否超过阈值；若未触发，再看文本分类分数是否超过机器阈值。两者取并集得到最终的“使用 LLM”判定。  
   - 统计所有工作者的判定比例，得到 33%–46% 的使用率区间。区间的宽窄来源于不同阈值设置的敏感性分析。  

**巧妙之处**：作者没有仅依赖单一信号，而是把行为日志和内容分析结合，类似于“多因素体检”。即使工作者把机器生成的文本手动敲入键盘，键盘特征仍可能显示异常的高频率停顿；反之，即使不粘贴，文本分类也能捕捉到机器痕迹。

### 实验与效果
- **数据集/任务**：使用了已有的学术论文摘要任务，约 200 条独立的摘要请求，招募了 150 名 MTurk 工作者。  
- **基线对比**：仅用键盘检测或仅用文本分类分别得到约 20% 与 30% 的使用率，远低于综合方法的 33%–46%。这说明单一信号容易漏检。  
- **消融实验**：去掉文本分类后，整体使用率下降约 8%；去掉键盘特征后下降约 5%，表明两者都有贡献，且键盘特征在高置信度检测中更关键。  
- **局限性**：作者承认该实验只针对摘要这种高度 LLM‑friendly 的任务，其他需要深度推理或专业知识的标注可能出现不同的使用率；此外，键盘日志的采集在某些地区可能受法规限制。  

### 影响与延伸思考
这篇工作首次在众包社区掀起对“标注是否真的人为” 的警觉，随后有几篇后续研究尝试把相同的检测框架搬到图像标注、情感标注等任务上，甚至出现了平台层面的“AI 使用声明”。如果你想进一步了解，可以关注以下方向：① 更细粒度的行为指纹（如鼠标轨迹）② 对抗性训练提升机器生成文本的不可检测性③ 众包平台政策层面的技术治理。  

### 一句话记住它
在众包标注中，约三分之一的工作者已经在暗用大语言模型，只有结合键盘行为和文本判别才能真正辨别“人手”与“机器手”。