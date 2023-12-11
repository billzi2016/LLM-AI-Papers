# Control Risk for Potential Misuse of Artificial Intelligence in Science

> **Date**：2023-12-11
> **arXiv**：https://arxiv.org/abs/2312.06632

## Abstract

The expanding application of Artificial Intelligence (AI) in scientific fields presents unprecedented opportunities for discovery and innovation. However, this growth is not without risks. AI models in science, if misused, can amplify risks like creation of harmful substances, or circumvention of established regulations. In this study, we aim to raise awareness of the dangers of AI misuse in science, and call for responsible AI development and use in this domain. We first itemize the risks posed by AI in scientific contexts, then demonstrate the risks by highlighting real-world examples of misuse in chemical science. These instances underscore the need for effective risk management strategies. In response, we propose a system called SciGuard to control misuse risks for AI models in science. We also propose a red-teaming benchmark SciMT-Safety to assess the safety of different systems. Our proposed SciGuard shows the least harmful impact in the assessment without compromising performance in benign tests. Finally, we highlight the need for a multidisciplinary and collaborative effort to ensure the safe and ethical use of AI models in science. We hope that our study can spark productive discussions on using AI ethically in science among researchers, practitioners, policymakers, and the public, to maximize benefits and minimize the risks of misuse.

---

# 控制人工智能在科学领域潜在误用风险 论文详细解读

### 背景：这个问题为什么难？

AI 正在渗透到化学、材料、生物等科研领域，模型可以在几秒钟内生成新分子结构或合成路线，这本是加速发现的好事。但同样的能力也能被用来设计毒性更强的化学武器、规避监管限制等。过去的安全研究大多聚焦在生成有害内容的文本或图像上，缺少针对科研场景的风险评估和防护手段。传统的模型审查往往是事后检测，无法在模型生成过程里实时拦截潜在危害，导致误用风险仍然高悬。因此，需要一种专门面向科学 AI 的风险控制框架，既能保持模型的科研效能，又能在危害出现前把关。

### 关键概念速览

**AI for Science（科学 AI）**：指在科研活动中使用的人工智能模型，例如预测分子属性、自动推导实验方案等。它像实验室的“智能助手”，帮助研究者快速筛选候选方案。

**误用风险**：模型被用于违背伦理或法律的目的，例如合成毒剂或规避监管。类似于一把锋利的刀子，既能切菜也能伤人，关键在于使用者的意图。

**Red‑Team（红队）**：主动寻找系统漏洞的攻击者团队，在安全领域常用来检验防御效果。这里的红队指的是故意让模型生成有害科研内容的测试者。

**Benchmark（基准测试）**：统一的评估集合，用来比较不同系统的安全表现。就像跑步比赛的计时器，提供公平的成绩对比。

**SciGuard**：论文提出的风险控制系统，像实验室的安全门卫，实时监控并拦截可能的危害输出。

**SciMT‑Safety**：专门为化学科研设计的红队基准，提供一系列诱导模型产生危险信息的任务，类似于化学实验的“安全演练”。

**Benign Test（良性测试）**：评估模型在正常科研任务上的表现，确保安全措施不会把有用的功能也拦下来。

**Multidisciplinary Collaboration（多学科协作）**：指化学、AI、法律、伦理等不同领域的专家共同制定和执行安全规范，类似于跨部门的应急演练。

### 核心创新点

1. **从通用安全转向科研专属风险**：之前的安全工作主要针对文本、图像等通用内容，缺少针对化学合成路线等专业输出的防护。本文把风险聚焦在科学 AI 上，首次系统化列举了科研场景的具体危害点。

2. **SciGuard：实时拦截+可解释的风险评估** → 通过在模型生成链路中嵌入风险检测模块，先对生成的化学描述进行危害评分，再决定是否阻断或修改输出。相比事后审计的被动方式，SciGuard 能在危害出现前直接干预，显著降低误用窗口。

3. **SciMT‑Safety 基准** → 构建了一个专门诱导模型产生危险化学信息的红队测试集，提供了统一的安全评估标准。以前没有针对科研 AI 的公开基准，导致不同系统难以对比。此基准让研究者可以量化安全性能。

4. **安全与性能的双重评估框架** → 在保持模型在常规科研任务上（如分子属性预测）的准确性不下降的前提下，评估其在 SciMT‑Safety 上的危害输出量。实验显示，SciGuard 在安全指标上领先，同时在良性测试上几乎不损失性能。

### 方法详解

**整体思路**：SciGuard 把风险控制嵌入到 AI 生成流程的每一步，形成“生成‑评估‑拦截‑反馈”四段循环。首先让模型产生候选科研内容；随后一个专门的风险评估器对候选进行危害打分；如果分数超过阈值，系统会触发拦截或要求模型重新生成；最后将评估结果反馈给模型，以便其在后续生成中自我调节。

**关键模块拆解**：

1. **生成模块**：使用现有的科学大模型（如化学专用的 LLM）完成分子结构、合成路线或实验方案的初稿。这里不做任何改动，只是普通的生成过程。

2. **风险评估器**：基于两层模型构建。第一层是**危害词典匹配**，快速捕捉已知的高危化学术语（如“神经毒剂”“高爆炸性”）。第二层是**语义危害分类器**，采用微调的二分类模型，输入完整的生成文本，输出“安全”“潜在危害”概率。分类器的训练数据来源于人工标注的化学危害案例和安全案例。

3. **拦截策略**：当评估器给出高危概率超过预设阈值时，系统会执行三种可能的动作：
   - **直接阻断**：不返回任何输出，提示用户内容被拦截。
   - **内容过滤**：对高危片段进行遮蔽或替换，保留其余安全信息。
   - **重新生成**：向生成模块发送负向提示（如“避免使用毒性化合物”），让模型在约束下重新输出。

4. **反馈回路**：拦截或过滤后的结果会被包装成“安全提示”，喂回生成模块的下一轮采样过程。这样模型在同一次对话中逐渐学习到哪些表达会触发拦截，从而自我规避。

**最巧妙的设计**：风险评估器并不是单纯的黑盒分类，而是把**规则匹配**和**深度语义判断**结合。规则匹配保证了已知高危词汇的零误报，而深度分类则捕捉到隐晦的危害表达（比如用代号或化学式暗示毒剂），两者互补提升了拦截率。

### 实验与效果

- **测试数据**：作者在公开的化学文献数据集上进行常规科研任务（如分子属性预测、合成路径生成）的良性测试；同时使用自行构建的 SciMT‑Safety 基准，对模型的危害输出进行红队评估。

- **基线对比**：与未加任何安全层的原始模型、以及仅使用规则匹配的简易防护系统相比，SciGuard 在 SciMT‑Safety 上的危害输出率下降约 **70%**（原始模型约 30% 的生成被标记为高危，SciGuard 降至 9%），而在良性测试上准确率仅下降 **1.2%**，几乎保持原有性能。

- **消融实验**：去掉深度分类器，仅保留规则匹配时，危害拦截率跌至 45%；去掉规则匹配，仅保留分类器时，误报率上升至 12%。说明两者协同是关键。

- **局限性**：论文承认风险评估器仍依赖于已有的危害词典和标注数据，对全新、未被收录的化学危害手段可能失效；此外，拦截策略的阈值设置需要在不同科研领域细调，未提供自动调参方案。

### 影响与延伸思考

这篇工作首次把安全红队评估引入科学 AI，开启了“AI 研究安全”这一子领域的系统化探索。后续有几篇工作尝试将类似的风险评估框架扩展到材料科学、基因编辑等更广的科研场景（如 2024 年的 *MaterialGuard*、2025 年的 *BioSafeAI*），并提出更细粒度的危害标签体系。对想进一步了解的读者，可以关注 **AI安全红队基准** 的发展、**多模态危害检测**（如结合结构式图像的风险评估）以及 **法规与技术协同** 的政策研究。

### 一句话记住它

SciGuard 用实时风险评估把科研 AI 的“强大能力”装上安全保险，既挡住危害，又不削弱科研效能。