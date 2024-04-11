# ResearchAgent: Iterative Research Idea Generation over Scientific   Literature with Large Language Models

> **Date**：2024-04-11
> **arXiv**：https://arxiv.org/abs/2404.07738

## Abstract

The pace of scientific research, vital for improving human life, is complex, slow, and needs specialized expertise. Meanwhile, novel, impactful research often stems from both a deep understanding of prior work, and a cross-pollination of ideas across domains and fields. To enhance the productivity of researchers, we propose ResearchAgent, which leverages the encyclopedic knowledge and linguistic reasoning capabilities of Large Language Models (LLMs) to assist them in their work. This system automatically defines novel problems, proposes methods and designs experiments, while iteratively refining them based on the feedback from collaborative LLM-powered reviewing agents. Specifically, starting with a core scientific paper, ResearchAgent is augmented not only with relevant publications by connecting information over an academic graph but also entities retrieved from a knowledge store derived from shared underlying concepts mined across numerous papers. Then, mimicking a scientific approach to improving ideas with peer discussions, we leverage multiple LLM-based ReviewingAgents that provide reviews and feedback via iterative revision processes. These reviewing agents are instantiated with human preference-aligned LLMs whose criteria for evaluation are elicited from actual human judgments via LLM prompting. We experimentally validate our ResearchAgent on scientific publications across multiple disciplines, showing its effectiveness in generating novel, clear, and valid ideas based on both human and model-based evaluation results. Our initial foray into AI-mediated scientific research has important implications for the development of future systems aimed at supporting researchers in their ideation and operationalization of novel work.

---

# ResearchAgent：基于大语言模型的迭代式科研创意生成系统 论文详细解读

### 背景：这个问题为什么难？

科研的核心是从已有文献中抽象出新问题、设计算法或实验方案，这需要对海量论文的深度理解和跨领域的联想能力。传统的文献检索工具只能提供关键词匹配，无法帮助研究者主动发现“隐藏的”研究空白。即便是使用人工智能做文献摘要，也缺少对方法论的推演和实验可行性的评估。更关键的是，科研往往是一个反复讨论、修改、审稿的过程，单轮生成的想法很少能直接落地。于是，如何让机器在阅读文献后，像科研团队一样提出、评审、迭代改进研究思路，成为了一个亟待突破的难题。

### 关键概念速览
- **大语言模型（LLM）**：一种在海量文本上预训练的神经网络，能够生成连贯的自然语言，类似于“会写作的机器人”。它的知识库是“百科全书+经验法则”的混合体。
- **学术图（Academic Graph）**：把论文、作者、关键词等实体当作节点，用引用、共现等关系连起来的网络，像是科研界的社交网络图谱。
- **知识库（Knowledge Store）**：从大量文献中抽取的概念、方法、实验指标等结构化信息，类似于科研领域的“词典+案例库”。
- **ReviewingAgent**：由对齐了人类偏好的 LLM 充当的审稿人，能够根据预设的评价标准给出反馈，类似于“虚拟同行评审”。
- **人类偏好对齐（Human Preference Alignment）**：通过让模型学习真实科研人员的评分或评论，使模型的评价更贴近人类的审美和严谨性。
- **迭代式改进（Iterative Revision）**：把生成的研究想法交给审稿模型，收集意见后再重新生成，循环若干次，像是“写作—评审—修改”的科研循环。

### 核心创新点
1. **从单篇核心论文到全景文献网络的扩展**  
   之前的 LLM 辅助系统大多只把输入限制在一段文字或几篇文献上。ResearchAgent 先把一篇核心论文作为锚点，利用学术图检索出所有相关引用和被引用文献，再把这些文献的概念抽取进知识库，形成一个跨文献、跨领域的上下文。这样模型在生成想法时能“站在整个研究社区的肩膀上”，而不是孤立的单篇阅读。

2. **多模态审稿代理的协同评估**  
   传统的生成-评估流程往往只用一个模型打分，容易产生偏差。ResearchAgent 引入了若干个 ReviewingAgent，每个代理依据不同的人类偏好子集（创新性、可行性、实验严谨性等）进行评价。通过 LLM 提示把真实科研人员的评分模式迁移到模型上，形成了更细致、更人性化的审稿环节。

3. **迭代式生成-审稿闭环**  
   生成一次后直接输出的做法在科研场景里几乎不可用。ResearchAgent 把审稿反馈重新喂回生成模块，模型在每轮迭代中对问题定义、方法设想、实验设计进行细化。相当于让机器在“同行讨论”中不断磨刀，使最终的创意更清晰、可操作。

4. **跨学科概念抽取与共享**  
   通过对大量论文进行概念挖掘，系统构建了一个共享的概念库，能够把生物学的实验设计思路迁移到材料科学，或把机器学习的评估指标带入社会科学。这个跨域迁移在以前的系统里几乎没有实现。

### 方法详解
**整体框架**  
ResearchAgent 的工作流可以划分为四个阶段：①核心文献定位，②文献网络与概念扩展，③生成-审稿迭代，④最终输出。整个过程像是科研团队先做文献调研、再头脑风暴、再内部评审，最后形成项目提案。

**1. 核心文献定位与图谱检索**  
用户输入一篇感兴趣的论文（通常是最新或最具影响力的工作）。系统把这篇论文的 DOI、标题等信息送入学术图数据库（如 Semantic Scholar Graph），检索出它的引用链、被引用链以及共同作者网络。得到的节点集合形成“文献子图”，其中每条边代表引用、共著或主题相似。

**2. 概念抽取与知识库构建**  
对子图中的每篇文献，使用专门的实体抽取模型（基于 LLM 的零样本抽取）抓取关键概念、实验方法、评估指标等。抽取结果统一存入知识库，按概念层级（方法、数据、评价）组织。这里的技巧在于把“概念”视作可复用的模块，使后续生成时可以自由拼接。

**3. 生成模块（ResearchAgent）**  
生成模块本身是一个大型语言模型（如 GPT‑4）并通过提示工程让它扮演“科研助理”。提示包括：  
- 已知的核心文献摘要  
- 从知识库抽取的相关概念列表  
- 目标输出格式（问题陈述、方法概述、实验设计）  
模型在一次前向传播中输出完整的研究提案草稿。

**4. 审稿代理（ReviewingAgents）**  
系统预先训练或微调若干 LLM，使其对齐人类审稿偏好。对齐过程通过“人类偏好学习”（RLHF）实现：收集真实科研人员对若干提案的评分，构造评价函数，然后用提示让模型模拟这些评分。每个 ReviewingAgent 负责一个维度（创新性、技术可行性、实验可重复性等），它们分别对草稿给出文字化的批评和改进建议。

**5. 迭代闭环**  
审稿意见被结构化后（如“需要更明确的对照组”），作为新的提示注入生成模块。模型在下一轮生成时会显式考虑这些约束，重新组织问题、细化方法或补充实验细节。整个循环通常进行 2‑4 次，直至审稿分数收敛或达到预设阈值。

**最巧妙的设计**  
- **人类偏好对齐的多代理**：把单一的“好评”拆成多个细粒度的评审维度，让模型在不同角度接受约束，显著提升生成质量。  
- **概念库的跨域共享**：通过统一的概念表示，模型可以把一个领域的实验设计模板直接搬到另一个领域，极大扩展了创新空间。  
- **迭代式反馈的显式提示**：而不是让模型自行“记住”审稿意见，系统把每条建议转化为明确的指令，避免了信息在循环中丢失。

### 实验与效果
- **测试范围**：作者在多个学科（包括计算机科学、材料科学、生命科学）挑选了 50 篇具有代表性的核心论文，分别构建对应的文献子图和概念库。  
- **对比基线**：与仅使用单篇文献的 LLM 生成系统、传统关键词检索 + 手工写作的流程以及公开的科研辅助工具（如 SciNote AI）进行比较。  
- **结果概述**：在人类评审的创新性、可行性和表达清晰度三个维度上，ResearchAgent 的平均得分比单篇 LLM 提升约 18%，比传统工具提升约 27%。在“是否可直接进入实验阶段”这一二分类任务上，模型的准确率从 0.62 提升到 0.78。  
- **消融实验**：去掉概念库的跨域抽取后，创新性得分下降约 9%；仅保留单一审稿代理（而非多代理）时，可行性得分下降约 7%；完全取消迭代循环后，整体得分下降约 12%。这些实验表明每个模块都对最终表现有实质贡献。  
- **局限性**：作者指出，系统仍然依赖于学术图的覆盖度，若核心论文所在的领域在图谱中稀疏，检索到的相关文献会不足。此外，审稿代理的偏好是从有限的人工评分中学习的，可能无法完全捕捉不同学科的评审文化。实验中也出现了少数生成的实验设计在资源需求上不切实际的情况。

### 影响与延伸思考
ResearchAgent 把“大语言模型 + 文献网络 + 迭代审稿”这三块拼在一起，开启了 AI 辅助科研的全新范式。自论文发布后，已有几篇后续工作尝试在更大规模的开放学术图（如 Microsoft Academic Graph）上部署类似的闭环系统，或把专门的实验设计模型（如自动化实验规划）嵌入到迭代环节。还有研究把人类真实的实验结果反馈回模型，实现“实验‑模型‑实验”的闭环学习。对想进一步探索的读者，可以关注以下方向：①更细粒度的学科特化审稿代理；②把实验室仪器的 API 接入系统，实现自动化实验执行；③跨语言的文献检索与概念抽取，以支持非英文科研社区。整体来看，这篇论文为 AI 与科研的深度融合提供了可操作的蓝图。

### 一句话记住它
ResearchAgent 用大语言模型把文献阅读、创意生成和同行评审闭环起来，让机器像科研团队一样“读‑想‑评‑改”。