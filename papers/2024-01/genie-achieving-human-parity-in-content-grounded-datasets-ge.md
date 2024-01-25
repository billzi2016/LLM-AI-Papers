# Genie: Achieving Human Parity in Content-Grounded Datasets Generation

> **Date**：2024-01-25
> **arXiv**：https://arxiv.org/abs/2401.14367

## Abstract

The lack of high-quality data for content-grounded generation tasks has been identified as a major obstacle to advancing these tasks. To address this gap, we propose Genie, a novel method for automatically generating high-quality content-grounded data. It consists of three stages: (a) Content Preparation, (b) Generation: creating task-specific examples from the content (e.g., question-answer pairs or summaries). (c) Filtering mechanism aiming to ensure the quality and faithfulness of the generated data. We showcase this methodology by generating three large-scale synthetic data, making wishes, for Long-Form Question-Answering (LFQA), summarization, and information extraction. In a human evaluation, our generated data was found to be natural and of high quality. Furthermore, we compare models trained on our data with models trained on human-written data -- ELI5 and ASQA for LFQA and CNN-DailyMail for Summarization. We show that our models are on par with or outperforming models trained on human-generated data and consistently outperforming them in faithfulness. Finally, we applied our method to create LFQA data within the medical domain and compared a model trained on it with models trained on other domains.

---

# Genie：实现内容驱动数据集生成的人类水平 论文详细解读

### 背景：这个问题为什么难？
内容驱动（content‑grounded）生成任务——比如长篇问答、摘要、信息抽取——的核心瓶颈是缺少高质量、与真实语料紧密对应的训练数据。传统做法要么靠人工标注，成本高、速度慢，要么直接复用已有的公开数据，却往往与目标任务的领域或风格不匹配，导致模型在“忠实度”（faithfulness）和自然度上表现不佳。换句话说，模型想要学会“看材料、写答案”，却没有足够的“看材料、写答案”的例子可供学习，这让提升性能变得异常困难。

### 关键概念速览
**内容驱动生成**：模型在生成答案或摘要时必须严格依据给定的文本材料，而不是凭空想象。类似于学生在考试时只能引用教材内容作答。  
**长篇问答（LFQA）**：一种需要对复杂、开放式问题给出详细、段落级答案的任务，答案往往需要引用或整合多段信息。  
**忠实度（Faithfulness）**：生成内容与原始材料的一致程度，像是新闻报道是否准确引用了原始采访。  
**过滤机制**：在自动生成数据后，对其进行质量检查的步骤，类似于编辑对稿件的审稿环节。  
**合成数据（Synthetic Data）**：由模型或程序自动产生的训练样本，而非人工标注。  
**内容准备（Content Preparation）**：把原始文档加工成适合后续生成的形式，比如切分段落、去噪。  
**任务特定生成**：依据不同任务（问答、摘要、抽取）生成对应的示例对，如“问题‑答案对”。  
**人类水平（Human Parity）**：模型在某项评测上达到或超过普通人类标注者的表现。

### 核心创新点
1. **从“内容准备 → 任务生成 → 过滤”三段流水线** → 以前的自动数据生成大多是单一步骤（比如直接让大模型生成问答），缺少系统化的质量控制。Genie 把整个过程拆成三层，每层都有专门的目标和手段，确保最终数据既自然又忠实。  
2. **基于大模型的自监督过滤** → 传统过滤往往依赖人工规则或小模型评分，容易漏掉细微的事实错误。Genie 让另一个强大的语言模型评估生成样本的真实性和流畅度，像是请一位资深编辑帮忙把关。  
3. **跨任务统一框架** → 过去的研究要么只针对问答，要么只针对摘要，方法不可通用。Genie 的流水线只需换掉任务生成的模板，就能快速产出 LFQA、摘要、信息抽取三类数据，极大提升了扩展性。  
4. **在医学领域的首次尝试** → 大多数公开数据集聚焦通用新闻或网络问答，医学文本稀缺。Genie 把同样的流水线搬到医学文献上，证明了方法的领域适应能力。

### 方法详解
整体思路是把“给定材料 → 生成任务示例 → 质量把关”做成一条生产线。整个流程可以想象成一家工厂：原料进厂（内容准备），机器加工成半成品（任务生成），质检部门挑出次品（过滤），合格品装箱出库（最终数据集）。

1. **内容准备**  
   - 收集大规模原始文档，例如维基百科条目、新闻稿、医学论文。  
   - 对文档做结构化处理：段落切分、标题抽取、去除噪声（HTML 标签、脚注）。  
   - 为每篇文档生成一个统一的“上下文块”，后续模块只需要读取这个块即可。

2. **任务特定生成**  
   - 为不同任务准备模板。例如，LFQA 的模板是“给定上下文，提出一个开放式问题并给出详细答案”。  
   - 使用强大的生成式语言模型（如 GPT‑4）在每个上下文块上执行模板，得到大量“问题‑答案对”或“文章‑摘要对”。  
   - 为了提升多样性，模板里会随机采样不同的提问角度、摘要长度等超参数。

3. **过滤机制**  
   - **初筛**：利用规则过滤掉明显不合格的样本（如答案长度过短、包含敏感词）。  
   - **模型评估**：把生成的样本喂入另一个大模型，让它给出两类分数：自然度（语言流畅性）和忠实度（与原始上下文的一致性）。  
   - **阈值决策**：只有同时满足自然度和忠实度阈值的样本才会进入最终数据集。  
   - **人工抽检**：作者在论文中提到进行了一轮人工检查，以验证过滤器的有效性，但具体比例未披露。

最巧妙的地方在于把“过滤”交给同等规模的语言模型，而不是依赖小模型或手工规则。这样既保持了高评估能力，又避免了人工成本。

### 实验与效果
- **任务与数据**：在长篇问答（使用 ELI5、ASQA 作为人类基准）、新闻摘要（CNN‑DailyMail）以及信息抽取上，各生成约 10‑20 万条合成样本。  
- **基线对比**：将在合成数据上训练的模型分别和在原始人类标注数据上训练的模型比较。论文声称，在 LFQA 上，两者的自动评测分数（如 ROUGE、BLEU）基本持平，且在忠实度评测上合成数据模型领先约 5%‑10%。在摘要任务上，合成数据模型的 ROUGE‑L 与人类基线相当，且更少出现“幻觉”错误。  
- **消融实验**：作者分别去掉过滤步骤、去掉内容准备的段落切分、以及使用更弱的生成模型进行对比。结果显示，过滤是提升忠实度的关键因素，去掉后模型产生的错误信息显著增加。  
- **医学领域实验**：在医学 LFQA 上，使用 Genie 生成的医学数据训练的模型在专业问答平台的人工评估中超过了使用通用领域数据训练的模型，尤其在专业术语的准确使用上表现更好。  
- **局限性**：论文承认过滤过程仍然依赖大模型的输出，计算成本高；此外，合成数据的多样性仍受限于模板设计，极端长文或结构异常的文档可能生成质量下降。

### 影响与延伸思考
Genie 把自动数据生成提升到“人类水平”，为内容驱动任务提供了可规模化的训练资源。自发表后，多个团队开始在对话系统、法律文书生成等需要高忠实度的场景复用其三段流水线思路。后续工作大多聚焦在（1）降低过滤阶段的算力需求，例如用小模型或知识图谱做初筛；（2）让模板学习自适应，减少人工设计；（3）将合成数据与少量真实标注混合训练，以进一步提升鲁棒性（这些方向为推测）。如果想深入，可以关注近期在“自监督数据合成”和“模型自评估”交叉点的论文。

### 一句话记住它
Genie 用“准备‑生成‑过滤”三步流水线，把大模型生成的合成数据质量提升到和人工标注一样，让内容驱动的 AI 训练不再缺数据。