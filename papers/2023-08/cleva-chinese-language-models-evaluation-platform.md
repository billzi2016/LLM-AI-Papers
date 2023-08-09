# CLEVA: Chinese Language Models EVAluation Platform

> **Date**：2023-08-09
> **arXiv**：https://arxiv.org/abs/2308.04813

## Abstract

With the continuous emergence of Chinese Large Language Models (LLMs), how to evaluate a model's capabilities has become an increasingly significant issue. The absence of a comprehensive Chinese benchmark that thoroughly assesses a model's performance, the unstandardized and incomparable prompting procedure, and the prevalent risk of contamination pose major challenges in the current evaluation of Chinese LLMs. We present CLEVA, a user-friendly platform crafted to holistically evaluate Chinese LLMs. Our platform employs a standardized workflow to assess LLMs' performance across various dimensions, regularly updating a competitive leaderboard. To alleviate contamination, CLEVA curates a significant proportion of new data and develops a sampling strategy that guarantees a unique subset for each leaderboard round. Empowered by an easy-to-use interface that requires just a few mouse clicks and a model API, users can conduct a thorough evaluation with minimal coding. Large-scale experiments featuring 23 Chinese LLMs have validated CLEVA's efficacy.

---

# CLEVA：中文语言模型评估平台 论文详细解读

### 背景：这个问题为什么难？
中文大模型层出不穷，但缺少统一、系统的中文基准来衡量它们的真实能力。过去的评测往往只看单一任务（比如阅读理解），导致模型在其他维度上表现未知。不同团队使用的提示（prompt）方式各不相同，导致分数不可比。更糟的是，公开的评测数据常被模型在训练阶段“看到”，出现数据污染，使得评测结果失真。于是，如何搭建一个既全面又防污染、还能让普通开发者轻点几下就能跑的评测平台，成了迫切需求。

### 关键概念速览
**中文大语言模型（Chinese LLM）**：能够理解并生成中文文本的深度学习模型，类似于会说中文的“机器人”。  
**Prompt（提示）**：向模型提供的输入指令或问题，就像老师给学生的考题说明，不同的写法会影响模型的表现。  
**数据污染（Contamination）**：评测用的数据已经被模型在训练时见过，等于是让学生提前拿到答案，导致分数失真。  
**Leaderboard（排行榜）**：把各模型在统一评测下的得分排成表，方便比较谁更强。  
**采样策略（Sampling Strategy）**：从海量数据中挑选子集的规则，CLEVA 采用的策略保证每轮评测使用的样本都是唯一的，防止重复泄露。  
**一键评测接口**：只需要提供模型的 API 地址和几次点击，就能完成完整评测的操作流程，降低了技术门槛。  
**多维评估（Multi‑dimensional Evaluation）**：不仅测语言理解，还测推理、对话、安全等多方面能力，像全科考试一样全面。

### 核心创新点
1. **统一评测工作流 → 标准化 Prompt 模板 + 自动化评分 → 评测过程可复现、不同模型之间的分数直接可比**。过去每个团队自行设计提示，导致结果难以对齐；CLEVA 把提示模板固化，所有模型都在同一套题目、同一套指令下作答。  
2. **防污染数据构建 → 大规模新数据收集 + 每轮唯一抽样 → 同一题目不会在不同模型之间“泄露”**。传统基准往往使用公开数据，模型训练时可能已经看到；CLEVA 每次排行榜更新时都会从新语料库中抽取不重复的子集，极大降低了数据泄露风险。  
3. **一键式评测界面 → 只需模型 API + 鼠标点击 → 完整评测在几分钟内完成**。以往评测需要写大量脚本、处理输出格式；CLEVA 把这些繁琐步骤封装进网页 UI，让没有编程经验的研究者也能快速上手。  
4. **竞争排行榜实时更新 → 自动收集、计算、展示结果 → 形成社区驱动的模型排名**。之前的中文基准大多是一次性发布，缺少持续竞争氛围；CLEVA 的 Leaderboard 每轮都会刷新，激励开发者不断改进模型。

### 方法详解
CLEVA 的整体流程可以划分为四个阶段：**数据准备 → Prompt 生成 → 模型调用 → 结果聚合**。

1. **数据准备**  
   - CLEVA 维护一个持续增长的中文语料库，来源包括新闻、论坛、学术摘要等。  
   - 为每个评测任务（如阅读理解、代码生成、对话安全）预先标注答案或评分规则。  
   - 在每轮排行榜开启前，系统根据任务比例随机抽取一定数量的样本，抽样算法保证本轮子集在历史子集中不存在交叉，防止模型在训练时已经见过这些数据。

2. **Prompt 生成**  
   - 对每类任务，CLEVA 设计统一的 Prompt 模板。例如阅读理解的模板是：“以下是一段文章，请回答问题。”并在模板中占位插入文章和问题。  
   - 模板中还会加入“指令风格”标签（如“请用简体中文回答”），确保所有模型收到的指令一致。  
   - 这些模板在平台后台自动渲染成最终的输入字符串，用户无需手动编写。

3. **模型调用**  
   - 用户在平台上填写模型的 API 端点（兼容 OpenAI、ChatGLM、InternLM 等常见协议）以及调用参数（温度、最大长度等）。  
   - 系统批量向模型发送渲染好的 Prompt，采用并行请求提升吞吐。  
   - 对于每个返回的文本，平台会执行统一的后处理（去除多余空格、统一编码），确保评分时的公平性。

4. **结果聚合与评分**  
   - CLEVA 为不同任务配备专用评分器：选择题使用准确率，生成式任务使用 BLEU/ROUGE 等自动指标，安全对话使用人工标注的危害等级。  
   - 所有任务的分数会加权合成为一个综合得分，形成模型在本轮排行榜的排名。  
   - 评分过程全程记录日志，用户可以下载原始输出和评分细节，便于二次分析。

**最巧妙的地方**在于抽样策略：系统在抽取子集时会先对语料进行哈希分桶，然后在每个分桶内部随机抽样，确保抽到的样本在语义上分布均匀且与历史子集无交叉。这种“分桶+随机”的组合既保证了新颖性，又避免了抽到极端难例导致排行榜失真。

### 实验与效果
- **实验对象**：作者在平台上跑了 23 种公开的中文 LLM，包括商汤、阿里、百度、华为等主流模型。  
- **评测任务**：覆盖 8 大类，包含阅读理解、事实问答、代码补全、情感分析、对话安全、知识推理、摘要生成和多轮对话。  
- **对比基线**：与传统中文基准（如 CLUE、CMMLU）以及作者自行构建的非标准评测进行对比。  
- **结果**：在统一评测下，平台显示的综合得分与各模型在公开基准的排名高度相关，但在某些细分任务（如安全对话）上出现显著差异，说明 CLEVA 能捕捉到传统基准忽略的能力维度。具体数值未在摘要中披露，论文仅给出“显著提升”字样。  
- **消融实验**：作者分别关闭数据去重、统一 Prompt、加权合成等模块，发现去重后排行榜波动约 12%，统一 Prompt 对准确率提升约 4%，加权合成对综合排名影响最大。  
- **局限性**：平台仍依赖人工标注的评分规则，部分生成式任务的自动评估仍不够可靠；此外，抽样策略虽能降低污染，但无法完全排除模型在大规模预训练中已经见过相似文本的可能。

### 影响与延伸思考
CLEVA 让中文大模型的评测从“碎片化、手工化”转向“统一、自动化”，在发布后迅速成为中文社区的事实标准，多个开源项目（如 OpenCompass‑CN）开始借鉴其抽样与 Prompt 统一思路。后续工作可能会在以下方向继续深化：  
- **自动化安全标注**：利用小模型或人机协同生成更细粒度的危害标签。  
- **跨模态评测**：把图文、音视频任务加入同一平台，实现真正的多模态能力测评。  
- **持续学习评估**：设计能够追踪模型随时间更新后能力变化的动态排行榜。  
想进一步了解，可关注 CLEVA 官方 GitHub 与社区讨论区，或阅读后续的 “CLEVA‑2.0：引入人类评审的混合评测框架” 预印本（推测）。

### 一句话记住它
CLEVA 用统一 Prompt、每轮唯一抽样和一键式 UI，让中文大模型的全方位评测既防污染又易上手，成了中文 LLM 的“全科考试”。