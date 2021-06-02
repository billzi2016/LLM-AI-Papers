# DynaEval: Unifying Turn and Dialogue Level Evaluation

> **Date**：2021-06-02
> **arXiv**：https://arxiv.org/abs/2106.01112

## Abstract

A dialogue is essentially a multi-turn interaction among interlocutors. Effective evaluation metrics should reflect the dynamics of such interaction. Existing automatic metrics are focused very much on the turn-level quality, while ignoring such dynamics. To this end, we propose DynaEval, a unified automatic evaluation framework which is not only capable of performing turn-level evaluation, but also holistically considers the quality of the entire dialogue. In DynaEval, the graph convolutional network (GCN) is adopted to model a dialogue in totality, where the graph nodes denote each individual utterance and the edges represent the dependency between pairs of utterances. A contrastive loss is then applied to distinguish well-formed dialogues from carefully constructed negative samples. Experiments show that DynaEval significantly outperforms the state-of-the-art dialogue coherence model, and correlates strongly with human judgements across multiple dialogue evaluation aspects at both turn and dialogue level.

---

# DynaEval：统一回合级与对话级评估 论文详细解读

### 背景：这个问题为什么难？

对话系统的好坏往往体现在多轮交互的整体流畅度上，但过去的大多数自动评估指标只看单句（回合）是否合适，忽视了前后句之间的衔接和整体走向。于是出现了两类问题：一是单句评分高的对话，整体可能显得跳脱或前后矛盾；二是现有的对话级评估往往基于手工特征或粗糙的相似度，难以捕捉细粒度的依赖关系。根本原因在于缺少一种能够同时“看见”每一句，又“感知”整段对话结构的模型。

### 关键概念速览
- **回合级评估**：对每一句用户或系统的回复进行独立打分，类似于给每道作文的句子打分。  
- **对话级评估**：把整段对话当作一个整体来打分，关注整体连贯性、主题一致性等。  
- **图卷积网络（GCN）**：一种在图结构上进行信息传播的神经网络，像在社交网络里让每个人把自己的观点传给邻居，再综合得到全局看法。  
- **对话图**：把对话中的每句话当作图的节点，节点之间的连边表示两句话之间的依赖（如前后顺序、引用关系）。  
- **对比学习**：让模型学会把“好”样本和“坏”样本拉开距离，类似于训练人辨别真伪图片。  
- **负样本构造**：人为制造不合理的对话（比如打乱顺序、插入无关句子），让模型学会识别这些错误。  
- **一致性相关系数**：衡量模型评分与人工评分之间的吻合程度，数值越高说明模型越靠谱。  

### 核心创新点
1. **统一的图结构 → 用图卷积网络一次性建模回合与整体**  
   过去的评估要么只看单句，要么用独立的整体特征。DynaEval 把每句话当作图节点，边连接所有可能的依赖关系，然后用 GCN 同时更新每个节点的表示和全局图的表示。这样既保留了细粒度的回合信息，又得到对话级的整体向量。  
   *改变*：实现了“一图两用”，评估结果在回合层面和对话层面上都能同步提升。

2. **对比学习驱动的负样本策略 → 用人为破坏的对话教模型辨别好坏**  
   传统方法往往只用正样本（真实对话）进行监督，模型容易学到表面的语言流畅度。作者专门设计了几种负样本：随机打乱句序、插入无关回复、替换关键句子等。通过对比损失，模型被迫学习到真正的对话连贯性。  
   *改变*：模型对不连贯、主题漂移等错误更敏感，评估更贴近人类直觉。

3. **统一评分机制 → 同时输出回合级分数和对话级分数**  
   在图卷积的最后，节点的向量直接映射为回合级得分，图的全局向量映射为对话级得分。这样不需要额外的回合评估模型，省去繁琐的两套系统。  
   *改变*：评估流程更简洁，且两层评分天然保持一致性。

### 方法详解
**整体框架**  
DynaEval 的流程可以概括为四步：① 将对话拆成句子并构建图；② 用图卷积网络在图上进行信息传播；③ 通过对比学习让模型区分真实对话和负样本；④ 从节点和全局向量分别得到回合级和对话级评分。

**1. 对话图构建**  
- 每一句话（包括用户和系统的发言）被编码成初始向量，常用预训练语言模型（如 BERT）得到。  
- 边的设计上，作者默认每句话与前一句相连，形成链式结构；此外，还会加入跨句边来捕捉引用或主题复现（比如系统回答前面用户提到的实体）。这让图既保留顺序，又能表达更灵活的依赖。

**2. 图卷积网络（GCN）**  
- GCN 的核心是“邻居聚合”：每个节点把自己和相邻节点的向量加权求和，再经过非线性变换，得到新的表示。  
- 经过若干层（通常 2~3 层）后，节点的向量已经融合了上下文信息，整个图的全局向量（可以通过对所有节点做平均或加权池化得到）则代表了对话的整体语义状态。

**3. 对比学习与负样本**  
- 正样本是原始对话图。负样本通过三种方式生成：  
  a. **顺序打乱**：随机重排句子顺序，破坏自然的对话流。  
  b. **无关插入**：在对话中插入与主题无关的句子。  
  c. **关键句替换**：用语义相似度低的句子替换关键回复。  
- 对比损失的目标是让正样本的全局向量与正标签（高分）更接近，而负样本的向量与负标签（低分）拉开距离。这样模型在训练时被迫学习到“连贯性”而不是单纯的语言流畅度。

**4. 统一评分**  
- **回合级**：每个节点的最终向量经过一个小的全连接层，输出一个标量分数，代表该句的质量。  
- **对话级**：全局向量同样经过一个全连接层，得到整体评分。因为全局向量是所有节点信息的汇总，两个评分天然共享底层特征。

**巧妙之处**  
- 负样本的构造不是随意的，而是围绕对话连贯性设计，直接把模型的学习目标对准人类评判的核心痛点。  
- 使用同一张图同时产生两层评分，避免了回合评估和对话评估之间的“信息孤岛”，提升了评估的一致性。

### 实验与效果
- **数据集**：作者在多个公开对话评估基准上验证，包括开放域聊天数据和任务导向对话数据（如 Persona‑Chat、DailyDialog 等）。  
- **对比基线**：与最先进的对话连贯性模型（如 Coherence‑BERT）以及传统回合级指标（BLEU、ROUGE）进行比较。  
- **结果**：实验显示 DynaEval 在与人类评分的相关系数上显著高于所有基线，尤其在对话整体连贯性和主题一致性两项上提升最为明显。  
- **消融研究**：去掉负样本或仅使用链式边（不加跨句边）都会导致相关系数下降，说明负样本和丰富的图结构是性能提升的关键因素。  
- **局限**：论文指出模型对极长对话（超过 30 轮）仍会出现信息稀释，GCN 的层数受限于计算成本；此外，负样本的构造仍依赖手工规则，自动化程度有待提升。

### 影响与延伸思考
DynaEval 把图结构引入对话评估后，后续不少工作开始探索更复杂的图神经网络（如图注意力网络）来捕捉更细致的依赖关系。还有研究尝试把对比学习与强化学习结合，让评估模型直接参与对话生成的奖励信号。对话系统的训练循环中加入 DynaEval 这类统一评估器，被认为是提升生成质量的潜在方向。想进一步了解，可以关注以下方向：  
- **动态图建模**：对话进行时动态更新图结构。  
- **自监督负样本生成**：让模型自己学习生成高质量的负样本。  
- **跨模态对话评估**：把视觉或声音信息也映射进同一图中，评估多模态对话的连贯性。

### 一句话记住它
DynaEval 用图卷积和对比学习把“每句话”和“整段对话”绑在同一张图上，实现了统一、连贯的自动评估。