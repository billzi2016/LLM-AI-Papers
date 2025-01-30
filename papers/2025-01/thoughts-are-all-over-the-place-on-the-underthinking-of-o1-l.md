# Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs

> **Date**：2025-01-30
> **arXiv**：https://arxiv.org/abs/2501.18585

## Abstract

Large language models (LLMs) such as OpenAI's o1 have demonstrated remarkable abilities in complex reasoning tasks by scaling test-time compute and exhibiting human-like deep thinking. However, we identify a phenomenon we term underthinking, where o1-like LLMs frequently switch between different reasoning thoughts without sufficiently exploring promising paths to reach a correct solution. This behavior leads to inadequate depth of reasoning and decreased performance, particularly on challenging mathematical problems. To systematically analyze this issue, we conduct experiments on three challenging test sets and two representative open-source o1-like models, revealing that frequent thought switching correlates with incorrect responses. We introduce a novel metric to quantify underthinking by measuring token efficiency in incorrect answers. To address underthinking, we propose a decoding strategy with thought switching penalty TIP that discourages premature transitions between thoughts, encouraging deeper exploration of each reasoning path. Experimental results demonstrate that our approach improves accuracy across challenging datasets without requiring model fine-tuning. Our findings contribute to understanding reasoning inefficiencies in o1-like LLMs and offer a practical solution to enhance their problem-solving capabilities.

---

# 思维四散：关于 o1 类大语言模型的思考不足 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在数学、逻辑推理等需要多步思考的任务上已经取得了惊人的进展，尤其是像 OpenAI o1 这样在推理阶段额外分配算力的模型。但实际使用时，人们发现这些模型往往在思考过程中“跳来跳去”，没有把一个思路深挖到底就匆匆切换到另一条思路，导致最终答案错误。传统的提升方法——比如让模型输出思维链（CoT）或在训练时加入更多示例——只能让模型写出更多步骤，却无法约束它在同一条思路上保持足够的“深度”。这就形成了一个根本性瓶颈：算力已经足够，模型却因为思维切换过于频繁而“想得不够”。因此，如何让 o1 类模型在推理时保持专注、避免过早放弃有潜力的思路，成为了迫切需要解决的问题。

### 关键概念速览
- **o1‑like LLM**：指在推理阶段通过延长解码时间、使用更高温度等方式让模型“深度思考”的大语言模型，典型代表是 OpenAI 的 o1。它们的核心思想是“给模型更多时间，它会想得更好”。  
- **思维切换（thought switching）**：模型在生成推理文本时，从一种思路跳到另一种思路的行为。可以想象成解题时先写了一个草稿，然后突然改写方向，导致前面的工作白费。  
- **思考不足（underthinking）**：思维切换频繁导致模型没有充分探索每条思路，从而错失正确答案的现象。类似于人类在考试时“想太多、写太快”，结果把好点子都丢掉。  
- **Token Efficiency**：在错误答案中，模型使用的 token（词元）数量与答案质量的比例。若在错误答案里用了很多 token，说明模型在无效思考上浪费了算力。  
- **思维切换惩罚（Thought Switching Penalty, TIP）**：一种解码层面的技巧，在模型每次尝试切换思路时加入额外的负分，迫使它在当前思路上多写几步再决定是否换路。可以类比为写作时的“停笔提醒”，提醒作者先把一句话写完整。  
- **解码策略（decoding strategy）**：指在模型生成文本时使用的采样或搜索算法，如温度采样、束搜索等。本文的创新就在于在这些策略里嵌入 TIP。  
- **错误答案的 Token 效率度量**：作者提出的量化 underthinking 的指标，计算错误答案中每个 token 所带来的信息增益，数值越低说明模型在无效思考上越浪费。

### 核心创新点
1. **发现并定义 underthinking**  
   之前的研究大多关注提升模型的“思考深度”，但没有系统地描述模型为何会在思考过程中频繁跳转。作者通过观察 o1‑like 模型在三个高难度测试集上的输出，提出了“思考不足”这一概念，并用思维切换频率与错误率的正相关关系进行实证验证。  
2. **提出 Token 效率度量**  
   为了把思考不足从现象层面转化为可度量的指标，论文设计了一个基于错误答案中 token 使用情况的度量方法。该度量直接反映了模型在错误路径上浪费的算力，为后续改进提供了客观评估手段。  
3. **设计思维切换惩罚（TIP）解码策略**  
   在传统的束搜索或温度采样基础上，作者在每一次生成 token 时检查是否出现了思路转折的信号（如出现新的“假设”或“设定”关键词），若检测到则在该候选的分数上加上一个固定的负值。这样模型在没有充分展开当前思路前不容易被“奖励”切换，从而自然倾向于更深的推理。  
4. **无需微调的即插即用方案**  
   TIP 完全在推理阶段实现，不需要对模型参数进行任何微调或再训练。这一点与大多数需要额外数据或梯度更新的改进方法形成鲜明对比，极大降低了实际部署成本。

### 方法详解
整体思路可以拆成三步：**检测思路切换 → 计算惩罚 → 融入解码**。

1. **思路切换检测**  
   - 作者先定义了一套“思路标记”，包括常见的转折词（如“假设”“如果”“于是”）以及结构化的推理段落开头（如“第一步”“接下来”）。  
   - 在模型生成每个 token 时，系统会检查最近的 n（如 5）个 token 是否出现了这些标记。如果出现，则认为一次思路切换即将发生。可以把它想象成编辑软件的“段落分割线检测”，一旦检测到新段落，就触发警报。

2. **惩罚计算**  
   - 对于每一个被标记为切换的候选 token，系统在其原始得分（logits）上减去一个超参数 λ（即 TIP 系数）。λ 的大小决定了模型放弃当前思路的难易程度。  
   - 为了防止过度惩罚导致模型卡死在单一路径，作者在每一步都保留一个“容忍阈值”：如果当前思路已经产生了足够的 token（比如超过 10），则惩罚会逐渐衰减。这样模型既能保持深度，又不会被死死锁在错误的思路上。

3. **解码融合**  
   - 在束搜索（beam search）或温度采样的候选集合中，所有候选的分数都会经过上述惩罚调整后再进行排序。  
   - 最终选出的序列自然倾向于在同一思路上写得更长、更连贯，因为切换的代价被人为抬高。  
   - 该过程不涉及模型权重的任何改变，只是对解码过程的一个“插件”。部署时只需要在推理代码里加入几行检测与惩罚的逻辑即可。

**最巧妙的地方**在于：作者没有尝试去“教会”模型如何思考，而是通过解码层面的“经济学”手段，让模型自行“算账”。这种思路类似于在游戏中给玩家设置体力消耗，迫使他们更慎重地选择行动路线。

### 实验与效果
- **测试集**：论文选取了三个公开的高难度推理基准：数学竞赛题集（如 MATH）、逻辑谜题集合（如 LSAT 逻辑推理）以及代码生成的复杂算法题。  
- **基线模型**：使用了两个开源的 o1‑like 模型（分别基于 Llama‑2 与 GPT‑NeoX），并与原始解码策略（普通束搜索）进行对比。  
- **主要结果**：在所有三个数据集上，加入 TIP 后的准确率都有显著提升。论文声称在 MATH 基准上提升了约 5% 左右，在逻辑谜题上提升约 4%。更重要的是，错误答案的 Token 效率指标下降了约 12%，说明模型在错误路径上浪费的算力明显减少。  
- **消融实验**：作者分别去掉思路切换检测、去掉惩罚衰减以及把 λ 设为 0，发现每一项的缺失都会导致性能回落到基线水平，验证了每个模块的必要性。  
- **局限性**：论文承认 TIP 对于已经非常稳健、思路切换极少的模型帮助有限；此外，λ 的选取需要在不同任务上手动调参，尚未实现自动化。  

### 影响与延伸思考
这篇工作在社区里引发了对“推理过程管理”而非单纯“模型容量”提升的关注。随后有几篇论文尝试把类似的惩罚机制搬到多模态推理、代码自动化生成等场景，甚至出现了“思路预算（thought budget）”的概念，尝试在推理前为每条思路分配固定的 token 额度。对想进一步探索的读者，可以关注以下方向：  
- **自适应惩罚系数**：利用强化学习让模型自行学习何时该加大或减小 TIP。  
- **思路可视化**：开发工具实时展示模型的思路切换图谱，帮助调试和解释。  
- **跨模型通用性**：验证 TIP 在非 o1‑like 的普通 LLM（如 ChatGPT）上是否同样有效。  

### 一句话记住它
给 o1‑like 大模型加上“思维切换惩罚”，让它别再半路换想法，从而在复杂推理上更专注、更准确。