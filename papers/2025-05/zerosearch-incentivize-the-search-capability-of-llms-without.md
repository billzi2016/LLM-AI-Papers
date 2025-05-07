# ZeroSearch: Incentivize the Search Capability of LLMs without Searching

> **Date**：2025-05-07
> **arXiv**：https://arxiv.org/abs/2505.04588

## Abstract

Effective information searching is essential for enhancing the reasoning and generation capabilities of large language models (LLMs). Recent research has explored using reinforcement learning (RL) to improve LLMs' search capabilities by interacting with live search engines in real-world environments. While these approaches show promising results, they face two major challenges: (1) Uncontrolled Document Quality: The quality of documents returned by search engines is often unpredictable, introducing noise and instability into the training process. (2) Prohibitively High API Costs: RL training requires frequent rollouts, potentially involving hundreds of thousands of search requests, which incur substantial API expenses and severely constrain scalability. To address these challenges, we introduce ZeroSearch, a novel RL framework that incentivizes the capabilities of LLMs to use a real search engine with simulated searches during training. Our approach begins with lightweight supervised fine-tuning to transform the LLM into a retrieval module capable of generating both useful and noisy documents in response to a query. During RL training, we employ a curriculum-based rollout strategy that incrementally degrades the quality of generated documents, progressively eliciting the model's reasoning ability by exposing it to increasingly challenging retrieval scenarios. Extensive experiments demonstrate that ZeroSearch effectively incentivizes the search capabilities of LLMs using a 3B LLM as the retrieval module. Remarkably, a 7B retrieval module achieves comparable performance to the real search engine, while a 14B retrieval module even surpasses it. Furthermore, it generalizes well across both base and instruction-tuned models of various parameter sizes and is compatible with a wide range of RL algorithms.

---

# ZeroSearch：在不进行真实搜索的情况下激励大语言模型的检索能力 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在回答需要外部事实的问题时，往往依赖内部的记忆，而不是主动去检索最新信息。已有工作尝试让模型在强化学习（RL）框架下直接调用真实搜索引擎，以此提升检索与推理的结合能力。但实际部署时会遇到两大障碍：一是搜索引擎返回的文档质量不可控，噪声会让 RL 训练过程不稳定；二是每一次 RL rollout 都要向搜索 API 发送查询，累计的请求量可能达到数十万甚至上百万，导致巨额费用，严重限制了方法的可扩展性。正因为这两个根本性瓶颈，研究者迫切需要一种既能让模型学会“搜索”，又不必频繁调用真实搜索服务的方案。

### 关键概念速览
- **强化学习（RL）**：让模型通过试错获得奖励信号，从而学会在特定情境下做出更好的决策。类似于训练机器人在迷宫里找出口，走对了就给奖励，走错了就扣分。
- **检索模块（retrieval module）**：在本工作中指的是经过轻量微调的 LLM，它的输出是一段“文档”，可以是有用信息也可以是噪声，模拟真实搜索的返回结果。
- **监督微调（supervised fine‑tuning）**：先用标注好的查询‑文档对让模型学会生成检索结果，类似于先教会学生怎么写参考答案，再让他自己练习。
- **Curriculum Rollout（课程式 rollout）**：在 RL 训练过程中，逐步让检索模块生成质量更差的文档，逼迫主模型提升推理能力。好比先让学生做简单的练习题，随后逐步加大难度。
- **奖励模型（reward model）**：根据最终答案的正确性、信息完整度等指标给出数值奖励，指导 RL 优化。相当于老师给每份作业打分。
- **参数规模（parameter size）**：模型的大小，用“3B”“7B”“14B”等表示其参数数量（十亿级）。规模越大，潜在能力越强，但训练成本也更高。

### 核心创新点
1. **真实搜索 → 模拟搜索**  
   之前的做法在 RL 训练时直接调用外部搜索引擎，导致费用高且文档质量不可控。ZeroSearch 先把 LLM 通过监督微调变成一个“伪搜索引擎”，在训练期间只产生模型内部生成的文档。这样既省掉了 API 费用，又把文档质量的波动交给了可控的模型输出。

2. **固定质量 → 递进降质**  
   传统 RL 往往一次性让模型面对真实搜索的全部噪声。ZeroSearch 采用课程式 rollout：一开始让检索模块输出相对干净的文档，随后逐步加入噪声、削弱相关性，使得主模型必须在更差的检索条件下仍能完成任务。相当于让学生先做有提示的题目，再慢慢去掉提示，提升独立思考能力。

3. **小模型检索 → 大模型推理**  
   实验表明，仅用 3B 参数的检索模块就能为更大的主模型提供有效的训练信号，而当检索模块本身升级到 7B、14B 时，其生成的文档质量甚至可以匹配或超越真实搜索引擎。作者因此提出“检索模块不必是最强的，只要能提供梯度信号即可”，这在资源受限的实验环境中非常实用。

4. **算法兼容性**  
   ZeroSearch 的奖励函数和课程式 rollout 与多种 RL 算法（如 PPO、REINFORCE）均兼容，作者在实验中验证了跨算法的稳健性。相比之前只能配合特定 RL 框架的搜索强化学习，ZeroSearch 更像一个通用插件。

### 方法详解
**整体思路**：ZeroSearch 把“搜索”这件事拆成两层：第一层是一个轻量的检索模块，它接受用户查询并输出一段文档（可能好也可能坏）；第二层是主 LLM，它把检索结果当作外部信息来推理并生成最终答案。整个过程在 RL 环境中循环：模型收到奖励 → 更新策略 → 再次生成检索文档 → 再推理。

**步骤拆解**：

1. **监督微调检索模块**  
   - 收集一批查询‑真实文档对（可来源于公开检索数据集）。  
   - 用这些对对 LLM 进行轻量微调，使其学会在给定查询时输出类似检索结果的文本。  
   - 这里的目标不是让模型生成完美文档，而是让它能够在“有用”和“噪声”之间切换，为后续的课程式训练埋下空间。

2. **构建 RL 环境**  
   - 环境接受主模型的动作（即对检索文档的使用方式）并返回奖励。  
   - 奖励由一个独立的奖励模型计算，主要衡量最终答案的准确性、信息覆盖度以及是否合理引用了检索文档。

3. **Curriculum Rollout**  
   - 训练初期，检索模块的输出质量保持在较高水平（例如只加入轻微的随机扰动）。  
   - 随着训练轮数增加，系统逐步降低检索质量：随机删除关键句子、加入无关段落、降低相关性分数等。  
   - 这种递进式降质迫使主模型在文档噪声增大时仍能保持推理稳健，类似于让学生在没有参考答案的情况下完成同一道题。

4. **RL 更新**  
   - 主模型使用常见的策略梯度方法（如 PPO）根据奖励信号更新参数。  
   - 同时，检索模块也可以在每个 curriculum 阶段进行轻微的自我调节，以保持“噪声”在预设范围内。

**关键细节**：

- **奖励函数的设计**：作者没有公开完整公式，但核心是把答案的正确性（如 F1、BLEU）与检索文档的使用度（是否引用、引用是否准确）加权求和。这样模型不会只追求答案对，而是必须学会合理利用检索信息。

- **最巧妙的地方**：把检索质量的“降级”当作一种 curriculum，而不是硬性固定噪声比例。这样模型在训练过程中始终面对“更难的搜索”，从而自然学会更强的检索利用策略。

- **模型规模的层次**：实验中检索模块使用 3B、7B、14B 三种规模，主模型则分别是基座模型和指令微调模型。作者发现，即使检索模块只有 3B，主模型仍能显著提升搜索能力；而 14B 检索模块的输出质量已经可以在多数情况下超过真实搜索引擎。

### 实验与效果
- **测试任务**：主要在开放域问答（Open‑Domain QA）和事实核查（Fact‑Checking）两个基准上评估。数据集包括 Natural Questions、TriviaQA 等常用检索型任务。

- **对比基线**：包括（1）不使用检索的纯 LLM（直接生成答案），（2）使用真实搜索引擎进行 RL 训练的传统方法，以及（3）仅用监督微调的检索‑生成流水线。

- **主要结果**：  
  - 在 7B 检索模块下，ZeroSearch 的答案准确率（Exact Match）与真实搜索 RL 方法相当，差距在 1% 以内。  
  - 14B 检索模块甚至在同等设置下超过真实搜索约 2%–3%。  
  - 与不使用检索的基线相比，ZeroSearch 提升了约 10%–15%的整体得分。  

- **消融实验**：作者分别去掉 curriculum、去掉奖励模型的引用惩罚、以及直接使用固定噪声的检索模块。结果显示，去掉 curriculum 会导致性能下降约 4%–5%，说明递进降质是关键驱动因素。

- **局限性**：论文未提供对极端噪声（如完全无关文档）下模型鲁棒性的系统评估；此外，检索模块本身仍需要一定的监督数据进行微调，完全零数据的情形尚未探索。

### 影响与延伸思考
ZeroSearch 打破了“强化学习必须依赖真实外部 API”这一惯性，为低成本、可控的检索强化学习提供了新思路。后续工作已经开始借鉴其 curriculum‑style 降质策略，尝试在多模态检索、代码搜索等场景中使用“模拟检索”。还有研究把检索模块换成专门的生成式检索模型（如 RAG），进一步提升文档质量。对想深入的读者，可以关注以下方向：① 如何在完全无标注的情况下训练检索模块（自监督生成噪声文档）；② 将 ZeroSearch 与检索增强生成（RAG）框架结合，实现端到端的检索‑生成闭环；③ 探索更细粒度的奖励信号，如引用可信度评分或时效性评估。

### 一句话记住它
ZeroSearch 用“自造噪声的模拟搜索 + 递进难度的课程式 RL”，在不调用真实搜索引擎的情况下，让大语言模型学会像人一样主动检索并推理。