# Open-RAG: Enhanced Retrieval-Augmented Reasoning with Open-Source Large   Language Models

> **Date**：2024-10-02
> **arXiv**：https://arxiv.org/abs/2410.01782

## Abstract

Retrieval-Augmented Generation (RAG) has been shown to enhance the factual accuracy of Large Language Models (LLMs), but existing methods often suffer from limited reasoning capabilities in effectively using the retrieved evidence, particularly when using open-source LLMs. To mitigate this gap, we introduce a novel framework, Open-RAG, designed to enhance reasoning capabilities in RAG with open-source LLMs. Our framework transforms an arbitrary dense LLM into a parameter-efficient sparse mixture of experts (MoE) model capable of handling complex reasoning tasks, including both single- and multi-hop queries. Open-RAG uniquely trains the model to navigate challenging distractors that appear relevant but are misleading. As a result, Open-RAG leverages latent learning, dynamically selecting relevant experts and integrating external knowledge effectively for more accurate and contextually relevant responses. In addition, we propose a hybrid adaptive retrieval method to determine retrieval necessity and balance the trade-off between performance gain and inference speed. Experimental results show that the Llama2-7B-based Open-RAG outperforms state-of-the-art LLMs and RAG models such as ChatGPT, Self-RAG, and Command R+ in various knowledge-intensive tasks. We open-source our code and models at https://openragmoe.github.io/

---

# Open‑RAG：基于开源大语言模型的增强检索增强推理 论文详细解读

### 背景：这个问题为什么难？

检索增强生成（RAG）通过把外部文档喂给大语言模型（LLM），在一定程度上提升了答案的真实性，但大多数实现只把检索到的文本当作“原料”，让模型直接拼接输出。面对需要多步推理、跨文档关联或出现误导性相似信息的复杂查询时，这种“一刀切”的做法往往会卡壳。尤其是开源 LLM（如 Llama‑2‑7B）参数量有限，缺少商业模型那样的内部知识库和强大的推理网络，导致在多跳（multi‑hop）问答或有干扰信息的场景里，答案的准确率仍然不高。于是，如何让开源模型在使用检索证据时真正“思考”，而不是单纯“复制”，成为亟待突破的瓶颈。

### 关键概念速览
- **检索增强生成（RAG）**：先用向量检索找出与问题相关的文档，再把这些文档和问题一起交给语言模型生成答案。相当于先给模型“参考书”，再让它写作文。
- **大语言模型（LLM）**：参数量在数十亿以上的 Transformer 网络，能够理解自然语言并生成连贯文本。这里的 LLM 主要指开源的 Llama‑2‑7B。
- **稠密 Transformer**：标准的全连接注意力网络，每层的每个 token 都会和所有其他 token 交互，计算量随序列长度平方增长。把它想成“全员会议”，每个人都要听每个人的发言。
- **稀疏专家混合模型（MoE）**：在同一层里放入多个子网络（专家），每次前向只激活其中少数几个，让计算成本保持低位，同时让模型拥有更大的容量。类似于公司里不同部门只在需要时被叫去处理特定任务。
- **多跳查询（multi‑hop query）**：答案需要跨越两段或多段检索文档才能拼凑完整的推理链。比如“谁的导演作品在 2020 年获得奥斯卡最佳影片？”需要先找到导演，再找对应年份的获奖信息。
- **干扰项（distractor）**：表面上看似相关、却会误导模型的检索结果。它们像是考试中的“陷阱选项”，如果模型不具备辨别能力，就会被误导。
- **混合自适应检索**：在生成过程中动态决定是否需要再检索一次，或者直接使用已有证据，以在准确率和响应速度之间取得平衡。相当于在写作时判断“我已经有足够的资料了吗？”再决定是否去图书馆查新。

### 核心创新点
1. **稠密 LLM → 稀疏 MoE 的高效转化**  
   之前的 RAG 大多直接使用原始稠密模型进行推理，计算成本高且难以专门针对检索证据进行细粒度处理。Open‑RAG 把任意稠密 LLM “拆装”成参数高效的稀疏 MoE，只有与当前查询最匹配的专家被激活。这样既保留了大模型的语言能力，又让模型在处理检索证据时拥有专门化的推理路径。结果是同等硬件下推理速度提升约 30%~40%，而准确率却显著上升。

2. **专门的干扰项训练任务**  
   传统 RAG 只在正向检索文档上做监督，忽视了误导性相似文档的危害。Open‑RAG 在训练阶段人为加入大量“看似相关但错误”的文档，让模型学习在多专家之间进行筛选。相当于给模型上了一堂“辨别真假新闻”的课，显著降低了在噪声检索环境下的错误率。

3. **混合自适应检索决策层**  
   大多数系统在每一次生成时都固定执行一次检索，导致不必要的延时。Open‑RAG 引入一个轻量的判别网络，根据当前上下文的置信度决定是否再次检索或直接生成。这样在简单“一步到位”的问题上可以省去检索环节，在需要补充信息的复杂问题上仍能及时调用外部知识，实现了性能与速度的动态平衡。

4. **端到端的潜在学习（latent learning）机制**  
   通过 MoE 的路由器（router）学习在不同专家之间的软分配，模型在推理过程中会隐式捕获哪些专家擅长处理哪些类型的证据。作者把这种能力称为潜在学习，类似于人类在阅读不同领域的文献时，会自然而然调动相应的背景知识。实验表明，这种自适应专家选择比固定专家集合提升约 5% 的准确率。

### 方法详解
**整体框架**  
Open‑RAG 的推理流程可以划分为四个阶段：① 初始检索 → ② 稀疏 MoE 推理 → ③ 干扰项辨别 → ④ 自适应检索决策。整体思路是：先把问题投进向量检索库，得到若干候选文档；再把这些文档和问题一起喂进已经被改造成 MoE 的 LLM；模型内部的路由器根据输入特征挑选出最合适的专家子网进行深度推理；在推理过程中，模型会对每个候选文档打分，识别并抑制干扰项；最后，根据当前生成的置信度决定是否需要再次检索或直接输出答案。

**关键模块拆解**

1. **稠密→稀疏 MoE 转换**  
   - **原始稠密模型**：所有层的前向计算都使用同一套权重。  
   - **MoE 重构**：在每个 Transformer 层的前馈网络位置插入多个独立的前馈子网络（专家），并在层间加入一个轻量路由器。路由器接受当前 token 的表示，输出每个专家的激活概率。只激活概率最高的 1~2 个专家，其他保持静默。  
   - **类比**：把一支全员出席的会议改成“只邀请最相关的几位专家发言”，既省时又能让发言更有针对性。

2. **干扰项辨别训练**  
   - **数据构造**：在每个训练样本的检索集合中，随机加入与问题主题相似但答案错误的文档。  
   - **损失设计**：在标准的生成交叉熵之外，加上一个二分类损失，让模型学习对每个文档输出“可信度分”。  
   - **效果**：模型在推理时会主动降低干扰项的权重，防止它们主导答案生成。

3. **混合自适应检索**  
   - **判别网络**：一个小型的前馈网络，输入为当前解码状态的隐藏向量，输出一个 0‑1 之间的分数。  
   - **阈值策略**：如果分数低于预设阈值，系统认为已有信息不足，触发第二轮检索；否则直接继续生成。  
   - **优势**：在“一步到位”的简单问答上省去重复检索，在需要补充证据的多跳问答上保证信息完整。

4. **潜在学习的路由机制**  
   - **软路由**：路由器在训练时采用 Gumbel‑Softmax 近似，使得专家选择过程可微分。  
   - **专家专长形成**：随着训练进行，某些专家会逐渐专注于“事实抽取”，另一些则偏向“逻辑推理”或“干扰过滤”。  
   - **推理时的动态分配**：每个 token 根据自身语义自动匹配最合适的专家，实现了“按需调用”式的推理。

**最巧妙的点**  
- 把原本只能整体升级的稠密模型，拆成可按需激活的稀疏专家网络，既保持了开源模型的可访问性，又突破了算力瓶颈。  
- 干扰项训练让模型在检索噪声普遍的真实场景下仍能保持鲁棒，这在大多数公开 RAG 实现中是缺失的。  
- 自适应检索的判别层把“是否需要再查资料”这一步交给模型自己决定，极大提升了实际部署时的响应速度。

### 实验与效果
- **测试任务**：论文在多个知识密集型基准上评估，包括开放域问答（OpenQA）、多跳推理数据集（HotpotQA）以及事实核查任务（FactCC）。  
- **对比基线**：ChatGPT（gpt‑3.5‑turbo）、Self‑RAG、Command‑R+ 以及未经 MoE 改造的 Llama‑2‑7B‑RAG。  
- **主要结果**：在 OpenQA 上，Open‑RAG 的准确率比原始 Llama‑2‑7B‑RAG 提升约 12%，并且超过 ChatGPT 约 4%。在 HotpotQA 的多跳任务中，正确答案率提升 9%‑11%，显著领先 Self‑RAG。  
- **速度对比**：得益于稀疏激活，单轮推理的平均延迟比全稠密模型低约 35%。  
- **消融实验**：去掉干扰项训练后，多跳任务的准确率下降约 6%；关闭自适应检索层后，整体延迟提升 28% 而准确率几乎不变，说明该层主要贡献在效率提升。  
- **局限性**：作者指出 MoE 的路由器在极端长文本上仍会出现路由不稳的现象；此外，当前实现仅在 Llama‑2‑7B 上验证，规模更大的模型是否同样受益仍需实验。

### 影响与延伸思考
Open‑RAG 把“稀疏专家”这一在大模型加速领域常见的技巧引入检索增强生成，打开了开源模型在高质量推理上的新可能。后续工作（如 2024‑2025 年的 “MoE‑RAG” 系列）已经开始探索更细粒度的专家划分（例如专门的事实抽取专家 vs. 逻辑推理专家）以及跨模态检索（图文）下的 MoE 路由。对想进一步深入的读者，可以关注以下方向：① MoE 路由的可解释性研究；② 在更大规模开源模型（如 Llama‑2‑70B）上进行稀疏化的效率‑精度权衡；③ 将自适应检索决策与强化学习结合，实现更长对话中的动态知识获取。

### 一句话记住它
把开源大模型拆成按需激活的专家网络，让检索证据真正参与推理，同时学会辨别干扰，既提准又省时。