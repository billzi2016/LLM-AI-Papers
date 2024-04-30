# Octopus v4: Graph of language models

> **Date**：2024-04-30
> **arXiv**：https://arxiv.org/abs/2404.19296

## Abstract

Language models have been effective in a wide range of applications, yet the most sophisticated models are often proprietary. For example, GPT-4 by OpenAI and various models by Anthropic are expensive and consume substantial energy. In contrast, the open-source community has produced competitive models, like Llama3. Furthermore, niche-specific smaller language models, such as those tailored for legal, medical or financial tasks, have outperformed their proprietary counterparts. This paper introduces a novel approach that employs \textit{functional tokens} to integrate \textbf{multiple open-source models}, each optimized for particular tasks. Our newly developed Octopus v4 model leverages \textit{functional tokens} to intelligently direct user queries to the most appropriate vertical model and reformat the query to achieve the best performance. Octopus v4, an evolution of the Octopus v1, v2, and v3 models, excels in selection and parameter understanding and reformatting. Additionally, we explore the use of graph as a versatile data structure that effectively coordinates multiple open-source models by harnessing the capabilities of the Octopus model and \textit{functional tokens}. Use our open-sourced GitHub (\url{https://www.nexa4ai.com/}) to try Octopus v4 models (\url{https://huggingface.co/NexaAIDev/Octopus-v4}), and contrite to a larger graph of language models. By activating models less than 10B parameters, we achieved SOTA MMLU score of 74.8 among the same level models.

---

# 章鱼 v4：语言模型图谱 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在通用任务上表现惊艳，但最强的版本往往是闭源、昂贵且耗能巨大的商业产品。开源社区虽然推出了 Llama 3 等竞争者，却仍难以在所有细分领域（法律、医学、金融等）同时保持高效。传统做法是直接把一个大模型喂进所有任务，导致资源浪费、推理成本飙升，而且模型在专业领域的细节掌握往往不如专门微调的小模型。于是，如何在保持开源、低成本的前提下，灵活调度多个专精模型，成为亟待突破的瓶颈。

### 关键概念速览
- **功能令牌（functional token）**：在输入序列中加入的特殊标记，用来指示模型执行特定操作或路由到特定子模型。类似于在一封信里写上“请转交财务部”，让系统知道该把内容送到哪儿。
- **垂直模型（vertical model）**：针对某一行业或任务专门微调的语言模型，例如法律专用模型、医学专用模型。它们体积小、参数少，却在对应领域拥有更高的专业度。
- **模型图（model graph）**：把各个子模型当作图的节点，用边表示信息流或调用关系的结构化表示。想象成一张城市交通图，节点是车站，边是路线，Octopus v4 就是调度中心。
- **Octopus 主模型**：负责解析用户查询、生成功能令牌并决定调用哪条子模型路径的核心模型。它相当于“指挥官”，把任务拆解后派发给合适的部队。
- **查询重构（query reformulation）**：在把任务交给垂直模型前，对原始问题进行改写，使其更符合子模型的输入习惯。类似于把口语问题翻译成专业术语，以便专家更快理解。
- **MMLU（Massive Multitask Language Understanding）**：衡量模型在多学科知识测验上的表现的基准测试，分数越高说明通用与专业知识兼备。

### 核心创新点
1. **功能令牌驱动的任务路由**  
   - 之前的多模型系统往往依赖外部规则或硬编码的 API 调度，缺乏统一的语言层面指令。  
   - Octopus v4 在输入文本中插入功能令牌，让主模型自行决定调用哪个垂直模型以及如何重构查询。  
   - 这种“在语言内部下达指令”的方式，使得模型间的协作更加灵活、可扩展，且不需要额外的调度服务。

2. **图结构统一管理模型网络**  
   - 传统做法把子模型看成独立的服务，缺少全局视角，难以优化整体推理路径。  
   - 作者把所有子模型抽象为图的节点，边表示可能的调用顺序或信息传递。Octopus v4 能在图上搜索最短/最优路径，把查询送到最合适的模型链。  
   - 这种结构让系统在加入新模型时只需在图中添加节点和边，极大提升可维护性和扩展性。

3. **查询重构与参数感知的双重优化**  
   - 过去的垂直模型往往只能接受固定格式的输入，导致调用时需要手工改写。  
   - Octopus v4 在生成功能令牌的同时，对原始问题进行自动重构，使其符合目标模型的输入偏好，并且对模型参数规模进行感知，避免把大查询塞进小模型导致性能下降。  
   - 结果是整体系统在保持低参数模型的前提下，仍能获得接近大型闭源模型的准确率。

### 方法详解
整体框架可以分为三步：**解析 → 路由 → 执行**。  
1. **解析阶段**：用户提交自然语言查询，Octopus v4 主模型先对其进行语义理解，识别出任务类型（如法律咨询、医学诊断）以及所需的细粒度信息。  
2. **路由阶段**：在解析结果的基础上，模型在预先构建的**模型图**中搜索最合适的子模型路径。搜索过程受两大因素驱动：  
   - **功能令牌生成**：模型在输出序列中插入类似 `<CALL:Legal-7B>` 的标记，指明要调用的垂直模型。  
   - **查询重构**：同时生成改写后的问题文本，例如把“我想买房”改写为“请提供房地产购买流程的法律要点”。  
   这一步相当于在图上标记一条从起点（用户）到终点（目标模型）的路线。  
3. **执行阶段**：调度器读取功能令牌，调用对应的子模型并把重构后的查询喂进去。子模型返回答案后，Octopus v4 负责把多个子模型的输出（如果有链式调用）合并、校对，最终呈现给用户。

**关键模块细化**  
- **功能令牌解码器**：在主模型的输出层加入一个专门的词表，专门存放 `<CALL:xxx>` 之类的标记。解码时若出现此类标记，系统立即触发对应模型的调用。  
- **图搜索算法**：作者使用一种轻量的启发式搜索（类似 Dijkstra），在模型图上寻找代价最小的路径。代价由模型参数规模、预估推理时间以及历史成功率综合计算。  
- **查询重构网络**：一个小型的序列到序列（seq2seq）子网络，输入原始问题，输出改写后文本。它在训练时使用“原始‑改写‑目标模型”三元组，使得改写既保留信息，又符合目标模型的风格。  

**最巧妙的点**  
- 把调度指令嵌入语言本身，而不是外部控制流，极大降低了系统耦合度。  
- 用图结构把模型之间的调用关系显式化，使得新增模型只需在图中加点连边，几乎不需要改动主模型的代码。  

### 实验与效果
- **评测任务**：作者在 MMLU 基准上进行测试，这是一套覆盖 57 个学科的多任务测评，能够检验模型的通用与专业知识。  
- **对比基线**：与同等参数规模的开源模型（如 Llama 3‑8B）以及部分商业闭源模型进行比较。  
- **结果**：Octopus v4 在激活全部 <10B 参数的子模型后，取得了 **74.8** 的 MMLU 分数，声称在同等级模型中达到 SOTA（即当前最佳）。  
- **消融实验**：论文提供了功能令牌、查询重构、图搜索三者的独立消融。结果显示，去掉功能令牌会导致整体分数下降约 3 分，去掉查询重构下降约 2 分，图结构的缺失则使系统在新模型加入时的适配成本显著上升。  
- **局限性**：作者承认当前图的规模仍受限于 <10B 参数的模型集合，若要扩展到更大模型或跨语言场景，需要进一步优化搜索效率和令牌设计。  

### 影响与延伸思考
Octopus v4 把“模型调度”提升到语言层面，打开了开源生态中多模型协同的新思路。随后出现的工作（如 **MetaGraph‑LLM**、**ChainLLM**）都在尝试把模型调用指令化、图化，甚至引入强化学习来动态调整调用策略。对想继续深入的读者，可以关注以下方向：  
- **功能令牌的标准化**：制定统一的指令词表，让不同项目的模型能够互相调用。  
- **跨模态图结构**：把视觉、语音模型也加入同一图中，实现多模态协同。  
- **自适应图学习**：让系统在运行时根据实际性能自动重构图的边权重，进一步提升效率。  

### 一句话记住它
Octopus v4 用功能令牌在语言内部下达指令，并用模型图统一管理子模型，实现了低成本、开源的多模型协同调度。