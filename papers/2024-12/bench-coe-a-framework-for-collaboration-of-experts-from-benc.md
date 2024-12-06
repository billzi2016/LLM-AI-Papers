# Bench-CoE: a Framework for Collaboration of Experts from Benchmark

> **Date**：2024-12-05
> **arXiv**：https://arxiv.org/abs/2412.04167

## Abstract

Large Language Models (LLMs) are key technologies driving intelligent systems to handle multiple tasks. To meet the demands of various tasks, an increasing number of LLMs-driven experts with diverse capabilities have been developed, accompanied by corresponding benchmarks to evaluate their performance. This paper proposes the Bench-CoE framework, which enables Collaboration of Experts (CoE) by effectively leveraging benchmark evaluations to achieve optimal performance across various tasks. Bench-CoE includes a set of expert models, a router for assigning tasks to corresponding experts, and a benchmark dataset for training the router. Moreover, we formulate Query-Level and Subject-Level approaches based on our framework, and analyze the merits and drawbacks of these two approaches. Finally, we conduct a series of experiments with vary data distributions on both language and multimodal tasks to validate that our proposed Bench-CoE outperforms any single model in terms of overall performance. We hope this method serves as a baseline for further research in this area. The code is available at \url{https://github.com/ZhangXJ199/Bench-CoE}.

---

# Bench‑CoE：基于基准的专家协作框架 论文详细解读

### 背景：这个问题为什么难？

LLM（大语言模型）已经可以完成翻译、写作、代码等多种任务，但单个模型的能力总有盲区。过去的做法要么让同一个模型硬撑所有任务，要么手动挑选几个模型拼凑，却缺少系统化的“谁负责哪个任务”。因为不同模型在不同基准上的表现差异大，直接把它们混在一起往往会出现冲突或资源浪费，导致整体性能不如最强单模型。于是，如何让多个专家模型在合适的时机、合适的任务上协同工作，成为一个迫切需要解决的难题。

### 关键概念速览
- **LLM（大语言模型）**：能够理解并生成自然语言的大规模神经网络，像 ChatGPT 那样可以完成对话、写作等任务。把它想象成“全能的语言机器人”。
- **专家模型（Expert Model）**：在某类任务或数据上经过专门微调的 LLM，擅长特定领域，就像不同专业的医生。
- **基准（Benchmark）**：公开的评测数据集，用来衡量模型在特定任务上的表现。相当于给模型打分的“考试”。
- **路由器（Router）**：一个轻量模型，负责把输入的查询分配给最合适的专家模型。可以类比为“客服中心的分流员”。
- **查询级路由（Query‑Level Routing）**：每一次具体的提问都单独决定走哪位专家，像是每个病人都要单独诊断。
- **主题级路由（Subject‑Level Routing）**：先把任务划分为大类（如翻译、代码），同一类的所有查询统一走同一个专家，类似科室分诊。
- **协作专家（Collaboration of Experts, CoE）**：多个专家模型在同一系统里配合完成任务的整体方案。

### 核心创新点
1. **把基准评测直接当作路由监督**  
   之前的路由器往往需要人工标注“哪个模型更好”。这篇论文把公开的基准成绩当成标签，让路由器在训练时学习“在这个基准上表现最好的模型是谁”。这样省去了额外标注成本，也让路由决策更贴合真实性能。

2. **统一的 Bench‑CoE 框架**  
   过去的多模型系统多是拼凑式的，缺少统一的接口和训练流程。Bench‑CoE 把专家池、路由器、基准数据三者封装进同一个框架，形成“专家库 + 分流员 + 考试卷”的闭环，便于复现和扩展。

3. **两种路由粒度的系统性对比**  
   作者分别实现了查询级和主题级两套路由策略，并在同一实验平台上评估它们的优缺点。查询级更灵活但计算开销大，主题级更高效但可能错失细粒度的优势，这种对比为后续研究提供了明确的选择依据。

4. **跨语言与跨模态的实证验证**  
   论文不仅在纯文本任务上做实验，还把视觉-语言（multimodal）任务加入评测，证明 Bench‑CoE 能在不同模态下统一提升性能，展示了框架的通用性。

### 方法详解
Bench‑CoE 的整体思路可以拆成三步：**准备专家、训练路由、执行协作**。

1. **准备专家模型**  
   研究者先挑选若干已经在不同基准上表现突出的 LLM，可能是专门微调过的代码模型、翻译模型或视觉语言模型。每个模型只负责自己擅长的那块儿，形成一个“专家库”。

2. **构建路由训练集**  
   取公开的基准数据集（比如 GLUE、VQA 等），对每条样本分别让所有专家跑一遍，记录它们的得分。于是每条样本就拥有一个“最佳专家标签”。这一步相当于让所有医生给每位患者做一次体检，找出最合适的专科。

3. **训练路由器**  
   路由器本身是一个轻量的分类模型（如小型 Transformer），输入是原始查询的特征向量，输出是对专家的概率分布。训练目标是让路由器的预测与基准上实际表现最好的专家对齐。这里的关键是把“基准成绩”转化为监督信号，让路由器学会“看分数挑医生”。

4. **查询级 vs 主题级路由**  
   - **查询级**：每一次推理时，路由器根据当前输入即时决定走哪个专家。好比每个患者都现场检查后再决定科室。  
   - **主题级**：先把任务划分为若干大类（如“翻译任务”“代码生成任务”），对同一类的所有查询统一使用同一个专家。相当于先把患者按症状分科，再统一由该科室的医生负责。

5. **执行协作**  
   推理时，系统先把用户的请求送入路由器，路由器输出最可能的专家 ID，随后把请求转发给对应的专家模型，得到最终答案。整个过程像是“客服先把电话转给最合适的技术支持”。

**最巧妙的点**在于把已有的基准评测直接当作路由的“老师”，省去人工标注，同时保证路由决策与真实性能高度一致。这种“让考试成绩教路由”的思路在多模型协作领域之前很少出现。

### 实验与效果
- **测试任务**：论文在多语言任务（如自然语言推理、机器翻译）和多模态任务（如图文检索、视觉问答）上做了实验，覆盖了文本和视觉两大模态。  
- **对比基线**：与单一最强模型、以及简单的模型集合（如平均投票）进行比较。论文声称 Bench‑CoE 在整体指标上超过所有单模型，并且在大多数基准上比集合方法有明显提升。  
- **消融实验**：作者分别去掉路由器、只使用查询级或只使用主题级，结果显示没有路由器的系统性能回落到单模型水平，说明路由器是提升的关键因素。  
- **局限性**：论文承认路由器本身的计算开销在查询级模式下会显著增加，且对专家库的质量高度依赖——如果专家模型本身在基准上差距不大，路由收益会减小。

### 影响与延伸思考
Bench‑CoE 为“模型即服务”提供了一个可复制的基准驱动路由方案，随后出现的工作多聚焦在**动态专家选择**、**更细粒度的路由特征**（如上下文历史）以及**大规模专家池的高效检索**。一些后续研究（如 Mixture‑of‑Experts 的自适应门控、跨语言专家共享）明显受到了 Bench‑CoE 思路的启发。想进一步深入，可以关注以下方向：  
- **无监督路由**：利用模型内部的表示相似度而非基准标签进行分流。  
- **跨模态专家协同**：让语言专家和视觉专家在同一查询上协同工作。  
- **成本感知路由**：在路由决策中加入计算资源或响应时间的约束。

### 一句话记住它
把公开基准成绩当作老师，让轻量路由器学会把每个任务精准分配给最擅长的专家，从而让多模型系统整体跑得比任何单一模型都快。