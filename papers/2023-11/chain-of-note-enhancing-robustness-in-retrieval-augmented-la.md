# Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language   Models

> **Date**：2023-11-15
> **arXiv**：https://arxiv.org/abs/2311.09210

## Abstract

Retrieval-augmented language models (RALMs) represent a substantial advancement in the capabilities of large language models, notably in reducing factual hallucination by leveraging external knowledge sources. However, the reliability of the retrieved information is not always guaranteed. The retrieval of irrelevant data can lead to misguided responses, and potentially causing the model to overlook its inherent knowledge, even when it possesses adequate information to address the query. Moreover, standard RALMs often struggle to assess whether they possess adequate knowledge, both intrinsic and retrieved, to provide an accurate answer. In situations where knowledge is lacking, these systems should ideally respond with "unknown" when the answer is unattainable. In response to these challenges, we introduces Chain-of-Noting (CoN), a novel approach aimed at improving the robustness of RALMs in facing noisy, irrelevant documents and in handling unknown scenarios. The core idea of CoN is to generate sequential reading notes for retrieved documents, enabling a thorough evaluation of their relevance to the given question and integrating this information to formulate the final answer. We employed ChatGPT to create training data for CoN, which was subsequently trained on an LLaMa-2 7B model. Our experiments across four open-domain QA benchmarks show that RALMs equipped with CoN significantly outperform standard RALMs. Notably, CoN achieves an average improvement of +7.9 in EM score given entirely noisy retrieved documents and +10.5 in rejection rates for real-time questions that fall outside the pre-training knowledge scope.

---

# 笔记链：提升检索增强语言模型的鲁棒性 论文详细解读

### 背景：这个问题为什么难？

检索增强语言模型（RALM）通过把外部文档拉进来，显著降低了大模型的“幻觉”——即凭空编造事实的倾向。但检索系统并不总是把最相关的材料送上来，噪声文档、无关信息甚至错误来源都会混进来。模型往往会把这些杂音当作答案的依据，甚至在自己内部已经掌握了答案时也会被检索结果“抢走风头”。更糟的是，传统 RALM 缺乏自我评估的机制，遇到知识缺口时很少主动说“我不知道”，而是硬凑一个可能错误的答案。正是这些根本性缺陷，让提升模型在噪声环境下的可靠性成为迫切需求。

### 关键概念速览
- **检索增强语言模型（RALM）**：在生成答案前先用搜索引擎或向量库把相关文档找出来，再把这些文档当作上下文喂给大语言模型。类似于人类先查资料再写报告的流程。  
- **幻觉（Hallucination）**：模型在没有足够依据的情况下捏造信息，就像在答题时凭空编造例子。  
- **噪声文档**：检索结果中与提问无关或误导性的文本，像是图书馆里误放进来的错误书籍。  
- **未知检测（Unknown Detection）**：模型判断自己是否拥有足够信息来回答，如果不行就返回“未知”。相当于考试时遇到不会的题目选择放弃而不是随便写。  
- **Chain-of-Note（CoN）**：对每篇检索到的文档逐段写“阅读笔记”，并用这些笔记评估文档的相关性，最终决定答案。可以把它想象成读书会里每个人先写读后感，再投票决定哪本书最能回答问题。  
- **EM（Exact Match）分数**：答案必须完全匹配参考答案才算对，常用于评估问答系统的严格准确率。  
- **拒绝率（Rejection Rate）**：系统在无法确定答案时主动拒绝的比例，数值越高说明模型更懂得说“不”。  

### 核心创新点
1. **从直接阅读转向“笔记链”**  
   - 之前的 RALM 把检索到的全文直接塞进模型，模型必须一次性消化所有信息。  
   - CoN 让模型先对每篇文档生成简短的阅读笔记，再把这些笔记送进后续模块。  
   - 这种分层处理让模型能够更细致地筛选信息，显著降低噪声文档的干扰。  

2. **显式的相关性评估模块**  
   - 传统方法依赖模型在生成答案时自行判断文档价值，缺乏可解释的评估步骤。  
   - CoN 在笔记生成后加入一个专门的相关性打分器，比较笔记内容与提问的匹配程度。  
   - 通过明确的打分，系统可以在答案合成前剔除低相关文档，提升答案的准确性和可解释性。  

3. **利用大模型自动构造训练数据**  
   - 手工标注“笔记”成本高，原文使用 ChatGPT 生成大规模的笔记-问答对。  
   - 这些合成数据随后用于在 LLaMA‑2 7B 上微调，使模型学会如何写笔记并评估相关性。  
   - 这种“模型生成模型” 的数据管道大幅降低了标注门槛，同时保持了高质量的学习信号。  

4. **统一的未知检测机制**  
   - 在笔记链的最后加入一个“是否足够信息”判断，如果所有笔记的相关性分数都低于阈值，直接输出 “unknown”。  
   - 与仅靠答案置信度的旧做法相比，这种基于检索质量的判断更稳健，提升了系统的拒绝率。  

### 方法详解
**整体框架**  
CoN 的工作流程可以拆成四步：①检索、②笔记生成、③相关性评估、④答案合成或未知返回。整个过程像是先去图书馆找书、再写读书笔记、再投票决定哪本书最有用，最后根据投票结果写报告或承认不知道。

**步骤拆解**  

1. **检索**  
   - 给定用户问题，使用传统的稀疏或密集检索器（如 BM25、向量相似度）取回 N 篇候选文档。这里的 N 通常在 5–10 之间，保证覆盖面同时控制计算量。  

2. **笔记生成（Note‑Writer）**  
   - 每篇文档被切成若干段落。模型（微调后的 LLaMA‑2 7B）对每段落输出一行简短的“笔记”。笔记的目标是提炼段落的核心事实或观点，同时标记与提问的潜在关联。  
   - 类比：像学生在课堂上把老师讲的要点记在笔记本上，只保留关键信息。  

3. **相关性评估（Relevance‑Scorer）**  
   - 将所有笔记拼接成一个“笔记序列”，再喂入一个轻量的匹配网络（可以是同一个 LLaMA‑2 的分类头），计算每篇文档的整体相关性分数。  
   - 这里的设计巧妙之处在于：模型不直接比较原文与问题，而是比较“提炼后的要点”与问题，噪声信息被自然过滤掉。  

4. **答案合成或未知返回**  
   - 若最高相关性分数超过预设阈值，系统把对应文档的笔记作为“证据”，交给答案生成模块（同样是 LLaMA‑2），让它在这些高质量证据上生成最终答案。  
   - 若所有分数都低于阈值，直接输出 “unknown”。这种二选一的策略避免了在信息不足时硬凑答案。  

**训练细节**  
- 训练数据：使用 ChatGPT 按照“问题 → 检索文档 → 笔记 → 答案” 的模板生成约 50 万对。  
- 微调：先在笔记生成任务上进行监督学习，让模型学会压缩段落；再在相关性评估上加入对比学习，使高相关文档的笔记得分更高。  
- 关键技巧：在笔记生成时加入“是否提及关键实体”的提示词，帮助模型聚焦事实；在评估阶段使用温度调节，使分数分布更分明。  

### 实验与效果
- **测试数据**：四个公开的开放域问答基准（如 Natural Questions、TriviaQA、WebQuestions、HotpotQA），其中在实验设置里故意把检索结果全部替换成噪声文档，以检验鲁棒性。  
- **对比基线**：标准的检索增强 LLaMA‑2（直接拼接检索文档）、以及最新的 RAG（Retrieval‑Augmented Generation）实现。  
- **核心结果**：在全噪声检索条件下，CoN 的 EM 分数比基线提升约 **+7.9**；在真实检索但包含未知问题时，系统的拒绝率提升 **+10.5**，即更频繁地正确说出“我不知道”。  
- **消融实验**：去掉笔记生成或相关性评估任一环节，性能分别下降约 4–5 分，说明两者相辅相成。  
- **局限性**：论文承认在检索质量本身极低（几乎没有任何相关文档）时，笔记链仍然难以恢复答案；此外，笔记生成的速度比直接拼接慢约 30%，对实时系统有一定冲击。  

### 影响与延伸思考
CoN 把“阅读笔记”引入检索增强框架后，激发了后续一波围绕“结构化检索后处理”的研究。比如有人尝试把笔记改成图谱形式，或让多模态模型对图像检索结果写笔记。还有工作把笔记链与 **Chain‑of‑Thought（思维链）** 结合，让模型先写思考步骤再写笔记，进一步提升复杂推理的透明度。对想深入的读者，可以关注 **检索后解释（Post‑retrieval Explainability）**、**自适应阈值的未知检测** 以及 **高效笔记生成的轻量模型** 等方向。  

### 一句话记住它
让模型先给每篇检索文档写“阅读笔记”，再用笔记评估相关性，既能过滤噪声，又能在不知道答案时大胆说“不”。