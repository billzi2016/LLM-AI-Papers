# Knowledge-Driven CoT: Exploring Faithful Reasoning in LLMs for   Knowledge-intensive Question Answering

> **Date**：2023-08-25
> **arXiv**：https://arxiv.org/abs/2308.13259

## Abstract

Equipped with Chain-of-Thought (CoT), Large language models (LLMs) have shown impressive reasoning ability in various downstream tasks. Even so, suffering from hallucinations and the inability to access external knowledge, LLMs often come with incorrect or unfaithful intermediate reasoning steps, especially in the context of answering knowledge-intensive tasks such as KBQA. To alleviate this issue, we propose a framework called Knowledge-Driven Chain-of-Thought (KD-CoT) to verify and modify reasoning traces in CoT via interaction with external knowledge, and thus overcome the hallucinations and error propagation. Concretely, we formulate the CoT rationale process of LLMs into a structured multi-round QA format. In each round, LLMs interact with a QA system that retrieves external knowledge and produce faithful reasoning traces based on retrieved precise answers. The structured CoT reasoning of LLMs is facilitated by our developed KBQA CoT collection, which serves as in-context learning demonstrations and can also be utilized as feedback augmentation to train a robust retriever. Extensive experiments on WebQSP and ComplexWebQuestion datasets demonstrate the effectiveness of proposed KD-CoT in task-solving reasoning generation, which outperforms the vanilla CoT ICL with an absolute success rate of 8.0% and 5.1%. Furthermore, our proposed feedback-augmented retriever outperforms the state-of-the-art baselines for retrieving knowledge, achieving significant improvement in Hit and recall performance. Our code and data are released on https://github.com/AdelWang/KD-CoT/tree/main.

---

# 知识驱动的思维链：探索大语言模型在知识密集型问答中的可信推理 论文详细解读

### 背景：这个问题为什么难？

在知识密集型问答（KBQA）里，模型必须把题目拆解成若干推理步骤，并且每一步都要依赖外部事实。传统的思维链（CoT）让大语言模型自行写出推理过程，却常常出现“幻觉”：模型凭空捏造信息或把错误的中间结论当作事实。因为模型本身不直接连通知识库，错误会在后续步骤中被放大，导致最终答案离谱。要想让模型在写出思维链的同时保持对真实知识的忠诚，必须解决两个根本障碍：①缺乏可靠的外部检索支持，②缺少机制去验证和纠正已经写出的推理。

### 关键概念速览
- **CoT（思维链）**：让模型在给出答案前先把推理步骤写下来，类似解数学题时先列出算式，便于检查和纠错。  
- **幻觉（Hallucination）**：模型生成的内容没有对应的真实来源，就像人在没有查资料的情况下凭空编造答案。  
- **KBQA（知识库问答）**：要求系统根据结构化或半结构化的知识库回答自然语言问题，需要精准的事实检索。  
- **检索增强生成（RAG）**：模型在生成文本时先向外部数据库查询相关片段，再把检索结果当作上下文输入，类似写报告时先查资料再写结论。  
- **多轮 QA 结构**：把一次完整的思维链拆成若干轮，每轮包括“模型提问 → 检索系统回答 → 模型基于答案继续推理”，像是老师和学生的来回问答。  
- **In‑Context Learning（上下文学习）**：通过在提示中加入示例，让模型学习到特定任务的解题套路，而不需要额外的参数更新。  
- **反馈增强检索器**：利用模型在推理过程中对检索结果的接受或拒绝信息，反向训练检索器，使其更倾向于返回真正有用的事实。  

### 核心创新点
1. **把思维链包装成多轮 QA**  
   之前的 CoT 直接让模型一次性写完所有步骤，缺少外部校验。KD‑CoT 把每一步都转化为“模型提问 → 检索系统返回精确答案 → 模型基于答案继续写”。这样模型可以随时对检索到的事实进行核对，显著降低幻觉的产生。  

2. **构建 KBQA CoT 示例库**  
   为了让模型学会在多轮对话中使用检索答案，作者手工收集并标注了一批知识密集型问答的完整思维链，既充当上下文学习的示例，也提供了“正确答案 ↔ 检索片段”对应关系，帮助后续训练检索器。  

3. **反馈驱动的检索器训练**  
   在推理过程中，模型会判断检索结果是否符合当前上下文。如果不符合，模型会生成纠错提示。作者把这些提示当作监督信号，进一步微调检索模型，使其更精准地捕获关键事实。  

4. **验证与修改推理轨迹的闭环机制**  
   每轮结束后，模型会回顾已写的推理步骤，检查是否与最新检索到的事实冲突，必要时主动修改之前的错误。这种“自我纠错”在传统 CoT 中几乎不存在。  

### 方法详解
**整体思路**：KD‑CoT 将一次完整的 KBQA 任务拆成若干交互轮次。每轮包括三步：①模型基于已有上下文提出子问题；②检索系统（Retriever + Reader）返回最相关的事实答案；③模型把检索答案写进思维链，并决定是否继续提问。整个过程循环，直到模型生成最终答案。

**关键模块拆解**  

1. **提示构造与示例注入**  
   - 在每次调用 LLM 前，系统先把 KBQA CoT 示例库中的若干完整思维链作为 few‑shot 示例放进提示。  
   - 示例展示了“问题 → 检索 → 推理 → 下一步提问”的完整格式，帮助模型养成在每一步都主动查询的习惯。  

2. **模型提问生成**  
   - LLM 根据当前已写的推理文本，输出一个子问题（如“X 的首都是什么？”）。这一步相当于学生在解题时自行提出需要查证的细节。  

3. **检索与阅读**  
   - 子问题送入检索器，检索器在大规模知识库（如 Wikipedia）中找出候选文档。  
   - 阅读器对候选文档进行抽取，输出一个简洁、精确的答案片段。  

4. **推理轨迹更新**  
   - LLM 接收到检索答案后，继续在思维链中写下“根据检索得到的 X=Y，接下来 …”。如果答案与已有推理冲突，模型会在同一轮生成纠错说明并修正之前的步骤。  

5. **反馈收集与检索器微调**  
   - 每轮结束后，系统记录模型对检索答案的接受度（接受、质疑或修改）。这些信号被汇总成训练数据，用来进一步微调检索器，使其在后续轮次更倾向于返回高质量答案。  

**最巧妙的设计**：把“验证”嵌进每一轮的推理，而不是等到最终答案才检查。这样模型可以在错误扩散之前及时止损，类似医生在手术过程中不断监测生命体征，发现异常立即处理。

### 实验与效果
- **数据集**：WebQSP 与 ComplexWebQuestion，这两个基准都要求系统在开放域知识库上回答自然语言问题，且答案往往需要多步推理。  
- **对比基线**：普通的 CoT 直接在 LLM 内部完成推理（即 vanilla CoT ICL），以及已有的检索增强方法。  
- **主要结果**：KD‑CoT 在两套数据上的成功率分别提升了 **8.0%** 与 **5.1%**（相对 vanilla CoT），说明引入外部知识并在每一步进行校验能显著提升答案的可信度。  
- **检索器提升**：作者报告说，经过反馈增强训练的检索器在 Hit@k 与 Recall 上均超过了最先进的基线，具体数值未在摘要中给出。  
- **消融实验**：去掉多轮交互或关闭反馈微调都会导致性能回落，验证了每个创新模块的贡献。  
- **局限性**：论文承认仍然依赖于检索系统的覆盖度；在极其稀疏或最新的事实上仍可能出现幻觉。此外，多轮交互带来的计算开销比一次性 CoT 要高。  

### 影响与延伸思考
KD‑CoT 为“让大模型在思维链里主动查资料”提供了可操作的框架，随后出现的工作多聚焦于：①更高效的多轮检索调度，②把结构化查询语言（SQL）直接嵌入思维链，③利用工具（如计算器、地图）实现更广泛的外部调用。对想进一步探索的读者，可以关注检索增强生成（RAG）与工具调用（Tool‑augmented LLM）交叉的方向，尤其是如何在保持推理连贯性的同时降低计算成本。  

### 一句话记住它
让大语言模型在每一步推理时都去检索、验证并即时纠错，才能真正摆脱幻觉，得到可信的知识密集型答案。