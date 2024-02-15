# A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts

> **Date**：2024-02-15
> **arXiv**：https://arxiv.org/abs/2402.09727

## Abstract

Current Large Language Models (LLMs) are not only limited to some maximum context length, but also are not able to robustly consume long inputs. To address these limitations, we propose ReadAgent, an LLM agent system that increases effective context length up to 20x in our experiments. Inspired by how humans interactively read long documents, we implement ReadAgent as a simple prompting system that uses the advanced language capabilities of LLMs to (1) decide what content to store together in a memory episode, (2) compress those memory episodes into short episodic memories called gist memories, and (3) take actions to look up passages in the original text if ReadAgent needs to remind itself of relevant details to complete a task. We evaluate ReadAgent against baselines using retrieval methods, using the original long contexts, and using the gist memories. These evaluations are performed on three long-document reading comprehension tasks: QuALITY, NarrativeQA, and QMSum. ReadAgent outperforms the baselines on all three tasks while extending the effective context window by 3.5-20x.

---

# 具有人类启发的阅读代理：超长上下文要点记忆 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在一次性处理文本时受限于固定的上下文窗口，常见的窗口只有几千个 token。面对几万甚至上百千字的长文档，模型要么截断，要么只能靠检索把片段喂进去，却失去全局连贯性。传统检索‑阅读管线把所有信息压缩成稀疏向量，检索到的片段往往缺少上下文的“整体感”，导致在需要跨段落推理的阅读理解任务上表现不佳。根本的瓶颈是：LLM 既没有长期记忆，也缺乏像人类那样在阅读过程中主动归纳要点、随时回溯的机制。

### 关键概念速览
**上下文窗口**：模型一次性能看到的文本长度上限，类似于一次性阅读的纸张大小。  
**记忆片段（Memory Episode）**：在阅读过程中把若干相邻段落打包在一起的单元，类似于人把章节划分为“章节摘要”。  
**要点记忆（Gist Memory）**：对记忆片段的高度压缩，只保留核心信息，像是把章节的要点写成几句话的笔记。  
**检索动作（Lookup Action）**：当模型发现要点记忆不足以完成任务时，主动去原文中查找对应段落，类似于读者翻回书页确认细节。  
**提示工程（Prompt Engineering）**：通过设计特定的文字提示，引导 LLM 按照预定流程执行任务。  
**阅读理解任务**：给定长文档和问题，要求模型给出正确答案的任务，常见数据集有 QuALITY、NarrativeQA、QMSum。  
**有效上下文长度**：指模型在完成任务时实际利用的文本信息量，可能远大于原始窗口大小。  

### 核心创新点
1. **从检索转向交互式阅读**：传统方法先把长文档切块、向量化，然后检索最相似的块交给 LLM。本文改为让 LLM 自己决定哪些块该一起记忆、哪些块需要压缩成要点记忆，模拟人类阅读时的“先浏览、后归纳”过程。这样做把记忆的组织权交给模型本身，而不是外部检索系统。  
2. **要点记忆的生成与使用**：在每个记忆片段结束后，模型生成一段简短的要点摘要（gist），并把它存入一个可随时查询的“记忆库”。要点记忆比原始片段短得多，却保留了关键线索，使得后续的推理可以在更小的上下文中进行。相比直接使用原文，这大幅提升了上下文利用率。  
3. **动态查阅机制**：当模型在回答问题时发现要点记忆不足，它会发出“查阅”指令，定位原文中对应的段落并重新读取。这个循环让模型在需要细节时能够回溯，而不是一次性把所有细节都压进要点记忆。  
4. **统一的提示框架**：所有上述行为都通过一套精心设计的提示模板实现，无需额外的模型结构改动，只利用现有的 LLM 能力即可部署。这样既保持了实现的简洁，又让方法可以直接迁移到不同的基础模型上。

### 方法详解
整体思路可以概括为三步循环：**划分‑压缩‑查阅**。首先把超长文档切成若干连续的段落，每个段落大约几百 token。接着让 LLM 按顺序读取这些段落，并在每读取完一个“记忆片段”（由若干段落组成）后执行两项操作：  
1. **生成要点记忆**：模型在提示中被要求用一句话或几句话概括该片段的核心信息。生成的要点记忆被存入一个列表，后续推理时可以直接拼接进提示。  
2. **决定是否继续**：模型评估当前要点记忆是否已经足够回答任务中的问题。如果不够，它会标记需要进一步查阅的原文位置。

当所有片段都处理完后，模型进入**答案生成阶段**。此时的提示包括：任务描述、所有要点记忆、以及一个“查阅指令列表”。如果在生成答案的过程中模型触发了查阅指令，它会根据指令中的段落编号，从原文中抽取对应的原始文本片段，重新加入提示并继续推理。这个过程可以循环多次，直到模型给出满意的答案或达到预设的查阅上限。

**关键模块的类比**：  
- **记忆片段** 像是读书时把章节划分为“章节块”。  
- **要点记忆** 类似于学生在每章节后写的“章节小结”。  
- **查阅动作** 就是考试时打开教材翻到对应页码确认细节。

**提示设计**：作者使用了两类模板。第一类用于“阅读‑压缩”阶段，提示模型先阅读指定段落，然后输出要点摘要；第二类用于“答题‑查阅”阶段，提示模型在已有要点的基础上尝试回答，并在需要时输出“LOOKUP: <段落编号>”。所有模板都明确指示模型只能在给定的记忆库和查阅得到的文本中寻找答案，防止它自行“幻觉”出信息。

**最巧妙的地方**：把长期记忆的构建交给 LLM 本身，而不是外部检索系统。模型在生成要点时已经对内容做了语义抽象，这比向量检索的相似度匹配更贴合任务需求。同时，查阅动作的触发是由模型自行判断的，形成了闭环的自适应阅读流程。

### 实验与效果
- **测试任务**：QuALITY（长篇小说阅读理解）、NarrativeQA（故事情节问答）和 QMSum（会议纪要摘要）。这三个数据集的文档长度都在数千到上万 token 之间，足以验证超长上下文能力。  
- **对比基线**：包括直接使用原始长文档（不做任何压缩）、传统检索‑阅读管线（先向量检索再交给 LLM）、以及仅使用要点记忆但不允许查阅的版本。  
- **主要结果**：在所有三个任务上，ReadAgent 的准确率或 ROUGE 分数均高出基线 3%~12% 不等。作者报告称有效上下文长度提升了 3.5‑20 倍，意味着模型在相同的硬件限制下能够利用更多信息。  
- **消融实验**：去掉要点记忆、去掉查阅动作、或把记忆片段大小固定不变，性能均出现显著下降。尤其是查阅模块的缺失导致在需要细节的问答上跌幅最大，验证了动态回溯的必要性。  
- **局限性**：实验主要在英文长文档上进行，中文或其他语言的适配尚未验证；查阅次数上限是人为设定的，若任务需要频繁回溯可能导致提示长度再次爆炸。作者也提到要点记忆的质量高度依赖 LLM 的压缩能力，若模型本身压缩不佳，后续推理会受影响。

### 影响与延伸思考
这篇工作把“阅读‑记忆‑回溯”三步循环形式化为可通过提示实现的通用框架，开启了在不改模型结构的前提下扩展上下文的思路。随后出现的研究多聚焦于更高效的要点生成（如利用专门的摘要模型）或把记忆库外部化为可检索的数据库（如长期向量存储），进一步提升了可扩展性。对想深入的读者，可以关注以下方向：  
- **可微分记忆**：把要点记忆的生成过程做成可学习的模块，直接在训练时优化。  
- **跨模态长文档**：把图像、音频等信息也纳入要点记忆，实现多模态阅读。  
- **自适应片段划分**：让模型根据内容复杂度动态决定记忆片段的大小，而不是固定长度。  
这些方向都有望把“人类式阅读”推向更真实的应用场景。

### 一句话记住它
让大模型自己划分、压缩并在需要时回溯原文，像人类做笔记一样，把有效上下文延伸到数十倍。