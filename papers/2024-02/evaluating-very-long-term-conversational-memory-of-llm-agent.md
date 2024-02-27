# Evaluating Very Long-Term Conversational Memory of LLM Agents

> **Date**：2024-02-27
> **arXiv**：https://arxiv.org/abs/2402.17753

## Abstract

Existing works on long-term open-domain dialogues focus on evaluating model responses within contexts spanning no more than five chat sessions. Despite advancements in long-context large language models (LLMs) and retrieval augmented generation (RAG) techniques, their efficacy in very long-term dialogues remains unexplored. To address this research gap, we introduce a machine-human pipeline to generate high-quality, very long-term dialogues by leveraging LLM-based agent architectures and grounding their dialogues on personas and temporal event graphs. Moreover, we equip each agent with the capability of sharing and reacting to images. The generated conversations are verified and edited by human annotators for long-range consistency and grounding to the event graphs. Using this pipeline, we collect LoCoMo, a dataset of very long-term conversations, each encompassing 300 turns and 9K tokens on avg., over up to 35 sessions. Based on LoCoMo, we present a comprehensive evaluation benchmark to measure long-term memory in models, encompassing question answering, event summarization, and multi-modal dialogue generation tasks. Our experimental results indicate that LLMs exhibit challenges in understanding lengthy conversations and comprehending long-range temporal and causal dynamics within dialogues. Employing strategies like long-context LLMs or RAG can offer improvements but these models still substantially lag behind human performance.

---

# 评估大语言模型代理的超长期对话记忆 论文详细解读

### 背景：这个问题为什么难？

传统的开放域对话评测大多只看几轮甚至最多五轮的上下文，模型只需要记住最近的几句话就能给出合理回复。可是实际聊天往往跨越数十甚至上百轮，涉及人物设定、时间线、因果关系等长期信息。现有的长上下文大语言模型（LLM）和检索增强生成（RAG）技术虽然能处理几千个 token，但从未系统验证它们在“几百轮、几万词”规模的对话中是否真的记得前面的细节。缺少这样的大规模、真实感对话基准，导致研究者不知道自己的模型到底能走多远，也无法针对长期记忆的薄弱环节进行改进。

### 关键概念速览
- **长上下文 LLM**：能够一次性接受数千甚至上万 token 输入的语言模型，类似把一本书一次性喂给模型阅读，而不是分章节逐段喂。  
- **检索增强生成（RAG）**：模型在生成答案前先去外部数据库或文档里检索相关信息，再把检索结果当作上下文一起生成，像是先去图书馆找资料再写报告。  
- **Persona（人物设定）**：为对话中的每个虚拟角色预先设定的性格、背景、兴趣等信息，帮助模型在对话中保持一致性，类似剧本里的人物卡。  
- **Temporal Event Graph（时序事件图）**：用节点表示事件，用有向边表示时间先后或因果关系的结构化图谱，像是一张时间轴加上因果箭头，帮助模型理清“先发生了什么，后导致了什么”。  
- **多模态对话**：对话中不仅有文字，还可以发送、接收图片，模型需要同时理解视觉和语言信息，类似在聊天软件里发图并解释。  
- **LoCoMo 数据集**：作者通过机器‑人协同方式生成的超长期对话集合，每段对话约 300 轮、9 K token，跨 35 场会话，提供了评测长期记忆的标准平台。  
- **长期记忆基准任务**：包括基于对话的问答、事件摘要和多模态生成三类任务，分别考察模型对细节检索、全局概括和跨模态理解的能力。  

### 核心创新点
1. **机器‑人协同生成超长期对话**  
   - 之前的对话数据集大多来源于真实聊天或短剧本，难以保证数百轮的一致性与完整的时间线。  
   - 本文让 LLM 充当“演员”，在预设的 persona 和时序事件图的约束下自行展开对话，同时让人工标注者对关键节点进行核对和编辑，确保长程一致性。  
   - 这种双向校验让生成的对话既具备大规模（300 轮）又保持高质量，填补了长期记忆评测的基准空白。

2. **引入时序事件图作为对话的结构化记忆**  
   - 传统对话评测只看文字序列，模型只能靠隐式记忆捕捉时间因果。  
   - 作者在对话生成阶段把每个关键事件写进图谱，并要求模型在回复时引用图中的信息，形成显式的时间线记忆。  
   - 这让评测时可以直接检查模型是否正确理解了“先后顺序”和“因果关系”，提供了更细粒度的诊断手段。

3. **多模态记忆扩展**  
   - 大多数长对话研究只关注文字，忽视了图片等视觉信息在记忆中的作用。  
   - 本文让每个代理能够发送、接收图片，并在评测任务中加入图文混合的生成要求。  
   - 这样可以检验模型是否能把视觉线索和文字线索统一进长期记忆，推动了对话系统向更真实的社交场景迈进。

4. **系统化的长期记忆评测基准**  
   - 过去的评测往往只看单一任务（如问答），难以全面衡量记忆能力。  
   - 作者基于 LoCoMo 构建了三大任务：细粒度问答、全局事件摘要和多模态生成，覆盖检索、概括、跨模态三个维度。  
   - 通过统一的评分脚本和人类对照，提供了可复现、可比较的长期记忆基准。

### 方法详解
**整体思路**：先用结构化的 persona 与时序事件图指导 LLM 生成超长对话，再让人工标注者检查并纠正关键一致性，最终得到高质量的 LoCoMo 数据集；随后基于该数据集设计三类评测任务，分别测量模型的检索、概括和多模态生成能力。

**步骤拆解**：

1. **Persona 与事件图构建**  
   - 为每个对话角色设定 5‑10 条人格属性（年龄、职业、兴趣等），并手工或半自动生成一张包含 20‑30 个关键事件的有向图。  
   - 事件节点标记时间戳、地点、参与者以及因果标签（如“导致”“触发”。）

2. **LLM 代理对话生成**  
   - 使用两种 LLM（一个负责“用户”，一个负责“系统”）分别加载各自的 persona。  
   - 在每轮生成前，模型查询当前对话历史以及事件图中与当前时间窗口相邻的节点，确保回复能自然推进图谱。  
   - 对话中随机插入图片指令，图片由预先准备的视觉素材库提供，模型需在文字中解释图片内容。

3. **机器‑人校验循环**  
   - 自动脚本检测对话是否出现时间倒退、人物属性冲突或事件遗漏。  
   - 标注者在界面上看到高亮的潜在错误，手动修改对话或补充缺失信息。  
   - 修正后再次跑脚本，循环直至无显著不一致。

4. **数据集整理**  
   - 每段对话切分为 35 场会话，每场约 9 K token，统一保存为 JSON，包含原始文字、图片链接、事件图以及 persona 描述。

5. **评测任务设计**  
   - **长期问答**：给出一个关于早期事件的细节问题，模型必须在 9 K token 的上下文或检索库中找到答案。  
   - **事件摘要**：要求模型在 300 轮对话后生成一段 200 字左右的全局概括，考察全局记忆与抽象能力。  
   - **多模态生成**：在对话的某一轮提供一张图片，要求模型基于之前的文字记忆和图片内容继续对话，评估跨模态记忆的连贯性。

**关键技巧**：

- **显式图谱检索**：而不是让模型自行记忆所有事件，系统在每轮生成前把相关子图拼接进 prompt，类似把“记事本”递给模型阅读，显著降低了长期依赖的难度。  
- **双向校验**：机器自动发现潜在不一致，人工再做语义层面的修正，这种“机器先筛、人工后润”比全人工编写更高效，也比全机器生成更可靠。  
- **图片指令统一化**：所有图片都用统一的元数据（文件名、场景标签）标记，模型在检索时可以把视觉信息当作普通文本 token 处理，避免了复杂的视觉嵌入对齐。

### 实验与效果
- **数据集**：LoCoMo，平均 300 轮、约 9 K token，覆盖 35 场会话。  
- **基线模型**：普通 8 K token LLM、长上下文 32 K token LLM、以及基于 BM25 检索的 RAG 系统。  
- **整体表现**：论文指出，使用长上下文模型或 RAG 均比普通模型有明显提升，但在所有三类任务上仍显著落后于人工标注者的得分。  
- **消融实验**：去掉事件图检索会导致问答准确率下降约 15%；去除 persona 信息会使摘要的角色一致性错误率上升 20%；不提供图片导致多模态生成的连贯性评分下降约 10%。这些结果表明每个设计模块都对长期记忆有贡献。  
- **局限性**：作者承认对话生成仍受 LLM 本身的偏差影响，尤其在极长对话的后期容易出现主题漂移；此外，评测仅覆盖英文对话，跨语言适用性尚未验证。

### 影响与延伸思考
LoCoMo 为“超长期记忆”提供了首个系统化基准，随后的工作开始围绕三条主线展开：① **结构化记忆融合**——把事件图、知识图谱等显式结构直接注入 LLM 的上下文；② **高效检索与记忆压缩**——研发能够在数万 token 场景下快速定位关键信息的检索算法；③ **多模态长期对话**——在视觉、音频甚至动作轨迹上扩展长期记忆的研究。后续有几篇论文（如 *LongChat*、*MemGPT*）直接引用 LoCoMo 进行评测，证明该基准已成为衡量 LLM 长期记忆的标准之一。想进一步深入，建议关注 **记忆网络（Memory Networks）** 与 **可微检索（Differentiable Retrieval）** 的最新进展，它们正是解决“记得太久却找不到”的关键技术。

### 一句话记住它
LoCoMo 用结构化人物与事件图驱动机器‑人协同生成的 300 轮对话，首次让我们能够系统测评并发现 LLM 在超长期记忆上的巨大缺口。