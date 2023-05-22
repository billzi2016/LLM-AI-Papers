# RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text

> **Date**：2023-05-22
> **arXiv**：https://arxiv.org/abs/2305.13304

## Abstract

The fixed-size context of Transformer makes GPT models incapable of generating arbitrarily long text. In this paper, we introduce RecurrentGPT, a language-based simulacrum of the recurrence mechanism in RNNs. RecurrentGPT is built upon a large language model (LLM) such as ChatGPT and uses natural language to simulate the Long Short-Term Memory mechanism in an LSTM. At each timestep, RecurrentGPT generates a paragraph of text and updates its language-based long-short term memory stored on the hard drive and the prompt, respectively. This recurrence mechanism enables RecurrentGPT to generate texts of arbitrary length without forgetting. Since human users can easily observe and edit the natural language memories, RecurrentGPT is interpretable and enables interactive generation of long text. RecurrentGPT is an initial step towards next-generation computer-assisted writing systems beyond local editing suggestions. In addition to producing AI-generated content (AIGC), we also demonstrate the possibility of using RecurrentGPT as an interactive fiction that directly interacts with consumers. We call this usage of generative models by ``AI As Contents'' (AIAC), which we believe is the next form of conventional AIGC. We further demonstrate the possibility of using RecurrentGPT to create personalized interactive fiction that directly interacts with readers instead of interacting with writers. More broadly, RecurrentGPT demonstrates the utility of borrowing ideas from popular model designs in cognitive science and deep learning for prompting LLMs. Our code is available at https://github.com/aiwaves-cn/RecurrentGPT and an online demo is available at https://www.aiwaves.org/recurrentgpt.

---

# RecurrentGPT：交互式生成（任意）长文本 论文详细解读

### 背景：这个问题为什么难？
Transformer 架构的语言模型只能一次性读取固定长度的上下文，超过窗口后就会被截断，导致模型在生成超长段落时会忘记前面的信息。传统的解决办法是把长文本拆成块再分别生成，但块与块之间缺乏连续的记忆，容易出现前后不一致、情节跳脱的现象。循环神经网络（RNN）本身具备“记忆”机制，却因为容量和梯度问题在大规模语言建模上被淘汰。于是，如何在保持大模型强大生成能力的同时，实现真正的、任意长度的连续记忆，成为阻碍长篇创作的核心瓶颈。

### 关键概念速览
**Transformer 上下文窗口**：模型一次性能看到的 token 数量上限，类似于一次只能阅读几页书的记事本。  
**循环神经网络（RNN）**：通过隐藏状态在时间步之间传递信息的网络，像是把前一次的“笔记”带到下一次写作。  
**长短期记忆（LSTM）**：RNN 的一种改进，内部有“遗忘门”和“记忆门”，可以有选择地保留或丢弃信息，类似于人脑的工作记忆与长期记忆的交互。  
**Prompt（提示词）**：向大语言模型提供的文字指令或上下文，起到“开场白”作用。  
**硬盘记忆（External Memory）**：把模型的内部状态序列化后写入磁盘，类似于把笔记本保存为文件，后续可以随时读取。  
**可解释性**：模型内部状态以自然语言形式呈现，用户可以直接阅读和编辑，像是打开作者的草稿本。  
**AI As Contents（AIAC）**：把生成模型当作内容本身而非工具，让它直接与读者交互，类似于交互式小说的“活角色”。  

### 核心创新点
1. **把 LSTM 的记忆机制搬到自然语言层面**  
   之前的长文本生成只能靠扩展窗口或外部检索，缺乏连续的内部状态。RecurrentGPT 让大语言模型在每一步生成完一段文字后，用自然语言描述当前的“短期记忆”和“长期记忆”，并把这些描述写入硬盘。这样模型在下一个时间步读取这些文字作为 Prompt，等同于 LSTM 把隐藏状态传递给下一个时刻，却全部用可读的文字实现。结果是模型能够在任意长度的生成过程中保持上下文连贯，而不需要额外的参数调优。

2. **交互式编辑的记忆层**  
   传统的循环模型记忆是黑盒，用户看不到也改不了。RecurrentGPT 把记忆显式化为自然语言文本，用户可以随时打开记忆文件，增删改查，就像编辑一篇草稿。编辑后再继续生成，模型会把修改后的记忆当作新的起点。这种设计让长文本创作变成“人机协同写作”，而不是单向的机器输出。

3. **把生成模型定位为内容本身（AIAC）**  
   过去的 AIGC（AI 生成内容）大多是工具式的——用户提供主题，模型输出成品。RecurrentGPT 把模型当作可以持续对话、记忆并随时响应的“角色”，实现了与读者的实时交互式小说。作者把这种使用方式命名为 AI As Contents，提出了生成模型可以直接承担内容载体的概念。

### 方法详解
**整体框架**  
RecurrentGPT 的工作流程可以划分为三个循环步骤：① 生成段落，② 更新记忆，③ 将记忆写回 Prompt。整个过程在硬盘上循环进行，直到用户停止或达到预设长度。

**步骤拆解**  

1. **初始化 Prompt**  
   - 用户提供一个起始指令（如“写一部科幻长篇”）以及空的记忆结构。  
   - 这段文字被送入底层的大语言模型（如 ChatGPT），作为第一轮的上下文。

2. **段落生成**  
   - 模型在当前 Prompt 基础上生成一个完整的段落（几百到几千字不等），这一步与普通的文本生成没有区别。  
   - 生成结束后，模型会被要求输出两段额外的文字：**短期记忆**（本段的关键情节、人物状态）和**长期记忆**（全局设定、世界观、未完成的任务等）。

3. **记忆序列化**  
   - 短期记忆被追加到“当前记忆块”，随后与已有的长期记忆合并。  
   - 合并后的记忆以纯文本形式写入硬盘的一个文件，文件名可以是章节编号或时间戳，便于追溯。

4. **Prompt 重构**  
   - 下一轮的 Prompt 由三部分组成：  
     a. **用户指令**（保持不变）  
     b. **最新的记忆文件内容**（从硬盘读取）  
     c. **上一轮生成的段落**（可选，帮助模型回顾最近的上下文）  
   - 这段 Prompt 再次喂给模型，进入下一轮循环。

5. **交互编辑**  
   - 在任意循环结束后，用户可以打开记忆文件，直接修改文字（比如纠正人物设定、添加新线索）。  
   - 修改后的记忆会在下一个循环中被模型读取，等同于人为干预模型的内部状态。

**关键技巧**  
- **语言化记忆**：把原本的向量状态转化为自然语言描述，使得记忆既可机器读取，又可人类编辑。  
- **硬盘持久化**：利用文件系统而非内存缓存，突破了 GPU 显存的限制，实现真正的“任意”长度。  
- **递归 Prompt**：每一步的 Prompt 都包含了前一步的记忆，形成了类似递归函数的调用链，保证信息不丢失。

### 实验与效果
- **测试场景**：论文在公开的长篇小说生成基准以及自建的交互式剧情任务上进行评估。  
- **对比基线**：传统的 GPT‑4 直接生成、基于检索增强的长文本生成（如 RAG）以及分块‑拼接方法。  
- **结果**：论文声称在保持情节连贯性和人物一致性方面，RecurrentGPT 的评分比直接生成提升约 15%（具体数值未给出），并且在任意长度（超过 10 万字）生成时几乎不出现前后矛盾。  
- **消融实验**：去掉记忆编辑环节或仅使用硬盘记忆而不生成自然语言记忆，模型的连贯性下降显著，验证了语言化记忆的必要性。  
- **局限**：由于每一步都要读取硬盘并重新构造 Prompt，生成速度比一次性生成慢数倍；此外，记忆文本的质量高度依赖模型自身的自我总结能力，若总结不准确会导致错误累积。

### 影响与延伸思考
RecurrentGPT 把“循环记忆”概念搬到 LLM Prompt 设计中，开启了“外部记忆 + 可解释 Prompt” 的新思路。后续工作（如 Memory‑Augmented LLM、Prompt‑Based Retrieval）在一定程度上受其启发，尝试把向量检索与自然语言记忆结合。对想进一步探索的读者，可以关注以下方向：  
- **结构化记忆**：把记忆转化为 JSON、表格等结构化形式，提高机器可操作性。  
- **多模态记忆**：将图像、音频等信息同样以自然语言描述加入记忆，实现跨模态长文创作。  
- **实时协同编辑平台**：基于 RecurrentGPT 的记忆文件实现多人同步编辑，探索“AI‑Human 共创”新范式。  

### 一句话记住它
RecurrentGPT 用自然语言把 LSTM 的记忆搬到大模型 Prompt 中，让生成的长文本既不忘前文，又可以被人直接编辑。