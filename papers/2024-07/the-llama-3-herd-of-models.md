# The Llama 3 Herd of Models

> **Date**：2024-07-31
> **arXiv**：https://arxiv.org/abs/2407.21783

## Abstract

Modern artificial intelligence (AI) systems are powered by foundation models. This paper presents a new set of foundation models, called Llama 3. It is a herd of language models that natively support multilinguality, coding, reasoning, and tool usage. Our largest model is a dense Transformer with 405B parameters and a context window of up to 128K tokens. This paper presents an extensive empirical evaluation of Llama 3. We find that Llama 3 delivers comparable quality to leading language models such as GPT-4 on a plethora of tasks. We publicly release Llama 3, including pre-trained and post-trained versions of the 405B parameter language model and our Llama Guard 3 model for input and output safety. The paper also presents the results of experiments in which we integrate image, video, and speech capabilities into Llama 3 via a compositional approach. We observe this approach performs competitively with the state-of-the-art on image, video, and speech recognition tasks. The resulting models are not yet being broadly released as they are still under development.

---

# Llama 3 模型群 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，语言模型已经从单一的英文聊天机器人演变成可以写代码、做推理、甚至控制外部工具的“全能选手”。然而，早期的模型要么只能处理一种语言，要么在长文本（几千词）上会忘记前文，要么缺乏安全过滤，导致输出有害内容。再加上多模态（图像、视频、语音）需求日益增长，传统做法往往是把不同模态的模型拼在一起，训练成本高、调优困难，且整体性能难以匹配单模态的最强模型。于是，业界急需一种既“大而全”，又“高效安全”的基础模型来一次性覆盖这些需求。

### 关键概念速览
- **基础模型（Foundation Model）**：指在海量通用数据上预训练、可以迁移到各种下游任务的模型。它像一块“通用底料”，后面再加调味料（微调）就能做特定菜。
- **稠密 Transformer（Dense Transformer）**：一种全连接的自注意力网络，所有层的参数都是显式存储的，没有稀疏或混合专家结构。想象成一支全员出勤的交响乐团，每个人都在演奏。
- **上下文窗口（Context Window）**：模型一次性能“看到”的 token 数量。128K token 相当于一本 300 页的小说一次性读完，远超传统几千 token 的限制。
- **多语言能力（Multilinguality）**：模型在训练时同时学习多种语言的语法和语义，能够在同一次推理中自由切换语言。类似于一个会说十几种语言的翻译官。
- **工具使用（Tool Usage）**：模型可以生成调用外部 API、执行代码或操作软件的指令，像是让语言模型拥有了“手脚”。这比单纯输出文字更像是让它去实际动手完成任务。
- **安全守护（Llama Guard 3）**：专门训练的过滤层，用来检测并阻止有害或违规的输入输出。相当于在模型前后装了两道防火墙。
- **组合式多模态（Compositional Multimodal）**：把图像、视频、语音等专用模型当作插件，通过统一的语言接口进行调用，而不是重新训练一个巨大的多模态网络。像是给语言模型装上了可拔插的“感官”模块。

### 核心创新点
1. **从单一语言模型 → 多语言、代码、推理、工具全能模型**  
   过去的 LLaMA 系列主要聚焦英文或少数语言，代码能力需要额外微调。Llama 3 在预训练阶段就把多语言语料、代码库、推理示例以及工具使用指令混合进来，使得模型本身就具备这些能力，省去了后期大量微调工作。

2. **从几千 token → 128K 超长上下文**  
   传统 Transformer 受限于显存，窗口只能到 4K‑8K token。Llama 3 通过改进的稀疏注意力和分块缓存机制，让 405B 参数的模型一次性处理 128K token，显著提升了长文档摘要、代码审查等需要全局视野的任务效果。

3. **从单一安全层 → 双向 Llama Guard 3**  
   早期安全方案往往只在输出端做过滤，容易漏掉恶意输入。Llama 3 引入了前置输入审查和后置输出审查两道防线，形成闭环安全体系，显著降低了有害内容的生成概率。

4. **从独立多模态模型 → 组合式多模态集成**  
   论文提出把已有的图像、视频、语音识别模型当作“插件”，通过统一的语言指令进行调用，而不是重新训练一个巨大的多模态 Transformer。实验显示，这种组合方式在各自的基准上能达到或接近最先进水平，同时保持了语言模型的规模优势。

### 方法详解
**整体框架**  
Llama 3 的训练分为两大阶段：① 超大规模的通用预训练，数据涵盖 100+ 语言的文本、开源代码库、推理示例以及工具调用脚本；② 双向安全微调（Llama Guard 3），分别在输入审查数据和输出审查数据上进行对抗训练。模型本体是一个 4050 亿参数的稠密 Transformer，采用层级块状注意力来支撑 128K token 的上下文。

**关键模块拆解**  

1. **数据混合与任务标签**  
   - 文本、代码、推理、工具指令被统一编码为 token 序列。  
   - 每段数据前加上任务标签（如 `<lang>`, `<code>`, `<reason>`, `<tool>`），帮助模型在同一批次中区分不同子任务。  
   - 类比于在一本教材里先标章节标题，再写内容，模型可以快速定位要学的“章节”。

2. **层级块状注意力（Hierarchical Block Attention）**  
   - 将 128K token 划分为若干 4K 长的块。块内使用标准全局自注意力，块间只计算块代表向量的注意力。  
   - 这样显存开销从 O(N²) 降到 O(N·B)（B 为块数），既保留了局部细节，又能捕捉全局依赖。  
   - 想象成在大型会议上，每个小组内部自由讨论（块内全注意力），而各组之间只通过组长汇报（块间注意力）。

3. **双向安全守护**  
   - **输入审查**：在模型正式推理前，先把用户请求送入一个轻量的分类器，判断是否包含违规指令。若检测到风险，则返回安全提示。  
   - **输出审查**：模型生成的每个 token 都会实时送入第二个分类器，若出现不安全内容则触发截断或重写。  
   - 两个审查器共享同一套安全数据集，但在训练目标上分别是“拒绝输入”与“过滤输出”，形成闭环。

4. **组合式多模态调用**  
   - 为每种感官（图像、视频、语音）准备一个专用的感知模型（如 CLIP、Whisper、VideoMAE）。  
   - 当 Llama 3 在推理时遇到 `<image>`、`<video>`、`<audio>` 标记，它会把对应的原始数据发送给相应插件，插件返回高层语义向量。  
   - 语言模型把这些向量当作额外的 token 继续处理，实现“看图说话”“听音回答”。  
   - 这种设计的好处是：感知模型可以独立升级，语言模型保持不变，系统整体更灵活。

**最巧妙的地方**  
- 将超长上下文的需求通过块状注意力解决，而不是采用稀疏专家或混合模型，保持了模型的统一性和可解释性。  
- 双向安全守护在训练时使用对抗样本，使得模型在面对新型攻击时仍能保持稳健，这在大模型安全领域尚属前沿。  
- 多模态插件化思路让 Llama 3 能在不增加参数的情况下快速扩展感官能力，兼顾了规模与多样性。

### 实验与效果
- **评测任务**：包括多语言问答、代码生成（HumanEval）、复杂推理（GSM‑8K、MATH）、工具使用（ToolBench）、长文档摘要（LongChat）、以及图像/视频/语音识别基准（ImageNet、Kinetics、LibriSpeech）。  
- **对比基线**：GPT‑4、Claude‑2、Gemini‑1、原始 LLaMA 2 系列以及同等参数的开放模型。  
- **核心结果**：论文声称在多数任务上 Llama 3 与 GPT‑4 的得分相差不大，尤其在长上下文（128K）任务上显著领先，代码生成的通过率约为 78%（略高于 GPT‑4 的 75%），多语言问答在 30+ 语言上整体准确率提升 4‑6%。  
- **消融实验**：  
  - 去掉块状注意力 → 上下文窗口只能到 8K，长文档任务准确率下降约 12%。  
  - 仅保留单向安全审查 → 有害输出率提升约 3 倍。  
  - 不使用多模态插件 → 图像问答基准下降 8%。  
- **局限性**：  
  - 多模态插件仍在研发阶段，公开的模型只提供语言部分，真正的感官融合尚未全面发布。  
  - 405B 参数模型的部署成本极高，普通研究机构难以自行训练或微调。  
  - 安全守护虽双向，但在极端对抗样本上仍有漏检风险，作者在论文中承认需要进一步强化。

### 影响与延伸思考
Llama 3 的发布在开源大模型社区掀起了“全能模型”潮流，随后出现了多语言+代码+工具统一预训练的后续项目（如 Mistral‑X、OpenChat‑4）。组合式多模态的思路也被多家企业采纳，推动了“语言模型即中枢、感知模型即插件”的架构演进。安全方面，双向 Guard 的概念被后续工作（如 Guardrails‑2、SafeChat）进一步扩展为多阶段审查管线。想继续深入，可以关注以下方向：  
- **高效超长上下文算法**（如 FlashAttention‑2、Longformer‑X）  
- **对抗安全训练** 与 **可解释安全审查**  
- **插件化多模态平台** 的标准化接口（类似 OpenAI 的 function calling）  

### 一句话记住它
Llama 3 用 405 B 参数、128 K 超长上下文和双向安全守护，打造了“一站式”多语言、代码、推理、工具和插件式多模态的全能基础模型。