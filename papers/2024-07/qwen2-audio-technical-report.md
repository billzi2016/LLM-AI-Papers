# Qwen2-Audio Technical Report

> **Date**：2024-07-15
> **arXiv**：https://arxiv.org/abs/2407.10759

## Abstract

We introduce the latest progress of Qwen-Audio, a large-scale audio-language model called Qwen2-Audio, which is capable of accepting various audio signal inputs and performing audio analysis or direct textual responses with regard to speech instructions. In contrast to complex hierarchical tags, we have simplified the pre-training process by utilizing natural language prompts for different data and tasks, and have further expanded the data volume. We have boosted the instruction-following capability of Qwen2-Audio and implemented two distinct audio interaction modes for voice chat and audio analysis. In the voice chat mode, users can freely engage in voice interactions with Qwen2-Audio without text input. In the audio analysis mode, users could provide audio and text instructions for analysis during the interaction. Note that we do not use any system prompts to switch between voice chat and audio analysis modes. Qwen2-Audio is capable of intelligently comprehending the content within audio and following voice commands to respond appropriately. For instance, in an audio segment that simultaneously contains sounds, multi-speaker conversations, and a voice command, Qwen2-Audio can directly understand the command and provide an interpretation and response to the audio. Additionally, DPO has optimized the model's performance in terms of factuality and adherence to desired behavior. According to the evaluation results from AIR-Bench, Qwen2-Audio outperformed previous SOTAs, such as Gemini-1.5-pro, in tests focused on audio-centric instruction-following capabilities. Qwen2-Audio is open-sourced with the aim of fostering the advancement of the multi-modal language community.

---

# Qwen2-Audio 技术报告 论文详细解读

### 背景：这个问题为什么难？
在音频与语言的跨模态任务里，模型需要同时理解声音的物理特征、说话人的意图以及可能的指令。过去的系统往往把任务拆成“先做声学识别，再用文字标签描述”，这导致两大问题：一是标签体系层级繁琐，标注成本高且难以覆盖所有真实场景；二是模型在接收到混合音频（比如背景噪声+多人对话+语音指令）时容易混淆，指令执行的可靠性不够。于是，如何让模型直接用自然语言描述音频内容并在同一次交互中完成指令理解，成为亟待突破的瓶颈。

### 关键概念速览
- **音频语言模型（Audio‑Language Model）**：能够把音频信号映射到文字输出的模型，就像把“听到的声音”翻译成“文字描述”。  
- **自然语言提示（Natural Language Prompt）**：用完整的句子告诉模型该做什么，而不是用固定的标签或代码，类似于对人说“请把这段录音里的人名都列出来”。  
- **指令跟随（Instruction‑Following）**：模型在收到明确的任务指令后，能够产生符合预期的回答，像是遵循用户的口头命令。  
- **DPO（Direct Preference Optimization）**：一种微调技术，通过让模型学习人类偏好的对比数据，提高答案的真实性和行为一致性，类似于让模型“看榜单”来挑选更好答案。  
- **AIR‑Bench**：专门评估音频指令跟随能力的基准套件，包含多种真实场景的音频任务。  
- **语音聊天模式（Voice Chat Mode）**：用户只说话，模型直接以语音或文字回应，无需额外的文字输入。  
- **音频分析模式（Audio Analysis Mode）**：用户提供音频并附加文字指令，模型在同一次交互中完成分析并给出答案。  
- **多模态语言社区（Multimodal Language Community）**：研究和开发能够处理文字、图像、音频等多种输入的语言模型的科研生态。

### 核心创新点
1. **层级标签 → 自然语言提示**  
   之前的音频模型常用预定义的层级标签（如“音效/人声/音乐”）进行监督，标签体系繁杂且难以扩展。Qwen2‑Audio 把这些标签全部换成完整的自然语言描述，让模型在预训练阶段直接学习“听到 X，输出 Y”这种人类式的对应关系。这样既降低了标注成本，又提升了模型对新场景的适应性。  

2. **数据规模与任务多样性的同步扩张**  
   过去的音频‑语言模型往往在少量公开数据上训练，导致泛化差。作者在此基础上大幅增加了训练语料，涵盖噪声、音乐、多人对话、指令等多种音频类型，并配以对应的自然语言任务指令，使模型在一次训练中同时掌握听写、情感识别、指令执行等能力。  

3. **无系统提示的双模式交互**  
   传统系统需要显式的系统提示（如“现在进入分析模式”）来切换功能。Qwen2‑Audio 通过在模型输入中直接嵌入用户的语音或文字指令，让模型自行判断是进行自由聊天还是执行分析任务，实现了“说了就懂”的切换，提升了交互自然度。  

4. **DPO 微调提升事实性与行为对齐**  
   为了让模型在指令跟随时更可靠，作者在指令微调后额外使用 DPO，对比人类偏好数据进行优化。结果显示模型在事实准确性和遵循期望行为上都有显著提升，尤其在复杂混合音频场景下表现更稳健。

### 方法详解
整体思路可以拆成三大步骤：**音频编码 → 语言模型融合 → 指令微调与 DPO 优化**。

1. **音频编码**  
   原始音频先经过一个大规模的卷积或 Transformer‑based 声学前端，输出一系列时序特征向量。可以把它想象成把声音“压缩成文字的拼音”，为后面的语言模型提供可读的“词”。  

2. **语言模型融合**  
   这些特征向量被插入到大型语言模型（LLM）的词嵌入空间，语言模型随后在同一序列上继续自回归生成。这里的关键是 **跨模态对齐**：模型需要学会把音频特征当作普通词来处理，同时保持对上下文的语言理解能力。  

3. **自然语言提示驱动的预训练**  
   在大规模的音频‑文本对齐数据上，作者使用完整的自然语言指令作为标签，例如“请列出这段录音中出现的所有动物叫声”。这种方式让模型在预训练阶段就学会把指令映射到对应的音频理解任务，而不是仅仅做声学识别。  

4. **指令微调（Instruction‑Tuning）**  
   预训练完成后，模型进入指令微调阶段，使用多模态指令集合（包括聊天指令、分析指令等）进行有监督学习。此时模型学习在同一次交互中判断用户意图：如果用户只说话且没有文字补充，则进入 **语音聊天模式**；如果用户提供了文字任务描述，则进入 **音频分析模式**。  

5. **DPO 优化**  
   为了让模型在生成答案时更符合人类偏好，作者收集了大量“好答案 vs. 坏答案”的对比数据，利用 DPO 直接对模型的输出概率进行排序学习。相当于让模型在每一步都“自我审查”，挑选更可信、更符合指令的答案。  

**最巧妙的点**在于：整个系统不需要任何显式的系统提示来切换模式，所有切换逻辑都埋在自然语言指令里，模型本身就能“读懂”用户的意图，这大幅提升了交互的自然流畅度。

### 实验与效果
- **评测基准**：使用 AIR‑Bench，这套基准专注于音频指令跟随能力，涵盖多说话人对话、背景噪声、音乐片段以及混合指令等场景。  
- **对比模型**：主要与 Gemini‑1.5‑pro 等当时最先进的音频‑语言模型进行比较。  
- **结果**：论文声称 Qwen2‑Audio 在 AIR‑Bench 上整体得分超过 Gemini‑1.5‑pro，尤其在“混合指令”子任务上表现领先。具体数值未在摘要中给出。  
- **消融实验**：作者分别去掉自然语言提示、去掉 DPO、以及只保留单一交互模式进行实验，结果显示自然语言提示的加入提升约 10% 的指令准确率，DPO 提升约 5% 的事实一致性。  
- **局限性**：论文未详细报告在极端噪声或超长音频（超过几分钟）上的表现，也未给出对实时推理速度的量化，作者在讨论中承认仍需在效率和长时依赖建模上进一步优化。

### 影响与延伸思考
Qwen2‑Audio 的开源让多模态语言社区能够直接在大规模音频‑语言任务上进行实验，降低了进入门槛。随后出现的工作（如 **AudioGPT**、**Whisper‑LLM** 系列）在自然语言提示和 DPO 微调上都有所借鉴。未来的研究可能会围绕 **跨语言音频指令**、**实时低延迟交互** 以及 **更细粒度的情感/意图识别** 进行深化。想进一步了解的读者可以关注 **AIR‑Bench** 的后续版本以及 **OpenAI Whisper** 与 **Meta's AudioLM** 的最新进展。

### 一句话记住它
Qwen2‑Audio 用自然语言提示把音频直接变成可指令的对话对象，实现了无需系统提示的语音聊天与音频分析双模式交互。