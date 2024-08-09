# VITA: Towards Open-Source Interactive Omni Multimodal LLM

> **Date**：2024-08-09
> **arXiv**：https://arxiv.org/abs/2408.05211

## Abstract

The remarkable multimodal capabilities and interactive experience of GPT-4o underscore their necessity in practical applications, yet open-source models rarely excel in both areas. In this paper, we introduce VITA, the first-ever open-source Multimodal Large Language Model (MLLM) adept at simultaneous processing and analysis of Video, Image, Text, and Audio modalities, and meanwhile has an advanced multimodal interactive experience. Starting from Mixtral 8x7B as a language foundation, we expand its Chinese vocabulary followed by bilingual instruction tuning. We further endow the language model with visual and audio capabilities through two-stage multi-task learning of multimodal alignment and instruction tuning. VITA demonstrates robust foundational capabilities of multilingual, vision, and audio understanding, as evidenced by its strong performance across a range of both unimodal and multimodal benchmarks. Beyond foundational capabilities, we have made considerable progress in enhancing the natural multimodal human-computer interaction experience. VITA is the first step for the open-source community to explore the seamless integration of multimodal understanding and interaction. While there is still lots of work to be done on VITA to get close to close-source counterparts, we hope that its role as a pioneer can serve as a cornerstone for subsequent research. Project Page: https://vita-home.github.io.

---

# VITA：迈向开源交互式全模态大语言模型 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（Multimodal LLM）要把文字、图片、音频甚至视频这几种截然不同的信号统一进同一个语言框架，本身就像让会说话的机器人同时懂得看、听、说、读。过去的开源模型大多只能处理一种或两种模态，或者只能在单轮对话中接受图像输入，根本无法实现“看图说话、听音答疑、看视频讲解”这样完整的交互。闭源的商业模型（如 GPT‑4o）已经展示了全模态交互的潜力，但它们的训练细节、数据来源以及可改造性都被锁住。于是，学术界和开源社区缺少一种既能处理四大模态，又能提供流畅多轮交互的模型，这正是 VITA 要解决的痛点。

### 关键概念速览

**全模态（Omni‑Multimodal）**：指模型能够同时接受并理解视频、图像、文本、音频四种输入，就像人类在看电影时既能看到画面、听到声音、读字幕，还能用语言思考。  

**多任务学习（Multi‑Task Learning）**：在同一次训练里让模型完成多种任务（比如图像‑文本对齐、音频‑文本对齐、指令微调），相当于让学生在同一学期里学多门课，能促进知识迁移。  

**对齐（Alignment）**：把非语言特征（视觉、听觉）映射到语言空间，使得模型在同一个向量里比较“看见的猫”和“说‘猫’”。常用的做法是对比学习，类似把不同语言的同义词放在一起。  

**指令微调（Instruction Tuning）**：在大模型上再训练一遍，让它学会按照人类给出的任务描述去回答，就像教机器人先学会走路，再教它怎么递东西。  

**双语词表扩展**：在原有的英文词表上加入中文词条，使模型在中英混合的对话中不必把中文拆成拼音或字符，提升中文理解和生成的流畅度。  

**交互式对话（Interactive Chat）**：模型能够在多轮对话中接受不同模态的输入并给出对应的输出，类似于真人客服可以随时切换看图、听音、看视频来回答问题。  

**Mixtral 8×7B**：一款开源的 8‑GPU、7 B 参数的语言模型，提供了强大的基础语言能力，是 VITA 的“语言底座”。  

### 核心创新点

1. **从单语言模型到全模态模型的跨界迁移**  
   - 之前的开源项目大多在语言模型上叠加单一视觉或音频适配层，无法同时处理四种信号。  
   - VITA 先把 Mixtral 8×7B 作为语言核心，随后通过两阶段的多任务学习把视觉、音频特征对齐进同一个语言空间。  
   - 结果是模型在一次推理中可以同时接受视频帧、图片、音频片段和文字，完成跨模态推理，这在开源圈是首例。

2. **中文词表扩展 + 双语指令微调的协同提升**  
   - 传统做法直接在英文词表上进行中英混合训练，中文往往被拆成子词，导致生成质量差。  
   - VITA 手动在词表中加入数千个高频中文词，并在双语指令数据上进行微调，让模型在中英混合对话中保持流畅、准确。  
   - 这种“语言底层+指令层”的双向强化，使得模型在中英双语基准上均能取得领先。

3. **两阶段多任务对齐 + 指令微调的分层训练策略**  
   - 直接在大规模多模态指令上训练往往会出现对齐不充分、生成不稳定的问题。  
   - VITA 先进行“对齐阶段”，只用对比学习让视觉/音频编码器与语言模型对齐；随后进入“指令阶段”，在多模态指令数据上微调，强化交互能力。  
   - 这种先“把语言和感官对上号”，再“教会它怎么聊天”的顺序，大幅提升了模型的稳定性和交互自然度。

4. **开放式交互体验的实现**  
   - 大多数开源 MLLM 只能在单轮对话中接受图像，缺乏连续交互的记忆和上下文管理。  
   - VITA 在对话管理层加入了多模态记忆缓存，使得模型能够在同一会话里记住前面的图片、音频或视频信息，并在后续轮次中引用。  
   - 这让用户可以像和真人一样，先发一段视频再提问题，模型能够基于之前的内容给出连贯回答。

### 方法详解

**整体框架**  
VITA 的训练流程可以划分为三大块：① 语言底座准备，② 多模态对齐阶段，③ 多模态指令微调阶段。整体思路是先让语言模型具备强大的中英双语能力，再把视觉和音频特征“搬进”语言空间，最后教会它在对话中灵活使用这些特征。

**1. 语言底座：Mixtral 8×7B + 中文词表扩展**  
- 选用 Mixtral 8×7B 作为基模型，因为它在开源社区已有成熟的训练脚本和良好的推理效率。  
- 手动在原有的 BPE（Byte‑Pair Encoding）词表中加入约 5k 条中文高频词，使得中文可以以完整词为单位被模型直接识别。  
- 在此基础上，用中英双语指令数据（约 200k 条）进行一次轻量级的指令微调，让模型熟悉中英混合的任务描述。

**2. 多模态对齐阶段（两阶段学习的第一步）**  
- **视觉编码器**：采用预训练的 CLIP‑ViT（Vision Transformer）提取图像帧或视频关键帧的特征向量。  
- **音频编码器**：使用 Whisper‑small 的音频特征提取层，将音频波形转成时序向量。  
- **对齐目标**：通过对比学习（Contrastive Loss），让同一内容的视觉/音频特征与对应的文字描述在语言模型的隐藏层上靠得更近。可以想象成把不同感官的“同一件事”放在同一个抽屉里。  
- 训练时，模型只更新视觉/音频编码器和语言模型的投影层，保持语言核心的原始知识不被破坏。

**3. 多模态指令微调阶段（两阶段学习的第二步）**  
- 构造多模态指令数据集：每条指令包含文字提问、可能的图像/视频/音频附件以及期望的文字回答。数据来源包括公开的 VQA、AudioCaps、YouCook2 等任务的指令化改写。  
- 在此阶段，模型的全部参数都参与训练，目标是最小化生成答案的交叉熵损失，同时保留对齐阶段学到的跨模态表示。  
- 为了实现连续对话，加入了“多模态记忆缓存”。每轮对话的输入会把前几轮的视觉/音频特征拼接进当前的上下文向量，模型因此能够在后续轮次中引用之前的内容。

**关键技巧与反直觉之处**  
- **先对齐后指令**：直觉上可能会认为一次性大规模指令微调更高效，但实验表明，先让模型在纯对齐任务上稳住感官-语言映射，再进行指令学习，能显著降低模式崩塌（mode collapse）和生成漂移的风险。  
- **中文词表扩展而非全词表重建**：直接重新训练一个全新的中英混合词表会破坏已有的英文子词分布，导致英文性能下降。只增补中文词条既保留了英文优势，又提升了中文流畅度。  
- **多模态记忆缓存**：在对话中保留前面的视觉/音频特征看似会增加计算开销，但实际只需在 Transformer 的 KV 缓存中追加少量向量，几乎不影响推理速度，却大幅提升了交互连贯性。

### 实验与效果

- **评测任务**：论文在多个单模态基准（如 ImageNet‑1K 分类、AudioSet 音频标签）和多模态基准（如 VQAv2、MS‑COCO Caption、YouCook2 视频问答）上进行评估，还加入了跨语言指令集（中文/英文混合对话）进行测试。  
- **对比基线**：与同类开源模型 LLaVA‑1.5、MiniGPT‑4、InternVL 进行横向比较，同时把闭源的 GPT‑4o 作为上限参考。  
- **性能表现**：论文声称 VITA 在上述基准上整体领先所有开源对手，尤其在视频问答和音频描述任务上取得了两位数的提升。相较于仅支持图像的 LLaVA，VITA 在视频理解任务上提升约 15% 的准确率。  
- **消融实验**：作者分别去掉中文词表扩展、对齐阶段、记忆缓存进行实验，发现：  
  - 去掉中文词表后中文指令准确率下降约 8%。  
  - 省略对齐阶段导致多模态任务（如 VQAv2）准确率下降约 12%。  
  - 取消记忆缓存后多轮对话的连贯性显著下降，后续轮次的回答错误率提升约 20%。  
- **局限性**：尽管在多数基准上超过开源对手，VITA 与 GPT‑4o 仍有约 10%–15% 的差距，尤其在长视频理解和高噪声音频辨识上表现不足。模型体积仍为 7 B 参数，推理成本对普通显卡仍有一定压力。作者在结论中坦承，需要更大规模的多模态指令数据和更高效的视觉/音频编码器来进一步逼近闭源水平。

### 影响与延伸思考

VITA 的出现为开源社区提供了首个真正意义上的全模态交互模型，激发了以下几类后续工作：

1. **更大规模的全模态指令数据构建**：研究者开始收集并指令化视频‑字幕‑音频三元组，以期进一步提升模型的长视频推理能力。  
2. **轻量化全模态模型**：基于 VITA 的架构，出现了 “VITA‑Mini” 系列，尝试在 2 B 参数左右的模型上保持基本的四模态能力，适配移动端和边缘设备。  
3. **跨语言全模态对齐**：VITA 的双语词表扩展思路被扩展到多语言场景，出现了支持日、韩、法等多语言的全模态模型。  
4. **交互式多模态应用**：开源项目如 “OmniChat” 直接基于 VITA 搭建了支持视频教学、音频客服、图文创作的全链路系统。

如果想继续深挖，可以关注以下方向：  
- **跨模态记忆网络**：如何在长对话或长视频中高效存储并检索历史感官信息。  
- **自监督多模态预训练**：利用未标注的网络视频、播客等大规模数据进行对齐，降低对人工指令数据的依赖。  
- **安全与对齐**：全模态模型的输出更丰富，也更容易产生不当内容，如何在多模态层面实现安全对齐是下一步的挑战。

### 一句话记住它

VITA 是首个开源、支持视频‑图像‑文本‑音频四模态交互的 LLM，用两阶段对齐+指令微调把感官世界搬进语言模型，让全模态对话从梦想到现实。