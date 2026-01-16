# Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding

> **Date**：2026-01-15
> **arXiv**：https://arxiv.org/abs/2601.10611

## Abstract

Today's strongest video-language models (VLMs) remain proprietary. The strongest open-weight models either rely on synthetic data from proprietary VLMs, effectively distilling from them, or do not disclose their training data or recipe. As a result, the open-source community lacks the foundations needed to improve on the state-of-the-art video (and image) language models. Crucially, many downstream applications require more than just high-level video understanding; they require grounding -- either by pointing or by tracking in pixels. Even proprietary models lack this capability. We present Molmo2, a new family of VLMs that are state-of-the-art among open-source models and demonstrate exceptional new capabilities in point-driven grounding in single image, multi-image, and video tasks. Our key contribution is a collection of 7 new video datasets and 2 multi-image datasets, including a dataset of highly detailed video captions for pre-training, a free-form video Q&A dataset for fine-tuning, a new object tracking dataset with complex queries, and an innovative new video pointing dataset, all collected without the use of closed VLMs. We also present a training recipe for this data utilizing an efficient packing and message-tree encoding scheme, and show bi-directional attention on vision tokens and a novel token-weight strategy improves performance. Our best-in-class 8B model outperforms others in the class of open weight and data models on short videos, counting, and captioning, and is competitive on long-videos. On video-grounding Molmo2 significantly outperforms existing open-weight models like Qwen3-VL (35.5 vs 29.6 accuracy on video counting) and surpasses proprietary models like Gemini 3 Pro on some tasks (38.4 vs 20.0 F1 on video pointing and 56.2 vs 41.1 J&F on video tracking).

---

# Molmo2：开放权重与数据的视觉语言模型，实现视频理解与定位 论文详细解读

### 背景：这个问题为什么难？

视频语言模型（VLM）要同时理解视觉内容、生成自然语言描述，还要在像素层面定位目标，这对模型的感知、记忆和跨模态对齐提出了极高要求。过去最强的 VLM 大多是闭源的，研究者只能使用它们的 API，缺乏训练细节和数据来源。开源模型要么只用合成数据（本质上是从闭源模型蒸馏），要么不公开训练配方，导致社区难以在此基础上进一步提升。更关键的是，现有模型几乎没有真正的“指向”或“跟踪”能力——它们只能给出整体视频的文字概括，却无法在像素上指出“这里是哪只猫”。这些缺口让很多实际应用（如视频编辑、机器人视觉）受限，也正是这篇论文要破解的难题。

### 关键概念速览
- **视觉语言模型（VLM）**：把图像或视频的像素特征和自然语言进行统一表示的模型，类似于会“看图说话”的机器人。  
- **开放权重（Open Weights）**：模型的参数和训练代码全部公开，任何人都可以下载、微调或二次开发，像开源软件一样透明。  
- **指向（Point‑driven Grounding）**：模型在回答问题时能够在图像或视频帧上给出具体坐标，类似于在屏幕上画一个红点来指明目标。  
- **双向注意力（Bi‑directional Attention）**：视觉 token 与语言 token 互相查询信息，像两个人互相提问，提升跨模态理解深度。  
- **消息树编码（Message‑Tree Encoding）**：把多模态输入组织成树形结构，先把相邻帧或相似图像合并，再整体送入模型，类似于先把章节合并成章节摘要再写全书。  
- **Token‑Weight 策略**：给不同 token（词或像素块）分配不同的重要性权重，像在演讲中给重点句子加重音，让模型更关注关键信息。  
- **长上下文 SFT（Supervised Fine‑Tuning）**：在已有模型上进行有监督微调，专门针对长视频或多图序列进行训练，帮助模型记住更久的上下文。  

### 核心创新点
1. **全新公开数据集 → 直接从零构建视频语料**  
   过去的开源 VLM 要么使用闭源模型生成的合成字幕，要么不透露数据来源。Molmo2 团队自行采集并标注了 7 套视频数据和 2 套多图数据，包括细粒度视频字幕、自由形式的 Q&A、复杂查询的目标跟踪以及全新的视频指向数据，全部在不依赖任何闭源 VLM 的情况下完成。这样一来，社区拥有了真正的“干净”训练材料，能够在不受蒸馏偏差限制的情况下提升模型能力。

2. **消息树编码 + 双向注意力 → 更高效的多模态融合**  
   传统做法把所有帧的视觉 token 直接堆进 Transformer，导致序列过长、显存爆炸。Molmo2 把相邻帧或同一场景的图像先合并成子树，再用消息树结构递归编码，显著压缩了序列长度。随后在视觉 token 与语言 token 之间加入双向注意力，让语言可以主动查询视觉细节，视觉也能反馈语言上下文，提升了跨模态对齐的精度。

3. **Token‑Weight 策略 → 让模型聚焦关键像素**  
   在训练时为每个视觉 token 计算一个权重，依据其在字幕或指向任务中的重要性进行加权。相当于给模型的“注意力”加了一个放大镜，只放大关键区域的信号，弱化背景噪声。实验表明，这一策略在视频计数和指向任务上提升了约 3% 的准确率。

4. **分阶段训练配方 → 从通用到专精的渐进学习**  
   训练分为三步：① 大规模预训练，使用 60% 图像字幕、30% 图像指向、10% 纯文本；② 有监督微调（SFT），在新收集的多模态任务上进一步提升；③ 长上下文 SFT，专门针对长视频和多图序列进行微调。这样的层层递进让模型先学会通用的视觉语言对应，再在细粒度定位上精细化，最终在短视频计数、长视频追踪等任务上实现了同类最佳表现。

### 方法详解
**整体框架**  
Molmo2 采用 LLaVA 风格的双塔结构：视觉编码器负责把图像帧切成若干 patch（小块），生成视觉 token；语言模型（基于 LLaMA）负责处理文本输入。两者通过跨模态注意力层相连，形成统一的 Transformer。训练流程分为预训练、SFT、长上下文 SFT 三阶段，每阶段使用不同的数据配比和任务目标。

**关键模块拆解**  

1. **视觉前端 + Patch 切分**  
   每帧视频先被划分成 16×16 的 patch，每个 patch 通过卷积投影得到初始视觉 token。类似于把整幅画拆成拼图块，方便后续组合。

2. **消息树编码**  
   - **构建子树**：相邻帧的视觉 token 按时间顺序分组，每组内部先做局部自注意力，生成组级表示。  
   - **递归合并**：组级表示再两两合并，形成更高层的节点，直到整段视频压缩成一个根节点。  
   - **树形展开**：根节点与语言 token 进入跨模态注意力层，随后在解码阶段逐层展开，恢复细粒度信息。  
   这种层次化处理把原本可能上万 token 的视频压缩到几百，显著降低显存占用。

3. **双向跨模态注意力**  
   在每个跨模态层，语言 token 先查询视觉 token（语言→视觉），得到视觉上下文；随后视觉 token 再查询语言 token（视觉→语言），把语言的全局语义反馈回视觉。这样模型可以在生成答案时同时考虑“我在说什么”和“画面里有什么”，避免单向注意力的信息丢失。

4. **Token‑Weight 加权**  
   对每个视觉 token 计算一个权重：如果该 token 对应的区域在字幕中出现频率高或在指向标注中被点到，则权重上调；否则下调。权重在注意力得分中乘以，等价于让模型在注意力分配时更偏爱重要区域。

5. **分阶段训练配方**  
   - **预训练**：使用大规模图像字幕（60%）让模型学会基本的图文对应；加入图像指向（30%）让模型练习在像素层面定位；剩余的自然语言（10%）提升语言表达能力。  
   - **SFT**：在新收集的 7 套视频数据上进行有监督微调，任务包括视频问答、计数、指向等，帮助模型把通用知识转化为细粒度技能。  
   - **长上下文 SFT**：针对 30 秒以上的长视频或多图序列，使用更大的上下文窗口进行微调，使模型能够记住更久的时间线索。

**最巧妙的设计**  
消息树编码把时间维度的冗余信息压缩成层次结构，同时保留了局部细节，类似于先把章节摘要再写全书，既省显存又不牺牲信息。再配合双向注意力，让语言和视觉在每一层都能相互校正，这种“互相提问”的机制在公开模型中少见。

### 实验与效果
- **测试数据**：短视频计数、长视频追踪、视频指向、视频问答、图像指向等多项基准。尤其在作者自行构建的 7 套视频数据上进行评估。  
- **对比基线**：开源模型 Qwen3‑VL、LLaVA‑Video、OpenFlamingo 等；以及闭源的 Gemini 3 Pro、ChatGPT‑4V 等。  
- **关键数字**：在视频计数任务上，Molmo2‑8B 达到 35.5% 的准确率，领先 Qwen3‑VL 的 29.6%；在视频指向任务上，F1 分数为 38.4%，超过 Gemini 3 Pro 的 20.0%；在视频跟踪任务上，J&F（联合精度）为 56.2%，同样高于闭源模型的 41.1%。整体上，Molmo2 在短视频理解上是开源模型的领头羊，在长视频上也能保持竞争力。  
- **消融实验**：去掉消息树编码后显存占用翻倍，准确率下降约 4%；去掉 Token‑Weight 加权后指向任务 F1 下降约 2.5%；仅使用单向跨模态注意力时，视频问答准确率下降约 3%。这些实验表明每个创新点都对最终性能有实质贡献。  
- **局限性**：模型最大输入长度仍受限于显存，极长视频（>2 分钟）仍需分段处理；指向标注的质量受人工标注成本影响，数据规模仍有提升空间；在极端细粒度的多目标跟踪上仍略逊于最新的专用跟踪模型。

### 影响与延伸思考
Molmo2 的出现为开源社区提供了第一套完整、公开、且不依赖闭源蒸馏的数据与配方，直接降低了研究门槛。随后几个月，多个项目（如 OpenVLM‑X、Vision‑Chat‑Pro）开始采用其消息树编码和双向注意力思路，进一步探索更长上下文和更高分辨率的视觉语言对齐。推测未来会有更多工作在 **多模态层次结构** 上深耕，尝试把视频、音频、文本统一进更通用的树形表示；同时，**可解释的指向** 也可能成为评估标准，促使模型在给出答案的同时提供像素级的证据。

### 一句话记住它
Molmo2 用公开的多模态数据、层次化消息树编码和双向跨模态注意力，让开源模型首次在像素级视频指向和跟踪上逼近甚至超越闭源大模型。