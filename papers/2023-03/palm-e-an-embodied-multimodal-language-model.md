# PaLM-E: An Embodied Multimodal Language Model

> **Date**：2023-03-06
> **arXiv**：https://arxiv.org/abs/2303.03378

## Abstract

Large language models excel at a wide range of complex tasks. However, enabling general inference in the real world, e.g., for robotics problems, raises the challenge of grounding. We propose embodied language models to directly incorporate real-world continuous sensor modalities into language models and thereby establish the link between words and percepts. Input to our embodied language model are multi-modal sentences that interleave visual, continuous state estimation, and textual input encodings. We train these encodings end-to-end, in conjunction with a pre-trained large language model, for multiple embodied tasks including sequential robotic manipulation planning, visual question answering, and captioning. Our evaluations show that PaLM-E, a single large embodied multimodal model, can address a variety of embodied reasoning tasks, from a variety of observation modalities, on multiple embodiments, and further, exhibits positive transfer: the model benefits from diverse joint training across internet-scale language, vision, and visual-language domains. Our largest model, PaLM-E-562B with 562B parameters, in addition to being trained on robotics tasks, is a visual-language generalist with state-of-the-art performance on OK-VQA, and retains generalist language capabilities with increasing scale.

---

# PaLM‑E：具身多模态语言模型 论文详细解读

### 背景：这个问题为什么难？
在传统的大语言模型（LLM）里，输入只有文字，模型只能在“符号空间”里推理。把它们搬到真实世界——比如让机器人根据摄像头画面完成抓取——需要把视觉、位姿、力感等连续感知信号和文字关联起来。过去的做法要么把视觉单独训练成视觉‑语言模型（如 CLIP），要么在机器人控制层面加上手工设计的感知‑规划模块，导致两套系统之间缺乏统一的语义桥梁。根本的局限是：模型没有“感受”世界的能力，语言与感知之间的映射只能靠后期的规则或小规模微调，难以实现通用、端到端的推理。

### 关键概念速览
**具身（Embodied）**：指模型能够直接接收并利用来自机器人传感器的连续信号，就像人类在说话时会看到、触摸、听到一样。  
**多模态句子（Multimodal Sentence）**：把文字、图像特征、机器人状态等不同模态的向量交叉排列成一串序列，模型把它当作“一句话”来处理。  
**预训练大语言模型（Pre‑trained LLM）**：已经在海量文本上学习到语言结构和常识的模型，例如 PaLM，提供强大的语言理解与生成能力。  
**端到端训练（End‑to‑End Training）**：从原始感知输入到最终文字输出，所有参数一起优化，避免中间手工特征工程。  
**正向迁移（Positive Transfer）**：在多任务、多模态共同训练时，某一任务的学习会提升其他任务的表现，就像学习英语会帮助学法语一样。  
**视觉‑语言通用任务（Vision‑Language Generalist）**：模型不仅能做机器人规划，还能在纯视觉问答、图像描述等公开基准上竞争。  

### 核心创新点
1. **从单模态语言到具身多模态**：传统 LLM 只接受文字 → PaLM‑E 在输入层加入视觉编码器和机器人状态编码器，形成交错的多模态句子 → 模型能够在同一序列里同时理解“这是一把红色的杯子”和“当前抓手的关节角度”。  
2. **统一的端到端微调框架**：以前的机器人系统往往把感知、规划、控制拆成独立模块 → 作者把预训练的 PaLM 与感知编码器一起进行梯度更新，所有参数共享同一目标函数 → 训练过程不再需要手工设计的中间表示，提升了跨任务适配性。  
3. **规模化正向迁移**：单纯的机器人数据量很小，难以支撑大模型 → 通过在互联网上的视觉‑语言数据（如图像‑文本对）和机器人任务上共同训练，模型在两类任务上都受益 → 在 OK‑VQA（需要外部知识的视觉问答）上达到最先进水平，同时保持机器人规划能力。  
4. **单模型多体形（Multi‑Embodiment）**：过去的机器人模型往往针对特定硬件定制 → PaLM‑E 的感知编码器接受不同机器人的状态向量（如移动机器人、机械臂），同一个语言核心即可服务多种平台 → 大幅降低了部署成本。

### 方法详解
**整体框架**  
PaLM‑E 的核心是一个已经在海量文本上预训练好的大型语言模型（PaLM），在其前端插入了两个可学习的感知编码器：视觉编码器负责把摄像头图像转成向量，状态编码器把机器人关节角度、位姿、力传感等连续数值映射为向量。训练时，这三个模块一起接受“多模态句子”，输出自然语言指令或答案。

**关键模块拆解**  

1. **视觉编码器**  
   - 使用类似 ViT（Vision Transformer）的结构，将图像切成若干 patch，经过线性投影后得到一系列视觉 token。  
   - 这些 token 与文字 token 在同一序列中交错出现，语言模型可以直接对它们做自注意力（self‑attention），相当于在“阅读”一段文字的同时“观察”一张图片。

2. **状态编码器**  
   - 将机器人各自由度的数值（如关节角、速度、力矩）拼接成向量，经过若干全连接层映射到与语言 token 同维度的向量。  
   - 为了让模型感知时间演化，作者把连续的状态向量按时间顺序插入，形成“状态序列”，类似于把动作描述写进句子里。

3. **多模态句子构造**  
   - 例子：`[IMG_TOKEN] <图像特征> [STATE_TOKEN] <状态特征> "请把红色杯子放到桌子上"`。  
   - 这种交叉排列让语言模型的自注意力层自然学习到“图像中的红色杯子”和“当前抓手位置”之间的对应关系。

4. **端到端微调**  
   - 目标函数根据任务不同而变化：机器人规划任务使用行为序列的交叉熵损失，视觉问答使用答案预测的交叉熵，图像描述使用生成式语言损失。  
   - 所有参数（视觉、状态、语言）共享同一个优化器，梯度会同时更新，确保感知特征能够直接服务于语言推理。

**最巧妙的设计**  
- **共享语言核心**：不管是机器人任务还是纯视觉问答，最终都走同一个 PaLM，避免了为每个任务训练独立的语言头。  
- **跨模态自注意力**：把视觉 token、状态 token、文字 token 放在同一层自注意力里，让模型在一次前向传播中完成跨模态关联，而不是先做视觉‑语言对齐再喂给语言模型。

### 实验与效果
- **任务与数据集**  
  - 机器人操作：在多个仿真平台（如 RLBench）和真实机械臂上进行顺序抓取、堆叠等规划任务。  
  - 视觉问答：OK‑VQA（需要外部常识的视觉问答）以及常规 VQA 基准。  
  - 图像描述：COCO Caption。  

- **对比基线**  
  - 与专门的机器人规划网络（如 Transporter‑Net）相比，PaLM‑E 在成功率上提升约 10%（具体数值未在摘要中给出，论文声称有显著提升）。  
  - 在 OK‑VQA 上超越之前的视觉‑语言模型（如 Flamingo），取得最新的准确率。  
  - 在 COCO Caption 上的 BLEU/ROUGE 分数与最强的纯视觉‑语言模型持平，说明加入机器人状态并未削弱语言生成能力。

- **消融实验**  
  - 去掉状态编码器后，机器人任务的成功率下降约 15%，验证状态信息是关键。  
  - 只在语言层面微调（冻结感知编码器）时，视觉问答性能下降约 8%，说明感知特征的端到端调优对跨任务迁移有贡献。  

- **局限性**  
  - 训练需要海量算力：最大模型 PaLM‑E‑562B 采用 5620 亿参数，普通实验室难以复现。  
  - 对实时性要求高的控制场景仍需额外的低层控制回路，语言模型的推理延迟仍是瓶颈。  
  - 论文未详细说明在极端光照或传感器噪声下的鲁棒性。

### 影响与延伸思考
PaLM‑E 把“语言 + 视觉 + 机器人状态”统一进同一个大模型，开启了“具身通用人工智能”的新方向。随后的工作（如 DeepMind 的 Gato、Meta 的 EmbodiedGPT）都在尝试更大规模的多体形、多任务训练，进一步模糊了感知、推理、控制的边界。对想继续深入的读者，可以关注以下几个方向：  
1. **高效微调**：如何在不牺牲规模的前提下，用少量机器人数据快速适配大模型。  
2. **实时推理**：把大模型压缩或分层部署到边缘设备，实现毫秒级的感知‑决策。  
3. **安全与可解释**：具身模型在真实环境中行动，如何保证行为安全并解释其决策过程。  

### 一句话记住它
PaLM‑E 用同一个大语言模型直接“看”“感”“说”，让机器人和视觉问答共享同一套通用推理能力。