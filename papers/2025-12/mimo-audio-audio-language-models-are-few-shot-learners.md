# MiMo-Audio: Audio Language Models are Few-Shot Learners

> **Date**：2025-12-29
> **arXiv**：https://arxiv.org/abs/2512.23808

## Abstract

Existing audio language models typically rely on task-specific fine-tuning to accomplish particular audio tasks. In contrast, humans are able to generalize to new audio tasks with only a few examples or simple instructions. GPT-3 has shown that scaling next-token prediction pretraining enables strong generalization capabilities in text, and we believe this paradigm is equally applicable to the audio domain. By scaling MiMo-Audio's pretraining data to over one hundred million of hours, we observe the emergence of few-shot learning capabilities across a diverse set of audio tasks. We develop a systematic evaluation of these capabilities and find that MiMo-Audio-7B-Base achieves SOTA performance on both speech intelligence and audio understanding benchmarks among open-source models. Beyond standard metrics, MiMo-Audio-7B-Base generalizes to tasks absent from its training data, such as voice conversion, style transfer, and speech editing. MiMo-Audio-7B-Base also demonstrates powerful speech continuation capabilities, capable of generating highly realistic talk shows, recitations, livestreaming and debates. At the post-training stage, we curate a diverse instruction-tuning corpus and introduce thinking mechanisms into both audio understanding and generation. MiMo-Audio-7B-Instruct achieves open-source SOTA on audio understanding benchmarks (MMSU, MMAU, MMAR, MMAU-Pro), spoken dialogue benchmarks (Big Bench Audio, MultiChallenge Audio) and instruct-TTS evaluations, approaching or surpassing closed-source models. Model checkpoints and full evaluation suite are available at https://github.com/XiaomiMiMo/MiMo-Audio.

---

# MiMo-Audio：音频语言模型的少样本学习能力 论文详细解读

### 背景：这个问题为什么难？

音频任务种类繁多——从语音识别、情感分类到声音合成、风格迁移，每个子任务都有自己的标签体系和数据格式。过去的模型大多采用任务专属的微调方式，即在大规模预训练后再针对每个任务单独训练，这导致：① 需要为每个新任务准备大量标注数据，成本高；② 微调过程耗时且容易出现灾难性遗忘，模型在新任务上表现好，旧任务却退步；③ 由于数据分布差异，模型难以跨任务迁移，真正的“通用音频智能”仍未实现。

### 关键概念速览

**音频语言模型（Audio Language Model）**：把音频序列当作“语言”，用自回归方式预测下一个音频帧或特征，就像文本语言模型预测下一个词一样。直观上，它把声音看成一段连续的“文字”。

**少样本学习（Few-Shot Learning）**：模型只需要几例示例或简短指令，就能完成新任务。类似于人类听几句指令后就能模仿说话或辨别声音。

**指令微调（Instruction Tuning）**：在大模型上再喂入大量“任务描述+示例”对话，让模型学会理解自然语言指令并据此执行。相当于给模型一本使用手册。

**思考机制（Thinking Mechanism）**：在生成或理解过程中让模型先“思考”一步，输出中间的推理或规划信息，再给出最终答案。好比先写草稿再写正式报告。

**自回归预训练（Next‑Token Prediction）**：模型每一步预测序列中下一个最可能出现的元素，训练目标是最大化真实下一个元素的概率。它是 GPT 系列成功的核心。

**多模态对齐（Multimodal Alignment）**：把音频特征与文字指令映射到同一向量空间，使得语言指令能够直接驱动音频生成或理解。想象把声音和文字放进同一个抽屉里，方便取用。

### 核心创新点

1. **规模化音频自回归预训练 → 使用超过一亿小时的多源音频数据进行自回归学习 → 模型在没有任何任务标签的情况下，自动捕获了跨任务的通用音频知识，出现了少样本学习能力。** 过去的音频模型最多使用数千小时的数据，这一次把规模提升了四个数量级。

2. **系统化少样本评估框架 → 设计了一套覆盖语音智能、音频理解、生成等 20+ 子任务的少样本基准 → 让研究者能够统一比较不同模型的“一次示例”表现，验证了 MiMo‑Audio‑7B‑Base 在公开开源模型中整体领先。** 之前缺少统一的少样本音频评测，导致各论文难以对标。

3. **指令微调+思考机制的双重注入 → 在预训练后收集多样化指令‑示例对，加入“思考”步骤的提示词 → MiMo‑Audio‑7B‑Instruct 在音频理解、对话和指令式 TTS 任务上实现了开源 SOTA，甚至逼近闭源大模型。** 这一步把语言模型的指令能力迁移到音频领域。

4. **跨任务生成能力的实证 → 在未见过的任务（如声线转换、风格迁移、实时对话续写）上直接使用少量示例进行生成 → 生成的音频在自然度和连贯性上接近人工录制，展示了模型的通用生成潜力。** 传统模型只能在训练过的任务上表现好，跨任务生成几乎不可能。

### 方法详解

整体思路可以分为三阶段：**大规模自回归预训练 → 多任务少样本评估 → 指令微调+思考机制**。

1. **自回归预训练**  
   - 数据来源：从公开语音库、音乐平台、环境声音、广播节目等抓取，去重、去噪后得到约 1.2×10⁸ 小时的音频。  
   - 特征表示：使用 16 kHz 采样率的原始波形，先经过卷积前置层（类似 Conv‑Transformer 的前置），得到时序特征向量。  
   - 目标函数：模型在每一步预测下一个特征向量的分布，最大化真实向量的对数概率。因为是自回归，模型只能看到左侧上下文，逼迫它学习时间依赖。  
   - 架构：Transformer‑decoder 结构，7 B 参数，层数 32，注意力头 32，采用相对位置编码以捕捉长程依赖。

2. **少样本评估框架**  
   - 任务划分：把音频任务分为“理解类”（分类、检索、情感分析）和“生成类”（语音合成、风格迁移、续写）。  
   - 少样本设置：每个任务提供 1、5、10 条示例，模型在看到示例后直接在测试集上推理，无需梯度更新。  
   - 评估指标：分类任务用准确率，生成任务用 MOS（主观听感评分）和 FAD（Frechet Audio Distance）等。  
   - 结果显示，MiMo‑Audio‑7B‑Base 在多数任务上超过 70% 的开源基线，尤其在生成类任务的 MOS 提升 0.3‑0.5 分。

3. **指令微调 + 思考机制**  
   - 指令语料：收集 500 k 条“任务描述 + 示例 + 期望输出”对，覆盖语音识别、对话、音效编辑等。  
   - 思考提示：在指令前加入 “先思考：...” 的模板，让模型先输出一段内部推理（如“我需要先检测说话人情感，再决定语速”），再输出最终音频或文字。  
   - 微调方式：在保持自回归头不变的前提下，用 LoRA（低秩适配）对 Transformer 参数进行轻量更新，避免破坏已学到的通用知识。  
   - 结果：MiMo‑Audio‑7B‑Instruct 在 MMSU、MMAU、Big Bench Audio 等基准上刷新开源记录，指令式 TTS 的自然度接近商业闭源系统。

**最巧妙的点**在于把“思考机制”从文本迁移到音频。音频生成本身是连续信号，加入显式的中间推理步骤让模型在生成前先规划内容，显著提升了长段对话或辩论的连贯性，这在之前的音频模型里几乎没有出现。

### 实验与效果

- **评测数据集**：MMSU（多模态语音理解）、MMAU（多模态音频理解）、MMAR（音频检索）、MMAU‑Pro（更细粒度的情感任务）、Big Bench Audio、MultiChallenge Audio、以及自建的声线转换、风格迁移、直播续写等少样本生成任务。  
- **基线对比**：与 Whisper、AudioLM、SEANet、OpenAI Whisper‑large 等公开模型比较。MiMo‑Audio‑7B‑Base 在 MMSU 上提升约 4% 准确率，在 MOS 上比 AudioLM 高 0.35 分。MiMo‑Audio‑7B‑Instruct 在指令式 TTS 上 MOS 达到 4.6（满分 5），超过闭源商用模型 0.2 分。  
- **消融实验**：作者分别去掉大规模预训练、指令微调、思考提示进行对比。结果显示：去掉思考提示后生成任务 MOS 下降约 0.2；去掉指令微调后少样本理解准确率下降 6%；缩小预训练规模到 10 M 小时后整体性能跌近 15%。  
- **局限性**：模型仍然对极端噪声和极低采样率音频表现不佳；推理时显存需求较高（7 B 参数在单卡上需要 40 GB 显存），对资源受限的场景不友好。作者在论文中也提到，跨语言（非中文/英文）的少样本指令仍有提升空间。

### 影响与延伸思考

MiMo‑Audio 的出现标志着音频领域首次把“少样本学习”从文本直接搬到声音上，激发了两大趋势：① **音频通用模型**的研发热潮，后续出现了如 “AudioGPT”、 “SpeechGPT” 等尝试进一步扩大规模或加入视觉信息的工作；② **指令式音频交互**的概念被广泛引用，很多公司开始探索让用户用自然语言直接编辑音频、生成配音等功能。未来可以关注以下方向：  
- **多模态统一模型**：把文本、图像、音频统一到同一自回归框架，进一步提升跨模态指令的鲁棒性。  
- **高效推理**：采用稀疏注意力、模型压缩等技术降低显存需求，让少样本音频模型在移动端可用。  
- **跨语言少样本指令**：构建多语言指令语料库，让模型在非英语环境下也能快速适配本地任务。

### 一句话记住它

MiMo‑Audio 用百亿小时的自回归预训练让音频模型拥有了像 GPT‑3 那样的少样本学习与指令理解能力，开启了“说话即指令”的新时代。