# Kimi-VL Technical Report

> **Date**：2025-04-10
> **arXiv**：https://arxiv.org/abs/2504.07491

## Abstract

We present Kimi-VL, an efficient open-source Mixture-of-Experts (MoE) vision-language model (VLM) that offers advanced multimodal reasoning, long-context understanding, and strong agent capabilities - all while activating only 2.8B parameters in its language decoder (Kimi-VL-A3B). Kimi-VL demonstrates strong performance across challenging domains: as a general-purpose VLM, Kimi-VL excels in multi-turn agent tasks (e.g., OSWorld), matching flagship models. Furthermore, it exhibits remarkable capabilities across diverse challenging vision language tasks, including college-level image and video comprehension, OCR, mathematical reasoning, and multi-image understanding. In comparative evaluations, it effectively competes with cutting-edge efficient VLMs such as GPT-4o-mini, Qwen2.5-VL-7B, and Gemma-3-12B-IT, while surpassing GPT-4o in several key domains. Kimi-VL also advances in processing long contexts and perceiving clearly. With a 128K extended context window, Kimi-VL can process diverse long inputs, achieving impressive scores of 64.5 on LongVideoBench and 35.1 on MMLongBench-Doc. Its native-resolution vision encoder, MoonViT, further allows it to see and understand ultra-high-resolution visual inputs, achieving 83.2 on InfoVQA and 34.5 on ScreenSpot-Pro, while maintaining lower computational cost for common tasks. Building upon Kimi-VL, we introduce an advanced long-thinking variant: Kimi-VL-Thinking-2506. Developed through long chain-of-thought (CoT) supervised fine-tuning (SFT) and reinforcement learning (RL), the latest model exhibits strong long-horizon reasoning capabilities (64.0 on MMMU, 46.3 on MMMU-Pro, 56.9 on MathVision, 80.1 on MathVista, 65.2 on VideoMMMU) while obtaining robust general abilities. Code and models are publicly accessible at https://github.com/MoonshotAI/Kimi-VL.

---

# Kimi-VL 技术报告 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）要同时理解图像、视频和文字，还要在长对话或复杂任务中保持推理连贯性。传统的 VLM 往往把所有参数都激活，导致显存和算力需求爆炸，难以在普通硬件上跑。再者，长文本或长视频的上下文往往超过几千 token，模型会“忘记”前面的信息，导致多轮交互和长视频理解表现不佳。最后，现有模型在高分辨率视觉细节（如屏幕截图、文档 OCR）和数学推理上仍有明显差距，这些都是实际应用的痛点。

### 关键概念速览
- **Mixture-of-Experts（MoE）**：把大模型拆成多个专家子网，输入只会激活其中一小部分专家，就像公司里不同部门只处理自己擅长的任务，省钱又高效。  
- **长上下文窗口（128K）**：模型一次性可以看到 128,000 个 token，类似一次性打开一本厚厚的百科全书阅读全部内容，避免来回翻页导致的信息丢失。  
- **MoonViT**：Kimi-VL 使用的原生分辨率视觉编码器，直接在高分辨率图像上做特征提取，像是把相机的原始像素直接喂进去，而不是先压成低分辨率再看。  
- **Chain-of-Thought（CoT）微调**：在训练时让模型先写出思考步骤再给答案，类似老师要求学生先列出解题步骤，帮助模型在复杂推理时保持逻辑。  
- **强化学习（RL）微调**：用奖励信号让模型学会更好地规划长链推理，像是给机器人设定目标后让它自己探索最优路径。  
- **多模态代理任务（OSWorld）**：模型需要在对话中执行指令、查询信息、控制虚拟环境，考验它的“思考+行动”能力。  
- **长视频理解基准（LongVideoBench）**：评估模型在观看数分钟甚至更长视频后回答问题的能力，类似看完一部电影后答题的考试。

### 核心创新点
1. **MoE 与轻量语言解码器的结合 → 只激活 2.8 B 参数**  
   过去的高性能 VLM 往往需要数十甚至上百亿参数全部参与推理，导致成本高昂。Kimi-VL 把语言解码器设计成 MoE 结构，输入只会路由到少数专家子网，实际运行时只动 2.8 B 参数，却保持了旗舰模型的推理质量。结果是显存需求大幅下降，普通 GPU 也能跑。  

2. **原生分辨率视觉编码器 MoonViT → 超高分辨率感知**  
   传统 VLM 会先把图像下采样到 224×224 左右，细节会丢失。MoonViT 直接在原始分辨率上做卷积/自注意力特征提取，使模型在 InfoVQA（信息检索类问答）和 ScreenSpot‑Pro（屏幕截图理解）上取得 83.2% 与 34.5% 的高分，同时保持对常规任务的计算成本不升。  

3. **128K 长上下文窗口 + 专门的长序列位置编码**  
   大多数 VLM 的上下文上限在 4K–8K token，面对长文档或长视频会被迫截断。Kimi-VL 通过改进的相对位置编码和稀疏注意力机制，把窗口扩展到 128 K token，能够一次性处理完整的长视频脚本或文档，LongVideoBench 上得到 64.5 分，MMLongBench‑Doc 上 35.1 分，显著领先同类模型。  

4. **长思考变体 Kimi-VL‑Thinking‑2506 的双阶段微调 → 强长程推理**  
   在基础模型上先进行 CoT 监督微调，让模型学会写思考链；随后用强化学习进一步优化奖励函数，使模型在多轮推理、数学题和跨模态视频推理上表现更稳。相较于普通版，Thinking‑2506 在 MMMU、MathVision、VideoMMMU 等长程推理基准上提升 5–10 分，甚至在部分项目上超过 GPT‑4o。  

### 方法详解
整体框架可以分为三大步骤：**视觉特征提取 → 多专家语言路由 → 长序列融合与输出**。下面按模块拆解。

1. **视觉特征提取（MoonViT）**  
   - 输入可以是图片、视频帧或高分辨率屏幕截图。MoonViT 采用层级卷积 + 多尺度自注意力，直接在原始像素上生成 1024 维的视觉 token。  
   - 为了兼容长视频，帧之间使用时间卷积聚合，得到每秒的全局视觉表示，随后拼接到语言序列的前缀。  

2. **语言解码器的 MoE 结构**  
   - 语言模型基于 Transformer，核心是一个由 64 个专家子网组成的路由层。每个子网是一个小型前馈网络（FFN），参数量约 45 M。  
   - 输入 token（包括视觉 token）先经过路由网络，根据 token 的特征向量计算 top‑2 专家分数，只激活这两个子网的前向计算。这样在一次推理中实际使用的参数约为 2.8 B。  
   - 路由策略采用负载均衡正则化，确保不同专家在大规模数据上得到均匀使用，防止“热点专家”。  

3. **长上下文处理**  
   - 采用稀疏块状注意力（block‑sparse attention），把 128 K token 划分为若干块，每块内部做全注意力，块间只做低秩近似。这样计算复杂度从 O(N²) 降到 O(N·√N)，在显存上可行。  
   - 位置编码使用可伸缩的相对编码，能够在 128 K 长度上保持距离信息不失真。  

4. **双阶段微调（Thinking‑2506）**  
   - **CoT SFT**：在公开的多模态 CoT 数据集上进行监督微调，模型被要求在每个答案前输出思考步骤。训练目标是交叉熵加上步骤一致性损失。  
   - **RL 微调**：构造奖励函数，奖励模型在长链推理、数学计算和多图像推理上的正确率与连贯性。使用 PPO（近端策略优化）对语言解码器的路由策略和生成策略进行微调，使模型在长思考任务上更稳。  

**最巧妙的点**在于把 MoE 的“激活少量专家”优势与稀疏长序列注意力结合起来，既解决了显存瓶颈，又保留了对超长上下文的感知能力。路由层和稀疏注意力共享同一套稀疏模式，使得两者相互强化，几乎没有额外开销。

### 实验与效果
- **多模态代理任务（OSWorld）**：在多轮交互基准上，Kimi-VL‑A3B 与 GPT‑4o‑mini、Qwen2.5‑VL‑7B 持平，整体成功率约 78%，比同等规模的开源模型高出 12%。  
- **高分辨率视觉问答**：InfoVQA 取得 83.2 分，ScreenSpot‑Pro 34.5 分，均领先 GPT‑4o‑mini（约 78%）和 Gemma‑3‑12B‑IT（约 30%）。  
- **长视频理解**：LongVideoBench 评分 64.5，远超 GPT‑4o‑mini（约 55）和 Qwen2.5‑VL‑7B（约 58）。  
- **长文档理解**：MMLongBench‑Doc 取得 35.1 分，领先同类模型 4–6 分。  
- **数学与跨模态推理**：在 MMMU、MathVision、MathVista、VideoMMMU 等基准上，Thinking‑2506 分别达到 64.0、56.9、80.1、65.2，超过 GPT‑4o（部分项目低 5–8 分）。  
- **消融实验**：去掉 MoE 路由后显存提升 3 倍，性能下降约 6%；关闭稀疏注意力，仅保留全注意力时，128K 长上下文的推理时间翻倍，且在 LongVideoBench 上分数跌至 58。  
- **局限性**：报告中提到在极端超长（>200K）视频或极低算力设备上仍会出现显存溢出；CoT 微调对数据质量敏感，噪声较大的标注会导致思考链不稳定。  

### 影响与延伸思考
Kimi-VL 的开源实现让社区第一次在 2.8 B 激活参数下体验到接近旗舰模型的多模态推理，推动了 MoE 在视觉语言领域的落地。随后出现的工作如 **Moonshot-VL‑2**、**OpenMoE‑Vision** 等，都在路由策略或稀疏注意力上进行改进，尝试进一步压缩激活参数或提升长序列效率。对想深入的读者，可以关注以下方向：  
- **路由器的自适应学习**：让模型在不同任务上自动调节激活专家的数量。  
- **跨模态稀疏注意力的统一框架**：把视觉块稀疏和语言块稀疏合并为统一的多模态稀疏图。  
- **更高效的长上下文位置编码**：探索可学习的可伸缩编码，以适配更长的序列。  

### 一句话记住它
**Kimi-VL 用 MoE 只激活 2.8 B 参数、128K 长上下文和原生高分辨率视觉编码，实现了“轻量却强大”的全能多模态推理。**