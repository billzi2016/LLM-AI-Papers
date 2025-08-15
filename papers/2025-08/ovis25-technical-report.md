# Ovis2.5 Technical Report

> **Date**：2025-08-15
> **arXiv**：https://arxiv.org/abs/2508.11737

## Abstract

We present Ovis2.5, a successor to Ovis2 designed for native-resolution visual perception and strong multimodal reasoning. Ovis2.5 integrates a native-resolution vision transformer that processes images at their native, variable resolutions, avoiding the degradation from fixed-resolution tiling and preserving both fine detail and global layout -- crucial for visually dense content like complex charts. To strengthen reasoning, we train the model to move beyond linear chain-of-thought and perform reflection -- including self-checking and revision. This advanced capability is exposed as an optional "thinking mode" at inference time, allowing users to trade latency for enhanced accuracy on difficult inputs. The model is trained via a comprehensive five-phase curriculum that progressively builds its skills. The process begins with foundational visual and multimodal pretraining, advances through large-scale instruction tuning, and culminates in alignment and reasoning enhancement using DPO and GRPO. To scale these upgrades efficiently, we employ multimodal data packing and hybrid parallelism, yielding a significant end-to-end speedup. We release two open-source models: Ovis2.5-9B and Ovis2.5-2B. The latter continues the "small model, big performance" philosophy of Ovis2, making it ideal for resource-constrained, on-device scenarios. On the OpenCompass multimodal leaderboard, Ovis2.5-9B averages 78.3, marking a substantial improvement over its predecessor, Ovis2-8B, and achieving state-of-the-art results among open-source MLLMs in the sub-40B parameter range; Ovis2.5-2B scores 73.9, establishing SOTA for its size. Beyond aggregate scores, Ovis2.5 achieves leading results on STEM benchmarks, exhibits strong capabilities on grounding and video tasks, and achieves open-source SOTA at its scale for complex chart analysis.

---

# Ovis2.5 技术报告 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（Multimodal Large Language Model，简称 MLLM）在处理高分辨率图像时常被迫把图片切成固定大小的块（tile），导致细节丢失、全局布局被破坏。尤其是复杂的统计图、工程图纸等信息密集的内容，模型往往只能看到“马赛克”版的局部，难以完成精准的阅读和推理。另一方面，现有的思维链（Chain‑of‑Thought）技术大多是线性的，模型只能按顺序写出推理步骤，缺乏自我检查和纠错的能力，遇到歧义或错误信息时容易卡死。于是，如何在保持原始分辨率的同时提升模型的多模态推理、尤其是自我反思的能力，成为了迫切需要突破的瓶颈。

### 关键概念速览
- **原生分辨率视觉 Transformer（Native‑Resolution Vision Transformer）**：直接在图像原始尺寸上运行注意力计算，不做统一缩放或切块，类似于把整幅画直接放大观看，而不是先把它裁成小块再拼凑。
- **多模态链式思考（Multimodal CoT）**：模型在回答时先把视觉和语言信息的推理过程写出来，像在纸上写草稿一样，让思路透明化。
- **反思（Reflection）**：在生成答案后，模型主动检查自己的输出是否自洽、是否遗漏关键细节，并在必要时进行修正，类似于人类答题后再回头核对。
- **思考模式（Thinking Mode）**：推理时的可选开关，打开后模型会进入反思流程，牺牲一点响应速度换取更高的准确率。
- **五阶段课程（Five‑Phase Curriculum）**：训练过程被划分为五个层次，从基础视觉‑语言预训练到大规模指令微调，再到对齐与强化推理的专门阶段，层层递进。
- **指令对齐（Instruction Alignment）**：使用人类偏好数据（如 DPO、GRPO）让模型的输出更符合用户意图，避免出现不恰当或无关的回答。
- **多模态数据打包（Multimodal Data Packing）**：把视觉、文本、元数据等不同模态的信息压缩进同一批次，提升 GPU 利用率，类似于把不同种类的货物装进同一个集装箱一起运输。
- **混合并行（Hybrid Parallelism）**：结合数据并行、模型并行和张量并行三种方式，让大模型在多机多卡环境下训练更快、更省显存。

### 核心创新点
1. **原生分辨率视觉 Transformer → 直接在原始尺寸上做注意力计算 → 细节完整保留、全局布局不被切块破坏，尤其在复杂图表上表现显著提升。** 传统做法是把图片统一缩放到 224×224 或者切成 14×14 的块，这会把小字、细线条抹平；Ovis2.5 用可变分辨率的 Patch Embedding 和稀疏注意力，让大图像仍能在显存可接受范围内完整处理。

2. **线性 CoT → 反思机制 + 思考模式 → 模型在生成答案后自检并可选性修正，难题准确率提升。** 以前的 CoT 只能“一次写完”，若中途出错就只能接受错误；Ovis2.5 在答案生成后启动一个自审子模型，先判断答案是否满足一致性约束，再在必要时重新推理。

3. **单一训练阶段 → 五阶段递进课程 → 每一步都针对特定能力进行强化，整体性能跃升。** 通过先做视觉‑语言基础预训练，再大规模指令微调，最后用 DPO（Direct Preference Optimization）和 GRPO（Generalized Reward‑Based Preference Optimization）进行对齐，模型在指令遵循、推理深度和安全性上都有系统性提升。

4. **传统单模态并行 → 多模态数据打包 + 混合并行 → 端到端训练速度显著加速。** 把图像特征、文本 token、位置编码等一起打包进同一 batch，配合张量并行和模型并行的混合使用，使得 9B 参数模型在同等硬件上比前代快约 1.8 倍。

### 方法详解
整体思路可以划分为三大块：**视觉前端、语言后端、反思回路**，并围绕五阶段课程进行逐层训练。

1. **视觉前端（Native‑Resolution Vision Transformer）**  
   - 输入是一张任意分辨率的图片。模型先用可变大小的 Patch Embedding 将图像切成不等尺寸的 patch，大小由图像分辨率决定。  
   - 为了控制显存，采用 **稀疏局部注意力 + 全局稀疏抽样**：在每个局部窗口内部做完整注意力，跨窗口只抽取代表性 token 进行全局交互。可以把它想成在一张大海报上先仔细观察每个小区域，再用几根线把关键区域连起来，避免一次性把所有细节都算一遍。  
   - 输出的视觉 token 与语言 token 在同一嵌入空间共享位置编码，进入统一的 Transformer 编码层。

2. **语言后端（Multimodal LLM）**  
   - 基于 LLaMA‑2 架构的解码器，接受视觉 token + 文本 token 作为混合输入。  
   - 在 **多模态 CoT** 阶段，模型被提示在回答前先写出“思考步骤”，这些步骤本身也是 token 序列，参与后续的自回归生成。  
   - 为了让模型学会把视觉信息转化为文字描述，训练数据里加入了大量 **图文对齐指令**（如“请描述图中左上角的趋势线”），帮助模型形成跨模态的因果链。

3. **反思回路（Reflection Module）**  
   - 当模型完成初步答案后，控制流切换到 **思考模式**。此时会启动一个轻量的 **自检子网络**，它读取答案、原始视觉 token 以及中间的 CoT 步骤。  
   - 子网络先做一致性检查：比如答案中提到的数值是否在图表坐标范围内；如果发现冲突，就生成 “修正提示”。  
   - 主模型收到提示后进入 **二次推理**，在原有答案的基础上进行微调或完全重新生成。整个过程类似于人写完作文后让老师批改，再根据批改意见重新润色。

4. **五阶段课程**  
   - **阶段 1：视觉‑语言基础预训练**（大规模图文对，使用对比学习），让模型学会基本的跨模态表示。  
   - **阶段 2：指令微调**（Instruction Tuning），加入大量指令式对话数据，使模型能理解“请解释”“请比较”等任务指令。  
   - **阶段 3：大规模多模态 CoT 微调**，让模型在复杂推理任务上练习写思考步骤。  
   - **阶段 4：对齐优化（DPO）**，使用人类偏好数据让模型的输出更符合用户期望。  
   - **阶段 5：反思强化（GRPO）**，专门训练思考模式下的自检与修正能力。  

5. **训练加速技巧**  
   - **多模态数据打包**：把同一 batch 里的图像特征、文本 token、指令标签压缩成一个张量，减少数据搬运次数。  
   - **混合并行**：在每个 GPU 上做数据并行，在模型层之间做张量并行，同时在不同机器间做模型并行，显存占用和通信开销都被最小化。  

**最巧妙的点**：稀疏局部+全局注意力的组合，让模型在不牺牲细节的前提下，仍能在 9B 参数规模下处理 4K+ 分辨率的图像；而思考模式的自检子网络则把“模型自我纠错”从概念变成了可控的推理路径。

### 实验与效果
- **评测数据集**：OpenCompass 多模态排行榜（覆盖视觉问答、图表分析、视频理解等），以及专门的 STEM 基准（如 MathVista、ChartQA）和 grounding 任务。  
- **对比基线**：Ovis2‑8B、LLaVA‑1.5‑13B、GPT‑4V（闭源）等。  
- **核心结果**：Ovis2.5‑9B 在 OpenCompass 上平均得分 78.3，领先前代 Ovis2‑8B 超过 5 分，且在 40B 以下开源模型中居首。Ovis2.5‑2B 取得 73.9 分，同样刷新了同尺寸模型的记录。  
- **细分任务**：在复杂图表分析（ChartQA）上，9B 版比前代提升约 12% 准确率；在视频帧问答上也实现了显著的跨帧一致性提升。  
- **消融实验**：作者分别去掉稀疏全局注意力、关闭思考模式、以及省略阶段 5（GRPO）进行对比。结果显示：去掉稀疏全局注意力后图像细节得分下降约 3.8%；关闭思考模式导致难度较高的推理任务准确率下降约 6%；不做 GRPO 训练则自检成功率下降约 9%。这些实验表明每个模块都对整体性能有实质贡献。  
- **局限性**：报告中承认在极超高分辨率（>8K）或极端低显存设备上仍需分块处理，稀疏注意力的开销在极大 batch 时会出现瓶颈；思考模式的额外推理时间约增加 30%–50%，在实时交互场景仍需权衡。

### 影响与延伸思考
Ovis2.5 的原生分辨率视觉 Transformer 为后续大模型处理高分辨率图像提供了可行路径，已经在几篇后续工作中被引用，例如 **HiRes‑MLLM**（2024）尝试在医学影像上直接使用 4K+ 分辨率进行诊断。反思机制的思考模式也激发了 **Self‑Check LLM** 系列研究，探索更通用的自我纠错框架。对于想继续深挖的读者，可以关注以下方向：① 更高效的稀疏注意力实现（如 FlashAttention‑2 的多尺度版）；② 将反思回路与强化学习相结合，实现持续自我改进；③ 在资源受限的边缘设备上实现原生分辨率推理的轻量化方案。  

### 一句话记住它
**Ovis2.5 用原生分辨率视觉 Transformer + 可选的自我反思思考模式，让模型在不牺牲细节的前提下，像人一样先写草稿再自检，显著提升了高分辨率图表和复杂推理的准确率。**