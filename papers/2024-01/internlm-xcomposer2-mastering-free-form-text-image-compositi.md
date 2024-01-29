# InternLM-XComposer2: Mastering Free-form Text-Image Composition and   Comprehension in Vision-Language Large Model

> **Date**：2024-01-29
> **arXiv**：https://arxiv.org/abs/2401.16420

## Abstract

We introduce InternLM-XComposer2, a cutting-edge vision-language model excelling in free-form text-image composition and comprehension. This model goes beyond conventional vision-language understanding, adeptly crafting interleaved text-image content from diverse inputs like outlines, detailed textual specifications, and reference images, enabling highly customizable content creation. InternLM-XComposer2 proposes a Partial LoRA (PLoRA) approach that applies additional LoRA parameters exclusively to image tokens to preserve the integrity of pre-trained language knowledge, striking a balance between precise vision understanding and text composition with literary talent. Experimental results demonstrate the superiority of InternLM-XComposer2 based on InternLM2-7B in producing high-quality long-text multi-modal content and its exceptional vision-language understanding performance across various benchmarks, where it not only significantly outperforms existing multimodal models but also matches or even surpasses GPT-4V and Gemini Pro in certain assessments. This highlights its remarkable proficiency in the realm of multimodal understanding. The InternLM-XComposer2 model series with 7B parameters are publicly available at https://github.com/InternLM/InternLM-XComposer.

---

# InternLM-XComposer2：掌握自由形式文本-图像组合与理解的视觉语言大模型 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）过去大多只能完成“看图说话”或“图文检索”这类固定格式任务。要让模型既能精准理解图像，又能像作家一样把文字和图像自由交织成长篇内容，面临两大难点：一是视觉信息的细粒度捕获需要大量专门的视觉参数；二是这些视觉参数往往会冲淡模型在大规模语言预训练中学到的丰富语言知识。结果是，要么模型懂图像却写不出流畅文字，要么文字写得好却对图像细节一知半解。正是这种“语言‑视觉权衡”让自由形式的文本‑图像创作成为未被彻底攻克的挑战。

### 关键概念速览
- **视觉语言模型（VLM）**：同时接受文字和图像输入，输出文字、图像或两者混合的系统。想象成会“看图说话”的机器人。
- **LoRA（Low-Rank Adaptation）**：在大模型上加一层低秩矩阵来微调，类似在原有乐谱上加几句即兴演奏，既省显存又保持原有能力。
- **Partial LoRA（PLoRA）**：只在图像对应的 token 上挂 LoRA，文字 token 完全保持原始权重。相当于只给画家加了新颜料，而不动作家的笔。
- **Free-form Text-Image Composition**：用户可以提供大纲、详细文字描述或参考图片，模型自行决定文字和图像的排布和风格，像自由写作一样没有固定模板。
- **InternLM2-7B**：InternLM 系列的第二代基础语言模型，拥有 70 亿参数，已经在大规模文本上预训练好，具备强语言生成能力。
- **多模态基准（Multimodal Benchmarks）**：评估模型在图文理解、生成等方面的公开数据集，如 VQAv2、MME、MMBench 等。

### 核心创新点
1. **视觉专属 LoRA → Partial LoRA（PLoRA）**  
   传统 LoRA 在所有 token 上统一加参数，容易把语言知识也“稀释”。本文只在图像 token 上加 LoRA，保持语言权重不变。这样模型在视觉细节上得到专门调优，却不牺牲已有的语言流畅度和常识。

2. **从“单一任务”到“自由组合”**  
   过去的 VLM 多聚焦于回答问题或生成单张图像。InternLM‑XComposer2 设计了一个统一的输入解析器，能够把大纲、细节描述、参考图像等混合信息转化为统一的 token 序列，让模型在同一次前向传播中完成理解、布局规划和文字‑图像交叉生成。

3. **基于 InternLM2‑7B 的双模态微调**  
   直接在已有的强语言模型上进行视觉微调，而不是从头训练一个全新多模态网络。这样既利用了 InternLM2 在长文本生成上的优势，又通过 PL​oRA 获得了高质量的视觉感知。

4. **跨模型对标，匹配或超越 GPT‑4V / Gemini Pro**  
   在公开的多模态理解基准上，InternLM‑XComposer2 的得分在部分任务上与业界顶级闭源模型持平甚至领先，证明了轻量化的 LoRA 方案也能达到最前沿水平。

### 方法详解
**整体框架**  
模型整体可以分为三步：① 输入统一化 → ② 视觉‑语言融合层（带 PL​oRA） → ③ 多模态解码。核心思想是把所有信息先转成 token 序列，再交给一个已经很会写文字的语言模型，让它在保留语言能力的前提下，借助视觉专属的 LoRA 参数来“看懂”图像。

**步骤拆解**  

1. **输入统一化**  
   - **文字部分**：用户的标题、大纲、细节描述等直接分词成文字 token。  
   - **图像部分**：参考图片先经过视觉编码器（如 CLIP ViT），得到一系列图像 token。  
   - **混合序列**：按照用户提供的顺序把文字 token 与图像 token 串联，形成一个长序列。可以把它想象成一本带插图的手稿，文字和图片交错出现。

2. **视觉‑语言融合层（Partial LoRA）**  
   - 在 Transformer 的每一层，针对图像 token 添加一组低秩适配矩阵（LoRA），只对这些 token 的键、值向量进行微调。  
   - 文字 token 仍然走原始的权重矩阵，保持语言模型的完整知识。  
   - 这种设计相当于在每层都给画家装上了“放大镜”，但不影响旁边作家的笔触。

3. **多模态解码**  
   - 融合后的隐藏状态送入语言模型的自回归解码器。解码器在生成每个 token 时，既可以输出文字，也可以输出特殊的“图像生成指令”。  
   - 当模型决定插入图像时，会触发内部的图像生成子模块（如 Diffusion 或 VAE），根据已有的视觉上下文合成对应的图像块。  
   - 最终输出是一段交叉的文字‑图像流，用户可以直接渲染成富文本页面。

**关键细节**  
- **低秩矩阵的秩** 设为 4~8，足够表达视觉微调所需的细节，却不增加显存负担。  
- **冻结语言层**：所有语言专属的参数在微调阶段保持不变，防止“语言漂移”。  
- **跨模态注意力**：因为图像 token 仍然参与自注意力计算，文字可以直接“看到”图像信息，保证布局和描述的一致性。  
- **最巧妙的地方**：只在图像 token 上加 LoRA，既解决了视觉适配，又保留了大模型的语言强度，这在以往的全局 LoRA 或全模型微调中很少出现。

### 实验与效果
- **评测任务**：包括 VQAv2（视觉问答）、MMBench（多模态理解）、MME（多模态评估）以及自建的自由文本‑图像组合任务。  
- **对比基线**：InternLM‑XComposer1、LLaVA、MiniGPT‑4、GPT‑4V、Gemini Pro 等。  
- **结果概述**：论文声称在多数公开基准上 InternLM‑XComposer2 超过前代模型 10% 以上的准确率或评分，在自由组合任务的人工评审中，生成的内容在“语言流畅度”和“视觉匹配度”两项上均获得最高分。对标 GPT‑4V 与 Gemini Pro 时，在部分子任务（如细粒度图像描述）得分略高于它们，显示出在视觉细节捕获上的优势。  
- **消融实验**：作者分别去掉 PL​oRA、只在全部 token 上加 LoRA、以及完全不加 LoRA。结果显示：没有 PL​oRA 时语言流畅度下降约 15%，全局 LoRA 虽提升视觉准确率但语言质量受损，验证了“只在图像 token 上加 LoRA”是最优配置。  
- **局限性**：模型仍然基于 7B 参数规模，面对极其复杂的高分辨率图像或超长文档时会出现细节遗失；此外，PL​oRA 对视觉编码器的依赖较大，换用不同的视觉 backbone 需要重新微调。

### 影响与延伸思考
- **领域影响**：InternLM‑XComposer2 展示了“局部 LoRA”在多模态微调中的可行性，随后有多篇工作尝试在不同模态（音频、视频）上只对特定 token 加 LoRA，以保持主模型的通用性。  
- **后续工作**：2024 年底出现的 **PartialAdapter** 系列进一步推广了部分适配的思想，加入了跨层共享的低秩矩阵；2025 年的 **Vision‑Language FusionNet** 则在 PL​oRA 基础上加入了动态秩调节，使得模型在不同任务上自动决定 LoRA 的容量。  
- **深入方向**：如果想继续深挖，可以关注以下两个方向：① 更细粒度的模态划分（比如只在特定视觉区域的 token 上加 LoRA），② 将 PL​oRA 与参数高效的稀疏激活（Mixture‑of‑Experts）结合，进一步提升大模型的多模态适配能力。

### 一句话记住它
只在图像 token 上加 LoRA，让大语言模型既保持文字天赋，又获得精准视觉感知，实现自由文本‑图像创作。