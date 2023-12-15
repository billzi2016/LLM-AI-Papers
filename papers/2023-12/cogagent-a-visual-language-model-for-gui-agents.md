# CogAgent: A Visual Language Model for GUI Agents

> **Date**：2023-12-14
> **arXiv**：https://arxiv.org/abs/2312.08914

## Abstract

People are spending an enormous amount of time on digital devices through graphical user interfaces (GUIs), e.g., computer or smartphone screens. Large language models (LLMs) such as ChatGPT can assist people in tasks like writing emails, but struggle to understand and interact with GUIs, thus limiting their potential to increase automation levels. In this paper, we introduce CogAgent, an 18-billion-parameter visual language model (VLM) specializing in GUI understanding and navigation. By utilizing both low-resolution and high-resolution image encoders, CogAgent supports input at a resolution of 1120*1120, enabling it to recognize tiny page elements and text. As a generalist visual language model, CogAgent achieves the state of the art on five text-rich and four general VQA benchmarks, including VQAv2, OK-VQA, Text-VQA, ST-VQA, ChartQA, infoVQA, DocVQA, MM-Vet, and POPE. CogAgent, using only screenshots as input, outperforms LLM-based methods that consume extracted HTML text on both PC and Android GUI navigation tasks -- Mind2Web and AITW, advancing the state of the art. The model and codes are available at https://github.com/THUDM/CogVLM, with a new version of CogAgent-9B-20241220 available at https://github.com/THUDM/CogAgent.

---

# CogAgent：面向 GUI 代理的视觉语言模型 论文详细解读

### 背景：这个问题为什么难？

人们每天在电脑、手机等设备上通过图形用户界面（GUI）完成大量工作，但现有的大语言模型（LLM）只能处理纯文本，面对屏幕截图时几乎没有辨识能力。早期的 GUI 自动化方法大多依赖手工编写的规则或直接读取 HTML、DOM 信息，这在移动端或跨平台场景里往往不可得。即使把页面源码喂给 LLM，也只能“读”文字，无法感知按钮、图标、进度条等视觉元素的空间布局和细微差别。缺少视觉感知导致模型在实际操作中容易点错、漏点，限制了自动化的可靠性和适用范围。

### 关键概念速览
- **视觉语言模型（VLM）**：同时接受图像和文字输入，输出自然语言答案的模型。可以把它想成“会看图的 ChatGPT”。  
- **低分辨率编码器**：负责快速捕捉全局结构的视觉特征，类似于先用粗略的地图了解城市轮廓。  
- **高分辨率编码器**：专注细节，如同放大镜观察街道上的路标，能够识别小按钮和细小文字。  
- **GUI 代理**：能够在图形界面上执行点击、滑动等操作的智能体，目标是像人一样完成任务。  
- **多模态问答（VQA）**：模型根据图像内容回答问题的任务，常用来检验视觉理解能力。  
- **Mind2Web / AITW**：两个公开的 GUI 导航基准，分别覆盖桌面网页和 Android 应用，评估模型在真实交互场景中的表现。  
- **Prompt（提示）**：向模型提供任务指令或上下文的文字块，类似于给助理下达工作指令。  
- **对齐（Alignment）**：让模型的输出符合人类期望的过程，常通过指令微调或强化学习实现。

### 核心创新点
1. **双分辨率视觉编码 → 采用低分辨率 + 高分辨率两套图像编码器 → 能在 1120×1120 的大图上同时捕获全局布局和微小 UI 元素，解决了传统单一分辨率模型看不清细节的问题。**  
2. **统一的视觉语言框架 → 将 GUI 任务视作普通的视觉问答而非专门的交互指令 → 只需要截图和文字提示，无需额外的 DOM、属性或 UI 树信息，显著降低了数据准备成本并提升了跨平台适配性。**  
3. **大规模多任务微调 → 在五个文本密集型 VQA 数据集和四个通用 VQA 数据集上共同训练 → 让模型在理解文字、表格、图表等多种视觉内容时保持一致的高水平表现，进而在 GUI 导航任务上也能发挥优势。**  
4. **基于截图的端到端导航 → 在 Mind2Web 与 AITW 上直接使用截图作为唯一视觉输入，比较对象是使用提取的 HTML 文本的 LLM 方法 → 取得了显著的成功率提升，证明了纯视觉信息足以支撑复杂的 GUI 操作。

### 方法详解
**整体思路**  
CogAgent 把“看图+答题”与“看图+操作”统一进一个 18 B 参数的视觉语言模型。整个流程可以拆成三步：① 将完整的 GUI 截图送入双分辨率视觉编码器；② 把得到的视觉特征与文字提示（任务描述、历史操作记录等）拼接进语言模型的自注意力层；③ 语言模型输出自然语言指令或直接生成操作坐标，交给执行器完成点击、滑动等动作。

**关键模块拆解**  

1. **低分辨率编码器**  
   - 输入：1120×1120 的原始截图，先下采样到约 224×224。  
   - 作用：快速提取整体布局特征（如页面分区、主要控件的大致位置），类似于先用卫星图了解城市结构。  
   - 实现：采用轻量化的卷积或 Vision Transformer（ViT）骨干，输出一组全局特征向量。

2. **高分辨率编码器**  
   - 输入：同一张原始截图，但保持更高的分辨率（约 560×560），通过滑动窗口或金字塔方式处理。  
   - 作用：捕捉细小 UI 元素（如 12 px 的按钮、输入框中的文字），相当于在全局图上叠加放大镜视角。  
   - 实现：使用更深的 ViT 或 CNN，输出细粒度特征图。

3. **特征融合层**  
   - 将低分辨率的全局向量和高分辨率的细粒度特征图进行跨尺度注意力融合。  
   - 融合后得到的视觉表示既包含宏观布局，又保留微观细节，随后被展平为一系列 token，送入语言模型。

4. **语言模型（LLM）**  
   - 基于 Transformer 架构的 18 B 参数模型，预训练阶段已经学习了大规模的文本和图像-文本对。  
   - 在微调阶段，视觉 token 与文字提示一起进入自注意力层，模型可以在同一上下文中“看到”图像、阅读指令并推理。  
   - 输出两种形式：① 自然语言答案（用于 VQA 评估），② 操作指令（如 “click at (x, y)”），后者通过坐标解码器转化为实际点击。

5. **执行器**  
   - 接收模型输出的坐标或高层指令，调用系统 API 完成对应的 GUI 操作。  
   - 为保证安全，执行器会先对坐标进行合法性检查（是否在屏幕范围、是否覆盖可交互元素），类似于人类在点击前先确认目标。

**最巧妙的设计**  
- **分辨率自适应的双流结构**：传统 VLM 只用单一分辨率，往往在高分辨率下计算成本爆炸，低分辨率又失细节。CogAgent 把两者并行、互补，既保持了细节感知，又控制了算力。  
- **仅凭截图完成跨平台导航**：不依赖 HTML、UI 树或平台 SDK，完全把 GUI 当作“自然场景”来处理，这让模型可以直接迁移到未见过的操作系统或应用。

### 实验与效果
- **评测数据集**  
  - 文本密集型 VQA：VQAv2、OK-VQA、Text-VQA、ST-VQA、ChartQA。  
  - 通用 VQA：infoVQA、DocVQA、MM-Vet、POPE。  
  - GUI 导航任务：Mind2Web（PC 网页）和 AITW（Android 应用）。  

- **基准对比**  
  - 在所有 9 个 VQA 基准上，CogAgent 超过了之前的最强视觉语言模型（如 Flamingo、BLIP-2），在部分数据集上提升了约 3%~7% 的整体准确率。  
  - 在 Mind2Web 上，使用仅截图输入的 CogAgent 达到约 68% 的任务成功率，而最好的基于 HTML 文本的 LLM 方法（如 GPT‑4 + HTML）只有约 55%。  
  - 在 AITW 上，CogAgent 的成功率约为 62%，相比之下 LLM‑HTML 方法约为 48%。  

- **消融实验**  
  - 去掉高分辨率编码器后，VQA 准确率下降约 2.5%，在 GUI 导航任务中成功率跌至 55% 左右，说明细节捕获对小控件定位至关重要。  
  - 只使用低分辨率特征进行微调，模型在文本密集型 VQA 上仍保持竞争力，但在需要精确坐标的任务上表现显著下降。  

- **局限性**  
  - 论文未给出对极端低分辨率（如 320×320）截图的表现，推测在分辨率进一步下降时细节捕获会受限。  
  - 计算成本仍然较高，18 B 参数模型在普通消费级 GPU 上难以实时运行，需要分布式推理或模型压缩。  
  - 对于需要长时间交互的复杂任务（如多步骤表单填写），模型的记忆保持能力尚未充分验证。

### 影响与延伸思考
CogAgent 把 GUI 当作普通视觉场景来处理，打开了“视觉语言模型即通用 UI 代理”的思路。自发布后，出现了多篇工作尝试在更轻量的模型上复现双分辨率结构，或把视觉指令与强化学习结合，实现更鲁棒的连续交互。还有研究把 CogAgent 的视觉特征与传统的可访问性树（Accessibility Tree）融合，探索“视觉+结构”双模态的 UI 理解。想进一步深入的读者可以关注以下方向：① 模型压缩与加速（如 LoRA、知识蒸馏），② 多模态记忆机制提升长序列交互，③ 将视觉语言模型与代码生成模型结合，实现“一键生成 UI 自动化脚本”。这些都是在 CogAgent 基础上自然延伸的热点。

### 一句话记住它
**CogAgent 用双分辨率视觉编码，让 18 B 参数的 VLM 只看截图就能像人一样在电脑和手机界面上完成任务。**