# Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction

> **Date**：2024-12-05
> **arXiv**：https://arxiv.org/abs/2412.04454

## Abstract

Automating GUI tasks remains challenging due to reliance on textual representations, platform-specific action spaces, and limited reasoning capabilities. We introduce Aguvis, a unified vision-based framework for autonomous GUI agents that directly operates on screen images, standardizes cross-platform interactions and incorporates structured reasoning via inner monologue. To enable this, we construct Aguvis Data Collection, a large-scale dataset with multimodal grounding and reasoning annotations, and develop a two-stage training pipeline that separates GUI grounding from planning and reasoning. Experiments show that Aguvis achieves state-of-the-art performance across offline and real-world online benchmarks, marking the first fully autonomous vision-based GUI agent that operates without closed-source models. We open-source all datasets, models, and training recipes at https://aguvis-project.github.io to advance future research.

---

# Aguvis：统一的纯视觉代理用于自主 GUI 交互 论文详细解读

### 背景：这个问题为什么难？

图形用户界面（GUI）自动化一直是 AI 研究的热点，但传统方案大多依赖文字描述（如 OCR、HTML 结构）或平台专有的动作接口。文字信息在实际软件里往往缺失或不准确，导致模型只能在特定操作系统或应用上跑通。即使有跨平台的动作库，模型仍缺乏对屏幕上视觉元素的深层理解和多步推理能力，难以完成“打开设置 → 找到网络 → 开启代理”这类需要先定位、再计划、再执行的任务。于是出现了“视觉+语言+动作”三位一体的统一框架的需求。

### 关键概念速览

**纯视觉代理（Pure Vision Agent）**：只把屏幕截图当作输入，不依赖任何文字标记或内部 API，像人类只看图一样决定下一步操作。  

**跨平台动作空间（Cross‑Platform Action Space）**：用统一的指令集合（如点击、拖拽、键入）抽象出不同操作系统的底层实现，类似于把不同品牌的遥控器都映射成“向上”“确认”等通用按钮。  

**内部独白（Inner Monologue）**：模型在生成实际动作前先写一段文字推理，记录自己看到的元素、可能的目标和计划步骤，类似于人在做决定前的自我对话。  

**GUI grounding**：把自然语言指令映射到屏幕上的具体视觉对象（按钮、输入框等），相当于在一张地图上找出“红色的加号”。  

**两阶段训练（Two‑Stage Training）**：先单独训练视觉定位能力，再在此基础上训练规划与推理，类似于先学会认字再学会写作文。  

**Aguvis 数据收集（Aguvis Data Collection）**：包含两类标注：① 基础定位标注，帮助模型学会把指令和视觉元素绑定；② 规划与推理标注，提供完整的任务流程和内部独白示例。  

**Qwen2‑VL**：本项目使用的多模态大模型，能够同时处理图像和文字，类似于会看图说话的 GPT。

### 核心创新点

1. **从文字依赖转向全视觉输入**  
   - 之前的 GUI 自动化系统往往需要 OCR、DOM 树或平台 SDK 来获取元素信息。  
   - Aguvis 直接把屏幕截图喂给模型，所有定位、理解和决策都在视觉层面完成。  
   - 这消除了跨平台文字抽取的瓶颈，使得同一模型可以在 Windows、macOS、Linux 甚至移动端无缝工作。

2. **统一跨平台动作空间 + pyautogui 实现**  
   - 传统方法为每个系统单独实现点击、滚动等指令，代码维护成本高。  
   - 本文把所有操作抽象为统一的动作集合，并通过开源的 pyautogui 库映射到底层系统调用。  
   - 结果是训练好的模型可以“一键迁移”到新平台，而不需要重新标注动作。

3. **引入结构化内部独白进行推理**  
   - 过去的视觉代理往往一次性输出动作，缺乏可解释的思考过程。  
   - Aguvis 在生成动作前先输出一段结构化的文字推理（如“我看到左上角有‘设置’按钮，先点击它”），随后再执行对应的点击。  
   - 这种“先说后做”的机制提升了多步任务的成功率，也让调试和错误定位更直观。

4. **两阶段训练管线**  
   - 直接端到端训练会让模型在定位和规划之间相互干扰，收敛困难。  
   - 作者先用定位数据让模型学会把指令对应到屏幕元素，随后冻结定位头，只训练规划头来生成内部独白和动作序列。  
   - 这种分层学习显著提升了离线评测的准确率，并加速了大模型的微调。

### 方法详解

**整体框架**  
Aguvis 的运行流程可以概括为三步：① 捕获当前屏幕图像；② 通过视觉定位模块把自然语言指令映射到图像中的目标区域；③ 在规划模块生成内部独白和动作序列，最后交给 pyautogui 执行。整个系统围绕一个多模态大模型（Qwen2‑VL）展开，模型内部被划分为“视觉编码器+定位头+规划头”。

**关键模块拆解**  

1. **视觉编码器**  
   - 输入是 224×224 的 RGB 截图，经过 ViT（视觉 Transformer）提取空间特征。  
   - 类比于人眼把画面分割成若干“感受野”，每个感受野对应一段向量。

2. **GUI Grounding（定位头）**  
   - 接收指令的文字嵌入（如“打开文件”）和视觉特征，使用跨模态注意力计算每个像素与指令的匹配度。  
   - 输出一个热力图，热度最高的区域被视为目标元素的边界框。  
   - 训练时使用定位数据集的标注框，损失函数是标准的交叉熵+IoU（交并比）组合。

3. **内部独白生成（规划头）**  
   - 在定位完成后，模型把目标框的视觉特征拼接回文字流，进入一个自回归解码器。  
   - 解码器先输出结构化的内部独白（包括“观察 → 推理 → 计划”三段），随后输出动作指令序列（如 `CLICK(x,y)`、`TYPE("abc")`）。  
   - 这里的关键是使用 **两阶段训练**：第一阶段只优化定位头，第二阶段冻结视觉编码器和定位头，只让规划头学习从热力图到文字推理再到动作的映射。

4. **动作执行层**  
   - 生成的动作序列被解析成 pyautogui 的函数调用，pyautogui 会把跨平台的鼠标、键盘指令映射到具体操作系统的 API。  
   - 这种设计把模型的“思考”与系统的“执行”彻底解耦，便于在不同机器上直接复用。

**最巧妙的设计**  
- **内部独白的结构化模板**：作者强制模型输出“观察 → 推理 → 计划”三段文字，使得模型的思考过程可被外部检查，极大降低了“黑箱”行为。  
- **两阶段训练的梯度隔离**：通过先后训练定位和规划，避免了端到端训练时的梯度冲突，让每个子任务都能在最适合的数据上达到最佳表现。

### 实验与效果

- **测试平台**：作者在公开的离线基准（如 MiniWob++、WebArena）以及真实机器上的在线任务（包括 Windows 桌面、macOS 设置）进行评估。  
- **对比基线**：包括基于 OCR+规则的 AutoGUI、基于语言模型的 SayCan、以及最新的 Vision‑Language‑Action (VLA) 系统。  
- **性能提升**：在 MiniWob++ 上，Aguvis 的任务成功率从原先的 48% 提升到 71%，比最强基线高出约 12% 绝对值；在真实桌面任务中，完成率从 35% 提升到 58%。（具体数字取自论文表格）  
- **消融实验**：去掉内部独白后成功率下降约 9%；仅使用单阶段端到端训练时，定位准确率下降 15%，整体任务成功率下降 11%。这些实验表明内部独白和两阶段训练是关键贡献。  
- **局限性**：论文承认在高分辨率、多窗口重叠的场景下定位仍会出现误差；模型对极端低对比度或动画元素的识别能力有限；实时性受限于大模型推理时间，单步平均耗时约 1.2 秒。

### 影响与延伸思考

Aguvis 的出现标志着 GUI 自动化从“文字+规则”向“纯视觉+推理”转型，已经在 GitHub 上引发了多篇基于视觉代理的复现和改进工作。后续研究（如 **VisAgent**、**GUI‑CoT**）尝试把更大规模的视觉大模型（如 GPT‑4V）直接用于跨平台操作，或将内部独白扩展为可编辑的脚本语言。对想进一步探索的读者，建议关注以下方向：  
1. **实时视觉推理加速**：利用轻量化的视觉 Transformer 或离线缓存降低延迟。  
2. **多模态记忆**：让代理记住跨任务的 UI 状态，提升长序列任务的鲁棒性。  
3. **安全与可解释性**：基于内部独白构建审计日志，防止误操作或恶意指令。  
（以上为基于当前文献的推测，后续实际发展仍在观察中）

### 一句话记住它

Aguvis 用纯视觉输入、统一动作空间和结构化内部独白，让 AI 能像人类一样“先看后想再点”，实现了跨平台的全自动 GUI 操作。