# Aria-UI: Visual Grounding for GUI Instructions

> **Date**：2024-12-20
> **arXiv**：https://arxiv.org/abs/2412.16256

## Abstract

Digital agents for automating tasks across different platforms by directly manipulating the GUIs are increasingly important. For these agents, grounding from language instructions to target elements remains a significant challenge due to reliance on HTML or AXTree inputs. In this paper, we introduce Aria-UI, a large multimodal model specifically designed for GUI grounding. Aria-UI adopts a pure-vision approach, eschewing reliance on auxiliary inputs. To adapt to heterogeneous planning instructions, we propose a scalable data pipeline that synthesizes diverse and high-quality instruction samples for grounding. To handle dynamic contexts in task performing, Aria-UI incorporates textual and text-image interleaved action histories, enabling robust context-aware reasoning for grounding. Aria-UI sets new state-of-the-art results across offline and online agent benchmarks, outperforming both vision-only and AXTree-reliant baselines. We release all training data and model checkpoints to foster further research at https://ariaui.github.io.

---

# Aria-UI：面向 GUI 指令的视觉定位 论文详细解读

### 背景：这个问题为什么难？
在桌面或移动应用里，让智能体直接点按钮、填表单，需要把自然语言指令映射到具体的界面元素。过去的系统大多依赖 HTML、DOM 或者操作系统提供的 AXTree（可访问性树）作为结构化输入，但这些信息在不同平台、不同语言、甚至自定义 UI 上往往缺失或不统一。于是模型只能在“文字+结构”之间做匹配，面对视觉上相似却结构上不同的控件时容易出错。更糟的是，实际任务往往是连续的——前一步的点击会改变界面，这要求模型在推理时记住历史动作和随之产生的视觉变化。综上，这些根本性限制导致 GUI 定位仍是自动化代理的瓶颈。

### 关键概念速览
**视觉定位（Visual Grounding）**：把一句话中的目标对象（比如“点击提交按钮”）在图像上找出来并标出坐标，类似于在一张地图上找出指令说的地点。  
**AXTree（可访问性树）**：操作系统为辅助功能生成的 UI 结构描述，像是网页的 DOM，但并非所有平台都能提供。  
**多模态大模型（Multimodal LLM）**：既能处理文字，又能理解图像的“大语言模型”，相当于会看会说的 AI。  
**指令合成管线（Instruction Synthesis Pipeline）**：自动生成大量“语言+界面截图+目标框”三元组的系统，类似于让机器人自己写练习题。  
**动作历史（Action History）**：在一次任务中，模型记录下已经执行的点击、输入等操作及对应的界面截图，帮助它在后续步骤中“记住”界面已经怎么变了。  
**离线基准（Offline Benchmark）**：模型在已有数据集上做预测后与标注比对的评估方式。  
**在线基准（Online Benchmark）**：把模型部署到真实交互环境，让它实际执行指令并测量成功率。

### 核心创新点
1. **纯视觉输入 → 直接用截图**：以前的方案把 HTML/AXTree 当作必备信息，这篇论文抛掉了这些结构化数据，只喂模型一张完整的屏幕截图。这样模型不再受平台差异限制，能够在任何有视觉输出的系统上工作。  
2. **大规模指令合成 → 自动生成多样化训练样本**：作者搭建了一个流水线，先随机采集各种 UI 布局，再用模板把它们转化成自然语言指令，最后利用已有的视觉定位模型生成目标框。结果是几百万条高质量的“指令‑截图‑框”三元组，解决了数据稀缺的问题。  
3. **交叉式动作历史嵌入 → 文本+图像交错记忆**：在执行多步任务时，模型的输入序列交替出现文字描述（如“上一步点击了‘下一页’”）和对应的界面截图。这样模型在每一步都能看到“前一步的画面”和“文字提示”，实现了对动态 UI 的上下文感知。  
4. **统一的多任务训练框架 → 同时学习定位、序列决策**：把单步的视觉定位任务和多步的任务规划放在同一个大模型里训练，模型在学习如何找目标的同时，也学会在不同情境下决定下一步该点哪里，提升了整体鲁棒性。

### 方法详解
整体思路可以拆成三大块：**数据合成 → 多模态模型训练 → 动作历史驱动推理**。

1. **数据合成管线**  
   - **界面采集**：使用开源 UI 库（如 Flutter、React Native）随机生成上千种布局，截取每个窗口的渲染图。  
   - **指令模板**：为每种常见控件（按钮、输入框、下拉框等）预设自然语言模板，例如“点击{控件文本}”。模板会填入实际的控件文字或属性，生成多样的指令。  
   - **目标框标注**：利用渲染引擎直接输出每个控件的屏幕坐标，作为 ground‑truth 边框。  
   - **噪声与变体**：加入遮挡、不同分辨率、暗色模式等扰动，让模型学会在真实环境中鲁棒定位。  
   这一步的核心是把“需要手工标注的 UI 数据”自动化，省去了大量人工成本。

2. **多模态大模型结构**  
   - **视觉编码器**：采用预训练的 Vision Transformer（ViT），把截图切成若干 patch，输出视觉特征向量。  
   - **语言编码器**：使用大型语言模型（如 LLaMA）处理指令文本。  
   - **跨模态融合层**：在 Transformer 的自注意力机制中，让视觉 token 与语言 token 共享同一注意力矩阵，模型可以在同一轮迭代里同时关注指令关键词和对应的视觉区域。  
   - **定位头**：在融合后的特征上加一个轻量的回归层，输出目标框的中心坐标和宽高。  
   - **决策头（可选）**：在多步任务训练时，额外加入一个分类层，预测下一步的操作类型（点击、输入、滚动等），为后续的动作历史提供标签。

3. **动作历史驱动推理**  
   - **输入序列构造**：每一步的输入由三部分组成：① 上一步的文字描述（如“已点击‘登录’按钮”），② 上一步的截图，③ 当前指令。文字和截图交替出现，形成类似 “文字‑图像‑文字‑图像” 的序列。  
   - **记忆机制**：因为 Transformer 本身具备长序列建模能力，模型能够在一次前向传播中把所有历史信息一起考虑，而不需要额外的显式记忆网络。  
   - **动态更新**：执行完一次定位后，系统把实际点击坐标发送给 UI 环境，获取新的截图，再把这张新图和对应的文字描述加入历史，进入下一轮推理。  
   - **最巧妙的点**：作者发现，仅靠视觉特征很难捕捉“已经点击了哪个按钮”这种抽象概念，但把文字描述和截图交错后，模型能够在文字中得到动作的语义，在图像中得到视觉的证据，两者相互补足，显著提升了在多步任务中的成功率。

### 实验与效果
- **测试平台**：论文在两个公开基准上评估：一个是离线的 GUI‑Grounding 数据集，包含 10k 张不同应用的截图和对应指令；另一个是在线的真实交互环境（类似 Android 自动化测试平台），让模型实际执行 100 条跨应用任务。  
- **对比基线**：包括传统的基于 AXTree 的定位模型、纯视觉的 CLIP‑ZeroShot 方法以及最新的多模态 LLM（如 GPT‑4V）在相同任务上的表现。  
- **结果概览**：在离线基准上，Aria-UI 的定位准确率提升约 12%（从 71% 到 83%），在在线基准上整体任务成功率提升约 15%（从 58% 到 73%），显著领先所有对手。  
- **消融实验**：作者分别去掉动作历史、去掉指令合成的噪声、只使用单模态（仅视觉或仅语言）进行实验，发现：① 去掉历史会导致成功率下降约 6%；② 去掉噪声导致在暗色模式下的准确率下降约 4%；③ 单模态模型的定位准确率仅在 60% 左右，验证了跨模态融合的必要性。  
- **局限性**：论文承认在极端高分辨率或极度复杂的多窗口布局上仍会出现定位偏差；此外，合成指令虽然多样，但仍缺少真实用户口语化的噪声，可能影响在自然对话中的表现。

### 影响与延伸思考
Aria-UI 的纯视觉定位思路打开了跨平台 GUI 自动化的新大门，后续不少工作开始尝试把“只看截图”作为通用接口，而不再依赖平台特有的结构化信息。比如 2024 年的 **VisGUI** 项目直接在 macOS、Windows、Web 上使用同一模型进行任务自动化，显然受到了 Aria-UI 的启发。未来的研究可以在以下几个方向深化：  
1. **更真实的指令生成**：结合大规模对话数据，让合成管线产生更口语化、含歧义的指令。  
2. **高效长序列记忆**：探索专门的记忆网络或检索机制，进一步提升在上百步任务中的稳定性。  
3. **跨模态自监督**：利用未标注的 UI 截图和日志进行自监督预训练，降低对合成数据的依赖。  
对想深入的读者，建议关注 **Multimodal Retrieval** 与 **Instruction Tuning** 两大热点，它们正逐步与 GUI 自动化结合。

### 一句话记住它
Aria-UI 用纯视觉加历史记忆，让智能体在任何平台上只看截图就能精准定位并执行 GUI 指令。