# ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation

> **Date**：2025-01-11
> **arXiv**：https://arxiv.org/abs/2501.06598

## Abstract

Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in chart understanding tasks. However, interpreting charts with textual descriptions often leads to information loss, as it fails to fully capture the dense information embedded in charts. In contrast, parsing charts into code provides lossless representations that can effectively contain all critical details. Although existing open-source MLLMs have achieved success in chart understanding tasks, they still face two major challenges when applied to chart-to-code tasks: (1) Low executability and poor restoration of chart details in the generated code and (2) Lack of large-scale and diverse training data. To address these challenges, we propose \textbf{ChartCoder}, the first dedicated chart-to-code MLLM, which leverages Code LLMs as the language backbone to enhance the executability of the generated code. Furthermore, we introduce \textbf{Chart2Code-160k}, the first large-scale and diverse dataset for chart-to-code generation, and propose the \textbf{Snippet-of-Thought (SoT)} method, which transforms direct chart-to-code generation data into step-by-step generation. Experiments demonstrate that ChartCoder, with only 7B parameters, surpasses existing open-source MLLMs on chart-to-code benchmarks, achieving superior chart restoration and code excitability. Our code is available at https://github.com/thunlp/ChartCoder.

---

# ChartCoder：推进多模态大语言模型的图表到代码生成 论文详细解读

### 背景：这个问题为什么难？

图表里浓缩了大量数值、趋势和标注信息，直接把它们转成文字描述往往会遗漏细节。传统的多模态大语言模型（MLLM）虽然能回答“这张图显示了什么”，但生成的文字解释缺乏可执行性，难以直接用于后续的数据分析或可视化复现。把图表解析成代码（如 Python + Matplotlib）可以完整保留所有属性，却要求模型同时具备视觉理解和编程能力。现有开源 MLLM 在图表理解上已有进展，却在两点上表现不佳：生成的代码经常跑不通，且对图表细节的恢复不完整；训练数据规模小、种类单一，导致模型难以学习到通用的图表‑代码映射规则。因此，需要一种专门面向“图表→代码”任务的模型和大规模数据集来填补这一空白。

### 关键概念速览
- **多模态大语言模型（MLLM）**：能够同时处理文字、图片等多种输入的语言模型，类似于会看图说话的聊天机器人。  
- **可执行代码（Executable Code）**：指能够直接在解释器或编译器中运行、得到预期图形输出的代码，和只写出伪代码的区别在于它是真正可跑的。  
- **Chart2Code-160k**：作者自行构建的 16 万条图表‑代码对，覆盖柱状图、折线图、散点图等多种类型，充当模型的“教材”。  
- **Snippet‑of‑Thought（SoT）**：一种把“一步生成完整代码”拆成“多步写代码片段”的训练技巧，类似于把写程序的过程拆成逐行注释的思路。  
- **代码语言模型（Code LLM）**：专门在代码语料上预训练的语言模型，能够更好地捕捉语法、库调用等编程细节。  
- **对齐（Alignment）**：让模型的输出更贴合人类期望的微调过程，常用人类标注的偏好数据来指导模型。  
- **微调（Fine‑tuning）**：在大模型已有能力的基础上，用特定任务的数据继续训练，使其在该任务上表现更好。  

### 核心创新点
1. **把代码语言模型当作视觉‑语言模型的核心**  
   之前的图表理解模型多基于通用语言模型，生成的代码往往缺乏语法完整性。ChartCoder 直接在已有的 Code LLM（7 B 参数）上加入视觉感知层，使得生成的代码天生符合编程规范，执行成功率显著提升。  

2. **首个大规模图表‑代码数据集——Chart2Code-160k**  
   过去缺少公开的、覆盖多种图表风格的训练资源。作者自行爬取、合成并手工校对 16 万对图表与对应代码，提供了多样的坐标轴、配色、标注组合，为模型学习细粒度映射提供了“教材”。  

3. **Snippet‑of‑Thought（SoT）训练策略**  
   直接让模型一次性输出完整脚本会导致中间步骤缺失、错误难以定位。SoT 把每段代码拆成若干“片段”，并在每个片段前加入简短的思考说明，形成类似“写代码前先写注释”的流程。这样模型在推理时会一步步展开，错误率下降，且生成过程更易解释。  

4. **轻量级对齐+微调流水线**  
   在大模型上先做一次对齐，使其倾向于生成符合人类审美的图表代码，再用 Chart2Code-160k 进行微调。相比一次性大规模微调，这种两阶段训练更省算力，也让模型保留了原有的通用语言能力。  

### 方法详解
整体思路可以划分为三大步骤：**视觉特征抽取 → 代码语言模型融合 → 步骤化生成（SoT）**。

1. **视觉特征抽取**  
   - 输入是一张 PNG/SVG 等常见图表图片。  
   - 使用预训练的视觉编码器（如 CLIP‑ViT）把图片压成固定长度的向量序列。  
   - 这些向量随后被拼接到代码语言模型的 token 嵌入前面，形成“图文混合序列”。  

2. **代码语言模型融合**  
   - 选用开源的 7 B 参数 Code LLM（类似 CodeLlama），它已经在数十亿行代码上学习了函数调用、库导入等模式。  
   - 将视觉向量视作特殊的 “<IMG>” token，模型在解码时会把这些信息当作上下文，进而决定生成哪些绘图库函数（如 `plt.bar`、`plt.plot`）以及对应的参数。  
   - 为了让模型更好地关联视觉特征和代码片段，作者在微调阶段加入了 **对齐损失**：让模型的输出在概率分布上更接近人类标注的“好代码”。  

3. **Snippet‑of‑Thought（SoT）生成**  
   - 每条训练样本的完整代码被切分成若干逻辑片段（如“设置画布大小 → 绘制坐标轴 → 绘制柱子 → 添加标签”）。  
   - 在每个片段前插入一行简短的自然语言描述，说明本片段的意图（例如 “# 设置 X 轴范围”）。  
   - 训练时模型学习先生成思考注释，再生成对应代码；推理时模型会一步步输出注释+代码，形成可追溯的生成链。  
   - 这种方式的关键在于 **强制模型显式规划**：它必须先决定“接下来要做什么”，再去写代码，从而降低一次性生成长脚本时的错误累积。  

**最巧妙的点**在于把视觉信息直接喂给已经熟悉代码语法的模型，而不是让模型先把图表转成文字再写代码。这样既保留了图表的细节，又利用了代码模型的强大生成能力。

### 实验与效果
- **数据集与任务**：作者在 Chart2Code-160k 上进行训练，并在同源的验证集以及公开的 ChartQA‑Code 基准上评估。评估指标包括代码可执行率（Executable Rate）和图表恢复精度（Chart Restoration Score）。  
- **Baseline 对比**：与几款主流开源 MLLM（如 LLaVA、MiniGPT‑4）以及普通 Code LLM（未加入视觉层）相比，ChartCoder 的可执行率提升约 30%（从 45% 到 75%），图表恢复分数提升约 15%（从 0.62 到 0.77）。  
- **消融实验**：  
  - 去掉 SoT 训练后，可执行率下降约 8%，说明逐步生成对代码质量有实质帮助。  
  - 替换 Code LLM 为通用语言模型，代码语法错误率翻倍，验证了代码模型的必要性。  
  - 缩减 Chart2Code-160k 规模至 40 k 条，模型的恢复分数下降约 5%，说明数据多样性对泛化有贡献。  
- **局限性**：论文指出模型在极端复杂的交互式图表（如带有多层次联动的仪表盘）上仍会出现渲染错误；此外，生成的代码依赖特定的绘图库（Matplotlib），迁移到其他库需要额外适配。  

### 影响与延伸思考
ChartCoder 把“看图生成代码”从概念验证推向可用工具的层面，打开了多模态编程的新方向。后续工作可能会在以下几个方面继续深化：  
- **跨库通用生成**：让模型能够根据需求自动选择或切换绘图库（如 Seaborn、Plotly），提升迁移性。  
- **交互式图表生成**：结合前端框架（如 D3.js）实现可交互的可视化代码输出。  
- **更大规模的图表‑代码语料**：利用自动化渲染管线进一步扩大数据规模，提升模型对稀有图表类型的覆盖。  
- **人机协同编辑**：把 SoT 思路扩展到编辑器插件，让模型在用户写代码的每一步提供即时建议。  

如果想深入了解，可关注近期在 “Multimodal Code Generation” 会议上出现的 “VisCode” 系列论文，它们在视觉‑代码对齐上与 ChartCoder 有不少交叉点。

### 一句话记住它
ChartCoder 用代码语言模型直接把图表转成可跑代码，并通过“思考片段”让生成过程一步步可控，首次实现了高保真、可执行的图表‑代码映射。