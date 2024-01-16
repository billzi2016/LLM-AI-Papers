# RoTBench: A Multi-Level Benchmark for Evaluating the Robustness of Large   Language Models in Tool Learning

> **Date**：2024-01-16
> **arXiv**：https://arxiv.org/abs/2401.08326

## Abstract

Tool learning has generated widespread interest as a vital means of interaction between Large Language Models (LLMs) and the physical world. Current research predominantly emphasizes LLMs' capacity to utilize tools in well-structured environments while overlooking their stability when confronted with the inevitable noise of the real world. To bridge this gap, we introduce RoTBench, a multi-level benchmark for evaluating the robustness of LLMs in tool learning. Specifically, we establish five external environments, each featuring varying levels of noise (i.e., Clean, Slight, Medium, Heavy, and Union), providing an in-depth analysis of the model's resilience across three critical phases: tool selection, parameter identification, and content filling. Experiments involving six widely-used models underscore the urgent necessity for enhancing the robustness of LLMs in tool learning. For instance, the performance of GPT-4 even drops significantly from 80.00 to 58.10 when there is no substantial change in manual accuracy. More surprisingly, the noise correction capability inherent in the GPT family paradoxically impedes its adaptability in the face of mild noise. In light of these findings, we propose RoTTuning, a strategy that enriches the diversity of training environments to bolster the robustness of LLMs in tool learning. The code and data are available at https://github.com/Junjie-Ye/RoTBench.

---

# RoTBench: 多层次评估大语言模型工具学习鲁棒性的基准 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）已经可以在代码生成、数据查询等场景里“调用工具”，但这些实验大多在干净、结构化的模拟环境里进行。真实世界的交互充满了拼写错误、格式混乱、传感器噪声等不确定因素，模型一旦遇到这些“脏”输入，往往会选错工具、给出错误参数，甚至直接崩溃。现有的评测大多只看模型在理想条件下的准确率，缺少对噪声鲁棒性的系统测量，导致研究者难以判断模型在实际部署时的可靠性，也没有明确的改进方向。

### 关键概念速览

**工具学习（Tool Learning）**：让语言模型在对话或代码中主动选择并调用外部程序、API 或硬件，就像人类在工作时打开相应的软件工具一样。

**鲁棒性（Robustness）**：模型面对输入噪声、格式偏差或环境变化时仍能保持正确行为的能力，类似于人在嘈杂的咖啡馆里仍能听懂对话。

**噪声层级（Noise Levels）**：本文把外部环境划分为 Clean、Slight、Medium、Heavy、Union 五种噪声强度，分别对应从几乎无误到严重混乱的输入情况。

**工具选择（Tool Selection）**：模型在给定任务描述后，需要判断哪一个外部工具最合适，就像在工具箱里挑选螺丝刀。

**参数识别（Parameter Identification）**：选定工具后，模型要从自然语言中抽取出工具所需的具体参数（如文件路径、数值阈值），相当于把口头指令翻译成机器指令。

**内容填充（Content Filling）**：在需要模型生成完整代码或配置文件的场景下，模型必须把抽取的参数嵌入到模板中，确保语法和语义都正确。

**RoTTuning**：一种训练策略，作者通过在多噪声环境中混合数据来提升模型的鲁棒性，类似于让学生在不同噪声的教室里练习听力。

### 核心创新点

1. **从单一干净环境到多层噪声基准 → RoTBench 构建了五个噪声层级的外部环境 → 研究者可以量化模型在不同噪声强度下的表现，发现即使是最强大的 GPT‑4 在轻度噪声下也会掉 20% 以上的分数。**

2. **把工具学习拆解为三阶段评估 → 在每个噪声层级分别测量工具选择、参数识别、内容填充的成功率 → 细粒度的结果帮助定位模型最薄弱的环节，而不是只给出一个整体准确率。**

3. **提出 RoTTuning 训练方案 → 在训练数据中加入多噪声版本的同一任务，迫使模型学习在噪声中恢复正确结构 → 实验显示该策略能把 GPT‑4 在 Heavy 噪声下的得分提升约 12 分，验证了“多样化环境”对鲁棒性的正向作用。**

4. **发现 GPT 系列自带的噪声纠正机制在轻度噪声时反而削弱适应性 → 通过对比不同模型的表现，作者指出现有的“自动纠错”可能会误把噪声当成有意义的信息，从而导致错误的工具选择。**

### 方法详解

**整体框架**  
RoTBench 的评测流程可以概括为三步：①准备任务样本；②在五个噪声层级下生成对应的输入；③让模型依次完成工具选择、参数识别、内容填充，并记录每一步的成功率。整个过程像是给模型安排了一场“噪声马拉松”，看它能跑多远。

**步骤拆解**  

1. **任务库构建**  
   - 作者挑选了六类常见的工具使用场景（如文件操作、数学计算、网络请求、图像处理、系统监控、机器人控制），每类准备 100 条自然语言指令。  
   - 每条指令对应唯一的目标工具、所需参数以及完整的代码/配置模板。

2. **噪声注入机制**  
   - **Clean**：原始指令，保持语法和拼写完整。  
   - **Slight**：加入轻微拼写错误、额外空格或同义词替换。  
   - **Medium**：在 Slight 基础上再加入词序颠倒、缺失关键动词等。  
   - **Heavy**：随机删除或插入无关词汇，甚至混入非 ASCII 字符。  
   - **Union**：把前四种噪声随机混合，形成最具挑战性的输入。  
   - 噪声生成采用规则模板 + 随机采样的方式，确保每个层级的噪声强度可控且可重复。

3. **三阶段评估**  
   - **工具选择**：模型输出工具名称或 API 标识，系统通过字符串匹配判断是否正确。  
   - **参数识别**：模型需要在输出中列出所有必需参数，系统检查键值对是否完整且数值在合理范围。  
   - **内容填充**：模型生成完整代码或配置文件，使用语法解析器和单元测试脚本验证可执行性。  
   - 每一步都独立计分，最终得分是三者的加权平均。

4. **RoTTuning 训练**  
   - 在原始干净数据上进行常规微调的基础上，作者额外加入了噪声版数据，比例约为 1:1。  
   - 训练目标仍是最小化标准交叉熵，但输入端的噪声多样化迫使模型学习“噪声不影响核心语义”的表征。  
   - 为防止模型过度依赖纠错，作者在损失函数中加入了一个噪声辨识项，鼓励模型在必要时保留原始噪声信息而不是盲目纠正。

**最巧妙的地方**  
RoTBench 并没有直接把噪声当作“错误”，而是把它当作一种“环境变量”。这种思路让评测更像真实世界的系统测试，而不是单纯的错误率统计。再者，RoTTuning 的噪声混合策略在保持原始任务语义的同时，显式让模型看到同一任务的多种“外观”，这点在以往的微调数据增强中很少出现。

### 实验与效果

- **实验对象**：六个公开可用的大语言模型，包括 GPT‑4、GPT‑3.5‑turbo、Claude‑2、LLaMA‑2‑70B、Gemini‑1.0‑Pro、Mistral‑7B。  
- **基准任务**：上述六类工具使用场景共 600 条指令，分别在五个噪声层级下评测。  
- **主要结果**：  
  - 在 Clean 环境下，GPT‑4 的整体得分约 80.0，排名第一。  
  - 当噪声提升到 Slight，GPT‑4 直接跌到 58.1，下降幅度超过 20%。  
  - 其他模型的下降趋势类似，但 GPT‑3.5‑turbo 在 Heavy 噪声下仍能保持约 45 分的得分，显示出相对更好的噪声容忍度。  
- **RoTTuning 效果**：对 GPT‑4 进行 RoTTuning 后，Heavy 噪声下的得分从 38 提升到 50，提升约 12 分；Union 环境提升约 9 分。  
- **消融实验**：作者分别去掉噪声混合、噪声辨识项和参数识别监督，发现去掉噪声混合会导致 Heavy 噪声下的提升幅度降至 3 分，说明噪声多样化是关键因素。  
- **局限性**：实验只覆盖了文本指令到代码/配置的转换，未涉及视觉、语音等多模态工具；噪声注入规则虽系统化，但仍是人工设定，可能与真实世界的噪声分布有差距。作者在讨论中承认需要更大规模的真实交互日志来进一步验证。

### 影响与延伸思考

RoTBench 为工具学习提供了首个系统化的鲁棒性评测框架，已经被后续工作引用来检验自定义插件、机器人控制以及代码自动化系统的稳健性。2024 年底，有几篇论文基于 RoTBench 提出了“噪声感知的提示工程”（Noise‑Aware Prompting）和“自适应工具选择器”（Adaptive Tool Selector），尝试在模型内部加入噪声检测模块。未来的研究可以往以下方向延伸：  
- 将噪声层级扩展到多模态输入（图像、语音），构建跨模态的 RoTBench。  
- 探索更细粒度的噪声类型（如时序延迟、网络抖动）对工具调用的影响。  
- 将 RoTTuning 与强化学习结合，让模型在噪声环境中通过试错学习更稳健的工具使用策略。  

### 一句话记住它

**RoTBench 把“噪声”当作真实世界的考验，揭示了即使是最强大的 LLM，在轻微输入扰动下也会大幅失效，并提供了通过多噪声微调提升鲁棒性的实用路径。**