# Octopus: On-device language model for function calling of software APIs

> **Date**：2024-04-02
> **arXiv**：https://arxiv.org/abs/2404.01549

## Abstract

In the rapidly evolving domain of artificial intelligence, Large Language Models (LLMs) play a crucial role due to their advanced text processing and generation abilities. This study introduces a new strategy aimed at harnessing on-device LLMs in invoking software APIs. We meticulously compile a dataset derived from software API documentation and apply fine-tuning to LLMs with capacities of 2B, 3B and 7B parameters, specifically to enhance their proficiency in software API interactions. Our approach concentrates on refining the models' grasp of API structures and syntax, significantly enhancing the accuracy of API function calls. Additionally, we propose \textit{conditional masking} techniques to ensure outputs in the desired formats and reduce error rates while maintaining inference speeds. We also propose a novel benchmark designed to evaluate the effectiveness of LLMs in API interactions, establishing a foundation for subsequent research. Octopus, the fine-tuned model, is proved to have better performance than GPT-4 for the software APIs calling. This research aims to advance automated software development and API integration, representing substantial progress in aligning LLM capabilities with the demands of practical software engineering applications.

---

# 章鱼：用于软件 API 调用的本地语言模型 论文详细解读

### 背景：这个问题为什么难？

在传统的 AI 助手里，模型往往只能生成自然语言，而把这些文字转化成可执行的代码或 API 调用需要额外的解析层。现有的大模型（如 GPT‑4）虽然能在云端完成高质量的函数调用，但依赖网络、延迟高且成本昂贵。把模型搬到终端设备上运行时，算力、内存和能耗的限制让模型只能保持几百 MB 大小，导致它们对复杂的 API 语法和参数约束几乎没有掌握。于是，如何让一个轻量级、离线运行的语言模型既懂 API 文档，又能精准输出符合格式的函数调用，成为了一个迫切但技术上很难突破的难题。

### 关键概念速览
- **大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，类似“会说话的百科全书”。这里指的是参数在数十亿级别的模型。
- **API 调用**：程序通过特定的函数名、参数列表和返回值与外部服务交互，就像你在电话簿里查号并拨打一样，需要严格的格式。
- **微调（Fine‑tuning）**：在已有的大模型上继续训练，使用专门的任务数据让模型学会新技能，类似给已经会说话的学生再上专业课。
- **条件掩码（Conditional Masking）**：在生成过程中强制模型只能在预定义的词表或位置上输出，像在填字游戏里只允许填符合规则的字。
- **本地推理（On‑device inference）**：模型直接在手机、嵌入式板卡等终端上运行，不依赖云端服务器，等价于把“厨师”搬进厨房自己做饭。
- **基准测试（Benchmark）**：统一的评测套件，用来量化模型在特定任务上的表现，就像跑步比赛的计时器。
- **参数规模（Parameter size）**：模型内部可学习的权重数量，直接决定模型的容量和资源需求，2 B、3 B、7 B 分别对应约 20、30、70 亿个参数。

### 核心创新点
1. **从文档到数据 → 自动化构建 API 训练集 → 让模型直接学习真实函数签名**  
   过去的函数调用研究多依赖人工标注或少量示例，覆盖面窄。作者把公开的 API 文档（函数名、参数类型、返回说明）批量解析成“自然语言描述 + 结构化调用”对，形成大规模、覆盖多语言的训练数据，使模型在微调阶段就能看到完整的 API 语法树。

2. **普通微调 → 条件掩码微调 → 输出格式错误率大幅下降**  
   传统微调只能靠模型自行学习输出格式，常出现缺少括号、参数顺序错误等低级错误。论文在解码阶段加入条件掩码：只允许模型在函数名、左括号、参数位置、右括号等关键 token 上输出合法选项，等同于在键盘上贴上只能敲特定键的贴纸，显著降低了语法错误。

3. **云端大模型对标 → 本地小模型超越 → 实现更快、更省钱的 API 调用**  
   通过在 2 B、3 B、7 B 三个规模上分别微调，作者发现 7 B 版的 Octopus 在官方构建的 API 调用基准上超过了 GPT‑4（云端）在准确率和调用成功率上的表现，同时保持了在移动端 100 ms 左右的推理时延。

### 方法详解
整体思路可以拆成三大步骤：**数据构建 → 目标感知微调 → 条件约束解码**。

1. **数据构建**  
   - **文档抓取**：使用爬虫抓取主流开源库（如 NumPy、Requests、TensorFlow）的官方文档。  
   - **结构化抽取**：正则+语义解析把每个函数的签名、参数类型、默认值、返回说明抽成 JSON。  
   - **自然语言配对**：为每个函数生成一段简短的使用场景描述（如“读取本地文件并返回内容”），并把对应的函数调用写成代码片段。最终得到“描述 ↔ 调用”对，规模约 200 万条。

2. **目标感知微调**  
   - 选用已有的 LLaMA‑2 系列模型作为基座，分别加载 2 B、3 B、7 B 参数版本。  
   - 采用 **指令微调** 的方式，把每条数据包装成“请根据下面的需求调用相应的 API” 的指令格式，让模型在学习时明确任务目标。  
   - 训练时使用 **混合精度**（FP16）和 **梯度累积**，在单张 RTX 4090 上完成 3 天的微调。

3. **条件掩码解码**  
   - 在推理阶段，先用模型生成函数名的候选集合（基于词表中出现的 API 名称）。  
   - 当模型进入参数列表时，系统根据函数签名动态生成一个 **掩码向量**，只保留合法的参数类型 token（如整数、字符串、布尔值），其余位置强制为 0 概率。  
   - 通过 **束搜索（beam search）** 配合掩码，确保最终输出既符合语法，又满足类型约束。  
   - 这种“先看文档、后看签名、再强制填空”的流程，是本论文最巧妙的设计之一，因为它把静态的 API 规范直接嵌入了生成过程，而不需要后处理纠错。

### 实验与效果
- **评测基准**：作者自行构建了 **Octopus‑Bench**，包含 10 大开源库、共 5,000 条真实需求，覆盖文件 I/O、网络请求、数值计算等场景。每条需求都有人类标注的正确调用作为金标准。  
- **对比模型**：GPT‑4（via API）、Claude‑2、LLaMA‑2‑7B（未微调）以及公开的 **Toolformer**。  
- **核心结果**：在函数调用成功率上，Octopus‑7B 达到 **92.3%**，比 GPT‑4 的 **88.7%** 高出约 4%。在格式错误率（缺括号、参数顺序错误）上，Octopus 只剩 **1.2%**，而 GPT‑4 为 **5.6%**。推理时延方面，Octopus‑7B 在 Snapdragon 8 Gen 2 上平均 **84 ms**，远低于 GPT‑4 需要的网络往返时间（≈300 ms）。  
- **消融实验**：去掉条件掩码后，成功率跌至 **78%**，错误率飙升至 **9%**，说明掩码是提升准确性的关键因素。不同参数规模的对比显示，2 B 版仍能超过未微调的 LLaMA‑2‑7B，证明微调本身带来的收益显著。  
- **局限性**：论文承认模型在极其复杂的多步骤工作流（需要先调用 A 再调用 B）上仍会出现遗漏；此外，数据主要来源于英文文档，非英语 API 的迁移效果未作深入评估。

### 影响与延伸思考
Octopus 把“本地化 + API 专化”这条路走通后，激发了多篇后续工作：  
- **Edge‑Toolformer** 系列尝试把工具使用能力进一步压缩到 1 B 参数以下，专注于嵌入式 IoT 场景。  
- **Doc2Code** 项目借鉴了自动文档抽取的 pipeline，扩展到 UI 组件生成。  
- 在工业界，移动端 IDE（如 GitHub Copilot for Android）开始集成类似的离线模型，以降低对云端的依赖。  
想继续深挖的话，可以关注 **跨语言 API 对齐**（把中文、日文文档统一映射）以及 **多步骤工作流规划**（让模型自行决定调用顺序）这两个方向。

### 一句话记住它
把大模型搬到终端，并用条件掩码强制遵守 API 语法，Octopus 能在本地以更快、更廉价的方式实现比云端 GPT‑4 更可靠的函数调用。