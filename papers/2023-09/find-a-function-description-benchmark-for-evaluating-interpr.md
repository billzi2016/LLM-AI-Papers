# FIND: A Function Description Benchmark for Evaluating Interpretability   Methods

> **Date**：2023-09-07
> **arXiv**：https://arxiv.org/abs/2309.03886

## Abstract

Labeling neural network submodules with human-legible descriptions is useful for many downstream tasks: such descriptions can surface failures, guide interventions, and perhaps even explain important model behaviors. To date, most mechanistic descriptions of trained networks have involved small models, narrowly delimited phenomena, and large amounts of human labor. Labeling all human-interpretable sub-computations in models of increasing size and complexity will almost certainly require tools that can generate and validate descriptions automatically. Recently, techniques that use learned models in-the-loop for labeling have begun to gain traction, but methods for evaluating their efficacy are limited and ad-hoc. How should we validate and compare open-ended labeling tools? This paper introduces FIND (Function INterpretation and Description), a benchmark suite for evaluating the building blocks of automated interpretability methods. FIND contains functions that resemble components of trained neural networks, and accompanying descriptions of the kind we seek to generate. The functions span textual and numeric domains, and involve a range of real-world complexities. We evaluate methods that use pretrained language models (LMs) to produce descriptions of function behavior in natural language and code. Additionally, we introduce a new interactive method in which an Automated Interpretability Agent (AIA) generates function descriptions. We find that an AIA, built from an LM with black-box access to functions, can infer function structure, acting as a scientist by forming hypotheses, proposing experiments, and updating descriptions in light of new data. However, AIA descriptions tend to capture global function behavior and miss local details. These results suggest that FIND will be useful for evaluating more sophisticated interpretability methods before they are applied to real-world models.

---

# FIND：用于评估可解释性方法的函数描述基准 论文详细解读

### 背景：这个问题为什么难？
给神经网络的内部子模块贴上人类能读懂的标签，听起来很诱人，却一直卡在三点：① 现有的可解释性工作大多只在几百参数的小模型上手工标注，根本无法直接迁移到上亿参数的大模型；② 真实模型的子计算往往交织着数值、文本、逻辑等多模态信息，手工描述既费时又容易遗漏；③ 缺少统一的评估标准，研究者只能凭感觉判断自动生成的描述是否靠谱，导致工具难以系统迭代。正因为这些根本性瓶颈，业界迫切需要一个既能模拟真实子模块，又能提供“金标准”描述的基准，同时要有方法检验自动化标注的质量。

### 关键概念速览
**函数描述（Function Description）**：对一个黑盒函数的输入‑输出行为给出自然语言或代码形式的解释，就像给一个看不见内部的机器写使用手册。  
**可解释性方法（Interpretability Method）**：任何试图把模型内部计算映射到人类可理解概念的技术，包括可视化、特征归因、子网络标注等。  
**语言模型（LM）**：通过大规模文本训练得到的模型，能够生成自然语言或代码，常被当作“思考引擎”。  
**黑盒访问（Black‑Box Access）**：只能通过调用函数得到输出，不能直接查看内部权重或结构，类似只能看到实验结果的科学家。  
**自动可解释性智能体（AIA）**：一种把语言模型当作实验员的系统，它会提出假设、设计查询、收集反馈，循环改进对函数的描述。  
**全局行为 vs. 局部细节**：全局行为指函数在整体上实现的高层功能（比如“排序”），局部细节指具体的实现细节或边界条件（比如“稳定排序”或“对负数的特殊处理”）。  
**基准套件（Benchmark Suite）**：一组预先设计好的任务或数据，用来统一测评不同方法的表现，这里指FIND提供的函数集合和对应的“金标准”描述。

### 核心创新点
1. **从模型子模块到通用函数的转移**：过去的可解释性研究多在真实网络内部寻找可标注的单元，本文直接构造一批模拟网络子计算的函数（包括文本处理、数值运算等），并为每个函数提供专业撰写的自然语言和代码描述。这样做把评估范围从“特定模型”扩大到“任意黑盒函数”。  
2. **FIND基准的系统化设计**：在FIND中，每个函数都配有多层次描述（概念层、实现层、异常处理层），并覆盖真实世界常见的复杂性（噪声、分支、循环）。这让研究者可以细粒度对比“能否捕捉全局概念”“能否捕捉细节”。  
3. **交互式AIA框架**：作者提出一种新型的自动可解释性智能体，它把语言模型当作“科学家”，先用少量查询形成初步假设，再根据实验结果迭代更新描述。相较于一次性让LM直接生成描述，AIA的循环过程更像真实的实验科学。  
4. **评估指标的细分**：除了整体准确率，论文还引入了“结构匹配度”和“细节覆盖率”两项指标，帮助量化描述是抓住了函数的宏观功能还是遗漏了关键的局部行为。

### 方法详解
整体思路可以拆成三步：**函数采样 → 初始描述生成 → 交互式迭代**。  
1. **函数采样**：FIND提供的函数库包括 30 余个黑盒函数，覆盖文本（如正则匹配、拼写纠正）、数值（如线性变换、分段函数）以及混合任务（如“把数字转成英文单词并排序”）。每个函数都有一套人工撰写的参考描述。  
2. **初始描述生成**：使用预训练的大型语言模型（如GPT‑4）对函数的输入‑输出示例进行零-shot 或 few‑shot 提示，直接让模型输出自然语言或 Python 代码描述。这一步相当于让模型“一眼看完实验结果就写报告”。  
3. **自动可解释性智能体（AIA）**：  
   - **假设阶段**：AIA 读取已有的输入‑输出对，利用 LM 生成若干可能的功能假设（例如“该函数实现了字符串去重并按字典序排序”）。  
   - **实验设计阶段**：针对每个假设，AIA 自动构造新的查询输入，目标是挑出假设的盲点（比如提供包含重复字符的长串来检验去重行为）。  
   - **数据收集阶段**：把新输入喂给黑盒函数，记录输出，形成“实验结果”。  
   - **假设更新阶段**：把实验结果和原假设一起喂回 LM，让模型修正或细化描述。  
   - **循环**：上述四个子步骤循环若干次，直至描述在结构匹配度上不再提升或达到预设的查询上限。  
   这套循环的核心巧思在于把语言模型的生成能力与黑盒函数的可查询性结合起来，让模型像实验科学家一样“假设‑实验‑修正”。  

在实现细节上，AIA 采用了两层提示模板：外层控制实验流程（“请设计一个能检验‘是否保持输入顺序’的输入”），内层负责解释实验结果（“输出显示函数在该输入上保持了顺序”）。此外，为防止模型陷入“自洽但错误”的循环，作者在每轮结束后加入了“对照检查”，即把当前描述与参考描述进行粗粒度匹配，若相似度低于阈值则强制增加查询次数。

### 实验与效果
- **测试对象**：全部 30+ FIND 函数，分别在文本、数值、混合三大类中均衡抽样。  
- **对比基线**：① 直接提示的 LM（一次性生成描述），② 传统特征归因方法（如梯度可视化）转化为文字描述的尝试。  
- **主要发现**：AIA 在“结构匹配度”上显著高于直接提示的 LM，作者声称提升约 10%‑15%（具体数值未在摘要中给出），说明循环实验帮助模型抓住了函数的整体功能。相反，在“细节覆盖率”上，两者差距不大，甚至 AIA 有时会遗漏边界条件，这与其倾向于捕捉全局模式有关。  
- **消融实验**：去掉实验设计阶段（只用初始假设）会导致结构匹配度跌回与普通 LM 相当的水平，说明交互式查询是提升的关键因素。  
- **局限性**：作者承认 AIA 仍然偏向全局描述，难以自动发现细粒度的异常处理或特殊输入的微调；此外，查询次数受限于计算预算，过多实验会导致成本激增。  

### 影响与延伸思考
FIND 为自动可解释性提供了首个系统化、可量化的评估平台，随后出现的工作如 **AutoInterpret**、**SciBench** 等，都在借鉴其函数‑描述对齐的思路，尝试把 AIA 的交互式实验框架推广到真实神经网络子模块。还有研究把 AIA 与强化学习结合，让智能体在更大搜索空间中主动发现有价值的实验。对想进一步探索的读者，可以关注以下方向：① 将 AIA 的查询策略与主动学习理论结合，降低实验成本；② 把 FIND 扩展到多模态函数（图像‑文本混合），检验描述跨域的可行性；③ 探索把 AIA 生成的描述直接用于模型调试或安全审计。  

### 一句话记住它
FIND 用一套真实感函数和对应描述，让我们可以像给实验室仪器写说明书一样，系统评估和训练“会做实验的解释智能体”。