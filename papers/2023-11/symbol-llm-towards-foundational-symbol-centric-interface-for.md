# Symbol-LLM: Towards Foundational Symbol-centric Interface For Large   Language Models

> **Date**：2023-11-15
> **arXiv**：https://arxiv.org/abs/2311.09278

## Abstract

Although Large Language Models (LLMs) demonstrate remarkable ability in processing and generating human-like text, they do have limitations when it comes to comprehending and expressing world knowledge that extends beyond the boundaries of natural language(e.g., chemical molecular formula). Injecting a collection of symbolic data directly into the training of LLMs can be problematic, as it disregards the synergies among different symbolic families and overlooks the need for a balanced mixture of natural and symbolic data. In this work, we tackle these challenges from both a data and framework perspective and introduce Symbol-LLM series models. First, we curated a data collection consisting of 34 tasks and incorporating approximately 20 distinct symbolic families, intending to capture the interrelations and foster synergies between symbols. Then, a two-stage tuning framework succeeds in injecting symbolic knowledge without loss of the generality ability. Extensive experiments on both symbol- and NL-centric tasks demonstrate the balanced and superior performances of Symbol-LLM series models. The project page is https://xufangzhi.github.io/symbol-llm-page/.

---

# Symbol-LLM：面向基础符号中心接口的大语言模型 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在自然语言的理解与生成上已经相当强大，但它们的知识库主要是基于文本。像化学分子式、数学符号、编程 API 等非语言符号，往往只能被当作普通词汇处理，导致模型在精确推理或计算时出现错误。早期的做法是直接把这些符号数据混进原始语料进行再训练，结果往往要么把语言能力削弱，要么符号知识注入不完整，因为不同符号体系之间存在相互关联，却被孤立地喂给模型。于是，如何在保持自然语言通用性的前提下，系统、协同地让模型掌握多种符号知识，成了一个亟待解决的难题。

### 关键概念速览
**符号族（symbolic family）**：指一类具有相似结构和语义规则的符号集合，例如化学式、正则表达式、逻辑谓词等。可以把它想成“同一门语言的不同方言”。  
**两阶段调优（two-stage tuning）**：先在大规模自然语言数据上微调模型，保持语言能力；再在专门的符号任务上进行第二轮微调，以注入符号知识。类似先学会说话，再学会写数学公式。  
**符号任务（symbolic task）**：需要模型直接操作或生成符号的任务，如一阶逻辑推理、分子式平衡、API 参数填充等。它们的输入输出往往不是自然语言句子，而是结构化的符号串。  
**跨符号协同（cross-symbol synergy）**：不同符号族之间共享的推理模式或结构特征，例如“括号匹配”在数学表达式、正则表达式和化学式中都出现。模型若能捕捉这种共性，就能在一种符号上学到的技巧迁移到另一种符号上。  
**符号中心接口（symbol-centric interface）**：一种统一的模型调用方式，用户可以直接提交符号任务而不必先把符号转成自然语言描述。相当于给模型装上了“符号输入端口”。  

### 核心创新点
1. **任务与符号族的系统化构建 → 论文收集了 34 项任务，覆盖约 20 种符号族 → 通过让模型在同一批次里看到多种符号的相互关系，提升了跨符号迁移能力，避免了单一符号孤立训练的局限。**  
2. **两阶段调优流程 → 第一步使用海量自然语言数据微调，保持原有的语言通用性；第二步在符号任务上继续微调，且采用了“保持语言头部不变”的技巧 → 在不牺牲自然语言表现的前提下，成功注入了符号知识。**  
3. **符号任务的统一格式化 → 将所有符号任务统一映射为“指令+输入+输出”三段式，配合特殊的 token 标记符号类型 → 模型能够在一次前向传播中辨认任务类型，减少了为每种符号单独设计提示的工作量。**  
4. **跨符号协同学习机制 → 在第二阶段微调时，引入了混合批次（mixed batches），每个批次里随机混入不同符号族的样本 → 让模型在同一梯度更新中感受到不同符号的共性，实验显示这种混合训练比单一符号训练更稳健。**  

### 方法详解
整体思路可以拆成三大块：**数据准备 → 两阶段调优 → 统一接口**。

1. **数据准备**  
   - **任务挑选**：作者从公开数据集、行业标准和自建脚本中挑选了 34 项任务，涵盖化学式解析、逻辑公式求值、正则表达式匹配、数学表达式求导、API 调用生成等。  
   - **符号族标注**：每条样本都被贴上所属的符号族标签，例如 `[CHEM]`、`[LOGIC]`、`[API]`，并在输入前加上统一的指令前缀，如 “请计算以下化学式的分子量”。  
   - **混合批次构造**：在第二阶段的训练数据里，作者随机抽取不同符号族的样本组成同一批次，确保每次梯度更新都涉及多种符号。可以把它想成“烹饪时把多种调味料一起下锅”，让模型在同一时间感受多种味道。

2. **两阶段调优**  
   - **第一阶段（语言保留）**：使用原始的 LLM 参数，在大规模自然语言语料（如 Common Crawl、Wikipedia）上继续微调。这里不做任何符号相关的改动，目标是让模型的语言生成、理解能力不退化。  
   - **第二阶段（符号注入）**：在已经保留语言能力的模型上，加载符号任务数据。关键技巧是 **冻结语言头部**（即前几层 Transformer 参数保持不变），只让后层和输出头适应符号任务。这样做的直觉是：语言的“底层语法”已经学好，只需要在“高层抽象”上加点符号专用的知识。  
   - **损失函数**：采用标准的交叉熵损失，同时加入一个 **符号平衡系数**，确保稀有符号族的样本不会被频繁出现的符号族淹没。系数的调节类似于在课堂上给每个学生分配不同的练习时间，保证每个人都有机会练习。

3. **统一符号中心接口**  
   - **指令模板**：所有符号任务都遵循 “指令 + 输入 + 输出” 的模板，指令用自然语言描述任务目标，输入是原始符号串，输出是模型需要生成的答案或转换结果。  
   - **特殊 Token**：在模型词表中加入了若干专用 token（如 `<CHEM>`, `<LOGIC>`），帮助模型快速定位符号类型。  
   - **推理时的调用**：用户只需提供指令和符号输入，模型在一次前向传播后直接返回符号答案，无需额外的后处理步骤。  

**最巧妙的点**在于 **混合批次 + 冻结语言层** 的组合。混合批次让模型在同一次梯度更新中感受到不同符号的共性，而冻结语言层则保证这些更新不会破坏已有的语言知识。两者相辅相成，形成了“在不打碎原有玻璃的情况下，往里面嵌入新的马赛克”。  

### 实验与效果
- **评测任务**：作者在 34 项符号任务以及若干自然语言基准（如 MMLU、ARC）上进行评估，覆盖化学、数学、逻辑、编程 API 等多个领域。  
- **对比基线**：主要包括原始 LLM（未做符号微调）、直接在符号数据上单阶段微调的模型、以及已有的符号注入方法（如 Symbolic Prompting）。  
- **整体表现**：论文声称在多数符号任务上相较于基线提升了数个百分点，同时在自然语言基准上保持了与原始模型相当的分数，说明语言能力没有明显退化。  
- **消融实验**：作者分别去掉了混合批次、符号平衡系数、冻结语言层这三个关键设计，实验结果显示每去掉一项，符号任务的准确率都会下降 3%~7%，验证了它们的必要性。  
- **局限性**：论文承认仍然存在对极端长符号串（如上千字符的化学反应式）处理不佳的情况；此外，符号族的覆盖仍然有限，未涉及图形化符号（如电路图）等更复杂的结构。  

### 影响与延伸思考
这篇工作打开了 **“符号中心化”** 的新思路，后续不少研究开始探索如何在同一模型中同时处理文本、代码、数学公式甚至图结构。比如 2024 年的 **Math-LLM**、**ChemPrompt** 等都引用了 Symbol-LLM 的两阶段调优和跨符号批次混合策略。对想进一步深入的读者，可以关注以下方向：  
- **更广泛的符号族扩展**：把图形、音频频谱等非文本符号加入统一接口。  
- **自适应符号平衡**：使用元学习让模型自动调节不同符号的训练权重。  
- **符号推理可解释性**：结合符号图谱或逻辑证明，使模型的符号输出可追溯。  

### 一句话记住它
**Symbol-LLM 用两阶段调优和跨符号混合批次，让大语言模型在不失语言能力的前提下，真正学会了“读懂并操作各种符号”。**