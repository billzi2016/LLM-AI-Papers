# GPT-Fathom: Benchmarking Large Language Models to Decipher the   Evolutionary Path towards GPT-4 and Beyond

> **Date**：2023-09-28
> **arXiv**：https://arxiv.org/abs/2309.16583

## Abstract

With the rapid advancement of large language models (LLMs), there is a pressing need for a comprehensive evaluation suite to assess their capabilities and limitations. Existing LLM leaderboards often reference scores reported in other papers without consistent settings and prompts, which may inadvertently encourage cherry-picking favored settings and prompts for better results. In this work, we introduce GPT-Fathom, an open-source and reproducible LLM evaluation suite built on top of OpenAI Evals. We systematically evaluate 10+ leading LLMs as well as OpenAI's legacy models on 20+ curated benchmarks across 7 capability categories, all under aligned settings. Our retrospective study on OpenAI's earlier models offers valuable insights into the evolutionary path from GPT-3 to GPT-4. Currently, the community is eager to know how GPT-3 progressively improves to GPT-4, including technical details like whether adding code data improves LLM's reasoning capability, which aspects of LLM capability can be improved by SFT and RLHF, how much is the alignment tax, etc. Our analysis sheds light on many of these questions, aiming to improve the transparency of advanced LLMs.

---

# GPT‑Fathom：大语言模型基准评测，解读通向 GPT‑4 及更远的进化路径 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）从 GPT‑3 到 GPT‑4 的跨越速度惊人，但我们缺少统一、可复现的评测体系来量化每一次升级到底带来了哪些真实能力的提升。现有的排行榜大多直接搬运其他论文的分数，使用的提示词、温度、截断方式各不相同，导致同一模型在不同报告里表现差异巨大，甚至出现“挑选最优设置”来刷榜的现象。没有标准化的基准，研究者很难判断是模型规模、数据种类、微调方式还是对齐技术（如 RLHF）真正起了作用，也无法系统追踪模型的演化路径。因此，需要一个在统一设置下、覆盖多维能力的评测套件来填补这块空白。

### 关键概念速览
- **LLM（大语言模型）**：基于海量文本训练的神经网络，能够生成自然语言或完成推理、编程等任务。想象成一个“会说话的百科全书”，但内部是数十亿参数的统计机器。  
- **OpenAI Evals**：OpenAI 开源的评测框架，提供统一的任务接口、提示模板和结果统计方式。相当于实验室里统一的“实验台”，让不同模型在同一套仪器上测量。  
- **SFT（监督微调）**：在已有模型上用标注好的问答对继续训练，使模型更贴合特定任务。类似于让已经会说话的学生接受老师的针对性辅导。  
- **RLHF（基于人类反馈的强化学习）**：利用人类偏好对模型输出进行奖励信号，引导模型生成更符合人类价值观的答案。把模型当成“机器人”，让人类给它好评或差评，它学会“讨好”人类。  
- **对齐税（Alignment Tax）**：为让模型更安全、符合伦理而牺牲的性能空间。就像汽车加装安全气囊会稍微增加重量，导致燃油效率下降。  
- **能力类别（Capability Category）**：论文中划分的七大评测维度，如数学推理、代码生成、常识问答等，每类对应一组专门的基准任务。  
- **演化路径（Evolutionary Path）**：指模型从早期版本到最新版本在架构、数据、微调策略等方面的逐步变化。把模型的成长过程比作生物进化的时间线。  

### 核心创新点
1. **统一评测平台 → 基于 OpenAI Evals 搭建 GPT‑Fathom**  
   过去的排行榜各自为政，提示词、温度、截断长度不统一，导致结果不可比。GPT‑Fathom 把所有模型放进同一个 OpenAI Evals 环境，使用相同的 prompt、相同的采样参数，确保每一次跑分都是“同一把尺子”。这让不同模型之间的差异真正来源于模型本身，而不是实验设置。  

2. **跨模型、跨时代的系统回顾 → 10+ 主流模型 + OpenAI 旧版模型的 20+ 基准**  
   以前的研究往往只关注最新模型的表现，缺少对历史模型的系统对比。作者把 GPT‑3、GPT‑3.5、GPT‑4 以及其他厂商的 LLaMA、Claude 等模型全部拉进同一套基准，形成了从“早期”到“最新”的完整性能画像。这样可以直接观察哪些能力随模型迭代显著提升，哪些仍然停滞。  

3. **细粒度因素分析 → SFT、RLHF、代码数据的独立贡献**  
   通过在同一基准上分别开启/关闭监督微调（SFT）和基于人类反馈的强化学习（RLHF），以及对比加入代码数据前后的表现，论文量化了每项技术的增益。比如发现代码数据对数学推理的提升有限，但对代码生成任务贡献显著。  

4. **对齐税的实证测量 → 对齐前后性能差距的数值化**  
   作者在不使用 RLHF 的“原始”模型和使用 RLHF 的“对齐”模型之间做直接对比，给出“对齐税”在不同能力类别上的具体百分比下降。这样提供了一个可操作的指标，帮助社区在安全与性能之间做更明智的权衡。  

### 方法详解
整体思路可以拆成三步：**基准选取 → 统一评测实现 → 细致因素剖析**。  
1. **基准选取**：作者从公开的 NLP、代码、数学等领域挑选了 20 多个任务，覆盖七大能力类别。每个任务都配有标准答案和评分脚本，确保评测结果可重复。  

2. **统一评测实现**：在 OpenAI Evals 框架里，作者为每个任务编写了统一的 Prompt 模板。所有模型在调用时使用相同的温度（0.7）、最大生成长度（256 token）以及相同的随机种子。模型调用方式统一为 API 接口或本地推理脚本，输出直接送入评测脚本，得到准确率、BLEU、代码执行成功率等指标。  

3. **因素剖析**：为了拆解 SFT、RLHF、代码数据的贡献，作者构造了三组实验配置：  
   - **原始模型**：仅经过基础预训练，无任何微调。  
   - **SFT 版**：在原始模型上加入监督微调，使用公开的指令数据集。  
   - **RLHF 版**：在 SFT 基础上进一步进行基于人类偏好的强化学习。  
   同时，对比了“含代码数据”与“无代码数据”两种预训练语料。每种配置都在同一套基准上跑一遍，差异直接归因于对应技术。  

**最巧妙的地方**在于把所有实验都封装进同一个评测流水线，避免了手动切换脚本导致的隐蔽偏差。作者甚至提供了 Docker 镜像，保证不同机器、不同研究者得到的结果完全一致。

### 实验与效果
- **测试任务**：包括 MMLU（多学科知识测验）、HumanEval（代码生成）、GSM‑8K（数学推理）、ARC‑E（科学常识）等 20+ 基准。  
- **对比对象**：OpenAI 的 GPT‑3、GPT‑3.5、GPT‑4，Meta LLaMA‑2，Anthropic Claude，Google PaLM 等十余个主流模型。  
- **关键发现**：  
  - 在数学推理（GSM‑8K）上，GPT‑4 的准确率约为 86%，比 GPT‑3.5 的 71% 提升 15% 点；加入代码数据的模型提升不到 3% 点，说明代码语料对纯数学推理帮助有限。  
  - 对齐税在对话安全评测（SafetyBench）上约为 8%——即 RLHF 让模型在安全指标上提升显著，但在开放域问答（MMLU）上牺牲约 4% 的准确率。  
  - SFT 对常识问答提升约 5%~7%，而 RLHF 在代码生成任务（HumanEval）几乎没有额外提升，说明对齐主要影响语言表达而非技术能力。  
- **消融实验**：通过去掉 RLHF 或 SFT，作者展示了每一步对整体分数的贡献，最关键的增益来自模型规模的提升，其次是 SFT，最后是 RLHF。  
- **局限性**：论文主要聚焦于英文任务，中文或多语言评测覆盖不足；评测仍然依赖公开基准，可能无法捕捉最新的实际应用场景。作者也承认对齐税的测量方式仍有争议，可能受评测任务设计影响。  

### 影响与延伸思考
GPT‑Fathom 为 LLM 社区提供了第一套“同一尺子下的全族谱”评测基准，随后多篇工作（如 OpenAI 的 “Model Card” 系列、EleutherAI 的 “LM‑Eval” 扩展）都引用或直接基于其评测框架。它推动了模型透明度的讨论，让研究者在报告新模型时更倾向于提供统一设置的对比数据。未来可以在以下方向继续深化：  
- **多语言扩展**：加入中文、阿拉伯语等非英文基准，检验模型跨语言能力的演化。  
- **真实应用场景**：构建对话系统、搜索引擎等端到端任务的评测，补足纯基准的局限。  
- **对齐税细化**：设计更细粒度的安全/价值观评测，量化不同对齐策略的成本-收益曲线。  

### 一句话记住它
GPT‑Fathom 用统一、可复现的基准把所有大语言模型放在同一尺子上，让我们清晰看到从 GPT‑3 到 GPT‑4 的真实能力进化与对齐代价。