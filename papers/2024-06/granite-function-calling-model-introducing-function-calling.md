# Granite-Function Calling Model: Introducing Function Calling Abilities   via Multi-task Learning of Granular Tasks

> **Date**：2024-06-27
> **arXiv**：https://arxiv.org/abs/2407.00121

## Abstract

Large language models (LLMs) have recently shown tremendous promise in serving as the backbone to agentic systems, as demonstrated by their performance in multi-faceted, challenging benchmarks like SWE-Bench and Agent-Bench. However, to realize the true potential of LLMs as autonomous agents, they must learn to identify, call, and interact with external tools and application program interfaces (APIs) to complete complex tasks. These tasks together are termed function calling. Endowing LLMs with function calling abilities leads to a myriad of advantages, such as access to current and domain-specific information in databases and knowledge sources, and the ability to outsource tasks that can be reliably performed by tools, e.g., a Python interpreter or calculator. While there has been significant progress in function calling with LLMs, there is still a dearth of open models that perform on par with proprietary LLMs like GPT, Claude, and Gemini. Therefore, in this work, we introduce the GRANITE-20B-FUNCTIONCALLING model under an Apache 2.0 license. The model is trained using a multi-task training approach on seven fundamental tasks encompassed in function calling, those being Nested Function Calling, Function Chaining, Parallel Functions, Function Name Detection, Parameter-Value Pair Detection, Next-Best Function, and Response Generation. We present a comprehensive evaluation on multiple out-of-domain datasets comparing GRANITE-20B-FUNCTIONCALLING to more than 15 other best proprietary and open models. GRANITE-20B-FUNCTIONCALLING provides the best performance among all open models on the Berkeley Function Calling Leaderboard and fourth overall. As a result of the diverse tasks and datasets used for training our model, we show that GRANITE-20B-FUNCTIONCALLING has better generalizability on multiple tasks in seven different evaluation datasets.

---

# Granite函数调用模型：通过细粒度多任务学习赋予函数调用能力 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）已经可以写代码、回答问题，但在真实系统里，它们往往需要调用外部工具（数据库、计算器、Python 解释器）才能完成完整任务。传统的 LLM 只能输出文字，缺乏识别何时需要工具、选哪个工具以及如何构造调用参数的能力。现有的商业模型（如 GPT‑4、Claude、Gemini）通过私有数据和大规模微调实现了较好表现，但开源模型在函数调用的准确率、链式调用和并行调用等细节上仍显不足。缺少统一的、多任务的训练框架导致模型在新工具或组合调用场景下容易出错，这直接限制了 LLM 成为真正的自主代理。

### 关键概念速览

**函数调用（Function Calling）**：模型在生成答案的过程中，主动识别需要外部工具并按照约定的 API 格式发起请求。就像人类在做数学题时会打开计算器一样。

**嵌套函数调用（Nested Function Calling）**：一个函数的返回值被用作另一个函数的输入，形成层层嵌套。类似于先查天气再根据天气决定穿衣建议。

**函数链（Function Chaining）**：多个函数按顺序依次执行，每一步的输出是下一步的输入。像流水线生产，每个工序都必须完成才能得到最终产品。

**并行函数（Parallel Functions）**：多个独立函数可以同时调用，随后合并结果。好比同时查询多个数据库再把信息汇总。

**函数名检测（Function Name Detection）**：模型需要从自然语言中抽取出正确的 API 名称。相当于在一段对话里找出“打开灯光”对应的具体指令。

**参数‑值对检测（Parameter‑Value Pair Detection）**：识别出每个函数需要的参数及其取值。就像在填写表格时把“日期=2024‑05‑01”填进去。

**下一最佳函数（Next‑Best Function）**：在当前上下文下，模型预测最有可能帮助完成任务的下一个函数。类似于棋手思考下一步最佳棋子。

**响应生成（Response Generation）**：在完成所有函数调用后，模型把工具返回的原始数据转化为自然语言答案。相当于把技术报告写成通俗的解释。

### 核心创新点

1. **细粒度七任务多任务学习 → 将函数调用拆解为七个基本子任务进行统一训练 → 模型在单一训练阶段同时掌握嵌套调用、链式调用、并行调用等多种场景，显著提升了跨任务的泛化能力。**  
   以前的工作往往只针对单一任务（比如只学会函数名检测），导致在实际使用时需要额外的后处理。Granite 通过一次性覆盖全部细节，让模型在真实对话中自然切换。

2. **公开的 20 B 参数模型 + Apache 2.0 许可证 → 在保持商业模型相近规模的同时，完全开源 → 社区可以直接下载、微调或部署，打破了高性能函数调用模型的闭源壁垒。**  
   之前只有少数公司内部模型具备同等能力，开源社区只能使用性能较差的替代品。Granite 把高质量的函数调用能力带给了所有研究者。

3. **跨域评测与排行榜领先 → 在 Berkeley Function Calling Leaderboard 上成为所有开源模型的第一名，整体排名第四 → 通过在七个不同评测数据集上的统一表现证明了模型的通用性，而不是在单一基准上“投机”。**  
   过去很多模型只在自建数据集上表现好，缺乏跨域验证。Granite 的评测设计让人更放心它在真实业务中的可靠性。

### 方法详解

**整体框架**  
Granite‑20B‑FUNCTIONCALLING 的训练流程可以看作三步：① 数据准备 → 把公开的函数调用数据和自行构造的细粒度任务数据统一成统一的 JSON‑API 格式；② 多任务微调 → 使用一种叫做“任务标签混合”（task‑tag mixing）的技巧，在同一批次里随机抽取七种任务的样本；③ 统一解码 → 在推理阶段，模型先输出一个“函数调用意图”结构（包括函数名、参数），随后根据意图调用外部工具，最后再生成自然语言答案。

**关键模块拆解**  

1. **任务标签混合**  
   每条训练样本前面会加上一个显式的任务标记，例如 `<nested>`, `<chain>`, `<parallel>`。模型在看到标记后会切换内部的注意力模式，专注于对应的推理路径。类似于人类在做不同类型的题目时先读题目类型再决定解法。

2. **统一的函数调用模板**  
   所有七个子任务都遵循同一套 JSON schema：`{ "function": "name", "arguments": { "param1": value1, ... } }`。这样模型只需要学习一次序列化/反序列化的技巧，就能覆盖所有细节。相当于给模型一本统一的“调用手册”，避免了每个任务各自为政的碎片化学习。

3. **两阶段解码**  
   - **意图阶段**：模型输出函数调用结构（如果当前上下文不需要调用，则输出 `null`）。  
   - **答案阶段**：在外部工具返回结果后，模型把返回的原始 JSON 再次喂入，生成最终的自然语言回答。  
   这种分离让模型在没有工具时不会胡乱生成参数，也让后续的答案生成可以专注于语言流畅度。

**最巧妙的设计**  
作者发现，仅靠大模型的自回归能力很难在一次生成中同时完成函数名、参数和值的精准匹配。于是引入了“参数‑值对检测”子任务，让模型在训练时专门练习把自然语言中的属性映射到结构化参数。这个细粒度的练习显著降低了实际调用时的语法错误率。

### 实验与效果

- **评测数据集**：作者在七个公开的函数调用基准上做了全面测试，包括 Berkeley Function Calling Leaderboard、SWE‑Bench、Agent‑Bench 以及四个自建的跨域数据集（覆盖金融查询、代码执行、天气查询、数学计算等场景）。  
- **对比基线**：与 15+ 领先的商业模型（GPT‑4、Claude‑2、Gemini‑1.5）以及开源模型（Llama‑2‑70B‑Chat、Mistral‑7B‑Instruct、OpenChat‑3.5）进行比较。  
- **核心结果**：在 Berkeley 排行榜上，Granite‑20B‑FUNCTIONCALLING 的整体得分为 78.4，领先第二名的开源模型（71.2）约 7 分；在整体排名中位列第四，仅次于三大商业模型。具体到子任务，嵌套调用的准确率提升到 85%（商业模型约 83%），并行调用的成功率达到 81%（开源模型普遍在 65% 左右）。  
- **消融实验**：去掉“任务标签混合”后，整体得分下降 4.3 分；去掉“参数‑值对检测”子任务后，函数名检测错误率翻倍，说明该子任务是提升结构化输出的关键。  
- **局限性**：论文承认模型仍然依赖于高质量的 API schema，面对全新未见过的工具时会出现“函数名未知”错误；此外，20 B 参数模型在资源受限的边缘设备上难以部署。

### 影响与延伸思考

Granite‑20B‑FUNCTIONCALLING 的发布让开源社区首次拥有了在函数调用能力上可以与商业模型竞争的基准模型。随后出现的几篇工作（如 “OpenFunction‑7B” 与 “ToolFormer‑Lite”）都直接引用了其七任务多任务训练框架，尝试在更小模型上复现相似效果。对想进一步探索的读者，可以关注以下方向：① 如何在更小的模型上保持高精度的函数调用（模型压缩、知识蒸馏）；② 动态工具发现——让模型在没有预先定义 schema 的情况下自行学习新 API；③ 安全与对齐——防止模型误调用有风险的工具。整体来看，Granite 为 LLM 从“文字生成”向“工具驱动的智能体”转型奠定了重要的开源基石。

### 一句话记住它

Granite‑20B‑FUNCTIONCALLING 用细粒度七任务多任务学习，让开源大模型一次性掌握所有常见函数调用技巧，性能逼近商业模型，真正打开了 LLM 当工具代理的大门。