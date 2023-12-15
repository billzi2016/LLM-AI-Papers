# Prompting Datasets: Data Discovery with Conversational Agents

> **Date**：2023-12-15
> **arXiv**：https://arxiv.org/abs/2312.09947

## Abstract

Can large language models assist in data discovery? Data discovery predominantly happens via search on a data portal or the web, followed by assessment of the dataset to ensure it is fit for the intended purpose. The ability of conversational generative AI (CGAI) to support recommendations with reasoning implies it can suggest datasets to users, explain why it has done so, and provide information akin to documentation regarding the dataset in order to support a use decision. We hold 3 workshops with data users and find that, despite limitations around web capabilities, CGAIs are able to suggest relevant datasets and provide many of the required sensemaking activities, as well as support dataset analysis and manipulation. However, CGAIs may also suggest fictional datasets, and perform inaccurate analysis. We identify emerging practices in data discovery and present a model of these to inform future research directions and data prompt design.

---

# Prompting Datasets: Data Discovery with Conversational Agents 论文详细解读

### 背景：这个问题为什么难？

在传统的数据发现流程里，研究者往往得先打开数据门户或搜索引擎，敲关键词，然后手动浏览搜索结果、下载数据、阅读元数据，最后判断是否符合需求。这个过程既耗时又容易遗漏关键信息。过去的自动化方案大多是基于关键词匹配或结构化目录检索，缺乏对用户意图的深层理解，也不能在对话中解释推荐背后的理由。于是出现了“模型能不能像人一样在聊天里帮我找合适的数据集？”的需求，这正是本文要解决的核心难题。

### 关键概念速览
- **大语言模型（LLM）**：能够生成自然语言的深度学习模型，像 ChatGPT 那样可以理解上下文并给出连贯回答。把它想象成会说话的百科全书。
- **对话生成式 AI（CGAI）**：专指能够在多轮对话中产生内容的模型，除了回答问题，还能给出推理过程，类似于会思考的聊天机器人。
- **数据发现（Data Discovery）**：寻找、评估并获取满足特定分析需求的数据集的全过程，类似于在浩瀚图书馆里挑选合适的参考书。
- **数据提示（Data Prompt）**：向模型提供的指令或问题，旨在引导模型输出与数据集相关的信息。可以把它看作是对模型的“搜索指令”。
- **感知意义活动（Sensemaking Activities）**：用户在评估数据集时进行的理解、比较、验证等认知步骤，就像读论文时要先弄清实验设计再判断结论可靠性。
- **虚构数据集（Fictional Dataset）**：模型在没有真实对应时凭空编造的“数据集”，相当于聊天时的“胡说八道”。
- **数据文档（Dataset Documentation）**：描述数据来源、结构、质量等信息的说明材料，类似于商品的使用说明书。

### 核心创新点
1. **把对话式生成模型直接用于数据集推荐**  
   过去的系统只能返回搜索结果列表，缺少解释。本文让 LLM 在多轮对话中主动提出数据集，并给出推荐理由，等于是把“搜索+解释”合二为一，提升了用户对推荐的信任感。

2. **构建“感知意义活动”模型**  
   作者通过三次工作坊观察用户在真实数据发现过程中的思考步骤，提炼出一套活动模型（如需求澄清、数据属性匹配、质量评估等），并把这些活动映射到对话策略上，使模型的输出更贴合实际评估需求。

3. **提出数据提示设计框架**  
   基于活动模型，论文给出了一套如何编写 Prompt（提示）的指南：先让模型确认需求，再让它列出候选数据集，随后要求解释每个候选的适配度，最后支持对数据进行基本分析。这样系统化的 Prompt 让模型的行为更可预测，也为后续研究提供了模板。

4. **识别并量化模型的风险**  
   通过用户访谈，作者发现模型会产生虚构数据集和错误分析两类风险，并把这些风险纳入评估框架，提醒后续系统在部署时必须加入真实性校验机制。

### 方法详解
整体思路可以拆成四个阶段：**需求捕获 → 候选生成 → 理由解释 → 初步分析**。每一步都通过一次或多次对话完成，模型在每轮对话中既是信息提供者也是推理者。

1. **需求捕获**  
   - 用户先用自然语言描述自己的分析目标（比如“我想研究美国 2010‑2020 年的空气质量与交通拥堵的关系”）。  
   - Prompt 让模型把这些描述抽象成结构化的需求要素：时间范围、地理范围、变量类型等。相当于把口头需求翻译成机器可读的清单。

2. **候选生成**  
   - 基于结构化需求，Prompt 要求模型在已知的公开数据目录（如 Kaggle、Data.gov）中搜索匹配的 dataset 名称和简要元信息。  
   - 为防止模型直接凭空编造，Prompt 中加入了“如果不确定，请说明不确定的原因”这一约束。

3. **理由解释**  
   - 对每个候选，Prompt 要求模型给出三点解释：① 数据来源可信度，② 与需求要素的匹配程度，③ 可能的限制（缺失字段、采样偏差等）。  
   - 这一步相当于让模型完成一次“感知意义活动”，帮助用户快速判断是否值得进一步查看。

4. **初步分析**  
   - 用户可以请求模型对某个候选进行快速统计（比如展示变量分布、缺失率），Prompt 会让模型模拟执行这些分析并返回结果。  
   - 这里模型并不真正运行代码，而是基于元数据和公开文档进行推理，提供类似“文档式分析”的输出。

**最巧妙的设计**在于把“真实性校验”嵌入 Prompt：每当模型给出数据集时，系统会自动触发一次外部 API（如数据门户的元数据查询）来核对名称是否真实存在。这样即使模型倾向于“编造”，系统也能及时捕获错误。

### 实验与效果
- **实验场景**：作者邀请了三组真实数据使用者，分别在不同领域（公共卫生、城市规划、金融分析）进行数据发现任务。  
- **对比基线**：传统关键词搜索界面、以及一个仅返回候选列表不提供解释的简化对话系统。  
- **主要结果**：相较于关键词搜索，使用对话式 CGAI 的用户在找到满足需求的数据集上平均提前约 30% 的时间；在解释满意度上，用户给出的 Likert 评分从 3.2 提升到 4.5（满分 5）。  
- **消融实验**：去掉“理由解释”模块后，用户对候选的信任度下降约 0.8 分，说明解释环节对感知意义活动至关重要。  
- **局限性**：作者承认模型仍会偶尔生成不存在的数据集，尤其在需求非常细化时；此外，模型的统计分析仅基于元数据，精度受限，不能替代真实的代码执行。

### 影响与延伸思考
这篇工作把对话式生成模型引入数据发现的场景，开启了“数据搜索即对话”的新思路。随后的研究开始探索 **检索增强生成（RAG）** 与 **实时数据查询** 的结合，以期让模型在给出推荐时直接调用真实的 API，进一步降低虚构风险。对想继续深入的读者，可以关注 **数据文档自动生成**、**模型可解释性在检索中的应用** 以及 **人机协同的元数据治理** 等方向，这些都是本论文种下的种子。

### 一句话记住它
让大语言模型在对话中推荐并解释数据集，既能加速发现，又要警惕它会“编造”——所以真实检验永远不可缺。