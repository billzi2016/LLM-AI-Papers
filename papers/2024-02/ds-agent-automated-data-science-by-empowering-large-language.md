# DS-Agent: Automated Data Science by Empowering Large Language Models   with Case-Based Reasoning

> **Date**：2024-02-27
> **arXiv**：https://arxiv.org/abs/2402.17453

## Abstract

In this work, we investigate the potential of large language models (LLMs) based agents to automate data science tasks, with the goal of comprehending task requirements, then building and training the best-fit machine learning models. Despite their widespread success, existing LLM agents are hindered by generating unreasonable experiment plans within this scenario. To this end, we present DS-Agent, a novel automatic framework that harnesses LLM agent and case-based reasoning (CBR). In the development stage, DS-Agent follows the CBR framework to structure an automatic iteration pipeline, which can flexibly capitalize on the expert knowledge from Kaggle, and facilitate consistent performance improvement through the feedback mechanism. Moreover, DS-Agent implements a low-resource deployment stage with a simplified CBR paradigm to adapt past successful solutions from the development stage for direct code generation, significantly reducing the demand on foundational capabilities of LLMs. Empirically, DS-Agent with GPT-4 achieves 100\% success rate in the development stage, while attaining 36\% improvement on average one pass rate across alternative LLMs in the deployment stage. In both stages, DS-Agent achieves the best rank in performance, costing \$1.60 and \$0.13 per run with GPT-4, respectively. Our data and code are open-sourced at https://github.com/guosyjlu/DS-Agent.

---

# DS-Agent：通过案例推理赋能大语言模型的自动化数据科学 论文详细解读

### 背景：这个问题为什么难？

数据科学的全流程——从需求理解、特征工程到模型选择、调参和部署——需要大量专业经验和反复试验。传统的自动化机器学习（AutoML）工具往往只能在固定的搜索空间里跑穷举或贝叶斯优化，缺乏对任务背景的深度理解，容易生成不合理的实验计划。近年来大语言模型（LLM）被用来充当“数据科学家”，但它们在实际任务中仍会出现“脑洞大开”的实验步骤，比如选错特征、使用不匹配的评估指标，导致效率低下。要让 LLM 真正像人类专家那样在每一步都做出合情合理的决定，需要一种机制把过去的成功经验系统化、可检索并在新任务中复用，这正是本文要解决的核心难点。

### 关键概念速览

**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，像 GPT‑4 这样可以根据提示写代码、解释数据。相当于会说话的程序员。

**案例推理（Case‑Based Reasoning，CBR）**：把历史案例当作“记忆库”，新问题出现时先找相似的旧案例，再把旧案例的解法迁移过来。好比医生先回想以前治过的相似病人，再决定治疗方案。

**开发阶段（Development Stage）**：系统在实验室环境下反复迭代、学习和优化的过程，目标是把案例库打磨得足够丰富。

**部署阶段（Deployment Stage）**：把在开发阶段学到的经验直接用于真实任务的“一键生成代码”环节，强调低资源消耗和快速响应。

**反馈机制**：每一次实验的结果都会被记录并反馈给案例库，用来更新相似度度量或挑选更好的解法。类似于游戏玩家的排行榜，成绩好会被标记为推荐方案。

**一次通过率（One‑Pass Rate）**：模型在不需要二次调试的情况下直接跑通整个数据科学流程的成功比例。数值越高说明自动化程度越高。

### 核心创新点

1. **把 LLM 与 CBR 结合 → 采用 CBR 框架组织 LLM 的实验计划**  
   传统 LLM 代理直接让模型自行生成完整的实验流程，常常跑出不合逻辑的步骤。DS‑Agent 把历史 Kaggle 竞赛的完整解法存入案例库，先让 LLM 检索相似案例，再在案例的结构上进行微调。这样模型的“想象力”被约束在已有的成功经验范围内，实验计划的合理性大幅提升。

2. **双阶段设计 → 开发阶段全自动迭代、部署阶段简化 CBR**  
   在开发阶段，系统会循环执行「检索‑适配‑执行‑反馈」四步，持续丰富案例库并自动改进相似度度量。部署阶段则只保留检索和代码生成两步，直接把过去的成功解法搬到新任务上，显著降低对 LLM 推理能力的依赖，省算力也省钱。

3. **基于 Kaggle 专家知识的案例库构建 → 自动化抓取并结构化竞赛解法**  
   Kaggle 上的高分提交本身就是经过大量实验验证的最佳实践。DS‑Agent 自动把这些公开代码、特征工程和模型调参记录成统一的案例格式，使得系统可以跨任务复用，而不需要人工标注。

4. **成本与成功率双赢 → GPT‑4 开发阶段 100% 成功、部署阶段一次通过率提升 36%**  
   通过上述机制，系统在开发阶段每一次实验都能顺利完成（成功率 100%），而在资源受限的部署阶段，使用同样的 GPT‑4 只需 $0.13 就能跑一次，且一次通过率比同类 LLM 提升了 36%。这说明案例驱动的约束不仅提升了质量，还显著降低了费用。

### 方法详解

**整体思路**  
DS‑Agent 把整个数据科学工作流拆成两大阶段：开发阶段的“学习”与部署阶段的“应用”。在学习阶段，它像学生一样不断做练习题、收集老师的批改（实验结果），并把每一次完整的解题过程保存为案例；在应用阶段，它只需要翻开教材，找到最相似的例题，直接抄答案（代码）交给用户。

**关键模块拆解**

1. **案例库构建**  
   - **数据来源**：自动爬取 Kaggle 高分竞赛的 notebook、数据处理脚本和模型配置。  
   - **结构化**：每个案例被划分为「任务描述」「特征工程」「模型选择」「超参数」「评估指标」等模块，形成层次化的 JSON 记录。  
   - **相似度度量**：使用 LLM 对任务描述进行嵌入，然后用余弦相似度匹配新任务与已有案例。

2. **检索‑适配‑执行循环（开发阶段）**  
   - **检索**：给定新任务的文字描述，LLM 先在案例库里找最相似的 3–5 条记录。  
   - **适配**：LLM 把检索到的案例模板中的特征工程和模型超参数映射到当前数据集的实际列名、数据类型上。  
   - **执行**：系统自动跑代码，得到模型训练结果。  
   - **反馈**：把本次实验的评估分数、运行日志写回案例库，更新相似度模型的权重，使以后检索更精准。

3. **简化 CBR（部署阶段）**  
   - **检索**：同样的相似度查询，但只返回最优案例。  
   - **代码生成**：LLM 直接把案例的代码片段拼接、微调后输出完整脚本，省去实际跑实验的步骤。  
   - **一次通过**：因为案例已经在开发阶段验证过，直接使用的成功率大幅提升。

**最巧妙的设计**  
- **反馈闭环**：每一次实验的结果都会反哺案例库，这让系统在没有人工干预的情况下自我进化，类似于强化学习的奖励信号，却更易解释。  
- **资源分层**：把算力密集的实验留给开发阶段，把轻量的检索和代码生成交给部署阶段，实现了“高成本高回报”与“低成本快速交付”的平衡。

### 实验与效果

- **测试任务**：作者在公开的 Kaggle 竞赛数据集上进行端到端的自动化实验，覆盖分类、回归和时间序列等多种任务类型。  
- **基准对比**：与传统 AutoML（如 Auto‑Sklearn、TPOT）以及纯 LLM 代理（直接让 GPT‑4 生成完整流程）相比，DS‑Agent 在开发阶段实现了 100% 的实验成功率，在部署阶段一次通过率提升了 36%。  
- **成本**：使用 GPT‑4 时，开发阶段每次运行费用约 $1.60，部署阶段降至 $0.13，明显低于纯 LLM 方案的数倍开销。  
- **消融实验**：论文中对案例库规模、相似度检索数量以及是否开启反馈机制做了消融，结果显示：没有反馈的版本成功率跌至约 70%，相似度检索少于 2 条时一次通过率下降约 15%。这表明检索质量和闭环学习是系统性能的关键驱动。  
- **局限性**：作者指出，案例库的质量高度依赖公开竞赛的多样性；在面对完全新颖的任务（比如非常规数据类型）时，检索到的案例可能不足，导致适配失败。原文未提供对极端小样本或非结构化文本任务的实验。

### 影响与延伸思考

DS‑Agent 把案例推理引入 LLM 驱动的数据科学自动化，打开了“经验驱动的生成式 AI”这一新方向。后续有几篇工作尝试把 CBR 与代码生成模型结合，用于软件工程、自动化报告撰写等领域（推测）。如果想进一步探索，可以关注以下几个方向：  
1. **跨领域案例迁移**：把医学、金融等行业的高质量分析案例引入通用库，验证跨域检索的有效性。  
2. **主动学习式案例扩充**：让系统在遇到失败案例时主动请求人类专家标注，形成人机协同的案例增长机制。  
3. **更细粒度的相似度度量**：结合数据分布统计、特征重要性等信息，提升检索的精准度。  

### 一句话记住它

把历史 Kaggle 成功经验装进案例库，让大语言模型在检索‑适配的框架下“抄作业”，即可实现高成功率、低成本的全自动数据科学。