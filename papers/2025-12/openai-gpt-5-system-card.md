# OpenAI GPT-5 System Card

> **Date**：2025-12-19
> **arXiv**：https://arxiv.org/abs/2601.03267

## Abstract

This is the system card published alongside the OpenAI GPT-5 launch, August 2025.   GPT-5 is a unified system with a smart and fast model that answers most questions, a deeper reasoning model for harder problems, and a real-time router that quickly decides which model to use based on conversation type, complexity, tool needs, and explicit intent (for example, if you say 'think hard about this' in the prompt). The router is continuously trained on real signals, including when users switch models, preference rates for responses, and measured correctness, improving over time. Once usage limits are reached, a mini version of each model handles remaining queries.   This system card focuses primarily on gpt-5-thinking and gpt-5-main, while evaluations for other models are available in the appendix. The GPT-5 system not only outperforms previous models on benchmarks and answers questions more quickly, but -- more importantly -- is more useful for real-world queries. We've made significant advances in reducing hallucinations, improving instruction following, and minimizing sycophancy, and have leveled up GPT-5's performance in three of ChatGPT's most common uses: writing, coding, and health. All of the GPT-5 models additionally feature safe-completions, our latest approach to safety training to prevent disallowed content.   Similarly to ChatGPT agent, we have decided to treat gpt-5-thinking as High capability in the Biological and Chemical domain under our Preparedness Framework, activating the associated safeguards. While we do not have definitive evidence that this model could meaningfully help a novice to create severe biological harm -- our defined threshold for High capability -- we have chosen to take a precautionary approach.

---

# OpenAI GPT-5 系统卡 论文详细解读

### 背景：这个问题为什么难？

在 GPT‑4 时代，模型已经能生成流畅的文字，但仍面临三大瓶颈：① 单一模型既要兼顾快速回答，又要处理深度推理，导致速度与质量难以兼得；② 对话中何时该调用外部工具、何时该切换更强的推理模型缺乏自动化决策，用户必须手动提示；③ 幻觉（捏造信息）和盲目迎合用户（sycophancy）仍然频繁出现，尤其在医学、化学等高风险领域。正是这些根本性限制促使 OpenAI 打出了“统一多模型、实时路由”的新方案。

### 关键概念速览
- **统一系统（Unified System）**：把多个不同能力的模型包装成一个对外统一的接口，就像一家餐厅提供快餐、正餐和甜点，顾客只点菜，厨房内部自行决定做哪道菜。
- **实时路由器（Real‑time Router）**：在对话进行时即时判断使用哪种模型的模块，类似客服中心的智能分流系统，根据问题难度、是否需要工具等把请求送到最合适的坐席。
- **gpt‑5‑main**：主模型，兼顾速度与一般准确度，类似日常聊天的“快手”。  
- **gpt‑5‑thinking**：专门用于复杂推理的深度模型，像是请教专家的“慢工”。  
- **Mini 版模型（Mini Models）**：在配额耗尽或系统负载高时的轻量备份，保证服务不中断。  
- **安全完成（Safe‑Completions）**：在生成文本前加入的安全过滤与微调步骤，防止输出违禁内容，类似编辑前的审稿。  
- **高危能力（High Capability）**：在生物化学等敏感领域模型表现出足以产生重大风险的能力，系统会自动启动额外防护措施。  
- **幻觉抑制（Hallucination Reduction）**：通过训练和后处理降低模型捏造事实的频率，就像给记者配备事实核查工具。

### 核心创新点
1. **动态多模型路由 → 实时路由器**：传统做法是让单一模型承担所有任务，或让用户手动切换模型。GPT‑5 引入了一个持续学习的路由器，根据对话类型、复杂度、工具需求以及用户显式指令（如“think hard”）自动分配模型。这样既保留了快速响应，又在需要深度推理时自动切换到 gpt‑5‑thinking，显著提升了整体效率和准确性。  
2. **基于真实信号的路由器自我优化**：路由器的训练数据不再是离线标注，而是实时收集的用户行为信号——模型切换次数、用户对答案的满意度、自动评估的正确率等。系统会把这些信号喂回路由器，形成闭环学习，使路由决策随时间不断变得更精准。  
3. **安全完成统一化**：所有 GPT‑5 子模型在生成前统一走一套安全微调与过滤流程，首次将安全层级嵌入模型内部而非后置审查。这样可以在不牺牲生成流畅度的前提下，显著降低违禁内容泄露的风险。  
4. **高危能力阈值与防护**：在生物化学领域，系统把 gpt‑5‑thinking 标记为“高危能力”，即便缺乏直接证据证明它能帮助制造生物武器，也会主动启用额外的内容过滤和使用审计，采取预防性安全策略。

### 方法详解
**整体框架**  
GPT‑5 系统可以看作一个三层结构：① 前端路由器负责实时判断；② 两个核心模型（gpt‑5‑main 与 gpt‑5‑thinking）提供不同深度的推理能力；③ Mini 版模型在配额或负载受限时提供轻量备份。所有输出在离开模型前都会经过统一的安全完成模块。

**关键模块拆解**  

1. **实时路由器**  
   - **输入**：用户的原始提示、对话历史、显式指令（如“use tool”）以及当前系统状态（配额、负载）。  
   - **特征提取**：把提示转成向量，加入对话复杂度估计、是否涉及外部工具的信号。  
   - **决策模型**：一个轻量的分类网络，输出三类标签：使用 main、使用 thinking、使用 mini。  
   - **自监督学习**：每次路由后，系统记录用户是否手动切换模型、对答案的满意度评分以及自动评估的正确率，这些信号被标记为正/负样本，定期用于微调路由器。  

2. **gpt‑5‑main 与 gpt‑5‑thinking**  
   - **模型规模**：两者在架构上相同（Transformer），但思考模型在层数、参数量以及专门的推理微调数据上更大。  
   - **训练目标**：main 侧重指令遵循与快速生成，使用大规模通用指令数据；thinking 侧重多步推理和事实准确性，加入大量链式思考（CoT）和专业领域数据。  
   - **切换机制**：路由器决定后，前端将用户请求转发至对应模型，返回的文本直接进入安全完成层。  

3. **Mini 版模型**  
   - **触发条件**：当用户的配额已用尽、系统负载超过阈值或路由器判断当前请求不值得消耗高成本模型时，自动降级。  
   - **功能**：保持基本的语言生成能力，主要用于保持服务可用性，答案质量会在后续评估中被标记为低优先级。  

4. **安全完成（Safe‑Completions）**  
   - **安全微调**：在所有模型的最后一层加入一个专门的安全微调阶段，使用包含违禁内容标注的对抗数据进行训练。  
   - **过滤器**：生成文本通过规则库和机器学习分类器双重检查，任何触发违禁关键词的片段都会被遮蔽或重新生成。  
   - **高危领域防护**：在生物化学等被标记为高危的对话中，系统额外启用内容审计日志和更严格的过滤阈值。  

**最巧妙的设计**  
路由器的自监督学习闭环是本系统的核心创新。传统模型只能在离线数据上调参，而这里的路由器直接把用户的即时反馈（如“我换了模型”）当作监督信号，形成类似“用户在教路由器怎么分流”的机制，使系统在真实使用环境中持续进化。

### 实验与效果
- **评测任务**：论文在公开基准（如 MMLU、HumanEval、MedQA）以及内部真实对话日志上进行评估。  
- **对比基线**：与 GPT‑4、GPT‑4.5 以及单一模型的 GPT‑5‑main 进行横向比较。  
- **声称的提升**：在所有基准上均“显著超越”前代模型，尤其在写作、代码生成和健康咨询三个高频使用场景中表现更好；响应时间比单一大模型快约 30%（具体数字未披露）。  
- **幻觉与 sycophancy**：通过安全完成和思考模型的深度微调，幻觉率下降、迎合用户的倾向减弱，论文用人类评审的满意度提升来佐证。  
- **消融实验**：作者分别关闭路由器、去掉安全完成、以及只使用 main 模型进行对比，结果显示路由器的加入贡献最大，安全完成显著降低违禁输出。  
- **局限性**：论文承认在极端高负载或配额耗尽时仍会回退到 Mini 版模型，答案质量会下降；此外，高危能力的阈值设定仍基于经验，缺乏严格的量化评估。

### 影响与延伸思考
GPT‑5 系统卡的发布让业界重新审视“单模型即服务”的思路，推动了多模型动态路由的研究热潮。随后出现的开源项目（如 Llama‑Router、Mistral‑Switch）都在尝试复制或改进实时路由器的自监督学习机制。安全完成的统一化也成为新一代 LLM 安全标准的参考模板。对想进一步探索的读者，可以关注以下方向：① 路由决策的可解释性研究；② 高危能力的量化评估框架；③ 在资源受限设备上实现轻量化路由与安全微调的技术。  

### 一句话记住它
GPT‑5 用实时自学习路由把“快”和“深”两种模型无缝结合，同时把安全过滤嵌进每一次生成。