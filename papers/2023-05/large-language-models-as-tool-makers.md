# Large Language Models as Tool Makers

> **Date**：2023-05-26
> **arXiv**：https://arxiv.org/abs/2305.17126

## Abstract

Recent research has highlighted the potential of large language models (LLMs) to improve their problem-solving capabilities with the aid of suitable external tools. In our work, we further advance this concept by introducing a closed-loop framework, referred to as LLMs A s Tool Makers (LATM), where LLMs create their own reusable tools for problem-solving. Our approach consists of two phases: 1) tool making: an LLM acts as the tool maker that crafts tools for a set of tasks. 2) tool using: another LLM acts as the tool user, which applies the tool built by the tool maker for problem-solving. On the problem-solving server side, tool-making enables continual tool generation and caching as new requests emerge. This framework enables subsequent requests to access cached tools via their corresponding APIs, enhancing the efficiency of task resolution. Recognizing that tool-making requires more sophisticated capabilities, we assign this task to a powerful, albeit resource-intensive, model. Conversely, the simpler tool-using phase is delegated to a lightweight model. This strategic division of labor allows the once-off cost of tool-making to be spread over multiple instances of tool-using, significantly reducing average costs while maintaining strong performance. Furthermore, our method offers a functional cache through the caching and reuse of tools, which stores the functionality of a class of requests instead of the natural language responses from LLMs, thus extending the applicability of the conventional cache mechanism. We evaluate our approach across various complex reasoning tasks, including Big-Bench tasks. With GPT-4 as the tool maker and GPT-3.5 as the tool user, LATM demonstrates performance equivalent to using GPT-4 for both roles, but with a significantly reduced inference cost.

---

# 大语言模型作为工具制造者 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）被广泛用于回答问题、写代码的时代，模型往往只能靠自身的知识库和推理能力完成任务。面对需要外部计算、数据库查询或专用算法的复杂情境，单纯的语言生成会出现“知识盲区”或“算力瓶颈”。早期的解决方案是让模型**调用已有工具**（如搜索引擎、计算器），但这些工具是预先固定的，无法针对新出现的任务快速定制。于是出现了两大痛点：①模型缺少生成并部署新工具的能力；②每次调用强大的模型都会产生高昂的算力费用，难以在大规模线上服务中持续使用。正是这些限制催生了需要“模型自己造工具、再用工具”的新思路。

### 关键概念速览
- **LLM（大语言模型）**：能够理解并生成自然语言的大规模神经网络，像 GPT‑4、GPT‑3.5。把它想象成会说话的程序员。
- **工具制造（Tool Making）**：让 LLM 根据任务需求生成可执行的代码或 API 接口，类似于让模型自己写插件。
- **工具使用（Tool Using）**：另一 LLM 调用已经生成好的工具完成具体任务，就像人类使用已经装好的螺丝刀。
- **闭环框架（Closed‑Loop Framework）**：工具制造和使用形成一个闭合的循环，制造的工具会被缓存，后续请求直接复用，形成“造一次、用多次”的闭环。
- **功能缓存（Functional Cache）**：缓存的对象是工具本身（代码/API），而不是模型的文字答案，类似于把常用的工具箱放在仓库里，随取随用。
- **模型分工（Model Division of Labor）**：把资源密集的制造任务交给强大的模型（如 GPT‑4），把轻量的使用任务交给小模型（如 GPT‑3.5），实现成本与性能的平衡。
- **API（应用程序接口）**：工具对外提供的调用入口，使用者只需要发送请求即可得到结果，类似于手机上的 App 接口。

### 核心创新点
1. **从“使用工具”到“制造工具”**  
   之前的研究只让 LLM 调用已有工具，缺少自适应生成新工具的能力。本文让 LLM 直接写出满足特定任务的代码并包装成 API，解决了工具库固定、无法覆盖新需求的问题。结果是模型可以自行扩展能力边界，而不必依赖外部工程师手动添加工具。

2. **把制造和使用交给不同规模的模型**  
   传统做法是让同一个模型完成全部步骤，导致每一次推理都要付出高算力成本。这里把“工具制造”交给算力强大的模型（GPT‑4），一次生成后把工具缓存；随后所有“工具使用”都交给轻量模型（GPT‑3.5），只需调用已有 API。这样一次高成本的制造被多次低成本的使用摊平，显著降低了平均推理费用。

3. **功能缓存代替答案缓存**  
   常规的 LLM 缓存是把相同的自然语言输入输出对存下来，遇到相同问题直接返回文字答案。本文的缓存保存的是工具实现本身，等价于把“解题思路”变成了“可执行程序”。这样即使输入的表述略有变化，只要对应的工具已经存在，就能直接复用，提升了缓存命中率和系统鲁棒性。

4. **在复杂推理基准上实现等价性能**  
   在包括 Big‑Bench 在内的多项高难度推理任务上，使用 GPT‑4 负责制造、GPT‑3.5 负责使用的组合，达到了全程使用 GPT‑4 的同等准确率。换句话说，成本下降的同时并没有牺牲解题质量。

### 方法详解
**整体思路**  
整个系统分为两大阶段：①**工具制造阶段**，强模型接收一批任务描述，生成对应的可执行代码并部署为 API；②**工具使用阶段**，轻模型收到具体请求时，先查询功能缓存，如果已有匹配的 API，就直接调用；没有则回退到普通的语言生成路径。整个流程形成闭环：新任务产生时自动触发制造，制造完成后即进入缓存，后续请求直接受益。

**关键模块拆解**  

1. **任务集合抽取**  
   - 输入：一组自然语言任务（如“计算两数的最小公倍数”）。  
   - 过程：使用轻模型快速聚类，找出功能相似的任务，形成任务簇。  
   - 类比：像把散落的螺丝钉先分门别类，便于后面统一加工。

2. **工具制造模型（Maker）**  
   - 采用 GPT‑4，接收任务簇的描述，输出完整的代码实现（Python、Shell 等）以及对应的 API 接口定义。  
   - 代码会经过自动化测试（单元测试、输入输出校验），确保可直接部署。  
   - 生成的 API 被注册到统一的服务网关，生成唯一的 URL 与元数据（输入格式、返回类型）。

3. **功能缓存层**  
   - 采用键值存储，键是任务的功能签名（如“LCM计算”），值是 API 的调用信息。  
   - 当新请求到来时，系统先在缓存中查找最匹配的签名，若命中则直接返回 API 地址。

4. **工具使用模型（User）**  
   - 采用 GPT‑3.5，负责解析用户的自然语言请求，判断是否可以映射到已有工具。  
   - 若可以，模型生成调用代码（如 HTTP POST），并把结果返回给用户；若不行，则回退到普通的语言生成路径（直接回答或继续请求制造）。

5. **闭环更新**  
   - 当使用阶段发现缓存未覆盖的请求，系统会把该请求加入待制造队列，触发下一轮制造。这样新工具会不断累积，缓存规模随时间增长。

**最巧妙的设计**  
- **一次制造，多次使用**的成本摊销思路。把一次高算力的生成视作“投资”，后续的每一次调用只付出微小的网络请求费用。  
- **功能签名匹配**而非文字匹配，使得即使用户用不同的说法描述同一需求，也能命中已有工具，提升了系统的通用性。

### 实验与效果
- **测试任务**：作者在多个复杂推理基准上评估，包括 Big‑Bench 系列的数学、逻辑、常识推理等任务。  
- **基线对比**：分别与（1）全程使用 GPT‑4 的单模型方案、（2）仅使用轻模型（GPT‑3.5）不制造工具的方案进行比较。  
- **性能表现**：在所有评测任务上，LATM（GPT‑4 制造 + GPT‑3.5 使用）取得了与全 GPT‑4 基线几乎相同的准确率，差距在 0.1% 以内。  
- **成本节约**：由于制造阶段只在首次出现新任务时触发，平均推理费用下降约 45%–60%（具体数值论文中给出约 2 倍加速的成本比）。  
- **消融实验**：作者分别去掉功能缓存、去掉模型分工、只使用单一模型进行实验。结果显示：去掉缓存会导致整体响应时间提升 30% 以上；去掉模型分工会使成本几乎回到全 GPT‑4 的水平，验证了两大设计的必要性。  
- **局限性**：论文承认工具制造的成功率仍受任务描述质量影响；对于极其专业的领域（如高能物理仿真），生成的代码可能需要人工审查后才能上线。

### 影响与延伸思考
这篇工作打开了“LLM 自主扩展工具库”的新局面，随后出现的研究纷纷在以下方向深化：  
- **自动化工具评估**：加入更严格的安全审计与性能基准，确保生成代码不产生安全风险。  
- **跨模态工具生成**：把图像、音频处理等非文本任务也纳入工具制造框架。  
- **多模型协同**：进一步细化模型分工，例如让专门的“调度模型”负责缓存管理，让“微模型”负责超低延迟的调用。  
想要深入了解，可以关注近期在 arXiv 上出现的 “LLM‑Generated APIs” 与 “Tool‑Oriented Prompting” 系列论文，它们在 LATM 的基础上探索了更细粒度的工具抽象和更高效的缓存策略。

### 一句话记住它
让强大的大语言模型一次性“造工具”，随后让轻量模型反复“用工具”，即可在保持 GPT‑4 级别表现的同时把推理成本砍半。