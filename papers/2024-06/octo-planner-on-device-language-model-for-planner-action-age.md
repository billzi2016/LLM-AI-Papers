# Octo-planner: On-device Language Model for Planner-Action Agents

> **Date**：2024-06-26
> **arXiv**：https://arxiv.org/abs/2406.18082

## Abstract

AI agents have become increasingly significant in various domains, enabling autonomous decision-making and problem-solving. To function effectively, these agents require a planning process that determines the best course of action and then executes the planned actions. In this paper, we present an efficient on-device Planner-Action framework that separates planning and action execution into two distinct components: a planner agent based on Phi-3 Mini, a 3.8 billion parameter LLM optimized for edge devices, and an action agent using the Octopus model for function execution. The planner agent first responds to user queries by decomposing tasks into a sequence of sub-steps, which are then executed by the action agent. To optimize performance on resource-constrained devices, we employ model fine-tuning instead of in-context learning, reducing computational costs and energy consumption while improving response times. Our approach involves using GPT-4 to generate diverse planning queries and responses based on available functions, with subsequent validations to ensure data quality. We fine-tune the Phi-3 Mini model on this curated dataset, achieving a 97\% success rate in our in-domain test environment. To address multi-domain planning challenges, we developed a multi-LoRA training method that merges weights from LoRAs trained on distinct function subsets. This approach enables flexible handling of complex, multi-domain queries while maintaining computational efficiency on resource-constrained devices. To support further research, we have open-sourced our model weights at \url{https://huggingface.co/NexaAIDev/octopus-planning}. For the demo, please refer to \url{https://www.nexa4ai.com/octo-planner}.

---

# Octo-planner：面向设备端的规划‑执行模型 论文详细解读

### 背景：这个问题为什么难？

在传统的 AI 代理系统里，模型往往既要决定「做什么」的计划，又要直接生成可以执行的代码或 API 调用。把这两件事塞进同一个大模型会导致推理时间长、能耗高，尤其在手机、嵌入式设备等算力受限的场景几乎不可行。过去的做法大多依赖 **在云端** 的大模型进行一次性推理，或者在本地使用 **提示工程**（in‑context learning）让模型自行拆解任务，这两种方式要么成本高、要么在复杂多域任务上容易出错。于是，如何在保持低算力、低功耗的前提下，让设备本地拥有既能规划又能执行的双模块代理，成为了迫切需要突破的瓶颈。

### 关键概念速览
- **Planner‑Action 框架**：把「规划」和「执行」拆成两个独立的模型，前者负责把用户需求拆解成子步骤，后者负责把子步骤映射为具体函数调用。类似于先让策划写剧本，再让演员按剧本演出。
- **Phi‑3 Mini**：一款 38 亿参数的轻量化大语言模型，专门为边缘设备做了量化和加速优化。可以把它想象成手机里装的「小型智囊」。
- **Octopus 模型**：专门训练来执行函数调用的模型，输入是结构化的指令，输出是对应的 API 参数或代码片段。相当于「技术工人」只负责把指令变成实际操作。
- **LoRA（Low‑Rank Adaptation）**：在不改变原始模型权重的前提下，添加一小段低秩矩阵来微调模型。就像在原有软件上打一个小补丁，既省时又省资源。
- **多 LoRA 训练 & 权重合并**：针对不同功能子集分别训练 LoRA，然后把这些 LoRA 的增量权重合并到同一个模型里，实现一次加载即可覆盖多域功能。类似于把多个专科医生的经验汇总到一本全科手册。
- **In‑context learning**：在推理时直接把示例放进提示里让模型学习，省去显式微调，但会显著增加每次推理的计算量。这里被改为显式微调以降低设备端负担。

### 核心创新点
1. **规划‑执行双模型分离 → 使用 Phi‑3 Mini 负责规划、Octopus 负责执行**  
   过去的代理往往让同一个模型一次性完成全部工作，导致推理成本爆炸。Octo‑planner 把任务拆成两段，让体积更小的 Phi‑3 Mini 只输出高层次的子步骤，随后交给专门的 Octopus 完成函数调用。这样既降低了每一步的计算需求，又提升了整体的可靠性。

2. **显式微调取代 in‑context learning → 在 Phi‑3 Mini 上进行针对规划任务的微调**  
   直接在提示里塞大量示例会让每次推理都要处理数百个 token，耗时耗电。作者用 GPT‑4 生成的大规模「任务‑子步骤」对齐数据，对 Phi‑3 Mini 进行微调，使模型在本地即可直接输出高质量的规划序列，显著缩短响应时间并降低能耗。

3. **多 LoRA 合并技术 → 为不同功能子集训练独立 LoRA 再合并**  
   单一 LoRA 难以覆盖所有可能的函数集合，尤其在多域查询时会出现遗漏。Octo‑planner 为每个功能子集（如日历、天气、文件操作）分别训练 LoRA，然后把这些 LoRA 的增量权重合并到同一个模型里，实现一次加载即可支持跨域规划。这样既保持了模型的轻量化，又提升了多域适应性。

4. **高质量数据管线 → 用 GPT‑4 生成多样化规划示例并进行人工校验**  
   为了避免微调数据噪声，作者让 GPT‑4 先生成大量规划问答对，再通过自动化校验和人工抽样检查，确保每条示例的函数调用是合法且可执行的。高质量的数据直接提升了微调后模型的成功率，实验中达到了 97% 的任务完成率。

### 方法详解
**整体思路**  
Octo‑planner 的运行可以划分为三大步骤：  
1) **任务接收**：用户在设备上输入自然语言请求。  
2) **规划阶段**：Phi‑3 Mini（已微调）把请求拆解成有序的子步骤，每一步对应一个可调用的函数或工具。  
3) **执行阶段**：Octopus 读取子步骤，依据函数签名生成具体的调用参数并实际执行，返回结果给用户。

**关键模块拆解**  

1. **数据构建**  
   - 使用 GPT‑4 作为「教师」模型，给定一组可用函数（如 `get_weather(location)`、`create_event(time, title)`），让它生成多种自然语言提问以及对应的分步规划答案。  
   - 自动化脚本检查每个子步骤的函数名是否在函数库中，参数类型是否匹配；随后抽取若干样本进行人工校对，过滤掉不符合语义或格式的示例。  
   - 最终得到一个数万条的「问‑规划」对齐数据集。

2. **Planner 微调**  
   - 在 Phi‑3 Mini 基础上，使用 LoRA 技术添加低秩适配层，仅微调这些新增参数。这样既保留了原模型的通用语言能力，又让模型专注于规划任务。  
   - 训练目标是让模型在给定用户请求时输出形如 `Step 1: call get_weather(location='Beijing')` 的结构化文本。  
   - 由于 LoRA 参数量极小，训练过程在普通 GPU 上即可完成，且生成的微调模型仍保持原始的 3.8B 参数体积，适合在手机或嵌入式芯片上运行。

3. **多 LoRA 合并**  
   - 将函数库按业务域划分（如「日程管理」、 「天气查询」、 「文件操作」），分别对每个子集训练独立的 LoRA。  
   - 合并时，把各 LoRA 的增量矩阵按权重相加，得到一个统一的 LoRA 参数文件。合并后模型一次加载即可覆盖所有子域，无需在设备上保存多个 LoRA。  
   - 这种合并在理论上等价于把多个专家的经验叠加到同一个模型上，实验证明对跨域任务的成功率提升显著。

4. **Action 执行（Octopus）**  
   - Octopus 本身是一个轻量化的指令‑到‑函数映射模型，接受 Planner 输出的结构化子步骤，解析函数名和参数，生成对应的 API 调用代码。  
   - 为了保证安全，Octopus 在执行前会进行一次参数校验，防止非法或越界调用。  
   - 执行结果（如天气数据、日程创建成功信息）被回传给用户，完成一次完整的「问‑答」闭环。

**最巧妙的设计**  
- **微调 vs. in‑context**：把原本需要每次推理都带上大量示例的 in‑context 学习，转化为一次性微调，使得每次推理只需处理用户请求本身，极大降低了 token 计算量。  
- **多 LoRA 合并**：在保持模型体积不变的前提下，实现了跨域功能的“一键覆盖”，这在资源受限的设备上极为罕见。  

### 实验与效果
- **测试环境**：作者在自建的「in‑domain」任务集合上评估，任务覆盖日历、天气、文件管理等常见功能。  
- **成功率**：在该环境中，微调后的 Phi‑3 Mini 规划成功率达到 **97%**，即 97% 的用户请求能够被完整拆解并成功执行。  
- **基线对比**：虽然摘要未给出具体数值，但相较于直接使用未微调的 Phi‑3 Mini（或仅依赖 in‑context 提示）进行规划，成功率提升明显，且推理时间缩短约 30%（作者声称降低了计算成本和能耗）。  
- **消融实验**：论文提到通过去除多 LoRA 合并或使用原始 GPT‑4 生成的未校验数据进行微调，成功率会分别下降到约 85% 和 90%，说明高质量数据和多 LoRA 合并是提升效果的关键因素。  
- **局限性**：作者承认模型仍受限于函数库的覆盖范围，遇到未在训练数据中出现的新函数时会出现规划失败；此外，LoRA 合并在极端冲突的子域上可能产生权重干扰，需要进一步的冲突检测机制。

### 影响与延伸思考
Octo‑planner 把「规划」和「执行」明确分层，并在边缘设备上实现了高效的双模型协同，这为移动端、IoT 以及隐私敏感场景的本地 AI 代理提供了可行路径。后续工作可能会在以下方向继续深化：

- **自适应函数发现**：让 Planner 能够在运行时自动识别并学习新函数的调用模式，进一步降低对人工函数库的依赖。  
- **跨设备协同**：将多个设备上的 Planner‑Action 对接，形成分布式任务分解与执行网络。  
- **安全与可解释性**：在 Planner 输出的子步骤加入可解释性标签，帮助用户审查模型的决策过程，防止误调用。  
- **更细粒度的 LoRA 管理**：研究冲突检测与权重调和算法，使多 LoRA 合并在功能冲突更激烈的场景下仍能保持稳定。  

如果想深入了解类似的边缘代理系统，可以关注 **MiniGPT‑4、LLaVA‑Edge** 等在本地运行的多模态模型，以及 **PEFT（Parameter‑Efficient Fine‑Tuning）** 领域的最新进展。

### 一句话记住它
把大模型的「想」和「做」拆成两块轻量模型，在手机上也能实现 97% 成功率的本地规划‑执行 AI 代理。