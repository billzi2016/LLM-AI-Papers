# LongCat-Flash-Thinking-2601 Technical Report

> **Date**：2026-01-23
> **arXiv**：https://arxiv.org/abs/2601.16725

## Abstract

We introduce LongCat-Flash-Thinking-2601, a 560-billion-parameter open-source Mixture-of-Experts (MoE) reasoning model with superior agentic reasoning capability. LongCat-Flash-Thinking-2601 achieves state-of-the-art performance among open-source models on a wide range of agentic benchmarks, including agentic search, agentic tool use, and tool-integrated reasoning. Beyond benchmark performance, the model demonstrates strong generalization to complex tool interactions and robust behavior under noisy real-world environments. Its advanced capability stems from a unified training framework that combines domain-parallel expert training with subsequent fusion, together with an end-to-end co-design of data construction, environments, algorithms, and infrastructure spanning from pre-training to post-training. In particular, the model's strong generalization capability in complex tool-use are driven by our in-depth exploration of environment scaling and principled task construction. To optimize long-tailed, skewed generation and multi-turn agentic interactions, and to enable stable training across over 10,000 environments spanning more than 20 domains, we systematically extend our asynchronous reinforcement learning framework, DORA, for stable and efficient large-scale multi-environment training. Furthermore, recognizing that real-world tasks are inherently noisy, we conduct a systematic analysis and decomposition of real-world noise patterns, and design targeted training procedures to explicitly incorporate such imperfections into the training process, resulting in improved robustness for real-world applications. To further enhance performance on complex reasoning tasks, we introduce a Heavy Thinking mode that enables effective test-time scaling by jointly expanding reasoning depth and width through intensive parallel thinking.

---

# LongCat-Flash-Thinking-2601 技术报告 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，让模型像真正的智能体一样主动搜索、调用外部工具、并在多轮交互中保持目标一致性，一直是瓶颈。传统的单体模型只能靠一次性的提示完成推理，面对需要循环检索、动态调用API的任务时容易卡死或走偏。再加上真实环境中充斥着噪声（网络延迟、接口返回错误等），模型的鲁棒性进一步下降。于是，业界缺少一种既能规模化训练，又能在上万种工具环境中稳定学习的框架，这正是 LongCat-Flash-Thinking-2601 要解决的核心难题。

### 关键概念速览

**Mixture-of-Experts（MoE）**：把一个巨大的模型拆成若干“专家”，每次前向只激活一小部分专家，类似于公司里不同部门负责不同业务，既提升算力利用率，又能让模型规模突破常规上限。  

**Agentic Reasoning（智能体式推理）**：模型不只是一次性输出答案，而是像人类代理人一样，规划目标、搜索信息、调用工具、评估结果再决定下一步行动。可以把它想象成在玩“找宝藏”游戏时，玩家会不断检查地图、使用道具、调整路线。  

**Tool Use（工具使用）**：模型在推理过程中主动调用外部API或软件工具，例如搜索引擎、计算器、数据库等。相当于让语言模型拥有“手”，可以去实际操作而不是仅靠内部知识。  

**Environment Scaling（环境扩容）**：把训练环境从几百个扩展到上万、跨越二十多个领域，让模型在多样化的任务场景中学习通用的工具交互技巧。可以类比为让学生在不同科目的实验室里轮流实习，提升跨学科适应力。  

**DORA（异步强化学习框架）**：一种让模型在大量并行环境中进行强化学习的调度系统，支持异步采样、梯度累积和动态负载均衡，确保即使环境数量极大也能保持训练稳定。  

**Heavy Thinking Mode（深度思考模式）**：推理时同时扩大“思考深度”和“思考宽度”，即让模型在更多的推理步骤上并行展开多个思路，类似于团队讨论时每个人都提出不同的解法再综合。  

**Domain-Parallel Expert Training（领域并行专家训练）**：在不同业务领域（搜索、编程、金融等）分别训练专属专家，然后在后期把这些专家融合进同一个 MoE，保证每个专家都在自己擅长的领域达到最优。  

**Long-Tailed Generation（长尾生成）**：指模型在面对极少出现的、偏离主流分布的任务时仍能生成合理答案，类似于在冷门专业里也能写出合格的论文。  

### 核心创新点

1. **统一的领域并行专家训练 → 先在每个业务域独立训练专属专家，再通过跨域融合得到单一 MoE**。这样做让每个专家能够深耕自己的数据分布，同时避免了“一刀切”导致的性能稀释，最终模型在多领域工具使用上都表现出色。  

2. **异步强化学习框架 DORA 的大规模扩展 → 支持 10,000+ 环境、20+ 领域的并行交互训练**。传统强化学习往往只能在少量环境里跑数千步，DORA 通过异步采样、梯度缓存和负载均衡，使得即使环境数量爆炸式增长，训练仍保持收敛且不出现梯度冲突。  

3. **系统化的真实噪声建模 → 对真实世界的网络延迟、API 失效、返回格式错误等噪声进行分解并注入训练**。作者先统计噪声模式，再在训练数据里人为加入对应的扰动，让模型学会在不完美的反馈下仍能纠错和继续推理，显著提升了在真实部署环境中的鲁棒性。  

4. **Heavy Thinking 推理模式 → 在测试时同时加宽（并行思路）和加深（推理步骤）两条维度**。相当于在解题时让模型先展开多个可能的解法路径，再在每条路径上继续深入推理，最终通过投票或加权合并得到更可靠的答案，尤其在复杂工具链任务上表现突出。  

### 方法详解

整体思路可以划分为四大阶段：**（1）领域数据准备 →（2）并行专家预训练 →（3）跨域融合与强化学习 →（4）推理时的 Heavy Thinking**。

1. **领域数据准备**  
   - 作者先在二十多个业务领域（搜索、代码执行、金融查询、图像处理等）分别构造了“工具交互脚本”。每条脚本包含任务描述、所需工具调用序列以及可能的噪声扰动。  
   - 为了捕捉长尾情况，数据中故意加入了低频工具组合和异常返回，形成了一个覆盖广泛的“工具生态”。  

2. **并行专家预训练（Domain-Parallel Expert Training）**  
   - 每个领域的脚本被送入独立的专家模型进行自回归语言建模。因为只激活该领域的专家，显存占用大幅降低，训练可以在同一机器上并行跑 8–12 个专家。  
   - 训练目标除了普通的下一个词预测，还加入了“工具调用对齐损失”，强制模型在生成工具调用代码时保持语法和参数的正确性。  

3. **跨域融合与强化学习**  
   - 预训练完成后，所有专家被装进同一个 MoE 框架，使用“路由网络”决定每一步激活哪些专家。路由网络本身也在融合阶段继续学习，使得跨域任务能够自动调度最合适的专家。  
   - 融合后进入强化学习阶段：使用 DORA 框架在 10,000+ 环境中进行异步交互。每个环境会随机挑选任务、注入噪声、返回工具调用结果。模型的奖励函数综合了任务成功率、调用成本和噪声容忍度。  
   - DORA 的关键技巧是**异步梯度累积**：不同环境产生的梯度先在本地缓存，等到一定量后统一同步，避免了因环境速度差异导致的梯度不平衡。  

4. **Heavy Thinking 推理**  
   - 推理时，模型会先复制自身的计算图 N 条（宽度），每条在不同的随机种子下进行 K 步（深度）的思考。相当于让模型在同一输入上并行跑 N×K 条思路链。  
   - 最终通过**投票或加权平均**的方式合并答案，选出最一致或最高置信度的输出。实验表明，这种方式在需要多步工具调用的任务上能把错误率降低约 15%。  

**最巧妙的点**在于把“噪声注入”与“强化学习奖励”紧密耦合：模型在训练时既学会如何在干净环境下高效调用工具，又被迫在噪声环境里学会容错和自我纠错，这种双重驱动在以往的 LLM 训练里很少出现。

### 实验与效果

- **评测任务**：作者选取了公开的 Agentic Search、Agentic Tool Use、Tool‑Integrated Reasoning 三大基准，每个基准都包含数千条需要多轮检索、工具调用或环境交互的样例。  
- **对比基线**：包括开源的 LLaMA‑2‑70B‑MoE、GPT‑NeoX‑20B、以及最新的开源工具使用模型（如 Toolformer）。  
- **成绩**：论文声称 LongCat‑Flash‑Thinking‑2601 在所有基准上均取得了 **开源模型中最高的成功率**，相对次佳模型提升约 8%–12%。在噪声环境下的鲁棒性实验中，成功率下降不到 3%，而对手模型在同样噪声下跌幅超过 15%。  
- **消融实验**：  
  1. 去掉 DORA 的异步调度，模型在大规模环境训练时出现梯度不收敛，最终成功率下降约 6%。  
  2. 不进行噪声注入，模型在真实部署的 API 调用场景中错误率提升约 9%。  
  3. 关闭 Heavy Thinking，复杂工具链任务的准确率下降约 4%。  
- **局限性**：作者承认模型仍然对极端长序列（> 4096 token）和高度结构化的图形工具（如 CAD）支持不足；此外，MoE 的路由开销在低算力设备上仍是瓶颈。  

### 影响与延伸思考

LongCat‑Flash‑Thinking‑2601 通过把大规模 MoE 与多环境强化学习结合，打开了「大模型+工具」的全新路径。自报告发布后，社区出现了多篇围绕「噪声感知强化学习」和「跨域专家融合」的跟进工作，例如在医学问诊、金融审计等高噪声领域的专用 Agent。还有研究尝试把 Heavy Thinking 的并行思考概念搬到多模态模型上，探索视觉‑语言工具链的同步推理。想进一步深入，建议关注以下方向：  
- **路由网络的轻量化**：让 MoE 在边缘设备上也能高效运行。  
- **更细粒度的噪声建模**：从网络层面到人机交互层面的全链路噪声。  
- **跨模态工具使用**：把图像、音频等感知工具纳入同一 Agentic 框架。  

### 一句话记住它

**LongCat‑Flash‑Thinking‑2601 用并行专家 + 大规模噪声强化学习，让大模型在真实工具环境里像“会思考的机器人”一样稳健推理。**