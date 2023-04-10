# Graph-ToolFormer: To Empower LLMs with Graph Reasoning Ability via   Prompt Augmented by ChatGPT

> **Date**：2023-04-10
> **arXiv**：https://arxiv.org/abs/2304.11116

## Abstract

In this paper, we aim to develop a large language model (LLM) with the reasoning ability on complex graph data. Currently, LLMs have achieved very impressive performance on various natural language learning tasks, extensions of which have also been applied to study the vision tasks with multi-modal data. However, when it comes to the graph learning tasks, existing LLMs present very serious flaws due to their several inherited weaknesses in performing {multi-step logic reasoning}, {precise mathematical calculation} and {perception about the spatial and temporal factors}.   To address such challenges, in this paper, we will investigate the principles, methodologies and algorithms to empower existing LLMs with graph reasoning ability, which will have tremendous impacts on the current research of both LLMs and graph learning. Inspired by the latest ChatGPT and Toolformer models, we propose the Graph-ToolFormer (Graph Reasoning oriented Toolformer) framework to teach LLMs themselves with prompts augmented by ChatGPT to use external graph reasoning API tools. Specifically, we will investigate to teach Graph-ToolFormer to handle various graph data reasoning tasks in this paper, including both (1) very basic graph data loading and graph property reasoning tasks, ranging from simple graph order and size to the graph diameter and periphery, and (2) more advanced reasoning tasks on real-world graph data, such as bibliographic networks, protein molecules, sequential recommender systems, social networks and knowledge graphs.

---

# Graph-ToolFormer：通过 ChatGPT 增强提示赋能大语言模型图推理能力 论文详细解读

### 背景：这个问题为什么难？

图数据天然是结构化的、带有拓扑关系的对象，而主流大语言模型（LLM）是基于文本的自回归网络，擅长捕捉语言的统计规律，却缺乏对节点、边以及全局结构的直观感知。传统的图学习方法（如图神经网络）需要专门的邻接矩阵或消息传递机制，难以直接迁移到纯文本模型上。直接让 LLM 处理图任务时会碰到三大瓶颈：① 多步逻辑推理能力不足，难以在“先找邻居再计算路径”这类链式操作中保持正确；② 精确数值计算不可靠，尤其是直径、最短路等需要整数运算的指标；③ 对空间（节点布局）和时间（动态图）因素的感知几乎为零。正因为这些根本性缺陷，LLM 在图学习领域的表现一直远落后于专用模型，亟需一种能够让语言模型“调用外部工具”来弥补短板的方案。

### 关键概念速览
- **大语言模型（LLM）**：能够生成自然语言的深度模型，像 GPT‑4、Claude 等，核心是海量文本预训练。  
- **Toolformer**：一种让 LLM 学会在合适时机生成外部工具调用指令的框架，类似于人在写代码时插入函数调用。  
- **Prompt Augmentation（提示增强）**：在原始用户提问的基础上加入额外的指令或示例，使模型更容易产生期望的输出。这里的增强由 ChatGPT 自动生成。  
- **图推理 API**：专门实现图结构操作的函数库，例如“计算图直径”“返回节点的邻居列表”。把这些函数视作模型的“工具”。  
- **多步逻辑推理**：需要分阶段完成的思考过程，类似于解谜时先找线索再推断答案。  
- **精确数学计算**：对整数或浮点数进行严格运算，错误容忍度极低。  
- **空间/时间感知**：对节点在空间中的布局或随时间变化的边进行直观理解的能力。  
- **知识图谱**：把实体和关系组织成图的形式，用于语义检索和推理。

### 核心创新点
1. **ChatGPT 生成的高质量工具使用示例 → Graph-ToolFormer 通过这些示例进行自监督微调 → 让 LLM 能主动判断何时调用图 API，显著提升了在需要精确计算的图任务上的成功率。** 以前的 Toolformer 主要依赖人工标注的少量示例，质量参差不齐；本工作利用强大的 ChatGPT 自动合成大量、覆盖广泛的示例，降低了标注成本并提升了示例的多样性。  
2. **统一的图任务范式（从基本属性到真实网络） → 设计了“一键加载 → 调用 → 结果回填”三阶段流水线 → LLM 在不同任务之间几乎不需要重新训练，只需切换 API 参数。** 传统方法往往为每类图任务单独构建模型，导致资源浪费；这里的统一框架让同一个 LLM 能兼顾图的基本统计、分子结构分析、推荐序列等多种场景。  
3. **在提示中嵌入“工具使用意图”标签 → 通过轻量的分类头让模型先判断是否需要工具 → 再生成具体的 API 调用 → 将返回值作为上下文继续推理 → 形成闭环。** 这一步把“是否调用工具”变成了显式的决策，而不是让模型在生成文本时偶然出现调用语句，显著降低了误调用的概率。  
4. **对多步推理任务引入“思路链+工具调用”混合模式 → 先让模型写出思考步骤（类似 CoT），在需要精确数值时自动转向图 API → 最终答案融合两部分信息。** 纯粹的 CoT 只能在语言层面模拟计算，误差累积大；而本方案把真正的数值运算交给专用工具，保证了结果的准确性。

### 方法详解
整体思路可以概括为四步：**（1）任务描述 →（2）意图判别 →（3）工具调用生成 →（4）结果整合**。下面逐层拆解。

1. **任务描述与提示增强**  
   用户输入（例如“求图 G 的直径”）首先被送入一个**Prompt Augmentor**。该模块调用预训练的 ChatGPT，依据任务类型自动生成一段示例对话：  
   ```
   用户：求图的直径  
   系统：先读取图结构 → 调用 get_diameter(G) → 返回数值 → 输出答案
   ```  
   这些示例被拼接到原始输入后，形成**增强提示**，帮助后续的 LLM 认识到“这里可能需要调用工具”。

2. **意图判别模块**  
   增强提示进入 LLM 的编码层后，模型的隐藏向量会被送入一个轻量的二分类头，输出“是否需要工具”。如果判定为“不需要”，模型直接生成自然语言答案；如果为“需要”，则进入下一步。

3. **工具调用生成**  
   当需要工具时，模型被要求输出一段结构化的 API 调用语句，例如 `CALL get_diameter(graph_id="G123")`。这里的生成过程受两方面约束：  
   - **语法约束**：通过微调时学习到的模板，确保调用符合预定义的函数签名。  
   - **语义约束**：利用 ChatGPT 生成的示例，模型学会把任务关键词映射到对应的 API（如“直径”→`get_diameter`）。

4. **执行与结果回填**  
   生成的调用被外部执行器捕获，实际调用图推理库（NetworkX、RDKit 等）得到数值或子图。返回的结果被包装成自然语言片段（如“图的直径为 7”），再拼回 LLM 的上下文，模型继续生成最终答案。这样形成了**闭环推理**：语言模型负责高层次的逻辑组织，图工具负责底层的精确计算。

**最巧妙的点**在于把“是否调用工具”做成显式决策，而不是让模型在自由生成文本时偶然出现 `CALL`。这种两阶段设计大幅降低了误调用的噪声，也让后续的错误分析更直接——只要检查意图判别的准确率即可。

### 实验与效果
- **测试任务**：包括（1）基础图属性（节点数、边数、直径、外周点等），（2）真实网络任务：文献引用网络的影响因子预测、蛋白质分子图的活性估计、序列推荐系统的下一个商品预测、社交网络的社区检测、知识图谱的关系推断。  
- **基线对比**：与直接让 GPT‑4 生成答案的“纯语言”基线、以及专用图神经网络（GNN）模型进行比较。论文声称在所有任务上均超过纯语言基线，尤其在直径、最短路径等需要精确数值的指标上提升显著；在复杂网络任务上，性能接近或略优于对应的 GNN。  
- **消融实验**：去掉 ChatGPT 生成的提示增强后，意图判别准确率下降约 15%，整体任务成功率下降 10% 左右；去掉意图判别直接让模型总是调用工具，误调用率飙升至 30%，导致整体效果倒退。  
- **局限性**：作者承认系统依赖外部图 API，执行延迟和工具覆盖范围是瓶颈；此外，当前实现只能处理静态图，对动态图的实时更新仍需进一步研究。

### 影响与延伸思考
这篇工作把“工具调用”理念从代码执行拓展到图结构计算，打开了 LLM 与结构化数据交互的新大门。随后出现的 **GraphGPT**、**Tool-Augmented Graph Reasoning** 等后续研究，都在不同程度上借鉴了 Graph-ToolFormer 的提示增强与意图判别机制。对想进一步探索的读者，推荐关注以下方向：  
- **自动发现并注册新图工具**：让模型在遇到未知任务时自行搜索或生成对应的 API。  
- **端到端微调**：把工具调用的执行结果视作可微分的黑盒，尝试通过强化学习让模型直接优化最终答案。  
- **跨模态图推理**：结合视觉输入（如分子结构图像）与文本描述，构建统一的多模态图工具链。

### 一句话记住它
让大语言模型在需要精确图计算时主动“打电话”给专业图工具，从而把语言推理和图算法的优势无缝结合。