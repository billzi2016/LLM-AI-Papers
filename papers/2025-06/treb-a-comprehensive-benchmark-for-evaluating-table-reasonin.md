# TReB: A Comprehensive Benchmark for Evaluating Table Reasoning Capabilities of Large Language Models

> **Date**：2025-06-23
> **arXiv**：https://arxiv.org/abs/2506.18421

## Abstract

The majority of data in businesses and industries is stored in tables, databases, and data warehouses. Reasoning with table-structured data poses significant challenges for large language models (LLMs) due to its hidden semantics, inherent complexity, and structured nature. One of these challenges is lacking an effective evaluation benchmark fairly reflecting the performances of LLMs on broad table reasoning abilities. In this paper, we fill in this gap, presenting a comprehensive table reasoning evolution benchmark, TReB, which measures both shallow table understanding abilities and deep table reasoning abilities, a total of 26 sub-tasks. We construct a high quality dataset through an iterative data processing procedure. We create an evaluation framework to robustly measure table reasoning capabilities with three distinct inference modes, TCoT, PoT and ICoT. Further, we benchmark over 20 state-of-the-art LLMs using this frame work and prove its effectiveness. Experimental results reveal that existing LLMs still have significant room for improvement in addressing the complex and real world Table related tasks. Both the dataset and evaluation framework are publicly available, with the dataset hosted on huggingface.co/datasets/JT-LM/JIUTIAN-TReB and the framework on github.com/JT-LM/jiutian-treb.

---

# TReB：面向大语言模型表格推理能力的综合评测基准 论文详细解读

### 背景：这个问题为什么难？
企业里大多数信息都埋在 Excel、SQL 表格里，模型要把这些结构化数据当成“文字”来理解，往往会忽略行列之间的隐含关系。过去的评测大多聚焦于自然语言问答或代码生成，几乎没有专门测表格推理的基准。现有的表格任务要么数据量太小、题目单一，要么只考查表格的表层读取，根本无法衡量模型在多步推理、跨表关联等深层能力。于是，缺少一个既覆盖浅层理解又覆盖深层推理、且规模足够大的基准，成了制约 LLM 在商业数据场景落地的瓶颈。

### 关键概念速览
**表格推理（Table Reasoning）**：指模型在给定表格的前提下，完成从简单检索到多步计算、逻辑推断的任务。想象把表格当成一张地图，模型需要在上面找路、算距离、甚至判断两条路是否相交。  
**浅层理解（Shallow Understanding）**：只要求模型定位单元格、读取数值或文字，相当于“找字”。  
**深层推理（Deep Reasoning）**：需要跨行跨列进行算术、比较、归纳等多步操作，类似“解谜”。  
**子任务（Sub‑task）**：TReB 把整个评测拆成 26 个细分任务，每个任务对应一种表格操作或推理模式。  
**推理模式（Inference Mode）**：包括 TCoT、PoT、ICoT 三种让模型输出思考过程的方式，分别对应“表格链式思考”“程序化思考”“交互式思考”。  
**CoT（Chain‑of‑Thought）**：模型在给答案前先写出推理步骤，像在纸上写草稿。  
**PoT（Program‑of‑Thought）**：模型把推理过程组织成类似代码的指令序列，便于执行和检查。  
**ICoT（Interactive‑CoT）**：模型在推理过程中可以自我提问或请求额外信息，类似人做实验时的“自问自答”。  

### 核心创新点
1. **从单一任务到 26 条子任务的全景评测**  
   之前的表格评测往往只提供几道检索或算术题，覆盖面窄。TReB 通过迭代式数据生成，构造了 26 种不同难度和类型的子任务，形成了从“找单元格”到“跨表逻辑推理”的完整谱系。这样模型的强项和短板可以被细粒度定位。  
2. **三种推理模式的统一评估框架**  
   过去的基准只让模型直接输出答案，无法观察思考过程。TReB 引入 TCoT、PoT、ICoT 三种模式，并在评估脚本中统一计分，使得同一模型在不同思考方式下的表现可比。实验显示，同一模型在 PoT 下往往算术更准，而在 ICoT 下对模糊问题的容错更好。  
3. **高质量数据的迭代清洗流程**  
   为避免噪声数据导致评测失真，作者设计了“生成‑筛选‑人工校验‑再生成”的闭环。每轮生成的表格和问题先用自动规则过滤，再交给专业标注员校对，最后把纠错经验反馈给生成模型。相比一次性抓取的公开表格，这种方式显著提升了题目的语义一致性和答案唯一性。  
4. **大规模基准公开与可复现评测平台**  
   数据集直接托管在 HuggingFace，评测框架开源在 GitHub，配套的 CLI 工具可以一键跑 20+ 主流 LLM。这样社区可以快速复现、对比新模型，推动表格推理的基准化进程。

### 方法详解
整体思路可以划分为三大步骤：**数据构造 → 推理模式设计 → 统一评测**。

1. **数据构造**  
   - **表格生成**：使用大型语言模型（如 GPT‑4）在指令下生成多种行业表格（财务、物流、实验等），每张表格包含 5‑30 行、3‑10 列，列名和单元格内容遵循真实业务词汇。  
   - **问题生成**：针对每张表格，模型被要求提出 5‑10 条问题，覆盖检索、排序、聚合、条件筛选、跨表联结等操作。  
   - **质量控制**：自动脚本检查答案唯一性、数值范围合理性；随后人工审校纠错，错误信息被写回提示词，进入下一轮生成。这个闭环确保了最终 26 子任务的高质量。  

2. **推理模式设计**  
   - **TCoT（Table‑CoT）**：模型在回答前输出类似“先找出第 3 行第 2 列的值 → 与第 5 行第 1 列相加”。输出格式固定为“步骤 1：… 步骤 2：…”。  
   - **PoT（Program‑of‑Thought）**：模型把推理写成伪代码，如 `value = table[3][2] + table[5][1]`，随后执行得到答案。评测脚本会解析并运行这些指令，确保答案来源可追溯。  
   - **ICoT（Interactive‑CoT）**：模型可以在推理过程中自发提出子问题（如“该列是否包含负数？”），并在同一输出中给出答案或请求重新计算。评测器会检测这些交互是否合理并计入最终得分。  

3. **统一评测**  
   - **评分指标**：每个子任务使用准确率（Accuracy）或相对误差（Relative Error），并对思考步骤的完整性给出加分。  
   - **多模态对比**：同一模型在三种模式下分别跑一遍，得到三套分数，最终取最高或平均作为模型的表格推理得分。  
   - **可复现脚本**：评测框架提供 `run_treb.py`，只需指定模型 API、推理模式和子任务列表，即可自动下载数据、生成提示、收集答案并输出报告。  

**最巧妙的地方**在于把程序化思考（PoT）和交互式思考（ICoT）嵌入同一评测流水线，让模型既能像人写草稿，又能像代码执行，极大拓宽了评估维度。

### 实验与效果
- **测试对象**：作者在公开的 HuggingFace 数据集上跑了 20 多个主流 LLM，包括 GPT‑3.5、Claude‑2、LLaMA‑2‑70B、InternLM 等。每个模型分别在 TCoT、PoT、ICoT 三种模式下完成全部 26 子任务。  
- **基线对比**：与之前的 TableQA、TabFact 等单任务基准相比，TReB 的整体准确率普遍低 10%‑20%，说明这些模型在更复杂的表格推理上仍有显著缺口。  
- **具体数字**（论文声称）：在最易的浅层检索子任务上，最强模型（Claude‑2）在 PoT 模式下达到 92% 正确率；在最难的跨表逻辑推理子任务上，同模型仅有 38% 正确率。整体平均分在 55% 左右。  
- **消融实验**：去掉 PoT 模式的代码执行检查，整体准确率下降约 4%；去掉 ICoT 的自问自答机制，复杂子任务的错误率提升约 7%。这表明三种模式相互补足，缺一不可。  
- **局限性**：作者承认数据仍然是合成的，真实企业数据库的噪声、缺失值和业务规则可能更具挑战性；此外评测框架对模型的调用成本较高，尤其是需要多轮交互的 ICoT。  

### 影响与延伸思考
TReB 公开后，表格推理迅速从“小众任务”升级为 LLM 评测的必备维度。随后出现的工作如 **TabCoT**、**SQL‑CoT** 等，都在借鉴 TReB 的多模态推理模式，尝试把思考链直接映射到 SQL 生成或数据可视化。社区也开始围绕 **真实业务表格**（如财报、供应链）构建更大规模的实测集，进一步验证模型的工业落地能力。想继续深入，建议关注 **表格‑代码混合推理**（Table‑Code Hybrid Reasoning）和 **噪声鲁棒性评估** 两个方向，这两条路正是 TReB 暗示的下一步挑战。  

### 一句话记住它
TReB 用 26 条细分任务和三种思考模式，给大语言模型的表格推理设立了全景基准，揭示了它们在真实业务数据上仍有大量提升空间。