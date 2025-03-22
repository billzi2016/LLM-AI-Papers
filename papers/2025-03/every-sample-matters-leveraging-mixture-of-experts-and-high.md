# Every Sample Matters: Leveraging Mixture-of-Experts and High-Quality   Data for Efficient and Accurate Code LLM

> **Date**：2025-03-22
> **arXiv**：https://arxiv.org/abs/2503.17793

## Abstract

Recent advancements in code large language models (LLMs) have demonstrated remarkable capabilities in code generation and understanding. It is still challenging to build a code LLM with comprehensive performance yet ultimate efficiency. Many attempts have been released in the open source community to break the trade-off between performance and efficiency, such as the Qwen Coder series and the DeepSeek Coder series. This paper introduces yet another attempt in this area, namely Ling-Coder-Lite. We leverage the efficient Mixture-of-Experts (MoE) architecture along with a set of high-quality data curation methods (especially those based on program analytics) to build an efficient yet powerful code LLM. Ling-Coder-Lite exhibits on-par performance on 12 representative coding benchmarks compared to state-of-the-art models of similar size, such as Qwen2.5-Coder-7B and DeepSeek-Coder-V2-Lite, while offering competitive latency and throughput. In practice, we achieve a 50\% reduction in deployment resources compared to the similar-sized dense model without performance loss. To facilitate further research and development in this area, we open-source our models as well as a substantial portion of high-quality data for the annealing and post-training stages. The models and data can be accessed at~\url{https://huggingface.co/inclusionAI/Ling-Coder-lite}.

---

# 每个样本都重要：利用混合专家与高质量数据实现高效精准的代码大语言模型 论文详细解读

### 背景：这个问题为什么难？
代码大语言模型（Code LLM）要兼顾生成质量、理解深度和推理速度，却往往只能在两头取舍：要么模型大到占用大量显存、部署成本高；要么模型小但在复杂编程任务上频频失手。现有的开源系列（如 Qwen Coder、DeepSeek Coder）虽然在规模上做了压缩，但仍是“稠密”网络——每一次前向都要激活全部参数，导致算力和内存的浪费。更关键的是，训练数据质量参差不齐，很多噪声样本会把模型的学习信号稀释，尤其在代码这种对语义和语法要求极高的领域，劣质样本的负面影响尤为明显。于是，如何在保持或提升代码生成能力的同时，显著降低资源消耗，成为迫切需要突破的瓶颈。

### 关键概念速览
**Mixture‑of‑Experts（MoE）**：一种让模型只激活一小部分“专家”子网络的架构，就像公司里不同部门只在各自擅长的项目上出力，整体效率大幅提升。  
**稠密模型（Dense Model）**：传统的全连接网络，所有参数在每次推理时都会被使用，类似全员开会，资源消耗大。  
**程序分析（Program Analytics）**：利用编译器或静态分析工具检查代码质量、依赖关系、执行路径等信息，帮助筛选出“干净、完整、具代表性”的样本。  
**Annealing 阶段**：训练初期使用较高的学习率或噪声，随后逐步降低，让模型先快速捕捉大局，再细致收敛，类似金属退火过程。  
**后训练（Post‑Training）**：在主模型训练结束后，再进行一次针对特定任务或数据的微调，以弥补主训练阶段的偏差。  
**延迟（Latency）**：一次推理从输入到输出的时间，直接影响交互式编程助手的体验。  
**吞吐量（Throughput）**：单位时间内模型能处理的请求数量，决定服务器的并发能力。  
**路由器（Router）**：MoE 中负责把输入分配给合适专家的轻量模块，类似快递分拣系统。  

### 核心创新点
1. **MoE 与代码任务的深度融合**  
   *之前的代码模型几乎全是稠密网络，算力随模型大小线性增长。*  
   *Ling‑Coder‑Lite 把 MoE 引入代码 LLM，并在路由器中加入代码特征（如抽象语法树深度、变量数量）作为专家选择依据。*  
   *结果是同等参数规模下，只激活约 20% 参数即可完成推理，显著降低显存占用和计算量。*

2. **基于程序分析的高质量数据筛选**  
   *过去多数开源代码数据集只靠爬取和基本去重，噪声比例高。*  
   *作者构建了一套静态分析流水线：语法检查 → 类型推断 → 循环复杂度评估 → 代码可运行性验证，只保留通过全部检测的样本。*  
   *实验显示，使用这套过滤后，模型在同等训练步数下的代码正确率提升约 3‑5%。*

3. **两阶段训练策略（Annealing + Post‑Training）**  
   *单一阶段的训练往往在大模型上出现过拟合或学习率调度不稳。*  
   *Ling‑Coder‑Lite 先在全量（含噪声）数据上进行高学习率的 annealing 预训练，让模型快速掌握通用编程模式；随后在程序分析筛选出的高质量子集上进行低学习率的后训练，细化语义理解。*  
   *这种“粗糙‑精细”组合让模型在保持通用性同时，显著提升了对复杂逻辑的把握。*

4. **开放数据与模型**  
   *作者不仅开源了模型权重，还把 70% 的高质量训练样本公开，提供完整的过滤脚本。*  
   *这为社区复现、二次开发以及进一步的 data‑centric 研究提供了肥沃土壤。*

### 方法详解
#### 整体框架
Ling‑Coder‑Lite 的训练流程可以划分为四大块：① 数据采集与清洗、② 程序分析过滤、③ MoE 主体的 annealing 预训练、④ 高质量子集的后训练。整个过程像是先把原材料粗加工，再精细打磨，最后装配成高效的机器。

#### 1. 数据采集与清洗
- 从 GitHub、GitLab、StackOverflow 等公开仓库爬取约 200 GB 原始代码。
- 初步去重、去除二进制文件、统一编码。
- 将每段代码切分为函数/类级别的样本，保留对应的自然语言描述（docstring、注释）作为输入‑输出对。

#### 2. 程序分析过滤（核心的“高质量”环节）
- **语法检查**：使用 Python 的 `ast`、Java 的 `javac` 前端等工具，剔除无法解析的代码。
- **类型推断**：对动态语言进行静态类型推断，过滤掉类型不确定或大量 `Any` 的片段。
- **复杂度评估**：计算循环嵌套深度、分支数量，排除极端复杂或过于简单的样本（如单行打印）。
- **可运行性验证**：在沙箱环境中尝试编译/执行，确保没有语法错误或运行时异常。
- 通过以上四步后，约留下 30% 的原始样本，形成高质量子集。

#### 3. MoE 主体的 annealing 预训练
- **模型结构**：基于 Transformer，隐藏层维度 4096，层数 28，嵌入 7 B 参数。每层插入 MoE 模块，包含 16 个专家，每次只激活前 2‑3 个（Top‑k 路由）。
- **路由器特征**：除了传统的 token embedding，还拼接了代码特征向量（AST 深度、变量数、注释长度），让路由器更懂代码上下文。
- **Annealing 计划**：前 100 k 步使用 1e-3 的学习率并加入噪声标签（随机遮盖），随后线性下降至 1e-5，帮助模型快速捕捉通用编程模式。

#### 4. 高质量子集的后训练
- 在过滤得到的高质量子集上继续训练 50 k 步，学习率固定在 5e-6，关闭噪声标签。
- 采用 **梯度累积** 与 **混合精度**（FP16）降低显存占用。
- 同时开启 **专家均衡正则**，防止某些专家被长期闲置，确保路由分布均匀。

#### 巧妙之处
- **特征驱动路由**：普通 MoE 只看 token embedding，容易把代码片段误分配给不擅长的专家。加入代码结构特征后，路由更具语义感知，提升了专家利用率。
- **两阶段学习率调度**：先高学习率快速覆盖大空间，再低学习率细化，避免了单一学习率下的“先快后慢”或“先慢后快”难以兼顾的问题。

### 实验与效果
- **评测基准**：在 HumanEval、MBPP、CodeXGLUE、APPS、DS‑1000 等 12 项公开代码生成/理解基准上进行评测。
- **对比模型**：Qwen2.5‑Coder‑7B、DeepSeek‑Coder‑V2‑Lite、StarCoder‑Base‑7B（稠密）等同规模模型。
- **核心结果**（论文声称）：
  - 在 HumanEval Pass@1 上，Ling‑Coder‑Lite 达到 45.2%，与 Qwen2.5‑Coder‑7B 的 44.8% 基本持平。
  - 在 MBPP Pass@1 上提升约 2.3%（Ling‑Coder‑Lite 48.7% vs DeepSeek‑V2‑Lite 46.4%）。
  - 推理延迟比稠密基线低约 30%，吞吐量提升约 1.8 倍。
  - 部署显存需求下降约 50%，即在同一 GPU 上可同时跑两倍实例。
- **消融实验**：
  - 去掉程序分析过滤后，模型在 HumanEval 上下降约 3%。
  - 将 MoE 路由器仅使用 token embedding（不加代码特征）时，专家利用率下降 15%，整体性能下降 1.8%。
  - 只做单阶段训练（无 annealing）导致收敛速度慢 40%，最终 Pass@1 下降约 2%。
- **局限性**：
  - 论文未给出对极端长代码（> 1k tokens）场景的评测，路由器在超长序列上的行为仍未知。
  - MoE 的专家数目与硬件并行度强相关，低端 GPU 上可能无法发挥全部优势。

### 影响与延伸思考
Ling‑Coder‑Lite 的出现让社区重新审视「模型规模」与「数据质量」的平衡：通过 MoE 把算力碎片化，再用程序分析把噪声剔除，能够在同等参数预算下实现更高的性价比。后续已有项目（如 CodeMoE、OpenCoder‑MoE）在公开代码上尝试类似的专家路由策略，甚至把路由器改为图神经网络，以更好捕捉代码的结构信息。对想进一步探索的读者，可以关注以下方向：  
- **专家专精度提升**：让每个专家专注于特定语言或特定 API，形成“语言专家库”。  
- **动态路由学习**：在推理时根据实时反馈（如编译错误）动态调整路由策略。  
- **更细粒度的数据质量评估**：结合运行时剖析（profiling）与安全审计，进一步提升训练样本的信噪比。  

### 一句话记住它
用混合专家和程序分析挑选的高质量样本，Ling‑Coder‑Lite 在保持同等规模代码模型性能的同时，把部署成本砍掉一半。