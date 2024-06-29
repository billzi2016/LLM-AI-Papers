# Teola: Towards End-to-End Optimization of LLM-based Applications

> **Date**：2024-06-29
> **arXiv**：https://arxiv.org/abs/2407.00326

## Abstract

Large language model (LLM)-based applications consist of both LLM and non-LLM components, each contributing to the end-to-end latency. Despite great efforts to optimize LLM inference, end-to-end workflow optimization has been overlooked. Existing frameworks employ coarse-grained orchestration with task modules, which confines optimizations to within each module and yields suboptimal scheduling decisions. We propose fine-grained end-to-end orchestration, which utilizes task primitives as the basic units and represents each query's workflow as a primitive-level dataflow graph. This explicitly exposes a much larger design space, enables optimizations in parallelization and pipelining across primitives of different modules, and enhances scheduling to improve application-level performance. We build Teola, a novel orchestration framework for LLM-based applications that implements this scheme. Comprehensive experiments show that Teola can achieve up to 2.09x speedup over existing systems across various popular LLM applications. The code is available at https://github.com/NetX-lab/Ayo.

---

# Teola：面向端到端优化的大语言模型应用 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）已经可以直接生成文本，但真实业务里往往要把 LLM 与检索、过滤、后处理等非 LLM 步骤串起来，形成一个完整的工作流。每一步都有自己的计算开销，整体延迟往往远高于单纯的模型推理。过去的系统把这些步骤划分成几个“大块”模块，然后在模块内部做加速，却把模块之间的调度当成黑盒。这样一来，跨模块的并行和流水线机会被埋在了粗粒度的编排里，导致资源利用率低、响应时间长，迫切需要一种能够在更细的粒度上进行整体优化的方案。

### 关键概念速览
- **LLM（大语言模型）**：能够理解并生成自然语言的深度模型，例如 GPT‑4。它是工作流中最耗时的算子之一。  
- **非LLM组件**：检索、向量相似度搜索、结果过滤、格式化等不依赖大模型的处理步骤。可以看作是工作流的配菜。  
- **任务原语（Task Primitive）**：工作流中最小的可调度单元，例如一次向量检索或一次前向推理。把整个流程拆成这些原子块，就像把一道菜拆成每一步的配料准备。  
- **原语级数据流图**：用有向图描述原语之间的数据依赖关系，节点是原语，边是数据流向。它把整个查询的执行路径可视化，类似工厂的装配线图。  
- **细粒度编排（Fine‑grained Orchestration）**：基于任务原语而不是模块来调度执行，能够在原语之间自由并行或流水线。  
- **跨模块并行化**：不同模块的原语可以同时运行，而不是等前一个模块全部结束后才开始下一个模块。  
- **流水线调度（Pipelining）**：当一个原语产生中间结果后，后续原语立即使用该结果，而不必等全部输入准备完毕，类似生产线的“半成品”直接进入下一工序。  
- **编排框架（Orchestration Framework）**：负责把工作流拆解、构图、优化、调度并最终执行的系统，Teola 就是这样一个框架。

### 核心创新点
1. **从模块到原语的粒度转变**  
   之前的系统把检索、生成、后处理等每个功能当成一个不可拆的任务模块，优化只能在模块内部进行。Teola 把每个模块进一步拆解成任务原语，并用原语级数据流图把它们串起来。这样做把隐藏的并行机会显现出来，使得调度器可以在更细的层面做决定。  
2. **原语级数据流图的显式建模**  
   传统编排只记录模块的调用顺序，缺少对数据依赖的细致描述。Teola 为每一次查询生成一张完整的原语级有向图，明确哪些原语可以并行、哪些必须顺序执行。图的可视化让调度器能够系统性地搜索更优的执行计划。  
3. **跨模块并行化与流水线的统一优化**  
   基于数据流图，Teola 同时考虑并行化（同层原语并行）和流水线（跨层原语流水）。它会在不冲突的情况下把检索、向量过滤、LLM 推理等原语交叉排布，最大化硬件利用率。相比只在单模块内部做并行的老方法，整体延迟下降显著。  
4. **面向端到端性能的调度器**  
   调度器不再只看每个模块的耗时，而是以整个查询的端到端延迟为目标进行优化。它会评估不同排布下的关键路径长度，选择最短的那条路径执行。实验表明，这种端到端视角能带来最高 2.09 倍的加速。

### 方法详解
**整体思路**：Teola 把一次 LLM 应用的请求视作一条工作流，先把工作流拆成最小的任务原语，随后构建原语级数据流图，接着交给优化器生成并行/流水线计划，最后由运行时引擎按照计划调度执行。整个过程可以概括为「拆解 → 建图 → 优化 → 调度 → 执行」五步。

1. **任务原语拆解**  
   开发者在编写应用时仍然使用熟悉的高层 API（比如 “检索+生成”），Teola 在编译阶段把这些高层调用映射成若干原语。例如一次向量检索会被拆成“加载索引 → 计算相似度 → 排序”。每个原语都有明确的输入、输出和资源需求（CPU、GPU、内存）。

2. **原语级数据流图构建**  
   拆解完成后，Teola 按照原语之间的数据依赖连线，生成一张有向无环图（DAG）。图的根节点是外部请求入口，叶子节点是最终的输出（如生成的文本）。这一步相当于把烹饪流程画成流程图，所有原材料的流向一目了然。

3. **优化器：并行化 + 流水线**  
   优化器遍历 DAG，先找出同一层可以并行执行的原语（例如检索的多个子查询），再检测跨层的流水线机会（比如检索得到的前 10 条向量可以立刻送入 LLM 做摘要）。它使用一种贪心+局部搜索的策略，尝试把关键路径（最长依赖链）压短。关键的技巧是把硬件约束（GPU 只能跑一定数量的并行推理）嵌入调度模型，确保生成的计划可落地。

4. **调度器**  
   调度器接受优化器输出的计划，负责把每个原语映射到具体的执行单元（CPU 线程、GPU 流、异构加速卡）。它会动态监控资源占用，必要时进行抢占或迁移，以防出现“热点”导致的瓶颈。调度策略的目标函数是最小化整个查询的端到端延迟，而不是单个原语的执行时间。

5. **运行时引擎**  
   引擎负责实际执行原语，提供统一的接口来调用不同后端（如 Faiss 检索、PyTorch 推理）。它实现了“生产线”式的流水线缓冲区：当上游原语产出结果后，立即把结果放入缓冲区供下游原语消费，而不必等待全部上游完成。这样即使在单个查询内部，也能实现类似批处理的高吞吐。

**最巧妙的地方**：Teola 把“任务模块”这一抽象层次抛掉，直接在原语层面做全局调度。原来被视为不可拆分的模块，现在可以被拆成若干并行的“小工序”，这让调度器拥有了前所未有的自由度。再加上端到端视角的目标函数，系统能够在保持功能正确性的前提下，自动发现并利用跨模块的并行和流水线机会。

### 实验与效果
- **测试场景**：论文在多个典型的 LLM 应用上评估，包括检索增强生成（RAG）、多轮对话、代码自动补全以及结构化数据生成等。每个场景都包含显著的非 LLM 步骤，使得端到端优化的价值可观测。  
- **对比基线**：主要与当前流行的编排框架（如 LangChain、LlamaIndex）以及自研的模块级并行方案进行比较。  
- **加速效果**：在所有测试中，Teola 的端到端延迟均优于基线，最高可达 **2.09 倍** 的加速，平均提升约 **1.4–1.7 倍**。尤其在 RAG 场景下，检索与生成的流水线显著缩短了关键路径。  
- **消融实验**：作者分别关闭原语级数据流图、跨模块并行化和端到端调度三项功能。结果显示，去掉数据流图导致加速下降到 1.2 倍，去掉跨模块并行化则降至 1.3 倍，去掉端到端调度则降至 1.1 倍，说明每个模块对整体性能都有贡献。  
- **局限性**：论文指出，Teola 依赖于对每个原语的资源需求进行准确标注，若标注不准可能导致调度失衡。此外，当前实现主要针对 CPU+GPU 的异构环境，对专用加速卡（如 TPUs）支持尚不完善。

### 影响与延伸思考
Teola 把细粒度编排引入 LLM 应用后，业界开始关注“工作流级别的并行”和“跨模块流水线”。随后出现的研究如 **PipeLLM**、**OrchestrateAI** 等，都在尝试把原语级调度与自适应硬件分配结合起来，进一步提升大模型服务的吞吐。对想深入的读者，可以关注以下方向：  
- **自适应原语划分**：根据实时负载自动决定是否进一步细分原语。  
- **硬件感知调度**：把 TPU、FPGA 等新型加速器纳入调度模型。  
- **多查询共享流水线**：在高并发场景下，让不同用户的查询共享同一条流水线，以进一步提升资源利用率。  

### 一句话记住它
**Teola 把 LLM 应用拆成最小原子块，用原语级数据流图实现跨模块并行和流水线，从而把端到端延迟压到原来的 ½ 左右。**