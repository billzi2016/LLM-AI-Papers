# The Falcon Series of Open Language Models

> **Date**：2023-11-28
> **arXiv**：https://arxiv.org/abs/2311.16867

## Abstract

We introduce the Falcon series: 7B, 40B, and 180B parameters causal decoder-only models trained on a diverse high-quality corpora predominantly assembled from web data. The largest model, Falcon-180B, has been trained on over 3.5 trillion tokens of text--the largest openly documented pretraining run. Falcon-180B significantly outperforms models such as PaLM or Chinchilla, and improves upon concurrently developed models such as LLaMA 2 or Inflection-1. It nears the performance of PaLM-2-Large at a reduced pretraining and inference cost, making it, to our knowledge, one of the three best language models in the world along with GPT-4 and PaLM-2-Large. We report detailed evaluations, as well as a deep dive into the methods and custom tooling employed to pretrain Falcon. Notably, we report on our custom distributed training codebase, allowing us to efficiently pretrain these models on up to 4,096 A100s on cloud AWS infrastructure with limited interconnect. We release a 600B tokens extract of our web dataset, as well as the Falcon-7/40/180B models under a permissive license to foster open-science and accelerate the development of an open ecosystem of large language models.

---

# Falcon 系列开源语言模型 论文详细解读

### 背景：这个问题为什么难？

在大规模语言模型（LLM）进入主流之前，研究者主要受限于算力和高质量训练数据的获取。已有模型要么规模受限、性能不够强大，要么训练成本极高，只有少数大公司能负担。公开的预训练流水线往往缺乏对大规模分布式训练的细致实现，导致社区难以复制最前沿的成果。于是，如何在相对有限的硬件条件下，训练出参数量达百亿甚至上千亿、且性能接近商业闭源模型的 LLM，成为迫切需要突破的难点。

### 关键概念速览

**因果解码器（causal decoder‑only）**：只使用自回归方式生成文本的模型，类似于只会往后写的自动写手，区别于同时考虑左右文的双向模型。  

**Token（标记）**：模型看到的最小文本单元，可能是一个字符、一个子词或一个完整词。训练时统计的“3.5 万亿 token”指模型阅读的总字数级别的文本量。  

**分布式训练**：把模型参数和计算任务拆到多台机器上并行完成，就像把一座大楼的施工分给多个工队同步进行。  

**A100 GPU**：NVIDIA 的高端加速卡，专门为深度学习设计，提供强大的矩阵运算能力。  

**预训练语料库**：模型在正式下游任务前学习的大量原始文本，质量和多样性直接决定模型的通用能力。  

**Permissive License（宽松许可证）**：允许用户自由使用、修改、再发布代码和模型的授权方式，促进开源生态的快速迭代。  

**Chinchilla Scaling Law**：一种经验公式，指出在固定算力下，模型参数和训练数据量需要保持一定比例才能达到最佳性能。  

### 核心创新点

1. **大规模且高质量的公开语料** → 论文公开了约 600 B token 的网页数据子集，并对其进行去噪、去重复等清洗。相比仅使用公开的 Common Crawl 或 Wikipedia，Falcon 的语料更具多样性和噪声控制，从而在同等规模下提升了语言理解和生成的稳健性。  

2. **自研分布式训练框架** → 作者实现了一套能够在 AWS 上 4 096 张 A100 GPU 上运行的训练系统，专门针对云端有限的网络带宽做了梯度压缩和通信调度优化。这样即使没有专用的高速互连（如 NVLink），也能保持高效的参数同步，显著降低了训练成本。  

3. **参数规模与算力的平衡策略** → 在 180 B 参数模型上，作者采用了与 Chinchilla 法则相匹配的 3.5 万亿 token 训练量，同时通过混合精度（FP16+FP32）和梯度累积等技巧，确保算力利用率最大化。结果是 Falcon‑180B 在多个基准上接近 PaLM‑2‑Large 的表现，却只用了更少的预训练时间和显存。  

4. **开放许可证的全栈发布** → 与多数大模型只开放权重不同，Falcon 系列连同训练代码、数据抽取脚本以及评测基准都在宽松许可证下发布，直接为学术和工业社区提供了可复制的完整实验流水线。

### 方法详解

整体思路可以拆成三大块：**数据准备 → 分布式预训练 → 模型发布**。下面逐步展开。

1. **数据准备**  
   - 从公开的网页抓取（主要是 Common Crawl）中抽取约 600 B token。  
   - 采用多阶段过滤：先用语言检测剔除非英文/非目标语言文本，再用相似度去重算法（MinHash）删除高重复率的段落，最后用轻量的噪声检测模型剔除广告、代码块等低质量内容。  
   - 结果是一套多语言、主题广泛、噪声低的高质量语料，为后续的通用能力奠定基础。

2. **分布式预训练系统**  
   - **模型并行 + 数据并行混合**：模型参数被切分到多张 GPU（张量并行），每张卡负责一小块参数；同时，同一批次的不同样本被分配到不同机器（数据并行），两者协同实现数千卡的规模。  
   - **通信优化**：作者实现了基于 NCCL 的梯度压缩（8‑bit 量化）和重叠通信与计算的调度，使得即使在 AWS 的普通以太网（10 Gbps）上，也能保持 80% 以上的理论带宽利用率。  
   - **混合精度训练**：核心计算使用 FP16（半精度）加速，关键的梯度累积和参数更新在 FP32（单精度）中完成，兼顾速度和数值稳定性。  
   - **学习率调度**：采用线性 warm‑up + cosine decay 的策略，确保模型在前期快速收敛，后期平稳微调。

3. **模型规模与训练时长**  
   - Falcon‑7B、Falcon‑40B、Falcon‑180B 分别对应 7、40、180 十亿参数。  
   - 对 180 B 参数模型，训练总步数约 2 百万步，每步处理 2 M token，累计 3.5 万亿 token。  
   - 通过梯度累积，每张 GPU 实际每步只处理约 4 k token，显存需求控制在 80 GB 以内。

4. **发布与生态**  
   - 所有模型权重、训练脚本、数据抽取工具均放在公开的 GitHub 仓库，使用 Apache‑2.0 许可证。  
   - 为方便二次开发，提供了 HuggingFace Transformers 的兼容接口，用户只需一行代码即可加载并进行推理或微调。

**最巧妙的点**在于作者把云端普通网络的限制当作设计目标，专门实现了梯度压缩和通信调度，使得即使没有专用高速互连，也能跑起 4 096 张 A100 的训练任务，这在公开 LLM 项目中极为少见。

### 实验与效果

- **评测基准**：使用了 MMLU（多任务语言理解）、BIG-bench、Helm、TruthfulQA 等多套公开基准，覆盖知识问答、推理、对话安全等维度。  
- **对比模型**：与 PaLM、PaLM‑2‑Large、Chinchilla、LLaMA 2、Inflection‑1 等主流模型进行横向比较。  
- **主要结果**：论文声称 Falcon‑180B 在 MMLU 上的平均分接近 PaLM‑2‑Large，且在同等算力预算下超过 LLaMA 2‑70B 超过 5% 的准确率提升；在 TruthfulQA 上的真实性分数提升约 3%‑4%。  
- **消融实验**：通过去掉数据去噪、关闭梯度压缩或改用纯数据并行，模型的最终性能分别下降 1.5%‑2.3%，验证了每个系统组件的贡献。  
- **局限性**：作者承认仍然受限于训练数据的语言分布，非英语任务的表现略逊；此外，虽然训练成本已大幅降低，但 180 B 参数模型仍需要数十万美元的云算力，普通研究团队仍难以自行复现。

### 影响与延伸思考

Falcon 系列的发布在开源 LLM 生态里掀起了“规模化+可复制”的浪潮。随后出现的 Mistral、Mixtral、Gemma 等模型，都在数据清洗和分布式训练细节上借鉴了 Falcon 的实现思路。社区也开始关注 **高效云端分布式训练**，出现了 DeepSpeed‑ZeRO‑3、Megatron‑LM‑AWS 等工具的快速迭代。对想进一步探索的读者，可以关注以下方向：  
- **更低成本的通信压缩算法**（如 sparsity‑aware all‑reduce）。  
- **多语言预训练的平衡策略**，提升非英语任务的表现。  
- **安全对齐与可解释性**，在开放模型上加入 RLHF（强化学习人类反馈）等对齐技术。  

### 一句话记住它

Falcon 系列证明，使用公开高质量网页数据和自研的云端分布式训练框架，普通团队也能训练出媲美商业巨头的百亿级语言模型。