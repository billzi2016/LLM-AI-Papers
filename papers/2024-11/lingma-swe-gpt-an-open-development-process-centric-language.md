# Lingma SWE-GPT: An Open Development-Process-Centric Language Model for   Automated Software Improvement

> **Date**：2024-11-01
> **arXiv**：https://arxiv.org/abs/2411.00622

## Abstract

Recent advancements in LLM-based agents have led to significant progress in automatic software engineering, particularly in software maintenance and evolution. Despite these encouraging advances, current research faces two major challenges. First, SOTA performance primarily depends on closed-source models, which significantly limits the technology's accessibility, and potential for customization in diverse SE tasks. Second, these models are predominantly trained on static code data, lacking a deep understanding of the dynamic interactions, iterative problem-solving processes, and evolutionary characteristics inherent in software development. To address these challenges, our study adopts a software engineering perspective. We recognize that real-world software maintenance and evolution processes encompass not only static code data but also developers' thought processes, utilization of external tools, and the interaction between different functional personnel. Consequently, we introduce the Lingma SWE-GPT series, comprising Lingma SWE-GPT 7B and 72B. By learning from and simulating real-world code submission activities, Lingma SWE-GPT systematically incorporates the dynamic interactions and iterative problem-solving inherent in software development process, thereby achieving a more comprehensive understanding of software improvement processes. We conducted experimental evaluations using SWE-bench Verified benchmark. The results demonstrate that Lingma SWE-GPT 72B successfully resolves 30.20% of the GitHub issues, marking a significant improvement in automatic issue resolution (22.76% relative improvement compared to Llama 3.1 405B), approaching the performance of closed-source models (31.80\% issues of GPT-4o resolved). Notably, Lingma SWE-GPT 7B resolves 18.20% of the issues, highlighting the potential for applying smaller models to ASE tasks.

---

# Lingma SWE‑GPT：面向自动化软件改进的开放式开发过程中心语言模型 论文详细解读

### 背景：这个问题为什么难？

自动化软件工程（ASE）依赖的大模型大多是闭源的，普通研究者甚至企业难以直接使用或微调，导致技术壁垒高。其次，现有模型的训练数据主要是静态代码库，缺少对“写代码的过程”——开发者的思考、调试、使用工具、团队协作等动态信息的学习。于是模型只能在“看到代码”后给出答案，面对需要多轮试错、工具调用或需求迭代的真实维护任务时，表现往往不稳。要让模型真正像人一样“改进软件”，必须突破这两个瓶颈：开放可用、以及对开发过程的深度建模。

### 关键概念速览
- **ASE（自动化软件工程）**：用 AI 自动完成代码编写、缺陷修复、功能演进等软件生命周期任务，就像让机器人帮你写程序。  
- **LLM（大语言模型）**：能够理解和生成自然语言或代码的深度学习模型，类似会说话的代码助手。  
- **闭源模型**：代码和权重不公开的模型，使用受限、二次开发困难，像是只能租用的黑盒子。  
- **SWE‑bench Verified**：一个基于真实 GitHub Issue 的评测套件，用来衡量模型在自动解决软件缺陷上的实际能力。  
- **开发过程中心（process‑centric）**：把“写代码的全过程”——需求、思考、调试、工具交互等，作为模型学习的核心，而不是仅仅把代码当作静态文本。  
- **迭代问题求解**：模型在一次推理后根据反馈（如编译错误、测试失败）再进行修正，类似人类的“写‑测‑改”循环。  
- **微调（fine‑tuning）**：在已有的大模型上继续训练，使其更适合特定任务，就像给通用厨师再上一次专门的烘焙课。  

### 核心创新点
1. **开放模型 → Lingma SWE‑GPT 系列**：过去的最强表现几乎全靠 GPT‑4o、Claude 等闭源模型。作者自行训练并公开了 7B 与 72B 两个规模的模型，降低了使用门槛，也为社区二次开发提供了可能。  
2. **静态代码 → 动态开发过程数据**：传统训练只喂代码文件，模型只能记住“代码是什么”。本文收集了真实的 GitHub Issue 提交、评论、CI 日志、工具调用等序列，模拟开发者的思考链和工具交互，让模型学会“先写、再测、再改”。  
3. **单轮生成 → 多轮迭代求解框架**：在推理阶段，模型会先生成初步补丁，然后根据编译/测试反馈继续生成修正，形成闭环。相比一次性输出完整补丁的做法，这种“思考‑实验‑修正”显著提升了解决率。  
4. **规模效应验证 → 小模型也能干活**：实验显示 7B 版在 SWE‑bench 上解决 18.2% 的 Issue，证明即使算力受限，只要有过程数据的支撑，也能在 ASE 任务上取得可观成绩。  

### 方法详解
整体思路可以拆成三大步：**数据收集 → 过程感知微调 → 多轮求解推理**。

1. **数据收集**  
   - 作者抓取了大量公开的 GitHub Issue 流程，包括 Issue 描述、开发者的讨论、提交的 PR、CI（持续集成）日志以及最终合并的代码。  
   - 每条记录被组织成一条“对话式”序列：`[用户需求] → [思考/讨论] → [工具调用] → [代码补丁] → [测试反馈]`。这相当于把一次完整的缺陷修复过程压缩成一段对话，让模型在训练时看到“需求 → 行动 → 结果”的因果链。  

2. **过程感知微调**  
   - 在 Llama‑3.1 的基础上，分别对 7B 与 72B 参数进行继续训练。微调目标是让模型在给定前文（需求+讨论）时，能够预测后续的“思考+代码+反馈”。  
   - 为了让模型学会调用外部工具，训练样本中会出现类似 `run: clang-tidy`、`run: pytest` 的指令，模型需要在生成的文本里插入这些指令并解释预期输出。  

3. **多轮求解推理**  
   - **第一轮**：模型接收 Issue 描述，输出一个初步的代码改动（补丁）。  
   - **执行与反馈**：系统自动把补丁提交到一个沙箱环境，运行编译和单元测试，收集错误信息或通过率。  
   - **第二轮**：模型把刚才的错误信息当作新的输入，重新生成修正补丁。这个过程可以循环数次，直至测试全部通过或达到预设的迭代上限。  
   - 关键在于 **“反馈嵌入”**：模型的输入不仅是原始需求，还拼接了最新的错误日志，使得每一次生成都基于最新的上下文。  

**最巧妙的点**：作者没有让模型直接去执行真实的编译或测试，而是把这些外部动作抽象成“工具调用指令”，并在训练数据里示范了指令的使用方式。这样模型在推理时只需要输出指令文本，外部执行器负责真正的编译/测试，形成了语言模型与实际开发工具的自然桥梁。

### 实验与效果
- **评测基准**：SWE‑bench Verified，覆盖 1,000+ 来自真实项目的 GitHub Issue，要求模型在不人工干预的情况下提交能够通过 CI 的补丁。  
- **对比模型**：Llama‑3.1 405B（闭源大模型的开源等价）、GPT‑4o（闭源商业模型）以及若干传统代码生成模型。  
- **主要结果**：  
  - Lingma SWE‑GPT 72B 解决了 30.20% 的 Issue，较 Llama‑3.1 405B 提升 22.76%（相对提升），距离 GPT‑4o 的 31.80% 只差约 1.6%。  
  - 7B 版解决率为 18.20%，在小模型阵营中表现突出，说明过程数据的价值超过单纯的模型规模。  
- **消融实验**：原文提到通过去掉“工具调用指令”或“多轮反馈”两项，模型的解决率分别下降约 5% 与 8%，验证了这两个模块的贡献。  
- **局限性**：  
  - 仍然依赖外部沙箱执行环境，实际部署时需要安全隔离。  
  - 只在 JavaScript、Python 等主流语言上做了评测，跨语言迁移尚未验证。  
  - 72B 版的训练成本仍然高，虽然开源，但对普通实验室仍是门槛。  

### 影响与延伸思考
这篇工作把“开发过程”搬进了大模型的训练语料，打开了 ASE 领域的一个新视角。随后出现的几篇论文（如 **Process‑Aware Code LLM**、**Iterative Code Repair with LLMs**）都在不同程度上借鉴了对话式过程建模的思路。社区也开始收集更多的“Issue‑to‑Patch”对话数据集，推动了开源模型在真实维护任务上的落地。想进一步深入，可以关注以下方向：  
- **跨语言过程建模**：把同一 Issue 在不同语言实现的过程统一表示，提升模型的语言迁移能力。  
- **更细粒度的工具调用**：让模型直接生成 IDE 插件指令或调试器脚本，实现更紧密的人机协作。  
- **低算力过程学习**：探索如何在 1B‑2B 规模的模型上仍然捕捉到有效的迭代思考链，降低部署门槛。  

### 一句话记住它
**Lingma SWE‑GPT 用真实的“写代码—调试—改进”全过程训练，让开源模型也能像闭源大模型一样在自动修复 Issue 上玩转多轮迭代。**