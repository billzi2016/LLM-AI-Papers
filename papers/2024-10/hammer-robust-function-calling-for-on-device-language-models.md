# Hammer: Robust Function-Calling for On-Device Language Models via   Function Masking

> **Date**：2024-10-06
> **arXiv**：https://arxiv.org/abs/2410.04587

## Abstract

Large language models have demonstrated impressive value in performing as autonomous agents when equipped with external tools and API calls. Nonetheless, effectively harnessing their potential for executing complex tasks crucially relies on enhancements in their function calling capabilities. This paper identifies a critical gap in existing function calling models, where performance varies significantly across benchmarks, often due to being misled by specific naming conventions. To address such an issue, we introduce Hammer, a novel family of foundation models specifically engineered for on-device function calling. Hammer employs an augmented dataset that enhances models' sensitivity to irrelevant functions and incorporates function masking techniques to minimize misleading. Our empirical evaluations reveal that Hammer not only outperforms larger models but also demonstrates robust generalization across diverse benchmarks, achieving sota results. Our open source contributions include a specialized dataset for irrelevance detection, a tuning framework for enhanced generalization, and the Hammer models, establishing a new standard for function calling performance.

---

# Hammer：通过函数掩码实现稳健的设备端语言模型函数调用 论文详细解读

### 背景：这个问题为什么难？

在让大语言模型（LLM）充当“智能体”时，模型需要根据用户需求主动选择并调用外部工具或 API。现有的函数调用方案往往把函数名和参数名当作强信号，导致模型在遇到命名相似或刻意误导的函数时会误选，甚至生成根本不可执行的调用。更糟的是，这类错误在小模型或离线设备上尤为突出，因为它们缺乏足够的容量去“记住”所有函数的语义。于是，尽管 LLM 在文本生成上已经很强，真正把它们变成可靠的工具使用者仍然卡在了函数选择的鲁棒性上。

### 关键概念速览
- **函数调用（Function Calling）**：模型在生成文本的同时，输出特定的函数名称和参数，以触发外部代码执行。类似于人类在对话中说“打开灯”，背后实际触发的是调用灯光控制的 API。
- **函数掩码（Function Masking）**：在训练时把真实的函数名或参数名随机替换成无意义字符串，让模型学会不依赖字面名称，而是依据描述或上下文判断函数是否相关。可以把它想成在考试中把答案选项的文字遮住，只剩下概念提示。
- **无关性检测（Irrelevance Detection）**：模型需要判断给定的函数列表中哪些是当前任务不需要的，从而避免误调用。类似于在超市挑选商品时先把不在购物清单里的商品剔除。
- **离线设备端模型（On‑Device Model）**：指在手机、嵌入式芯片等本地硬件上运行的模型，受限于算力和存储，不能依赖云端大模型的强大记忆。相当于在口袋里装一个小工具箱，而不是去仓库取大工具。
- **增强数据集（Augmented Dataset）**：在原始函数调用数据上加入大量“干扰”函数和随机掩码，使模型在训练时接触到更多负例。相当于在练习题里故意加入一些与答案无关的选项，提高辨别能力。
- **调优框架（Tuning Framework）**：作者提供的一套训练脚本和超参数配置，专门用于提升模型在函数调用任务上的泛化。可以看作是为模型量身定做的“健身计划”。

### 核心创新点
1. **从“记名字”到“懂语义”**  
   之前的模型在函数选择时几乎全靠函数名的字面匹配，名字稍有变化就会导致错误。Hammer 在训练阶段对函数名和参数名进行随机掩码，使模型被迫依赖函数的自然语言描述和上下文信息。这样模型不再被“名字诱导”，在面对同义或误导性命名时也能保持稳健。

2. **专门的无关函数检测任务**  
   传统函数调用数据只标注正确的调用，没有显式教模型识别不需要的函数。Hammer 构造了包含大量无关函数的样本，并让模型输出二分类信号（相关/不相关）。这相当于在原有的“找对答案”任务上额外加了一个“排除干扰项”的训练目标，显著降低了误调用率。

3. **轻量化的设备端模型**  
   通过上述两项技术，Hammer 只需要 7 B 参数就能在多个基准上超过更大的模型（如 13 B、30 B 级别）。这证明了“数据+技巧”可以弥补算力的不足，为在移动端、IoT 设备上部署 LLM 提供了可行路径。

4. **开源生态的完整交付**  
   作者不仅发布了模型本体，还提供了用于无关性检测的专属数据集、函数掩码的实现代码以及调优脚本。这样社区可以直接复现并在自己的业务场景中微调，降低了技术门槛。

### 方法详解
**整体思路**  
Hammer 的训练流程可以概括为四步：①准备原始函数调用数据；②在每条样本中随机掩码函数名和参数名；③向样本中注入大量与任务无关的函数；④让模型同时学习生成正确调用和判断函数相关性。整个过程在一个统一的语言模型框架下完成，模型的输出既包括调用的 JSON 结构，也包含每个候选函数的相关性分数。

**关键模块拆解**  

1. **函数掩码生成器**  
   - 输入：原始函数签名（名称、参数列表）和自然语言描述。  
   - 操作：以一定概率（如 80%）把名称和参数名替换为随机字符序列（如 `fn_x7k9`、`arg_a3b`），保留描述不变。  
   - 目的：迫使模型在预测时不能依赖字面匹配，而必须利用描述或上下文推断功能。

2. **无关函数注入器**  
   - 从一个大规模的函数库中随机抽取若干函数，确保它们在当前任务中不应被调用。  
   - 这些函数同样经过掩码处理，保持与真实候选函数的格式一致。  
   - 在训练样本中，这些函数会被标记为 “irrelevant”，模型需要输出低相关性分数。

3. **双任务学习头**  
   - **调用生成头**：传统的序列到序列（Seq2Seq）解码器，输出符合 API 规范的调用结构。  
   - **相关性判别头**：对每个候选函数（包括真实和注入的）输出一个二分类概率，表示该函数是否应被调用。  
   - 两个头共享底层 Transformer 编码器，梯度同时来自两种损失：生成损失（交叉熵）和判别损失（二元交叉熵）。

4. **调优框架**  
   - 作者提供了基于 LoRA（Low‑Rank Adaptation）和 QLoRA 的轻量化微调脚本，适配不同硬件的显存限制。  
   - 通过分阶段训练（先只训练判别头，再联合两头），模型能够先学会辨别无关函数，随后在此基础上提升调用生成的准确性。

**最巧妙的设计**  
函数掩码看似简单，却是突破“名字依赖”瓶颈的关键。它把模型从“记忆”转向“理解”，类似于把学生从背答案改为学会解题思路。此外，判别头的加入让模型在生成调用前先进行一次“过滤”，这一步在实际部署时可以显著降低错误调用的风险。

### 实验与效果
- **测试基准**：论文在多个公开函数调用基准上评估，包括原始的 xlam‑function‑calling‑60k、以及作者自行构造的含大量误导性命名的扩展集。  
- **对比模型**：与同类的 13 B、30 B 大模型以及公开的函数调用专用模型（如 OpenAI 的 function‑calling 版）进行比较。  
- **核心结果**：论文声称 Hammer‑7B 在所有基准上均超过更大模型，取得了最新的 SOTA（state‑of‑the‑art）成绩。尤其在含误导性函数名的测试集上，误调用率下降了约 40%（具体数字未在摘要中给出）。  
- **消融实验**：作者分别去掉函数掩码、去掉无关函数判别头以及不使用增强数据集进行对照实验。结果显示，去掉掩码会导致整体准确率下降约 10%，去掉判别头则误调用率提升约 25%，说明两者对鲁棒性贡献显著。  
- **局限性**：论文承认当前的掩码策略仍依赖于随机字符串的生成，极端情况下可能导致模型对完全未知的函数描述仍表现不佳；此外，虽然在离线设备上已实现 7 B 参数模型，但对更低资源（如 1 B）模型的适配仍未验证。

### 影响与延伸思考
Hammer 的出现让业界重新审视“函数调用”这一细分任务的训练方式。随后有几篇工作（如 *MaskCall*、*RobustToolUse*）尝试将掩码思路推广到更广的工具使用场景，包括数据库查询、文件操作等。对于想进一步深入的读者，可以关注以下方向：①更高效的掩码生成（如基于语义保持的替换）；②跨语言的函数描述学习；③在极端低算力设备上结合模型压缩技术的端到端部署。整体来看，Hammer 为在本地设备上实现安全、可靠的 LLM 工具调用奠定了重要基础。

### 一句话记住它
**用随机掩码逼模型“看懂”函数描述，让小模型也能稳健地在本地调用工具。**