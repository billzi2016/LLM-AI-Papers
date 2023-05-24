# Who Wrote this Code? Watermarking for Code Generation

> **Date**：2023-05-24
> **arXiv**：https://arxiv.org/abs/2305.15060

## Abstract

Since the remarkable generation performance of large language models raised ethical and legal concerns, approaches to detect machine-generated text by embedding watermarks are being developed. However, we discover that the existing works fail to function appropriately in code generation tasks due to the task's nature of having low entropy. Extending a logit-modifying watermark method, we propose Selective WatErmarking via Entropy Thresholding (SWEET), which enhances detection ability and mitigates code quality degeneration by removing low-entropy segments at generating and detecting watermarks. Our experiments show that SWEET significantly improves code quality preservation while outperforming all baselines, including post-hoc detection methods, in detecting machine-generated code text. Our code is available in https://github.com/hongcheki/sweet-watermark.

---

# 谁写的这段代码？代码生成的水印技术 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在代码生成上已经能写出几乎可以直接编译的程序，这让版权归属和责任划分变得模糊。传统的文本水印技术通过在生成的文字里植入可检测的概率偏移来辨别机器产出，但代码的“信息密度”与自然语言不同——同一功能的实现往往只有少数几种合法写法，导致生成过程的熵（不确定性）极低。熵低意味着模型在大多数位置只能选出极少的候选 token，水印的概率偏移很容易被压平，从而既不影响代码质量，又难以被检测。于是，直接把现有文本水印搬到代码上会出现两大问题：检测率跌到几乎为零，或者水印导致代码出现语法错误、逻辑缺陷。

### 关键概念速览
**大语言模型（LLM）**：能够根据上下文预测下一个词或代码片段的深度神经网络，类似于“会写作文的机器人”。  
**熵（Entropy）**：衡量模型在某一步预测的不确定程度，熵高像是多选题，熵低像是填空题只有唯一答案。  
**Logit**：模型在每个候选 token 上的原始得分，得分越高该 token 被选中的概率越大。  
**Watermark（水印）**：在生成过程中人为调节 logit，使得特定模式出现的概率略高，像在文字里暗埋的“指纹”。  
**Selective Watermarking（选择性水印）**：只在模型预测不确定的高熵位置植入水印，避免在确定性强的低熵位置干扰。  
**Entropy Thresholding（熵阈值）**：设定一个熵上限，只有超过该阈值的生成步骤才会被标记为“可水印”。  
**Post‑hoc Detection（事后检测）**：不在生成时加入任何干预，而是事后分析文本特征来判断是否机器生成。

### 核心创新点
1. **低熵段的排除 → 通过熵阈值只在高熵位置植入水印 → 检测信号更强且不破坏代码**  
   之前的水印方法在每一步都强行调节 logit，导致在代码这种低熵任务上出现语法错误或逻辑偏差。作者提出先计算每一步的熵，若低于预设阈值则直接放行，不做任何改动。这样既保留了原始模型的高质量输出，又把水印限制在模型本身已经“犹豫不决”的地方。

2. **基于 logit 的可逆调制 → 在高熵位置对 logits 做微小、可逆的偏移 → 检测时只需逆向恢复偏移模式**  
   传统水印往往使用固定的 token 子集或随机噪声，难以在检测时区分是噪声还是自然偏好。本文的做法是把可检测的偏移写进 logits 本身，检测器只要重新计算这些 logits 的相对大小，就能恢复出水印的二进制序列。

3. **统一检测框架 → 同时利用生成时的熵信息和检测时的偏移模式 → 超越所有事后检测基线**  
   事后检测只能靠统计特征，面对高度相似的机器生成代码时效果有限。SWEET 在检测阶段把生成时记录的熵阈值信息一起喂入模型，形成双重线索，使得检测准确率显著提升。

### 方法详解
整体思路可以拆成三步：**熵评估 → 选择性调制 → 双向检测**。

1. **熵评估**  
   在每一次 token 采样前，模型先把当前上下文的 logits 送入 softmax，得到每个候选 token 的概率分布。随后计算该分布的熵值（即信息熵），如果熵 > τ（τ 为经验设定的阈值），说明模型在此位置有多种合理选择，进入下一步；否则直接使用原始 logits，保持模型的原始输出。

2. **选择性调制**  
   对于被标记为“高熵”的位置，作者在 logits 上加入一个细微的偏移向量。偏移的方向和幅度由预先生成的二进制水印序列决定：比如水印位为 1 时，对属于“水印子集”的 token 增加 δ，位为 0 时则对同一子集的另一个 token 增加 δ。δ 的大小远小于模型原始 logits 的差距，确保不会改变最终采样的 token（除非模型本身在该位置真的犹豫），但足以在统计上留下可检测的偏好。

3. **双向检测**  
   检测阶段，首先对待检测的代码重新跑一次模型，记录每一步的熵并标记出高熵位置。然后在这些位置重新计算 logits（不做采样），检查是否出现了与水印子集对应的概率偏移模式。因为偏移是可逆的，只要统计这些偏移的出现频率，就能恢复出原始的二进制水印并判断该代码是否由带水印的模型生成。

**最巧妙的点**在于把熵信息当作“开关”。低熵位置直接跳过水印，避免了代码质量的退化；高熵位置才打开水印开关，既保证了检测信号的可观测性，又让水印的植入对最终代码几乎没有影响。

### 实验与效果
- **测试任务**：作者在公开的代码生成基准（如 HumanEval、MBPP）上进行评估，覆盖 Python、JavaScript 等主流语言。  
- **对比基线**：包括传统的 logit‑based 水印、基于 token 子集的硬编码水印以及几种主流的事后检测方法（如 perplexity‑based、n‑gram fingerprint）。  
- **结果声称**：SWEET 在保持原始模型生成代码的通过率（即可编译、单元测试通过率）基本不下降的前提下，检测准确率比最强的事后检测提升了数个百分点，且显著优于所有直接在每一步植入水印的基线。  
- **消融实验**：作者分别去掉熵阈值、去掉可逆偏移、仅使用事后检测三种设置，发现熵阈值的加入是提升代码质量的关键，而可逆偏移是提升检测率的主要因素。  
- **局限性**：论文未给出在极低熵（如单行函数）几乎没有高熵位置时的检测表现；此外，阈值 τ 的选取需要在不同语言和模型规模间手动调参，尚未提供自动化方案。

### 影响与延伸思考
SWEET 把“信息熵”引入水印设计，为代码生成这种低熵任务提供了可行的防伪思路。自发表后，已有工作尝试把类似的熵阈值机制搬到文本摘要、对话生成等高确定性场景，进一步验证了“只在模型犹豫的地方植入信号”这一原则的普适性。未来可以探索 **自适应阈值学习**（让模型自行决定何时开启水印）以及 **跨模型水印迁移**（在不同模型之间共享同一水印序列），这些方向都有望让水印技术更稳健、更易部署。

### 一句话记住它
只在模型“不确定”的代码位置偷偷调小概率，让水印既不破坏代码，又能被精准捕捉——这就是 SWEET 的核心。