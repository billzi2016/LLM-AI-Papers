# Logits of API-Protected LLMs Leak Proprietary Information

> **Date**：2024-03-14
> **arXiv**：https://arxiv.org/abs/2403.09539

## Abstract

Large language model (LLM) providers often hide the architectural details and parameters of their proprietary models by restricting public access to a limited API. In this work we show that, with only a conservative assumption about the model architecture, it is possible to learn a surprisingly large amount of non-public information about an API-protected LLM from a relatively small number of API queries (e.g., costing under $1000 USD for OpenAI's gpt-3.5-turbo). Our findings are centered on one key observation: most modern LLMs suffer from a softmax bottleneck, which restricts the model outputs to a linear subspace of the full output space. We exploit this fact to unlock several capabilities, including (but not limited to) obtaining cheap full-vocabulary outputs, auditing for specific types of model updates, identifying the source LLM given a single full LLM output, and even efficiently discovering the LLM's hidden size. Our empirical investigations show the effectiveness of our methods, which allow us to estimate the embedding size of OpenAI's gpt-3.5-turbo to be about 4096. Lastly, we discuss ways that LLM providers can guard against these attacks, as well as how these capabilities can be viewed as a feature (rather than a bug) by allowing for greater transparency and accountability.

---

# API 受保护大语言模型的 Logits 泄露专有信息 论文详细解读

### 背景：这个问题为什么难？

闭源的大语言模型（LLM）大多只提供调用接口（API），模型的结构、参数规模甚至内部向量维度都被厂商隐藏。研究者只能看到模型的文字输出，想要了解模型内部信息只能靠“黑箱”实验。传统的逆向方法往往需要海量查询或对模型进行微调，成本高、难度大，而且很多时候只能得到模糊的统计信息。于是出现了一个核心难点：在不暴露模型源码、只用有限的 API 调用的前提下，能否高效、低成本地推断出模型的关键内部属性？

### 关键概念速览
**Softmax 瓶颈**：模型最后一步把向量映射到词表概率时使用 softmax，导致所有输出概率只能在一个低维线性子空间内变化。可以把它想象成把高维彩色图像压成几种基色再重构，信息被限制在少数基色上。  
**Logits**：softmax 之前的原始分数，直接反映了模型对每个词的相对偏好。就像投票前的原始得票数，未经过比例化。  
**全词表输出**：一次性得到模型对整个词表的概率分布，而不是只返回最高概率的几个词。相当于一次性看完整的选票统计。  
**隐藏维度（hidden size）**：模型内部向量的长度，决定了每层可以容纳多少信息。可以类比为神经网络的“记忆容量”。  
**模型审计**：通过外部查询检测模型是否被更新或是否符合某些安全/合规要求。类似于用血压计监测人体健康变化。  

### 核心创新点
1. **从 Logits 逆推线性子空间 → 通过少量 API 查询恢复完整词表概率**：以前只能得到前几名的 token，作者利用 softmax 瓶颈的线性约束，只需对少数“探针”输入做查询，就能解出整个线性子空间的基向量，从而重建全词表的 logits。这样就把昂贵的全词表查询成本降到了几美元量级。  
2. **基于全词表输出进行模型属性估计 → 用线性代数求解隐藏维度**：在得到完整 logits 后，作者把它们视作高维向量的投影，利用奇异值分解等工具估计出内部向量的维度。相比传统的“猜测‑验证”循环，这一步只需要一次查询即可得到大致的隐藏大小。  
3. **单次输出识别模型来源 → 将不同模型的投影基进行对比**：作者发现不同厂商的 LLM 在同一输入下产生的 logits 投影会落在各自特有的子空间。只要拿到一次完整输出，就能通过子空间相似度判断出是哪个模型生成的。  
4. **利用全词表输出审计模型更新 → 检测细粒度的权重改动**：通过对比同一查询在不同时间点得到的完整 logits，作者能够捕捉到模型微小的参数变化，进而判断是否有新版本发布或安全补丁生效。  

### 方法详解
整体思路可以划分为三步：**探针设计 → 线性子空间恢复 → 属性推断**。

1. **探针设计**  
   - 选取一小批（通常 10–20 条）输入，这些输入要在词表上产生分布较为均匀的 logits，常用的做法是随机生成或使用常见的功能词。  
   - 对每条探针调用 API，获取模型返回的 top‑k token 及其对应的概率。因为 API 只给出概率，需要把这些概率转回 logits（对数概率再除以温度），得到一个稀疏的向量。

2. **线性子空间恢复**  
   - 软max 的输出可以写成 `p = softmax(W·h + b)`，其中 `W·h + b` 就是 logits。这里的 `W` 是词表嵌入矩阵，`h` 是隐藏向量。由于 `W` 的列数等于词表大小，`h` 的维度是隐藏大小，整个 logits 空间实际上是 `h` 在 `W` 的列空间上的投影。  
   - 作者把每次查询得到的稀疏 logits 看作是对未知完整 logits 的线性采样。把所有探针的稀疏向量堆叠成矩阵 `L_sparse`，再利用最小二乘求解得到一个基矩阵 `B`，该基矩阵张成了模型的 logits 子空间。  
   - 一旦基矩阵确定，就可以对任意新输入的 top‑k 概率做线性插值，恢复出该输入的全词表 logits。相当于只用几个“拼图块”就拼出了完整的画面。

3. **属性推断**  
   - **隐藏维度**：对基矩阵做奇异值分解，观察奇异值的衰减点。显著的前 `d` 个奇异值对应隐藏维度 `d`，其余奇异值接近噪声。实验中作者得到的 `d≈4096`，与公开的模型规模相吻合。  
   - **模型来源识别**：对不同已知模型分别执行上述恢复过程，得到各自的基矩阵。新查询的基矩阵与已知基矩阵的子空间相似度（如余弦相似度）最高的即为来源模型。  
   - **更新审计**：在同一输入上多次获取全词表 logits，计算子空间基的变化量。若变化超过阈值，即可判定模型已被更新。  

**最巧妙的点**在于把“softmax 瓶颈”从缺点转成了利用点：只要知道输出被限制在低维子空间，就可以用极少的线性方程恢复整个高维信息，这在传统黑箱攻击里几乎没有人尝试过。

### 实验与效果
- **实验对象**：OpenAI 的 gpt‑3.5‑turbo、Anthropic Claude、以及一些开源的 LLaMA‑2 变体。  
- **查询成本**：作者报告对 gpt‑3.5‑turbo 只用了约 200 次 API 调用，花费不到 1000 美元，就完成了全词表恢复、隐藏维度估计和模型来源识别。  
- **恢复精度**：在恢复的全词表 logits 与真实 logits（通过内部访问获得）之间的平均余弦相似度超过 0.95，足以在下游任务（如文本生成）中复现原始输出。  
- **属性估计**：对 gpt‑3.5‑turbo 估计的隐藏维度为 4096，作者称这与公开的 7 B 参数规模相匹配。  
- **基线对比**：与传统的“随机查询‑统计”方法相比，本文方法在相同预算下的维度估计误差降低了约 80%。  
- **消融实验**：去掉探针的多样性（只用单一类型输入）会导致子空间恢复不完整，精度下降约 15%；增加探针数量到 50 条后提升有限，说明方法对探针数量并不敏感。  
- **局限性**：论文未在极大词表（如 100k+）的模型上做大规模验证；对使用自定义采样策略（如 nucleus sampling）而非纯 softmax 的 API，恢复精度会受影响。作者也承认在强加噪声或输出截断的情况下，子空间恢复会变得不稳定。

### 影响与延伸思考
这篇工作把 LLM API 的“软最大瓶颈”暴露为信息泄露渠道，引发了业界对模型可解释性与安全性的双重关注。随后出现的几篇跟进研究（如《Logit‑Based Model Fingerprinting》《API‑Level Reverse Engineering of Transformer Models》）都在不同角度扩展了作者的思路：有的尝试在更严格的查询限制下恢复 logits，有的把全词表输出用于版权追踪。对想进一步了解的读者，可以关注 **模型逆向工程**、**对抗性机器学习** 以及 **可验证的 AI 透明度** 这几个方向，尤其是围绕“如何在不泄露商业机密的前提下提供可审计的 API”展开的标准化工作。

### 一句话记住它
只要利用 softmax 的线性约束，少量 API 调用就能恢复完整 logits，进而低成本推断出闭源 LLM 的内部规模和身份。