# Competition-Level Code Generation with AlphaCode

> **Date**：2022-02-08
> **arXiv**：https://arxiv.org/abs/2203.07814

## Abstract

Programming is a powerful and ubiquitous problem-solving tool. Developing systems that can assist programmers or even generate programs independently could make programming more productive and accessible, yet so far incorporating innovations in AI has proven challenging. Recent large-scale language models have demonstrated an impressive ability to generate code, and are now able to complete simple programming tasks. However, these models still perform poorly when evaluated on more complex, unseen problems that require problem-solving skills beyond simply translating instructions into code. For example, competitive programming problems which require an understanding of algorithms and complex natural language remain extremely challenging. To address this gap, we introduce AlphaCode, a system for code generation that can create novel solutions to these problems that require deeper reasoning. In simulated evaluations on recent programming competitions on the Codeforces platform, AlphaCode achieved on average a ranking of top 54.3% in competitions with more than 5,000 participants. We found that three key components were critical to achieve good and reliable performance: (1) an extensive and clean competitive programming dataset for training and evaluation, (2) large and efficient-to-sample transformer-based architectures, and (3) large-scale model sampling to explore the search space, followed by filtering based on program behavior to a small set of submissions.

---

# AlphaCode：竞争级代码生成 论文详细解读

### 背景：这个问题为什么难？

在编程助手的早期阶段，模型大多只能把自然语言描述直接翻译成代码，类似把“把两个数相加”变成一行函数。面对竞赛编程题目，要求选手先读懂复杂的题干、推敲算法思路、再写出高效实现，这种多层次的推理远超单纯的翻译。此前的代码生成模型在简单的函数填空或 API 调用上还能取得不错成绩，但在需要深度算法理解、时间空间复杂度分析的题目上几乎没有突破。根本的瓶颈在于：训练数据缺乏高质量的竞赛题目，模型规模不足以捕捉长程推理，且没有系统的搜索与验证机制来筛选出真正可运行的解答。

### 关键概念速览
- **竞争编程数据集**：收集自 Codeforces 等平台的题目、样例输入输出和优秀提交，类似把所有历届数学竞赛的试题和答案整理成一本参考书，供模型学习。
- **Transformer（变压器）模型**：一种基于自注意力机制的神经网络，擅长捕捉序列中远距离的关联，就像在长篇小说里快速定位前后呼应的情节。
- **大规模采样（sampling）**：在同一个输入下让模型生成成千上万的候选代码，类似让多位选手分别写解法，然后挑选最好的。
- **行为过滤（behavioral filtering）**：把生成的代码跑在真实的测试用例上，只保留通过或表现优秀的提交，像裁判只给通过所有测试的选手计分。
- **排名百分位（ranking percentile）**：在真实竞赛中把模型的提交与所有人类选手的成绩进行比较，百分位越低表示表现越好。

### 核心创新点
1. **从通用代码库到专属竞赛数据**  
   - 之前的模型大多使用 GitHub 上的开源项目做训练，代码风格杂乱、题目深度不足。  
   - AlphaCode 构建了一个规模庞大且经过清洗的竞争编程数据集，包含题目描述、样例、评测脚本以及高分人类提交。  
   - 这让模型在学习时直接接触到算法思路和评测逻辑，显著提升了对复杂题目的理解能力。

2. **超大 Transformer 与高效采样**  
   - 早期代码生成模型参数在几亿级别，采样速度慢且多样性受限。  
   - AlphaCode 采用了 41 B 参数的 Transformer，配合分布式并行采样，使得一次推理可以产生上万条候选代码。  
   - 大模型提供了更丰富的潜在解空间，而高效采样保证了在可接受的计算预算内探索到高质量解。

3. **两阶段过滤：行为验证 + 评分模型**  
   - 传统做法只靠语言模型的概率打分，容易保留语法正确但逻辑错误的代码。  
   - AlphaCode 首先在轻量级的测试用例上执行所有候选，剔除运行错误或超时的程序；随后用一个专门训练的评分网络对剩余代码进行细粒度排序。  
   - 这种“先跑后挑”的策略把最终提交的成功率从几百分点提升到超过 50% 的竞争排名。

### 方法详解
AlphaCode 的整体流程可以划分为四个阶段：**数据准备 → 大模型训练 → 大规模采样 → 行为过滤与排序**。

1. **数据准备**  
   - 从 Codeforces 下载过去几年的比赛题目，提取题目正文、约束条件、样例输入输出以及官方评测脚本。  
   - 对每道题目收集多位人类选手的提交，去除抄袭、编译错误的代码，只保留通过全部评测的高质量解。  
   - 最终得到一个“题目‑代码‑评测”三元组的数据库，供后续模型学习。

2. **大模型训练**  
   - 使用标准的自回归 Transformer 架构，输入为题目描述（自然语言）+ 已知的函数签名，输出为完整的程序文本。  
   - 训练目标是最大化生成代码的似然概率，同时加入代码特有的正则化（如缩进、括号匹配）来提升语法正确率。  
   - 由于数据量巨大，训练采用混合精度和模型并行技术，使 41 B 参数模型在数千 GPU 上完成。

3. **大规模采样**  
   - 对每个新题目，模型在 **温度采样**（控制随机程度）和 **核采样**（限制候选词的概率质量）下生成约 10 k–30 k 条代码。  
   - 采样过程类似让模型“思考”多次，每一次都可能走向不同的算法路径（比如使用贪心、动态规划或二分搜索）。

4. **行为过滤与排序**  
   - **第一层过滤**：把所有候选代码提交到轻量级的本地评测器，只运行公开的样例和若干隐藏的随机生成用例。任何编译错误、运行时异常或超时的程序直接被淘汰。  
   - **第二层排序**：对通过第一层的代码，用一个专门训练的二分类模型预测其在完整评测集上的成功概率。该模型的输入包括代码的 token 分布、执行时间特征以及前一步的测试通过率。  
   - 最终选取得分最高的前 5–10 条提交作为正式答案提交到真实的竞赛平台。

**最巧妙的点**在于把“语言模型的创造力”和“程序的可执行性”两者结合起来：模型负责产生多样化的创意，行为过滤负责把不符合物理约束的创意剔除，二者形成闭环，使得最终提交的代码既新颖又可靠。

### 实验与效果
- **评测平台**：使用 Codeforces 近三年的公开比赛，单场参赛人数常在 5 k–10 k 之间。  
- **基准对比**：与当时最强的代码生成模型（约 6 B 参数、未使用大规模采样）相比，AlphaCode 在同样的题目上平均排名提升约 30% 的百分位。具体而言，AlphaCode 在所有测试赛中取得 **前 54.3%** 的平均排名，意味着它的提交比超过一半的真实选手更好。  
- **消融实验**：作者分别关闭（1）竞争编程数据集、（2）大规模采样、（3）行为过滤，发现排名分别跌至 78%、71% 和 84% 的水平，说明每个模块对整体性能都有显著贡献。  
- **局限性**：AlphaCode 仍然依赖大量计算资源，采样与过滤的成本在普通实验室难以复现；对极端长代码或需要跨文件组织的项目表现不佳；评测仅覆盖了已知的竞赛题目，通用编程任务的迁移能力尚未充分验证。

### 影响与延伸思考
AlphaCode 的成功让业界认识到“规模+专用数据+行为回路”是提升代码生成模型的关键路径。随后出现的 **Codex、CodeGen、StarCoder** 等模型，都在数据质量和后处理验证上借鉴了类似思路。更进一步，研究者开始探索 **自我验证（self‑verification）**、**程序合约生成** 以及 **基于搜索的程序合成**，试图把运行时反馈直接嵌入训练循环。想深入了解的话，可以关注以下方向：  
- 大模型在 **程序合成搜索空间** 中的采样策略优化；  
- **神经符号混合** 方法，把传统算法模板与语言模型的创造力结合；  
- **低资源高效采样**，让普通实验室也能复现 AlphaCode 的核心思路。

### 一句话记住它
AlphaCode 用大规模竞赛数据 + 超大模型 + 先跑后挑的过滤链，让机器在编程竞赛中跑进前半圈。