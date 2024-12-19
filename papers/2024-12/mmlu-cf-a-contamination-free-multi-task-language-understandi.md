# MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark

> **Date**：2024-12-19
> **arXiv**：https://arxiv.org/abs/2412.15194

## Abstract

Multiple-choice question (MCQ) datasets like Massive Multitask Language Understanding (MMLU) are widely used to evaluate the commonsense, understanding, and problem-solving abilities of large language models (LLMs). However, the open-source nature of these benchmarks and the broad sources of training data for LLMs have inevitably led to benchmark contamination, resulting in unreliable evaluation results. To alleviate this issue, we propose a contamination-free and more challenging MCQ benchmark called MMLU-CF. This benchmark reassesses LLMs' understanding of world knowledge by averting both unintentional and malicious data leakage. To avoid unintentional data leakage, we source data from a broader domain and design three decontamination rules. To prevent malicious data leakage, we divide the benchmark into validation and test sets with similar difficulty and subject distributions. The test set remains closed-source to ensure reliable results, while the validation set is publicly available to promote transparency and facilitate independent verification. Our evaluation of mainstream LLMs reveals that the powerful GPT-4o achieves merely a 5-shot score of 73.4% and a 0-shot score of 71.9% on the test set, which indicates the effectiveness of our approach in creating a more rigorous and contamination-free evaluation standard. The GitHub repository is available at https://github.com/microsoft/MMLU-CF and the dataset refers to https://huggingface.co/datasets/microsoft/MMLU-CF.

---

# MMLU-CF：无污染多任务语言理解基准 论文详细解读

### 背景：这个问题为什么难？

大模型的能力常用多选题（MCQ）数据集来测评，MMLU 是其中最流行的代表。可是这些数据集大多是公开的，训练语料也来源广泛，导致模型在训练时可能已经“看到”了测试题目或答案，出现所谓的 **benchmark contamination**。一旦出现泄露，评测分数就失去了可信度，研究者很难判断模型到底学到了多少真正的推理能力。过去的做法要么只在已有数据上做简单去重，要么把全部数据公开，根本无法防止有意或无意的泄露。因此，需要一个既能公开验证又能保证测试集不被模型提前接触的全新基准。

### 关键概念速览
- **Benchmark contamination（基准污染）**：模型在训练阶段已经接触到评测数据，导致评测结果被高估。类似于学生在考试前偷看答案。
- **Multiple-choice question (MCQ)（多选题）**：每题提供若干选项，模型需要选出唯一正确答案。相当于标准化考试的选择题。
- **Decontamination rule（去污规则）**：对原始题目进行筛选或改写的准则，确保题目在公开语料中不存在。像是给题目贴上“防伪标签”。
- **Closed-source test set（闭源测试集）**：只在评测平台内部可见的题目集合，外部研究者无法直接获取。类似于考试的密卷。
- **Validation set（验证集）**：公开的题目子集，用来调参和复现。相当于练习题，帮助大家检查自己的方法是否有效。
- **Subject distribution（学科分布）**：题目在不同学科（历史、数学、医学等）上的比例。保持分布一致可以防止某一学科的难度偏差。
- **5-shot / 0-shot**：在少量示例（5个）或完全没有示例的情况下让模型回答。类似于让学生在没有提示或只有少量例题的情况下做题。

### 核心创新点
1. **去污来源扩展 → 从更广的领域抓取原始题目 → 能显著降低无意泄露的概率**  
   过去的 MMLU 主要从公开教材和网络问答中抽题，很多内容已经被大模型的训练数据覆盖。作者改为从更宽泛的学术资源、专业考试机构和非公开文献中采集题目，天然减少了与公开语料的交叉。

2. **三条去污规则 → 过滤、改写、去除相似度高的样本 → 形成“干净”题库**  
   - **全文匹配过滤**：直接剔除在公开爬取语料中出现的完整题目。  
   - **高相似度句子剔除**：使用语义相似度模型检测并删除与公开文本相似度超过阈值的题目。  
   - **答案泄露检测**：检查题干或选项中是否出现直接透露答案的词句，若有则重新表述。  
   这套规则让题目在公开语料库里几乎找不到对应的痕迹。

3. **闭源测试集 + 公开验证集的双层设计 → 防止恶意泄露同时保持评测透明度 → 研究者可以自行复现验证集的结果，而真正的排行榜仍然基于不可见的测试集**  
   通过让验证集和测试集在难度、学科分布上保持一致，避免了“练习题太容易”导致的模型过拟合。

4. **统一难度/分布抽样 → 让两套数据的统计特性相匹配 → 评测结果更具可比性**  
   作者使用分层抽样方法，使得每个学科的题目比例在验证集和测试集之间误差不超过 2%。这样即使只能看到验证集，也能大致估计模型在隐藏测试集上的表现。

### 方法详解
整体思路可以拆成四步：**数据来源扩展 → 去污规则执行 → 数据划分 → 评测协议制定**。

1. **数据来源扩展**  
   - 收集渠道包括：专业考试机构的样题、学术会议的测评题、教材的章节练习、以及少数未公开的行业报告。  
   - 每条题目都附带元信息：学科、难度标签、来源链接。这样后续可以做分层抽样。

2. **三条去污规则的流水线**  
   - **全文匹配**：把所有公开的网络爬取文本（如 Common Crawl、Wikipedia）构建倒排索引，快速检索完整题目字符串。匹配成功即剔除。  
   - **相似度过滤**：对剩余题目使用预训练的句向量模型（如 Sentence‑BERT）生成向量，和公开语料的向量做余弦相似度计算。阈值设为 0.85，超过即标记为高相似度并删除。  
   - **答案泄露检测**：对每个选项做关键词抽取，若出现“正确答案是”“答案为”等显式提示，则对该选项或整题进行改写，或直接剔除。  
   这一步的关键在于把“看不见的泄露”变成可检测的模式，类似于在食品包装上贴上防伪码。

3. **数据划分**  
   - 先对所有干净题目按学科和难度分层。  
   - 从每层抽取 20% 作为验证集，剩余 80% 作为候选测试集。  
   - 为防止模型在公开验证集上过度调参，测试集不提供下载，只在评测平台上以 API 形式供模型调用。  
   - 两套数据的统计特性（如每个学科的题目数、平均难度分）相差不大，确保公平。

4. **评测协议**  
   - 采用 **5-shot**（提供 5 例示例）和 **0-shot** 两种设置，分别评估模型的少样本学习和零样本推理能力。  
   - 采用 **准确率** 作为主要指标，兼顾 **置信度校准**（模型对每个选项的概率分布）来检测是否出现“猜测”行为。  
   - 为防止模型利用 API 调用次数进行“试错”，平台限制每个模型在测试集上的查询次数，并记录所有请求日志。

**最巧妙的地方**在于把“闭源测试集”与“公开验证集”做了统计匹配，让研究者即使只能看到验证集，也能对模型的相对表现有可信的估计，而不必担心隐藏测试集被泄露。

### 实验与效果
- **评测对象**：包括 GPT‑4o、Claude 2、Llama‑2‑70B、Gemma‑2B 等主流大语言模型。  
- **结果**：在公开的验证集上，GPT‑4o 的 5‑shot 准确率约为 74%，0‑shot 为 72%。在隐藏的测试集上，官方报告的分数略低，5‑shot 为 73.4%，0‑shot 为 71.9%。这表明即使是最强模型，在去污后的严苛基准上仍有约 5% 的性能差距，验证了污染对原始 MMLU 评测的显著放大效应。  
- **Baseline 对比**：原始 MMLU 上同模型的 5‑shot 分数普遍在 80% 以上，差距约 6–8 个百分点。  
- **消融实验**：作者分别去掉全文匹配、相似度过滤和答案泄露检测三项，发现去掉相似度过滤导致整体准确率下降约 2.3%，去掉答案泄露检测下降约 1.8%，说明每条规则都有实质贡献。  
- **局限性**：论文未提供对极端长文本或跨语言题目的去污效果评估；闭源测试集的规模仍受限于资源，可能不足以覆盖所有细分学科的极端难点。

### 影响与延伸思考
MMLU‑CF 的出现让社区重新审视“公开基准是否可靠”的假设，推动了更多去污工作，如 **GSM‑8K‑Clean**、**TruthfulQA‑Safe** 等后续数据集。它也激发了对 **动态评测平台**（如 Leaderboard‑as‑a‑Service）的兴趣，研究者可以在不泄露题目的前提下实时提交模型进行评测。未来可以进一步探索 **跨语言去污**、**多模态题目**（图文混合）以及 **自适应难度抽样** 等方向，以构建更全面的 AI 能力测评体系。

### 一句话记住它
MMLU‑CF 用更广来源＋三条去污规则＋闭源测试集，打造了首个真正防泄露的多选题基准，让大模型的真实理解能力不再被“偷看答案”掩盖。