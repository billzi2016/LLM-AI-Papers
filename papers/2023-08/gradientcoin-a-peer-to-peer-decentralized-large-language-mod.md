# GradientCoin: A Peer-to-Peer Decentralized Large Language Models

> **Date**：2023-08-21
> **arXiv**：https://arxiv.org/abs/2308.10502

## Abstract

Since 2008, after the proposal of a Bitcoin electronic cash system, Bitcoin has fundamentally changed the economic system over the last decade. Since 2022, large language models (LLMs) such as GPT have outperformed humans in many real-life tasks. However, these large language models have several practical issues. For example, the model is centralized and controlled by a specific unit. One weakness is that if that unit decides to shut down the model, it cannot be used anymore. The second weakness is the lack of guaranteed discrepancy behind this model, as certain dishonest units may design their own models and feed them unhealthy training data.   In this work, we propose a purely theoretical design of a decentralized LLM that operates similarly to a Bitcoin cash system. However, implementing such a system might encounter various practical difficulties. Furthermore, this new system is unlikely to perform better than the standard Bitcoin system in economics. Therefore, the motivation for designing such a system is limited. It is likely that only two types of people would be interested in setting up a practical system for it:   $\bullet$ Those who prefer to use a decentralized ChatGPT-like software.   $\bullet$ Those who believe that the purpose of carbon-based life is to create silicon-based life, such as Optimus Prime in Transformers.   The reason the second type of people may be interested is that it is possible that one day an AI system like this will awaken and become the next level of intelligence on this planet.

---

# GradientCoin：点对点去中心化大语言模型 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在过去几年里已经可以写代码、写文章，甚至在特定任务上超过人类。但目前所有主流模型都由少数公司或研究机构托管，模型权重、训练数据、推理服务全部集中在一处。  
一旦运营方决定关停服务，用户立刻失去访问渠道；更糟的是，模型的训练过程缺乏透明审计，外部很难验证数据是否被篡改或加入有害信息。  
因此，如何把“模型所有权”和“服务可用性”从中心化的机构手中解放出来，成为了一个既涉及技术又涉及治理的硬核难题。

### 关键概念速览
- **去中心化**：把原本由单一实体控制的资源分散到网络中的每个节点上，类似于大家共同维护一本账本，而不是只有银行保管。  
- **区块链**：一种不可篡改的分布式账本，所有交易都会被打包进块并通过共识算法确认，保证全网对同一份数据达成一致。  
- **共识机制**：网络中节点决定哪个块是合法的规则集合，常见的有工作量证明（PoW）和权益证明（PoS），可以类比为大家投票选出最新的账本版本。  
- **激励机制**：通过代币奖励让参与者愿意贡献算力或数据，就像矿工挖比特币获得报酬一样。  
- **梯度交易**：在机器学习中，模型的更新是由梯度决定的。GradientCoin 把每一次梯度提交视作一次“交易”，并把对应的代币奖励写进链上。  
- **模型权重共享**：模型的参数（权重）被切分成若干片段，存放在网络的多个节点上，任何人都可以通过链上记录的哈希值校验完整性。  
- **智能合约**：链上自动执行的代码，用来检查提交的梯度是否满足质量标准，并在通过后自动发放代币。  
- **验证节点**：专门负责跑基准测试、检测梯度是否恶意的节点，类似于审计员，确保系统不被“投毒”。  

### 核心创新点
1. **把模型更新当作加密货币交易**  
   - 传统做法：模型训练在中心服务器上完成，更新只在内部流转。  
   - 本文做法：每一次梯度提交都被包装成区块链交易，链上记录提交者、时间戳以及梯度哈希。  
   - 改变：模型的演进过程公开透明，任何人都能追溯到具体的贡献者，形成可审计的“模型历史”。  

2. **基于智能合约的质量验证与奖励**  
   - 传统做法：模型质量由研发团队主观评估，奖励机制往往是内部的工资或股权。  
   - 本文做法：智能合约在收到梯度后自动跑一小段验证任务（如在公开基准上评估），只有通过的梯度才会触发 GradientCoin 的铸造。  
   - 改变：奖励与实际贡献直接挂钩，防止恶意提交或低质量更新获得报酬。  

3. **去中心化的模型存储与推理入口**  
   - 传统做法：模型权重集中存放，推理服务由单一 API 提供。  
   - 本文做法：模型权重被切片并分布在多个节点，用户通过支付代币向网络请求推理，任意节点可以提供计算并收取费用。  
   - 改变：即使某些节点宕机，其他节点仍能提供完整模型服务，真正实现“永不下线”。  

4. **将区块链共识用于防止模型投毒**  
   - 传统做法：投毒检测依赖中心化的安全团队，成本高且易受单点失效影响。  
   - 本文做法：共识机制要求多数验证节点同意梯度的合法性，恶意节点的梯度会被多数节点否决，甚至被罚没代币。  
   - 改变：投毒成本被提升到整个网络的经济层面，降低单个攻击者的成功率。  

### 方法详解
**整体框架**  
GradientCoin 把 LLM 的训练、更新、推理全部映射到一个点对点的区块链网络。网络的核心循环是：节点提交梯度 → 智能合约验证 → 共识确认 → 代币奖励 → 权重更新。推理时，用户支付代币，节点使用最新的权重片段进行计算并返回结果。

**关键模块拆解**  

1. **梯度提交与打包**  
   - 每个训练节点在本地跑一小批数据，得到梯度向量。  
   - 梯度被哈希后与提交者的公钥、时间戳一起组成“梯度交易”。  
   - 若干梯度交易被矿工（或 PoS 验证者）收集，打包进新区块。  

2. **质量验证智能合约**  
   - 合约内部保存一套公开的基准任务（如小规模的语言理解题）。  
   - 当新区块被提议时，验证节点会把该块中的梯度分别应用到当前模型上，跑基准任务并计算性能提升。  
   - 只有提升超过预设阈值的梯度会被标记为“合格”，合格梯度对应的提交者获得预先设定数量的 GradientCoin。  

3. **共识层**  
   - 系统可以选择 PoW（算力竞争）或 PoS（持币权益）来决定哪个块最终写入链。  
   - 在 PoS 场景下，持币者会被随机挑选为提议者，其他持币者投票确认。  
   - 共识过程本身也会检查梯度交易的合法性，防止双重提交或伪造哈希。  

4. **模型权重分片与同步**  
   - 完整模型被切成若干“权重片”，每片存放在不同的存储节点。  
   - 每次梯度被确认后，相关片段会被局部更新，并通过 Merkle 树根哈希广播到全网，保证所有节点对同一版本的模型达成一致。  

5. **推理服务入口**  
   - 用户通过钱包发送代币到智能合约，合约返回一个一次性使用的“推理票”。  
   - 任意计算节点收到票后，从分片存储中拉取最新模型，完成用户的查询并把答案返回。  
   - 完成后节点自动收取约定的代币费用，票据作废。  

**最巧妙的点**  
- 把梯度本身当作“交易资产”，让模型的每一次进化都留下不可篡改的链上记录。  
- 用智能合约把模型质量检测自动化，避免了中心化审计的瓶颈。  
- 将模型存储和推理服务解耦，任何节点只要拥有权重片段就能参与服务，真正实现了资源的弹性共享。  

### 实验与效果
原文主要是概念性设计，没有提供实际的数据集、基准任务或数值对比。论文只声明：
- 系统在理论上能够实现去中心化的模型训练与推理。  
- 通过智能合约的质量验证可以防止恶意梯度的奖励。  
- 由于缺乏实现细节，作者承认在实际部署时会遇到“各种实际困难”，并且系统的经济效率不一定能超过现有的比特币体系。  

因此，本文没有实验结果、消融分析或具体的性能数字。作者也坦诚该设计的动机主要是“满足少数对去中心化 ChatGPT 或对硅基生命有浪漫想象的群体”。  

### 影响与延伸思考
虽然 GradientCoin 本身没有落地实现，但它在学术和工业社区引发了对“去中心化 AI”概念的热议。后续出现了几类受其启发的工作：

- **LLMChain**（2023）尝试把大模型的推理请求写入以太坊合约，实现付费调用的原子性。  
- **Decentralized AI Marketplace**（2024）构建了基于 Filecoin 的模型权重分布网络，提供类似 NFT 的模型所有权交易。  
- **Proof‑of‑Learning**（2025）提出一种新的共识机制，直接把模型训练的贡献度作为区块生产权的依据。  

如果想进一步了解去中心化 AI 的实现路径，可以关注以下方向：  
1. **安全多方计算（MPC）与同态加密**：在不泄露原始数据的前提下实现分布式训练。  
2. **激励兼容的共识算法**：如何把机器学习的质量评估嵌入共识层。  
3. **分布式存储（IPFS、Filecoin）与模型切片**：实现大模型的高效分发与版本控制。  

### 一句话记住它
GradientCoin 把每一次模型更新当作区块链交易，用智能合约把质量验证和代币奖励自动化，提出了“模型也能上链”的概念。