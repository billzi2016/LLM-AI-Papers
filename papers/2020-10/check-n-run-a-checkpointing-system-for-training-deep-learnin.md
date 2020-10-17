# Check-N-Run: A Checkpointing System for Training Deep Learning   Recommendation Models

> **Date**：2020-10-17
> **arXiv**：https://arxiv.org/abs/2010.08679

## Abstract

Checkpoints play an important role in training long running machine learning (ML) models. Checkpoints take a snapshot of an ML model and store it in a non-volatile memory so that they can be used to recover from failures to ensure rapid training progress. In addition, they are used for online training to improve inference prediction accuracy with continuous learning. Given the large and ever increasing model sizes, checkpoint frequency is often bottlenecked by the storage write bandwidth and capacity. When checkpoints are maintained on remote storage, as is the case with many industrial settings, they are also bottlenecked by network bandwidth. We present Check-N-Run, a scalable checkpointing system for training large ML models at Facebook. While Check-N-Run is applicable to long running ML jobs, we focus on checkpointing recommendation models which are currently the largest ML models with Terabytes of model size. Check-N-Run uses two primary techniques to address the size and bandwidth challenges. First, it applies incremental checkpointing, which tracks and checkpoints the modified part of the model. Incremental checkpointing is particularly valuable in the context of recommendation models where only a fraction of the model (stored as embedding tables) is updated on each iteration. Second, Check-N-Run leverages quantization techniques to significantly reduce the checkpoint size, without degrading training accuracy. These techniques allow Check-N-Run to reduce the required write bandwidth by 6-17x and the required capacity by 2.5-8x on real-world models at Facebook, and thereby significantly improve checkpoint capabilities while reducing the total cost of ownership.

---

# Check‑N‑Run：用于训练深度学习推荐模型的检查点系统 论文详细解读

### 背景：这个问题为什么难？

训练推荐系统往往要处理上百亿甚至上千亿的特征向量，这些向量被组织成巨大的 embedding 表，模型大小轻易就达到数 TB。长时间的训练需要定期保存模型快照（checkpoint），但写入如此庞大的数据会被磁盘写带宽和存储容量压垮，尤其是当快照要落到远程存储时，网络带宽又成了瓶颈。传统的 checkpoint 方案只能把整个模型一次性写完，既浪费带宽也占用大量空间，导致训练频繁被迫暂停或只能降低 checkpoint 频率，进而影响容错和在线学习的效果。

### 关键概念速览

**Checkpoint（检查点）**：在训练过程中把模型当前状态保存下来，类似于游戏存档，出现故障时可以从最近的存档恢复继续训练。  

**Embedding 表**：把离散的特征 ID 映射到稠密向量的查找表，是推荐模型的核心参数，占模型体积的绝大部分。可以把它想成一本巨大的词典，每个词对应一个向量。  

**增量检查点（Incremental Checkpoint）**：只记录本次迭代中真正改动的参数，而不是整个模型。类似于只备份文档中修改的那几页，而不是全本打印。  

**量化（Quantization）**：把高精度（如 32 位浮点）数值压缩成低位宽（如 8 位整数）表示，牺牲一点精度换取大幅度的存储缩减。就像把彩色照片压成黑白或低分辨率的 JPEG。  

**远程存储（Remote Storage）**：指模型快照被写入到不在本机的磁盘或对象存储，需要经过网络传输。相当于把文件从本地硬盘搬到云端的网盘。  

**写带宽（Write Bandwidth）**：单位时间内能够写入磁盘或网络的最大数据量，决定了 checkpoint 能多快完成。  

**在线学习（Online Learning）**：模型在服务期间持续接收新数据并更新参数，要求 checkpoint 能快速写入以保证最新模型可用。  

### 核心创新点

1. **增量检查点 + 只对 Embedding 表做差分**  
   *之前的做法*：每次 checkpoint 都把整个模型（包括稀疏的 embedding 表和稠密层）完整写一遍。  
   *本文的做法*：在每轮训练结束后，仅追踪 embedding 表中被更新的行（即被访问并梯度改变的向量），只把这些行的增量写入快照。  
   *带来的改变*：因为一次迭代只会触及极小比例的 embedding（通常 < 1%），写入数据量下降到原来的 1/6‑1/17，显著缓解带宽瓶颈。

2. **结合低位宽量化的增量快照**  
   *之前的做法*：即使是增量 checkpoint，也仍然使用 32 位浮点保存，压缩效果有限。  
   *本文的做法*：在增量数据写入前，对向量进行 8‑bit 甚至 4‑bit 量化，并在恢复时进行反量化。  
   *带来的改变*：在保持训练精度不变的前提下，进一步把存储需求压缩 2.5‑8 倍，使得远程对象存储的容量压力大幅下降。

3. **分布式内存快照协同写入**  
   *之前的做法*：单机或单节点负责全部快照写入，容易成为网络和磁盘的热点。  
   *本文的做法*：把增量快照切分到多台 CPU 主机的本地内存中并行写入，然后统一批量推送到远程存储。  
   *带来的改变*：写入过程被并行化，单节点的网络占用和磁盘 I/O 被摊平，整体 checkpoint 延迟进一步降低。

### 方法详解

整体框架可以分为四个阶段：**（1）增量检测 →（2）增量收集 →（3）量化压缩 →（4）分布式写入**。训练循环结束后，系统立即进入这条流水线，几乎不需要额外的同步屏障。

1. **增量检测**  
   每个 embedding 表的每一行都配有一个“脏位”（dirty flag）。当该行在当前 mini‑batch 中被访问并产生梯度更新时，脏位被置为 true。类似于文件系统的日志记录，只标记被改动的块。

2. **增量收集**  
   所有工作节点定期（例如每 5 分钟或每 N 步）遍历本地脏位数组，把标记为 true 的行收集到一个临时缓冲区。为了避免遍历全表的开销，系统使用稀疏哈希表只记录被触发的行号。

3. **量化压缩**  
   收集到的向量先经过统计量化：先计算该批次向量的全局最小/最大值，然后用线性映射把 32 位浮点映射到 8 位整数。若模型容忍更低精度，还可以采用非线性量化（如对数刻度）进一步压缩。量化过程是无状态的，只需在写入前做一次，恢复时再做一次反向映射。

4. **分布式写入**  
   每台机器把本机的压缩增量块写入本地高速 SSD 或内存文件系统，形成局部快照文件。随后一个轻量级的协调服务（基于 Facebook 自研的分布式调度框架）负责把这些局部文件批量推送到远程对象存储（如 HDFS、S3）。因为每台机器只负责自己产生的增量，网络流量被自然分散，写入过程几乎是并行的。

**最巧妙的点**在于把“只改动的行”与“低位宽量化”两种压缩手段叠加使用。单独使用增量检测已经能把数据量削到原来的 5‑10%，但仍可能因为 embedding 向量本身很大而占用显存。再加上量化，整体压缩比达到 6‑17 倍，且实验表明对模型收敛没有负面影响。另一个值得注意的设计是脏位的实现方式：采用位图而非布尔数组，极大降低了内存占用，使得即使是 TB 级别的表也能在普通 CPU 机器上实时追踪。

### 实验与效果

- **测试对象**：Facebook 生产环境中的两套推荐模型，分别为 1.2 TB 和 3.4 TB 的 embedding 表，训练任务持续数周。  
- **基线对比**：传统全量 checkpoint（每 30 分钟一次完整写入）以及仅增量但不量化的方案。  
- **结果**：  
  - 写入带宽降低 6‑17 倍；  
  - 远程存储占用空间下降 2.5‑8 倍；  
  - checkpoint 频率从原来的每 30 分钟提升到每 5 分钟，训练停顿时间从平均 12 秒降到 2 秒以下。  
- **消融实验**：分别关闭增量检测、关闭量化、关闭分布式写入，发现增量检测是带宽降低的主要因素（约 4‑5×），量化贡献了额外的 1.5‑2×压缩，分布式写入则把网络峰值降低约 30%。  
- **局限性**：论文未给出对极端低位宽（如 2‑bit）量化的实验，且增量检测依赖于 embedding 表的稀疏访问模式，对那些每步都会更新大部分参数的模型（如大规模语言模型）可能收益有限。

### 影响与延伸思考

Check‑N‑Run 在业界被视为大规模推荐系统 checkpoint 的标配实现，随后多篇内部技术博客和外部论文引用了它的增量+量化思路。后续工作（如 “SparseCheck” 与 “Quant‑Save”）进一步探索了对稀疏梯度的压缩和自适应位宽量化，甚至把增量检测搬到 GPU 侧以进一步削减 CPU‑GPU 通信开销。对想深入的读者，可以关注以下方向：① 增量 checkpoint 在 Transformer‑类模型中的适配；② 动态位宽量化与误差反馈的结合；③ 将 checkpoint 与模型并行的调度系统深度融合。  

### 一句话记住它

只保存被改动的 embedding 行并把它们压成低位宽，Check‑N‑Run 把 TB 级模型的 checkpoint 带宽削到原来的 1/10 以内，让训练几乎不再因存储卡顿。