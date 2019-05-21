<center><font size=5><b>Deep Neural Networks for YouTube Recommendations<sup>[1]</sup></b></font></center>
<center>Paul Covington, Jay Adams, Emre Sargin</center>
### 摘要
YouTube是目前最大规模和最复杂的工业级推荐系统之一。本文从较高的层面上对该系统进行了描述，并着重讲解了深度学习给该系统的性能带来的提升。本文按照经典的信息检索二段法进行划分：首先，详细介绍了一个深度候选生成模型，然后，介绍了一个单独的排序模型。同时，本文分享了从设计、迭代、维护一个大规模的推荐系统得到的经验和见解。
### 1. 介绍
YouTube是世界上最大的视频内容创作、分享、发掘平台。Youtube致力于帮助超过10亿用户，从不断增长的视频集中，发掘个性化的内容。本文重点介绍深度学习最近对YouTube视频推荐系统产生的巨大影响。
从以下三个角度来看，YouTube视频推荐是一项极富挑战性的工作：
- *Scale*：现有很多算法在小数据集上表现良好，但是无法在大数据集上运行。高度定制化的分布式学习算法和高效的服务系统，对于处理YouTube庞大的用户群和语料库是至关重要的。
- *Freshness*：YouTube的语料库每秒会上传很多小时时长的视频。推荐系统应该能够对新上传的内容和用户最近的行为及时响应并建模，因此存在exploration/exploitation的问题。
- *Noise*：数据的稀疏性和各种不可观察的外部因素导致YouTube的历史用户行为本身很难被预测。作者基本无法获取用户满意度的真实反馈，只能对带有噪声的隐式反馈信号进行建模。此外，与内容相关联的元数据结构很差，没有明确定义的本体。 所设计的算法需要对训练数据的这些特性具有鲁棒性。

本文的模型学习了大约10亿个参数，在数千亿的训练样本上进行了训练。
### 2. 系统概述
<center>
<img src="https://github.com/zhaibowen/Reading-Papers/blob/master/Image/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-19%20%E4%B8%8A%E5%8D%8810.59.07.png?raw=true" width="500" hegiht="313"/>

<font size=3 face="STKaiti">图1. 候选视频经过检索和排序呈现给用户</font>
</center>
整个推荐系统的框架如图1所示，系统由两个神经网络组成，一个用于候选生成，一个用于排序。

候选生成网络使用用户的历史行为数据作为输入，从视频库中检索出一小部分视频子集。该子集要求在与用户的相关性上具有高准确率。候选生成网络仅通过协同过滤提供宽泛的个性化特征。用户之间的相似性通过粗粒度的特征来表示，例如视频ID，搜索词以及人口统计学特征。
***notice: 什么是准确率？***
准确率(precision)$P=\frac{TP}{TP+FP}$，即在所有神经网络预测的正类中，真实的正类所占的比例。
<center>
<table style="text-align: center;">
<tr>
    <td rowspan="2" colspan="2">混淆矩阵</td>
    <td colspan="2">预测值</td>
</tr>
<tr>
    <td>正类</td>
    <td>负类</td>
</tr>
<tr>
    <td rowspan="2">真实值</td>
    <td>正类</td>
    <td>TP</td>
    <td>FN</td>
</tr>
<tr>
    <td>负类</td>
    <td>FP</td>
    <td>TN</td>
</tr>
</table>
</center>

***notice: 为什么是通过协同过滤提供宽泛的个性化特征？***
从一个列表中选出具有高召回率的一小部分"最好的"的视频进行推荐，需要细粒度的表示来区分视频之间的相对重要性。排序网络通过使用一个描述视频和用户的丰富特征集，根据期望的目标函数给每个视频打分，来完成这项任务。按照分数排序之后，最高分数的视频集被呈现给用户。
***notice: 召回率(recall)$R=\frac{TP}{TP+FN}$，文中采用的是哪个性能的召回率？为什么采用召回率？***
两段推荐法使得作者可以从非常大的视频库中进行推荐，同时确保展现在设备上的那一小部分视频是个性化的，并且对用户有吸引力。此外，这种设计可以混合其他来源的候选视频，例如早期的工作中所描述的那些<sup>[2]</sup>。
在开发过程中，作者广泛使用离线指标（精确度，召回率，排名损失等）来指导系统的迭代改进。 然而，为了最终确定算法或模型的有效性，作者依靠实时实验进行A / B测试。 在实时实验中，可以衡量点击率，观看时间以及衡量用户参与度的许多其他指标的细微变化。 这很重要，因为实时A / B结果并不总是与离线实验相关。
### 3. 候选生成
在候选生成过程中，从巨大的资源库中筛选出可能与用户相关的数百个视频。 本文描述的推荐系统的前身是在排序损失下训练的矩阵分解方法<sup>[3]</sup>。 本文采用的神经网络模型的早期方案模拟了这种分解行为，只采用了嵌入用户观看历史的浅网络。 从这个角度来看，本文的方法可以被视为分解技术的非线性推广。
#### 3.1 将推荐视为分类问题
本文将推荐视为一个多分类问题，基于用户$U$和上下文$C$，从资源库$V$的上百万个视频$i$中，预测在时刻$t$观看的视频$w_t$等于$i$的概率，
$$P(w_t=i|U,C)=\frac{e^{v_iu}}{\sum_{j\in V}e^{v_ju}}，$$其中，$u\in\mathbb{R}^N$表示用户和上下文在高维空间中的映射，$v_j \in\mathbb{R}^N$表示每个候选视频的映射。深度神经网络的任务是根据用户历史和上下文学习映射$u$，并通过softmax分类器区分不同的视频。
尽管YouTube存在显式反馈机制（拇指向上/向下，产品调查等），但本文使用隐式反馈来训练模型，即将用户观看完一个视频作为正例。 这是因为隐式反馈数据具有更高的数量级，可以在显示反馈数据极其稀少的情况下做出有效推荐。
***高效的超多分类***
为了有效地训练包含数百万个类的模型，本文依靠一种技术从背景分布中负采样，然后通过重要性加权来校正这个采样<sup>[4]</sup>。对于每个样本，对真实标签和采样得到的负类，最小化其交叉熵损失函数。在实践中，对数千个负样本进行采样，相对于传统softmax加速了100倍以上。另一种流行的替代方法是采用层级softmax<sup>[5]</sup>，但无法达到相当的准确度。 在层级softmax中，遍历树中的每个节点涉及区分通常不相关的类集，使得分类问题更加困难并且降低性能。
***notice: 使用层级softmax时，不相关的类集是指什么？***
在服务时，需要计算出最可能的N类（视频）呈现给用户。 数十毫秒的严苛的服务延迟内对数百万个物品打分需要一个亚线性于类别数量的近似打分方案。 YouTube以前的系统依赖于哈希<sup>[6]</sup>，本文的分类器采用类似的方案。 由于在服务时不需要来自经过softmax输出层校准后的似然概率，因此评分问题简化为在点积空间中的最近邻搜索，可以使用通用库来实现<sup>[7]</sup>。 本文发现A / B实验的结果对最近邻搜索算法的选择不是特别敏感。
***notice: 什么是点积空间的最近邻搜索？***
#### 3.2 模型架构
<center>
<img src="https://github.com/zhaibowen/Reading-Papers/blob/master/Image/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-19%20%E4%B8%8B%E5%8D%887.56.12.png?raw=true" width="500" hegiht="313"/>

<font size=3 face="STKaiti">图2. 候选生成网络的模型架构</font>
</center>
模型架构如图2所示。首先用Embedding层将用户历史观看视频ID映射成高维向量，然后通过简单平均（相比于求和或求最大值表现更优），将不定长的观看序列变成定长向量。Embedding层通过反向梯度传播和模型的其他参数放在一起训练。平均后的向量和其它特征连成很宽的第一层，后面是使用RELU作为激活函数的全连接层。

***notice: Embedding放在整个模型里一起训练，也就是在反向传播时，同时更新了softmax中的$u$和$v_i$。***
#### 3.3 异构信号
神经网络相对于矩阵分解方法的一个关键优势是可以将任意的连续特征和类别特征加入到模型中。搜索历史的处理方法和观看历史类似，每个query通过unigram和bigram符号化。人口学特征对推荐系统给新用户做出推荐非常重要。 用户的地理区域和使用设备同样经过嵌入层被级联。 简单的二元或连续特征（例如用户的性别，登录状态和年龄）作为实数直接输入到网络中，并标准化为[0,1]之间。
***“样本年龄” 特征***
每秒都会有数小时的视频上传到YouTube。推荐最近上传的（“新鲜”）内容对于YouTube来说非常重要。 我们始终注意到用户更喜欢新鲜内容，但不以牺牲相关性为代价。 除了简单推荐用户想要观看的新视频的一阶效应之外，还有一个关键的次要现象：引导和传播内容<sup>[8]</sup>。
机器学习系统通常表现出隐式的对历史的偏置，这是因为它们由历史样本训练，并去预测未来。视频流行度的分布是非常不稳定的，但是由推荐产生的语料库的多项式分布反映的是过去几周的训练窗口内的平均观看似然概率，因此，本文将样本年龄作为一个训练特征。在服务时，此特征置零（或略微为负），以反映模型在训练窗口的尾部进行预测。图3展示了这种方法对任意选择的视频的效果。
<center>
<img src="https://github.com/zhaibowen/Reading-Papers/blob/master/Image/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-20%20%E4%B8%8B%E5%8D%885.30.34.png?raw=true" width="500" hegiht="313"/>

<font size=3 face="STKaiti">图3. 加入样本年龄前后某个视频的类别概率对比</font>
</center>

***notice: 什么是样本年龄？***
假设所有训练样本中最新的样本产生时间为t1，某条样本的产生时间为t2（用户观看了某条视频），则样本的年龄为t1-t2
***notice: 推荐产生的语料库为什么是多项式分布？***
n个样本可以看成n次独立实验，每个样本经过模型和softmax层，输出m种分类的概率，m种分类的概率之和为1，满足多项式分布的定义。
***notice: 图3的经验分布是怎么得到的？***
***notice: 为什么不直接使用视频上传时间作为特征？***
因为每个视频的上传时间都不一样，神经网络输出$u$之后，通过softmax计算所有类别（视频）的概率，因此无法针对每一个视频单独设置一个上传时间。
***notice: 为什么样本年龄是有效的特征？***
假设某个视频已经产生了10天，那么[0,10]中的每一天观看它的人数应该符合图4绿色曲线的经验分布，每一天由该视频产生的训练样本的相对数量如图5所示，在训练中，样本的相对数量会转化为类别概率，例如在样本年龄等于5，并且其他输入特征相似的情况下，有10个该视频的样本（一个用户观看了一个视频为一个样本），90个其他视频的样本，那么训练出来的模型在预测相似特征时，该类别的输出概率大致为10%。因此图5也是预测类别概率随样本年龄变化的曲线，如果设置为负值，那么预测出的曲线应该是原有曲线的延伸（类似于直线的延伸），这解释了为什么图4中红色曲线的峰值相对于绿色曲线有一天的延迟。

<center>
<img src="https://github.com/zhaibowen/Reading-Papers/blob/master/Image/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-20%20%E4%B8%8B%E5%8D%885.30.34.png?raw=true" width="500" hegiht="313"/>

<font size=3 face="STKaiti">图3. 加入样本年龄前后某个视频的类别概率对比</font>
</center>

![图5. 样本相对比例随样本年龄变化示意图](https://upload-images.jianshu.io/upload_images/17943149-4d8854ea083a2cc8.png?imageMogr2/auto-orient/strip%7CimageView2/2/480)***notice: 如何处理新产生的视频？***
在模型训练好之后，进行服务时，新产生的视频是没有单独的Embedding表示的，Embedding层会用默认值表示未见过的ID，这样的话新视频的类别概率肯定不高，因此视频刚产生时，应该是从其他渠道呈现给一部分用户（例如图1的other candidate sources，或主动搜索），然后模型在不断的迭代过程中（例如小时级别的迭代），包含该视频的样本越来越多，模型学到的该视频的Embedding也越来越精确。
#### 3.4 标签和上下文的选择
推荐通常涉及解决代理问题并迁移结果到特定的上下文。 一个典型的例子是假设准确预测评分可以有效推荐电影<sup>[9]</sup>。 我们发现，代理学习问题的选择对A / B测试的性能非常重要性，但很难通过离线实验来衡量。
***notice: 什么是代理问题？***
训练样本是从YouTube已观看视频中产生的，而不仅仅是我们所产生的推荐。否则新内容很难被展现出来，系统会过分倾向于exploitation。如果用户通过我们的推荐以外的方式发现视频，我们希望能够通过协同过滤将此发现快速传播给其他人。 改进实时指标的另一个关键是为每个用户生成固定数量的训练样本，从而每个用户的权重在损失函数中都是相等的。 这阻止了一小群高度活跃的用户主导损失函数。
***notice: 每个用户生成固定数量的训练样本***
有些反直觉的是，必须非常小心地对分类器隐瞒一些信息，以防止模型利用网站结构并过拟合代理问题。 以用户刚刚发出“taylor swift”搜索查询的情况为例。 由于我们要预测下一个观看的视频，给定此信息的分类器将预测最有可能观看的视频是出现在具有“taylor swift”搜索结果页面上的视频。 毫无疑问，使用用户最后一个搜索页面作为推荐主页是非常糟糕的。 通过丢弃序列信息并用无序的token包表示搜索查询，使得分类器不再直接知道标签的来源。
随机held-out的方法会导致数据泄露，采用预测未来的方法性能会好得多（图6）。
![图6. 预测held-out观看与预测未来观看](https://upload-images.jianshu.io/upload_images/17943149-128e749517b1b22d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#### 3.5 特征和深度的实验
如图7所示，添加特征和深度可显着提高预测准确性。在这些实验中，视频和搜索token的词汇表都是1M的，通过Embedding被映射成256维的向量，每个样本最多50个最近的观看记录和50个最近的搜索记录。softmax层在相同的1M视频类别上输出多项式分布。神经网络是典型的塔式结构，深度为0的网络实际上是线性分解方案，与前一个系统非常相似。增加宽度和深度，直到增量收益减少并且收敛变得困难：
•深度0：256 线性层
•深度1：256 ReLU
•深度2：512 ReLU→256 ReLU
•深度3：1024 ReLU→512 ReLU→256 ReLU
•深度4：2048 ReLU→1024 ReLU→512 ReLU→256 ReLU
![增加特征和深度提高了平均预测精度（MAP）](https://upload-images.jianshu.io/upload_images/17943149-8fe040a52f7da28b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/480)
### 4. 排序
### 参考文献
>[1] P. Covington, J. Adams, E. Sargin. Deep neural networks for youtube recommendations[C]. Proceedings of the 10th ACM conference on recommender systems. ACM, 2016: 191-198.
>[2] J. Davidson, B. Liebald, J. Liu, P. Nandy, T. Van Vleet, U. Gargi, S. Gupta, Y. He, M. Lambert, B. Livingston, and D. Sampath. The youtube video recommendation system. In Proceedings of the Fourth ACM Conference on Recommender Systems, RecSys ’10, pages 293–296, New York, NY, USA, 2010. ACM.
>[3] J. Weston, S. Bengio, and N. Usunier. Wsabie: Scaling up to large vocabulary image annotation. In Proceedings of the International Joint Conference on Artificial Intelligence, IJCAI, 2011.
>[4] S. Jean, K. Cho, R. Memisevic, and Y. Bengio. On using very large target vocabulary for neural machine translation. CoRR, abs/1412.2007, 2014.
>[5] F. Morin and Y. Bengio. Hierarchical probabilistic neural network language model. In  AISTATS, pages 246–252, 2005.
>[6] J. Weston, A. Makadia, and H. Yee. Label partitioning for sublinear ranking. In S. Dasgupta and D. Mcallester, editors, Proceedings of the 30th International Conference on Machine Learning (ICML-13), volume 28, pages 181–189. JMLR Workshop and Conference Proceedings, May 2013.
>[7] T. Liu, A. W. Moore, A. Gray, and K. Yang. An investigation of practical approximate nearest neighbor algorithms. pages 825–832. MIT Press, 2004.
>[8] L. Jiang, Y. Miao, Y. Yang, Z. Lan, and A. G. Hauptmann. Viral video style: A closer look at viral videos on youtube. In Proceedings of International Conference on Multimedia Retrieval, ICMR ’14, pages 193:193–193:200, New York, NY, USA, 2014. ACM.
>[9] X. Amatriain. Building industrial-scale real-world recommender systems. In Proceedings of the Sixth ACM Conference on Recommender Systems, RecSys ’12, pages 7–8, New York, NY, USA, 2012. ACM.
### 个人理解
######1. 准确率(precision)与召回率(recall)

