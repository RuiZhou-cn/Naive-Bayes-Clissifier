### Math

$P(A\and B)=P(A|B)P(B)=P(B|A)P(A)$

$P(A\or B)=P(A)+P(B)-P(A\and B)$

$P(B)=\sum_{i=1}^nP(B|A_i)P(A_i)$

$P(h|D)=\frac{P(D,h)}{P(D)}=\frac{P(D|h)P(h)}{P(D)}$ 

* h: hypothesis

* D: Data

* $P(h|D)$: 后验概率

* $p(h)$: 先验概率

* $p(D|h)$: 条件概率

---

对于给定的训练集，首先基于特征条件独立假设学习输入输出的联合概率分布；然后基于此模型，对给定的输入 x，利用贝叶斯定理求出后验概率最大的输出 y。

$h_{MAP}=arg \max_hP(h|D)=arg \max_hP(D|h)P(h)$

因为$P(D)$是一样的，可以忽略，如果要算，用全概率公式计算$P(D)=\sum_{i=1}^nP(D|h_i)P(h_i)$。要想计算后验概率只需计算类条件概率和先验概率

 对先验概率的处理方式：

1. 所有h都给一个相同的先验概率，计算时可忽略
2. 每一类样本出现的频率代替概率

### 朴素贝叶斯

类条件概率在D维数较高时较难计算，朴素贝叶斯分类器通过假设每一维都是独立来减少运算量，但可能会因为数据的某些维度有关联，导致效果不好。

$P(D|h)=P(d_1,...,d_2|h)=\prod_iP(d_i|h)$，在朴素贝叶斯假设下，计算联合概率分布时直接连乘数据每一维的概率。在非朴素贝叶斯前提下，计算时要考虑他的分布如高斯。

改进方法：

* 贝叶斯信念网络(BBN）-假设某些维度存在关联
* 高斯混合模型

#### 文本分类

在文本分类场景下，$h$是文本类别，最终是根据后验概率大小决定是哪一类文本。

##### Learning

###### 先验概率：

可以用已有知识或者用每一类样本出现频率代替。

###### 条件概率：

我们用每个位置出现的单词的概率来分析文本，如有like, dislike两类，每个文本19个单词，词汇表中总共有5000个单词，则总共需要计算2x19x5000个概率，太复杂了，因此我们不考虑单词的位置，只考虑单词的统计个数。

$P(w_i|h_j)=n_i^j/n_j$，hj类中wi单词出现的概率 = wi单词在hj类中出现的次数 除以 这一类文本的单词数量

但实际应用时我们不用上述公式计算类条件概率，而是用下式：

$P(w_i|h_j)=(n_j^i+1)/(n_j+|Vocabulary|)$

因为某些单词如果在训练集中没有出现，那么他的概率为0，会导致整个后验概率为0，因此在分子加1，在分母加整个词汇大小

##### 分类