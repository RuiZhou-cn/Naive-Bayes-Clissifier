# 使用朴素贝叶斯分类器实现文本分类

数学原理与代码解析参见Naïve Bayesian Classifier on 20NewsGroup.md

可执行代码参见NaiveBayesclassifier.ipynb

## 环境配置

安装sklearn包`conda install scikit-learn`

如果需要在Jupyter Botebook使用conda创建的环境，首先把刚才创建的环境添加到Jupyter Notebook中，下载ipykernel`pip install --user ipykernel`，然后在需要添加的环境下执行`python -m ipykernel install --user --name=yourenv`

## 代码实现

### 以20Newsgroups为数据集进行朴素贝叶斯分类器的训练与测试

#### 训练过程

得到Vocabulary

​		把所有文档单词拿出来，但要过滤掉重复的单词

计算先验概率$p(h_j)$和类条件概率$P(w_i|h_j)$（此处共20类，i=Vocabulary大小）

​		for each target value $h_i$ in H

​				把所有hi类的文章拿出来，构成一个子集=$artics_j$

​				$P(h_j)=\frac{|artics_j|}{|Training data|}$ # 计算每一类的先验概率

​				计算$artics_j$中总单词个数$n_j$

​				for each word $w_i$ in Vocabulary # 计算每个单词$w_i$的类条件概率

​						计算$artics_j$类文档中$w_i$单词出现的次数=$n_j^i$

​						$P(w_i|h_j)=\frac{n_j^i+1}{n_j+|Vocabulary|}$ # 在j类文档中wi的类条件概率

#### 测试过程

$h_{NB}=argmax_{h_j\in H}P(h_j)\prod_{i=1}^{N}P(w_i|h_j)$

概率值最大的类对应文章的类别

#### 用Sklearn中自带的贝叶斯分类器实现

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
x_train = tf.fit_transform(x_train)
x_test = tf.transform(x_test)

from sklearn.naive_bayes import MultinomialNB

bayes = MultinomialNB(alpha = 1.0)
bayes.fit(x_train, y_train)
import numpy as np
y_predict = bayes.predict(x_test)
y_predict = np.array(y_predict)
print("预测文章的类型：", y_predict)
print("The accuracy: ", bayes.score(x_test, y_test))
```

#### 自己的实现

导入数据

```python
from sklearn.datasets import fetch_20newsgroups # 导入sklearn package
news = fetch_20newsgroups(subset="all")
```

划分训练集与测试集

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25) # 划分训练集与测试集
```

数据预处理

创建wordlist

* 全部转化为小写
* 去掉标点符号
* 去掉换行符
* 将数据以空格划分

```python
import string
from collections import Counter
table = str.maketrans(dict.fromkeys(string.punctuation))

word_set = set()
counter_per_cat = [Counter() for i in range(len(news.target_names))]  # 创建文章类别个数的Counter()，计算每个单词在所有文档类别出现的频率 
for sample_id, sample in enumerate(x_train):
    sample_label = y_train[sample_id]
    words = str(sample).lower().translate(table).strip().split()#lower()转化为小写 translate()去掉标点符号 strip()去掉换行符 split()将数据以空格划分
    counter_per_cat[sample_label].update(words)
    word_set.update(words)
word_list = list(word_set)
```

###### Classifier learning

* prob_mat: 类条件概率
  * size of[label_num（类别数）, word_num（Vocabulary大小）]
  * 类条件概率
* Get expected statistics from the training data
  * total_freq : total word frequency in articles with certain label（具有特定标签的文章的总词频）
  * label_prob : the class prior probabilities of training data（先验概率）
  * empty_prob : for word doesn’t appear in training data（处理没有出现在训练数据中的词）
  * prob_mat : probability of each word appears in each article label
* Use these statistics to classify articles in the test data

```python
total_freq = [] # 每类文章的总词频：nj
label_prob = [] # 先验概率
empty_prob = [] # 没有出现在训练数据中的词
prior_prob = Counter(y_train)
for label_id, label in enumerate(news.target_names):
    total_freq.append(sum(counter_per_cat[label_id].values())) # 每类文章的总词频：nj
    label_prob.append(prior_prob[label_id]/len(y_train)) # 先验概率
    empty_prob.append((1)/(total_freq[label_id] + len(word_list))) # 没有出现在训练数据中的词

import numpy as np
prob_mat = np.zeros((len(news.target_names), len(word_set))) # 类条件概率
for label_id, label in enumerate(news.target_names):
    freq_list = counter_per_cat[label_id]
    for word_id, word in enumerate(word_set):
        if word not in freq_list:
            freq_label = 0
        else:
            freq_label = freq_list[word]
        freq_all = total_freq[label_id]
        prob = (freq_label + 1) / (freq_all + len(word_set)) # 计算类条件概率
        prob_mat[label_id, word_id] = prob
```

###### News article classification

* The steps for testing:
  * Tokenize the articles in test data (数据预处理)
  * Calculate the probability of the article belongs to each label（计算文章属于每一类的概率）
  * Classify it with the highest one（将文章分类为上面计算概率最大的类）
* Issue:
  * To avoid the product of probabilities getting too close to0, we use log likelihood equation: Convert product to addition（因为概率都是小数，连乘后结果很小，因此用log likelihood把连乘变成累加）

```python
import math
from tqdm import tqdm
label_prob = np.log(np.array(label_prob))
empty_prob = np.log(np.array(empty_prob))
prob_mat = np.log(np.array(prob_mat))
predict_labels = np.zeros(len(x_test), dtype = int)
for sample_id, sample in tqdm(enumerate(x_test), total = len(x_test)):
    probs = np.zeros(len(news.target_names))
    words = str(sample).lower().translate(table).strip().split() # Tokenize
    sample_len = len(words)
    word_freq = Counter(words)
    words = list(set(words).intersection(set(word_list))) # 返回words和word_list的交集
    for label_id, label in enumerate(news.target_names): # Calculate the probality for each category
        prob_label = label_prob[label_id]
        len_a = 0
        for word in words:
            word_id = word_list.index(word)
            prob_cur = prob_mat[label_id, word_id]
            prob_label += word_freq[word] * (prob_cur)
            len_a += word_freq[word]
        len_b = sample_len - len_a
        if len_b > 0:
            prob_label += len_b * (empty_prob[label_id])
        probs[label_id] = prob_label
    predict_label = np.argmax(probs) # Category with the highest probality
    predict_labels[sample_id] = predict_label
```

###### Results

* Calculate the accuracy of the prediction among all of the categories

```python
# accuracy of the prediction among all of the categories
comp = predict_labels - np.array(y_test)
accuracy = (len(np.where(comp == 0)[0])) / len(x_test)
print(accuracy)

# Detailed accuracy
y_test = np.array(y_test)
for i in range(len(news.target_names)):
    cat_data_num = (np.where(y_test == i))[0].shape[0]
    pred_cat_tmp = predict_labels[np.where(y_test == i)]
    pred_correct = np.where(pred_cat_tmp == i)[0].shape[0]
    accuracy_per_cat = pred_correct/cat_data_num
    print(news.target_names[i])
    print("total data number of this category: " + str(cat_data_num))
    print("number of correctly prediction: " + str(pred_correct))
    print("predicition accuracy for this category: " + str(accuracy_per_cat))
```

