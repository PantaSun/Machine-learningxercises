
## 朴素贝叶斯
### 0.前言
> 在机器学习中，朴素贝叶斯分类器是一系列以假设特征之间强（朴素）独立下运用贝叶斯定理为基础的简单概率分类器。
朴素贝叶斯自20世纪50年代已广泛研究。在20世纪60年代初就以另外一个名称引入到文本信息检索界中，并仍然是文本分类的一种热门（基准）方法，文本分类是以词频为特征判断文件所属类别或其他（如垃圾邮件、合法性、体育或政治等等）的问题。通过适当的预处理，它可以与这个领域更先进的方法（包括支持向量机）相竞争。它在自动医疗诊断中也有应用。

>朴素贝叶斯分类器是高度可扩展的，因此需要数量与学习问题中的变量（特征/预测器）成线性关系的参数。最大似然训练可以通过评估一个封闭形式的表达式来完成，只需花费线性时间，而不需要其他很多类型的分类器所使用的费时的迭代逼近。

>在统计学和计算机科学文献中，朴素贝叶斯模型有各种名称，包括简单贝叶斯和独立贝叶斯。所有这些名称都参考了贝叶斯定理在该分类器的决策规则中的使用，但朴素贝叶斯不（一定）用到贝叶斯方法；《Russell和Norvig》提到“‘朴素贝叶斯’有时被称为贝叶斯分类器，这个马虎的使用促使真正的贝叶斯论者称之为傻瓜贝叶斯模型。”

>朴素贝叶斯是一种构建分类器的简单方法。该分类器模型会给问题实例分配用特征值表示的类标签，类标签取自有限集合。它不是训练这种分类器的单一算法，而是一系列基于相同原理的算法：所有朴素贝叶斯分类器都假定样本每个特征与其他特征都不相关。举个例子，如果一种水果其具有红，圆，直径大概3英寸等特征，该水果可以被判定为是苹果。尽管这些特征相互依赖或者有些特征由其他特征决定，然而朴素贝叶斯分类器认为这些属性在判定该水果是否为苹果的概率分布上独立的。

>对于某些类型的概率模型，在监督式学习的样本集中能获取得非常好的分类效果。在许多实际应用中，朴素贝叶斯模型参数估计使用最大似然估计方法；换而言之，在不用到贝叶斯概率或者任何贝叶斯模型的情况下，朴素贝叶斯模型也能奏效。
----摘自[维基百科](https://zh.wikipedia.org/wiki/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%88%86%E7%B1%BB%E5%99%A8)

### 1. 收集数据
以在线社区的留言板为例，如果某条留言使用了负面或者侮辱性的语言，那么就将该留言标识为内容不当，用1表示；正常留言用0表示。下文中，使用的一篇文档就相当于一条留言。

### 2. 准备数据：从文本中构建词向量


```python
def load_data_set():
    """创建实验样本，假设这六个列表为六篇文章"""
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    class_vec = [0, 1, 0, 1, 0, 1] # 1代表含有侮辱性文字的言论；0代表正常的言论
    return posting_list, class_vec
```


```python
def create_vocab_list(data_set):
    """创建一个包含在所有文档中不重复出现的词的列表"""
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)  # 创建并集
    return list(vocab_set)
```


```python
def words_set2vec(vocab_list, input_set):
    """将一篇文档中所有不重复的词转换为0-1向量"""
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return return_vec
```


```python
posting_list, class_vec = load_data_set()
posting_list
class_vec
```




    [0, 1, 0, 1, 0, 1]




```python
my_vocab_list = create_vocab_list(posting_list)
print(my_vocab_list) # 可以看到此表中没有重复的词
```

    ['food', 'ate', 'cute', 'worthless', 'mr', 'buying', 'dalmation', 'maybe', 'has', 'problems', 'stop', 'please', 'love', 'is', 'steak', 'posting', 'how', 'to', 'stupid', 'licks', 'quit', 'flea', 'my', 'him', 'park', 'dog', 'not', 'I', 'help', 'so', 'garbage', 'take']
    


```python
"""检查my_vocab_list中每个词在第一篇（索引为0）文章中是否出现"""
my_vec = words_set2vec(my_vocab_list, posting_list[0]) 
print(my_vec)
```

    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    


```python
"""检查my_vocab_list中每个词在第4篇（索引为3）文章中是否出现"""
my_vec = words_set2vec(my_vocab_list, posting_list[3]) 
print(my_vec)
# len(my_vec)
# import numpy as np
# a = np.zeros(len(my_vec))
# print(a + my_vec)
```

    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    

可以看到my_vocab_list中的第四个词‘problems’是在第一篇文章中出现，所以被标记为“1”。
##### 将所有文档都转化为词向量：


```python
train_mat = []
for doc in posting_list:
    train_mat.append(words_set2vec(my_vocab_list, doc))
print(train_mat)
```

    [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
    

### 3.分析数据
   首先我们要知道训练数据中每篇文档（留言）是属于哪种类型？侮辱性或非侮辱性。然后我们计算词向量中每个词在已知文档分类的条件下，该词出现的概率和侮辱性文档占总文档的概率。最后根据贝叶斯准则，见下式，计算出我们想要的结果。
   设w为一个向量，它由多个数值组成，数值个数与词向量中的词个数相同。
$$ p(c_i|w) = \cfrac{p(w|c_i)p(c_i)}{p(w)}$$
$c_i$表示第$i$个分类。

在计算$p(w|c_i)$时，要用到朴素贝叶斯中的“朴素”二字，就是假设$w$中的每一个值都是相互独立的，那么$p(w|c_i)=p(w_0,w_1,...,w_N|c_i)$就等于$$\prod_{j=0}^{N}p(w_j|c_i) = p(w_0|c_i)p(w_1|c_i)p(w_2|c_i)...p(w_N|c_i), i=0,1$$ $w_j$表示$w$中第$j+1$个值。

### 4.训练算法：从词向量训练算法
##### 算法伪码：
    计算每个类别中文档的个数
    对每篇训练文档：
        对每个类别：
            如果词条出现在文档中->增加该词条的计数值
            增加所有词条的计数值
    对每个类别：
         对每个词条：
             将该词条的数目除以总词条数目得到概率
    返回每个类别的条件概率


```python
import numpy as np
# 获取留言的数目，共6条
num_train_docs = len(train_mat)
num_train_docs
```




    6




```python
# 获取词向量的长度，32
num_words = len(train_mat[0])
num_words
```




    32




```python
# 侮辱性留言占总留言数的比例
pAbusive = sum(class_vec)/float(num_train_docs) 

# 存放非侮辱性留言（即类型为0）中每个词出现的个数
p0num = np.zeros(num_words) 
# 存放侮辱性留言（即类型为1）中每个词出现的个数
p1num = np.zeros(num_words)

# 存放类型为0的所有留言中所有词的出现的总次数
p0Denom = 0.0
# 存放类型为1的所有留言中所有词的出现的总次数
p1Denom = 0.0
```


```python
pAbusive
```




    0.5




```python
p0num
```




    array([ 0.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  3.,  2.,  0.,  1.,
            0.,  1.,  1.,  1.,  0.,  0.])




```python
p1num
```




    array([ 1.,  0.,  0.,  2.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,
            0.,  0.,  1.,  0.,  1.,  3.,  0.,  1.,  0.,  0.,  1.,  1.,  2.,
            1.,  0.,  0.,  0.,  1.,  1.])




```python
for i in range(num_train_docs): # 对留言矩阵进行循环
        if class_vec[i] == 1:  # 如果该留言类别为1，即侮辱性留言
            p1num += train_mat[i]  
            p1Denom += sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0Denom += sum(train_mat[i])
```


```python
p1num, p1Denom
```




    (array([ 1.,  0.,  0.,  2.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,
             0.,  0.,  1.,  0.,  1.,  3.,  0.,  1.,  0.,  0.,  1.,  1.,  2.,
             1.,  0.,  0.,  0.,  1.,  1.]), 19.0)




```python
p0num, p0Denom
```




    (array([ 0.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,
             1.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  3.,  2.,  0.,  1.,
             0.,  1.,  1.,  1.,  0.,  0.]), 24.0)




```python
# 计算在类型为1时，词向量中每个词出现的概率
p1vec = p1num / p1Denom
# 计算在类型为0时，词向量中每个词出现的概率
p0vec = p0num / p0Denom
```


```python
p1vec, p0vec
```




    (array([ 0.05263158,  0.        ,  0.        ,  0.10526316,  0.        ,
             0.05263158,  0.        ,  0.05263158,  0.        ,  0.        ,
             0.05263158,  0.        ,  0.        ,  0.        ,  0.        ,
             0.05263158,  0.        ,  0.05263158,  0.15789474,  0.        ,
             0.05263158,  0.        ,  0.        ,  0.05263158,  0.05263158,
             0.10526316,  0.05263158,  0.        ,  0.        ,  0.        ,
             0.05263158,  0.05263158]),
     array([ 0.        ,  0.04166667,  0.04166667,  0.        ,  0.04166667,
             0.        ,  0.04166667,  0.        ,  0.04166667,  0.04166667,
             0.04166667,  0.04166667,  0.04166667,  0.04166667,  0.04166667,
             0.        ,  0.04166667,  0.04166667,  0.        ,  0.04166667,
             0.        ,  0.04166667,  0.125     ,  0.08333333,  0.        ,
             0.04166667,  0.        ,  0.04166667,  0.04166667,  0.04166667,
             0.        ,  0.        ]))




```python
"""将上述过程合成一个函数"""
import numpy as np
def trainNB0(train_mat, train_category): 
    num_train_docs = len(train_mat) # 文档数目，即网站上的留言数目
    num_words = len(train_mat[0])
    pAbusive = sum(train_category)/float(num_train_docs) # 侮辱性留言占总留言数的比例
    p0num = np.zeros(num_words) 
    p1num = np.zeros(num_words)
    p0Denom = 0.0
    p1Denom = 0.0
    
    for i in range(num_train_docs): # 对留言矩阵进行循环
        if train_category[i] == 1:  # 如果该留言类别为1，即侮辱性留言
            p1num += train_mat[i]  
            p1Denom += sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0Denom += sum(train_mat[i])
    p1vec = p1num / p1Denom
    p0vec = p0num / p0Denom
        
    return p1vec, p0vec, pAbusive
            
    
```


```python
# 调用该函数
p1vec, p0vec, pAbusive = trainNB0(train_mat, class_vec)
```


```python
pAbusive
```




    0.5




```python
p1vec
```




    array([ 0.05263158,  0.        ,  0.        ,  0.10526316,  0.        ,
            0.05263158,  0.        ,  0.05263158,  0.        ,  0.        ,
            0.05263158,  0.        ,  0.        ,  0.        ,  0.        ,
            0.05263158,  0.        ,  0.05263158,  0.15789474,  0.        ,
            0.05263158,  0.        ,  0.        ,  0.05263158,  0.05263158,
            0.10526316,  0.05263158,  0.        ,  0.        ,  0.        ,
            0.05263158,  0.05263158])




```python
p0vec
```




    array([ 0.        ,  0.04166667,  0.04166667,  0.        ,  0.04166667,
            0.        ,  0.04166667,  0.        ,  0.04166667,  0.04166667,
            0.04166667,  0.04166667,  0.04166667,  0.04166667,  0.04166667,
            0.        ,  0.04166667,  0.04166667,  0.        ,  0.04166667,
            0.        ,  0.04166667,  0.125     ,  0.08333333,  0.        ,
            0.04166667,  0.        ,  0.04166667,  0.04166667,  0.04166667,
            0.        ,  0.        ])



### 5.测试算法
    测试前要对算法进行改进，因为在计算概率相乘，如果其中某个概率为0，那么最后的结果也就是0，就无法进行相应的分类了，因此将所有的词出现的次数初始值设为1，并将分母初始化为2。
   
     还有一个问题就是下溢出，由于大部分概率比较小，计算机在进行四舍五入时，最后的结果可能得到0，于是我们采用log的形式，根据高等数学可知，有如下式子。
$$ln(a*b) = ln(a) + lin(b)$$
    
    所以将那些概率先进性取对数，再相加，就可以避免下溢出，于是就有了朴素贝叶斯分类函数。


```python
"""修改函数"""
import numpy as np
def trainNB0(train_mat, train_category): 
    num_train_docs = len(train_mat) # 文档数目，即网站上的留言数目
    num_words = len(train_mat[0])
    pAbusive = sum(train_category)/float(num_train_docs) # 侮辱性留言占总留言数的比例
    p0num = np.ones(num_words) 
    p1num = np.ones(num_words)
    p0Denom = 2.0
    p1Denom = 2.0
    
    for i in range(num_train_docs): # 对留言矩阵进行循环
        if train_category[i] == 1:  # 如果该留言类别为1，即侮辱性留言
            p1num += train_mat[i]  
            p1Denom += sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0Denom += sum(train_mat[i])
    p1vec = np.log(p1num / p1Denom)
    p0vec = np.log(p0num / p0Denom)
        
    return p1vec, p0vec, pAbusive
            
```

##### 朴素贝叶斯函数


```python
def classifyNB(vec2classify, p0vec, p1vec, pclass1):
    p1 = sum(vec2classify * p1vec) + np.log(pclass1)
    p0 = sum(vec2classify * p0vec) + np.log(pclass1)
    if p1 > p0:
        return 1
    else:
        return 0
```

##### 测试函数


```python
def testingNB():
    posting_list, class_list = load_data_set()
    my_vocab_list = create_vocab_list(posting_list)
    train_mat = []
    for doc in posting_list:
        train_mat.append(words_set2vec(my_vocab_list, doc))
    p1v, p0v, pAb = trainNB0(train_mat, class_list)
    test1 = ['love', 'my', 'dalmation']
    test1_vec = words_set2vec(my_vocab_list, test1)
    print(test1,"classified as:",classifyNB(test1_vec, p0v, p1v, pAb))
    
    test2 = ['stupid', 'garbage']
    test2_vec = words_set2vec(my_vocab_list, test2)
    print(test2,"classified as:",classifyNB(test2_vec, p0v, p1v, pAb))
        
```


```python
testingNB()
```

    ['love', 'my', 'dalmation'] classified as: 0
    ['stupid', 'garbage'] classified as: 1
    

从上述结果可以看出，分类结果和我们预想的一样。

### 6.其他模型
之前做的是将每个词是否出现作为一特征，这种方式被称作是词集模型。如果一个词在文档中出现不止一次，这可能意味着包含该词是否出现在文档中所不能表达的某种信息，这种方法被称为词袋模型。下面为词袋模型的词向量构造函数。


```python
def words_bag2vec(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec
```

*参考文献：《机器学习实战》Peter Harrington*
