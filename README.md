### 最终排名-26/2
##### 方案
```
步骤:
1. 词向量 glove + idf
2. 降维 : triple loss（源自15年facenet） + stack autoencoder
采样方式，正采样1：4，采样两次，训练样本164w
3. 采样优化, 半监督，样本选取：正采样距离 < 负采样距离
4. 聚类kmeans++

```

### 随记
```
# 官方思路
任务描述：给定一堆拥有同名作者的论文，要求返回一组论文聚类，使得一个聚类内部的论文都是一个人的，不同聚类间的论文不属于一个人。最终目的是识别出哪些同名作者的论文属于同一个人。

参考方法：解决这一问题的常用思路就是通过聚类算法，提取论文特征，定义聚类相似度度量，从而将一堆论文聚成的几类论文，使得聚类内部论文尽可能相似，而类间论文有较大不同，最终可以将每一类论文看成属于同一个人的论文。[7] 是一篇经典的使用聚类方法的论文，它使用了原子聚类的思想，大致思路是首先用较强的规则进行聚类，例如：俩篇论文如果有俩个以上的共同作者，那么这俩篇论文属于同一类，这样可以保证聚类内部的准确率，随后用弱规则将先前的聚类合并，从而提高召回率。有些工作考虑了传统特征的局限性，所以利用了低维语义空间的向量表示方法，通过将论文映射成低维空间的向量表示，从而基于向量使用聚类方法 [2]。

[2].  Yutao Zhang, Fanjin Zhang, Peiran Yao, and Jie Tang. Name Disambiguation in AMiner: Clustering, Maintenance, and Human in the Loop. In Proceedings of the Twenty-Forth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'18).

[7].    Wang, F. , Li, J. , Tang, J. , Zhang, J. , & Wang, K. . (2008). Name Disambiguation Using Atomic Clusters. Web-Age Information Management, 2008. WAIM '08. The Ninth International Conference on.

```

```
# 疑问
0.统计重名文章的比例； 机构-名称，pair <存在单名称多机构，但是不能确定是否是单人多机构>
1.标题和内容，训练词向量，得到关键词的关系；
2.不同作者的关键词；
```
```
# 字段
关键词表
venue表
姓名机构表
标题
摘要
```

```
# 实验步骤
先根据内容和关键词，对文章进行聚类；
0.同名的作者下，文章共用多个作者的情况；
1.文章向量化；
2.计算各个维度的特征向量，分析差异性；
3.选择和组合特征；

0.关键词聚类

```

```
# 先验假设
1. 同名字的作者下，同一个人的研究领域是类似的；
2. 同一人文章内容具有相似性；<1.paper embedding;><2.tf-idf;>
```
