# IMDb_Review  IMDb影评情感分析
## Zhixiang Wang

### 1.项目简介
互联网电影资料库(Internet Movie Database，简称IMDb)是一个关于电影演员、电影、电视节目、电视明星和电影制作的在线数据库。<br>
因此如果可以对IMDb的影评进行分析，判断每条影评留言的情感倾向，我们或许可以在评分以外更精准地判断一部影视作品的好坏优劣，甚至可以判别出一部影视作品的评分过高或者过低是否有人为因素的影响。
#### 数据集：
    (1)已经有情感倾向的训练文件laberedTrainData.csv，里面有25000条影评以及对应的情感倾向标识；
    (2)待测试文件testData.tsv，同样也有25000条电影评论；
    (3)还有一份无标注但是数据量更大的影评文件unlabeldTrainData.tsv；
    (4)最后是一份样例文件sampleSubmission.csv用来告知参赛者最终结果的提交格式。
由于情绪是非常难以精准描述的，因此在对于电影评论文本进行情感分类时，仅分为正面评论和负面评论，即为一个二分类问题。因此我们在首先尝试基于TF-IDF的向量化方法之后，并通过一些二分类模型来进行预测及评判，常见的模型有朴素贝叶斯、逻辑回归，集成学习等等。

### 2.数据处理与特征提取
* 样本简介<br>
每条样本包括ID，评论文本以及情感倾向（1：正面评论；2：负面评论）
```
       id	 sentiment	  review
0	 5814_8	         1	  With all this stuff going down at the moment w...
1	 2381_9	         1	  \The Classic War of the Worlds\" by Timothy Hi...
2	 7759_3	         0	  The film starts with a manager (Nicholas Bell)...
3	 3630_4	         0	  It must be assumed that those who praised this...
4	 9495_8	         1	  Superbly trashy and wondrously unpretentious 8...
```
* 文本数据预处理<br>
利用beautifulsoap，re，stopwords去掉电影评论文本中的html标记，非字母字符以及停用词，使文本变成我们需要的无不相关字符的且仅有主干词汇的文本，便于之后的特征抽取。
```
def review_to_text(review, remove_stopwords):
    #去掉html标记
    raw_text = BeautifulSoup(review, 'html.parser').get_text()
    #去掉非字母字符
    letters = re.sub('[^a-zA-Z]', ' ', raw_text)
    words = letters.lower().split()
    #去掉停用词
    if remove_stopwords:
        all_stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in all_stop_words]
 
    return words
```

### 3.文本特征抽取
* TF-IDF词频矩阵<br>
词频(Term Frequency，TF)指某一个给定的词语在该文件中出现的次数。这个数字通常会被归一化(一般是词频除以文章总次数)，以防止它偏向长的文件。
                                        
    TF = 在某一类中词条出现的次数/该类中所有的词条数目
逆向文件频率(Inverse Document Frequency，IDF)：主要思想是如果包含词条t的文档越少，IDF越大，则说明词条具有很好的类别区分能力。
    
    IDF = log⁡(语料库中的文档总数/包含该词条的文档数+1)	  
