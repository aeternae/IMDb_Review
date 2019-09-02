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

### 2.数据预处理
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
在项目中我们对文本特征提取的过程就是将文本数据转化成特征向量的过程，使用了比较常用的文本特征表示法——词袋法。<br>
词袋法的原则如下：
        
    1)不考虑词语出现的顺序，每个出现过的词汇单独作为一列特征。
    2)这些不重复的特征词汇集合为词表。
    3)每一个文本都可以在很长的词表上统计出一个很多列的特征向量。
    4)如果每个文本都出现的词汇，一般被标记为停用词不计入特征向量。

* Countvectorizer<br>

Countvectorizer旨在通过计数来将一个文档转换为向量，只考虑词汇在文本中出现的频率。

* TF-IDF词频矩阵<br>

词频(Term Frequency，TF)指某一个给定的词语在该文件中出现的次数。这个数字通常会被归一化(一般是词频除以文章总次数)，以防止它偏向长的文件。
                                        
    TF = 在某一类中词条出现的次数/该类中所有的词条数目<br>

逆向文件频率(Inverse Document Frequency，IDF)：主要思想是如果包含词条t的文档越少，IDF越大，则说明词条具有很好的类别区分能力。
    
    IDF = log⁡(语料库中的文档总数/包含该词条的文档数+1)	  

综上：某一特定文件的高词语频率，以及该词语在整个文件集合的低文件频率，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语。即：TF-IDF = TF*IDF

### 4.模型构建
#### 多项式朴素贝叶斯(Naïve Bayes, NB)
特征向量表示由多项式分布生成的特定事件的频率。这是用于文本分类的典型的事件模型。
```
def MNB_tfidf_Classifier():              #采用管道将步骤进行封装
    return Pipeline([
        ('tfidf_vec', TfidfVectorizer()),  
        ('mnb', MultinomialNB())       #训练贝叶斯模型
    ])

mnbt_clf = MNB_tfidf_Classifier()
mnbt_clf.fit(X1_train, y1_train)
mnbt_clf.score(X1_test, y1_test)
0.8598
```
#### 逻辑回归(Logistic Regression，LR)
logistic回归通过Sigmoid函数将ax+b对应到一个隐状态p = S(ax+b)，然后根据p与1-p的大小决定因变量的值。
```
def LogisticRegression_c():              #采用管道将步骤进行封装
    return Pipeline([
        ('count_vec', CountVectorizer()),  
#         ('poly', PolynomialFeatures(degree=degree)),            #添加多项式项
        ('logistic', LogisticRegression(C=0.1, penalty='l2'))     #训练逻辑回归模型
    ])

polyc_log_reg = LogisticRegression_c()
polyc_log_reg.fit(X1_train, y1_train)
polyc_log_reg.score(X1_test, y1_test)
0.8806
```
#### 随机森林(Random Forest)
随机森林是一种集成算法(Ensemble Learning)，它属于Bagging类型，通过组合多个弱分类器，最终结果通过投票或取均值，使得整体模型的结果具有较高的精确度和泛化性能。其可以取得不错成绩，主要归功于“随机”和“森林”，一个使它具有抗过拟合能力，一个使它更加精准。
```
def RFC_t():
    return Pipeline([
        ('tfidf_vec', TfidfVectorizer()),  
        ('rfc', RandomForestClassifier(n_estimators=500, max_depth=3, random_state=666, n_jobs=-1)) 
    ])
rfct_clf = RFC_t()
rfct_clf.fit(X1_train, y1_train)
rfct_clf.score(X1_test, y1_test)
0.8308
```
#### XGBoost
XGBoost在Gradient Boosting框架下实现机器学习算法。 XGBoost提供了并行树提升(也称为GBDT，GBM)，可以快速准确地解决许多数据科学问题。
```
def XGB_t():
    return Pipeline([
        ('tfidf_vec', TfidfVectorizer()),  
        ('XGB', xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                                  subsample=.7, colsample_bytree=0.6, gamma=0.05))
    ])
xgbt_clf = XGB_t()
xgbt_clf.fit(X1_train, y1_train)
xgbt_clf.score(X1_test, y1_test)
0.8598
```
#### Voting Classifier
考虑到AUC(Area Under Curve)同时考量了精准率和召回率的优秀特性，因此我们最后采取了预测模型少数服从多数的投票方式进行预测，并最终用AUC作为我们评论情感分析是否准确的评判标准。
```
voting_clf = VotingClassifier(estimators=[
    ("mnb_clf", MNB_tfidf_Classifier()),
    ("log_clf", LogisticRegression_c()),
    ("rfc_clf", RFC_c()),
    ("etc_clf", ETC_c()),
    ("xgb_clf", XGB_t()),
    ("lgb_clf", LGB_t())
], voting='soft')    
voting_clf.fit(X1_train, y1_train)
roc_auc_score(y1_test, y_predict)
0.8843170087255745
```

### 5.预测结果
单模型中逻辑回归模型和极限树模型有更高的准确度，同时多模型组合后的预测结果明显优于单个模型的预测结果。<br>
最终组合模型的AUC(Area Under Curve)为0.8843。

|模型|预测结果|
|:------:|:------:|
|朴素贝叶斯|0.8534|
|逻辑回归|0.8806|
|随机森林|0.8378|
|极限树|0.877|
|ADABOOST|0.8426|
|XGBOOST|0.8598|
|LIGHTGBM|0.8598|
|组合模型|0.884|

