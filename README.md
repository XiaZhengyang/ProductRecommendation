# Product Recommendation
**This is a recommender system based on collaborative filtering**

---

The original data is in .csv format. To decode it, use the “import csv” command in python and store every unique line in the .csv file as a list, so that all data will be in a list of lists.
原始数据的格式是csv。使用python中的csv包来把文件中的每一行不重复的行存到一个python中包含了多个列表的列表里。




#### 1. Construct the infoMatrix and the column vector called “label”
创建储存样本信息的矩阵和储存分类标签的向量

Import numpy to create a matrix. Scan through ‘data’, an store the useful information into the newly created matrix. Currently, the features used include the following:
使用numpy创建一个矩阵，扫描上一步创建的列表，把有用的信息存入这个矩阵内。现时用到的特征信息包括：

0. 所申请的种类: 信优贷/信薪贷/信薪佳人贷/薪期贷(网薪期)
1. 申请的期限: 24/36/48
2. 申请的金额
3. 贷款目的 消费/个人资金周转/经营周转/其他
4. 最高可接受的每月还款额
5. 家人是否知情 是/否
6. 住房类型 无按揭购房/商业按揭房/公积金按揭购房/自建房/单位住房/亲属住房/租用
7. 本市生活时长
8. 学历 大学本科/高中及中专/大专/硕士/初中及以下
9. 婚姻状态 已婚/未婚/离异 （建议增加“丧偶”和/或”分居“的选项，以增加准确性）
10.性别 男/女
11.车辆价值
12.工作岗位 一般正式员工/中级管理人员/一般管理人员/派遣员工/高级管理人员/负责人
13.工作单位类别 '机关事业单位/外资企业/私营企业/国有股份/合资企业/民营企业/个体
In the abovementioned 14 features, 2,4,7,11 are numerical features and the remaining are catagorical features.
(Age is to be added to the list of features)
在上述特征中，2、4、7、11是数值型特征，其他都是分类型特征。


Column vector (named 'label') has four possible labels, namely:
向量'label'有四个可能的取值，这四个标签是：

0.薪期贷07/薪期贷10/薪期贷13/薪期贷17
1.信优贷17_A11/信优贷19/信优贷21
2.信薪贷23/信薪贷25/信薪贷27
3.信薪佳人贷21

During the creation of infoMatrix and labels[], if any missing value is encountered, np.nan will be stored. They will later be imputed by the method below.
创建这个矩阵和向量的过程中，如果发现有任何数据缺失或空白，将往矩阵中写入np.nan。这些空白数据将用下文介绍的方法进行推测、填补。



***
#### 2. Data preprocessing
数据预处理

###### Step1. Data Imputation
第一步 缺失数据填补

Use the Imputer class from scikit-learn library to perform imputation on all missing data values. For catagorical features, use the most frequent value in the column to fill the blank; for numerical features, use the mean of all other values to fill the blank.

填补缺失的数据将用到scikit-learn这个库里面的Imputer这个类。对于分类型特征，空白的数据点将用这个特征里的众数来填补；对于数值型特征，空白的点将用其他数据点的平均数来填补。

###### Step2. One-hot encoding
第二步 独热编码

Ten features out of fourteen are catagorical values, and such columns(except gender which only has 2 options) need to be expanded into several columns which only contain 0 and 1. For example, a column [0; 2; 1; 3] needs to be expanded to [1 0 0 0;0 0 1 0;0 1 0 0;0 0 0 1], and these four new columns represent whether this example is 0/1/2/3 respectively.

Columns with numerical values will not go through one-hot encoding. They will be pushed to the right of the matrix, and will be remain the same in the matrix after encoding.

在上述14个特征中，10个是分类型特征。在这10个分类型特征中，除了性别（只有两个选项）之外，其他所有特征均应进行“独热编码”，也就是把一个含有几种代号的列展开成几个只含有0和1的列。例如，[0; 2; 1; 3]这一列应展开成[1 0 0 0;0 0 1 0;0 1 0 0;0 0 0 1]，这四个新的列分别代表该样本是否为0、是否为1、是否为2、是否为3。
数值型特征不需要进行独热编码。储存了数值型特征的列将会被推到整个矩阵的最右边，且他们的数值在独热编码前后不会有变化。

###### Step3. Scaling
第三步 数据缩放

The infoMatrix needs to be scaled so that the training result will not be dominant by the columns with naturally large values (e.g. Value of vehicle compared to those 0/1 columns). The 'whiten' function from Scipy is used. In the resulting 'scaledMatrix', all columns will have unit variance.

The matrix named 'scaledMatrix' is the end product of data collection and data preprocessing. Its size is 1303*45, where 1303 means there are 1303 data samples (observations), and there are in total 45 features (including dummy ones). This matrix, along with the column vector 'label', are ready for classification in any of the major supervised learning algorithms.

存有数据的矩阵应进行一次缩放，从而使训练结果不会只由数值较大的特征决定，而数值较小的特征效果不大（例如，车辆价值和只含0和1的列）。此处将会用到scipy库里面的whiten函数，经此函数处理后得到的'scaledMatrix'中，所有的列的方差都是1。
这个叫做'scaledMatrix'的新矩阵是数据收集和数据预处理的最终产品。它的尺寸是1303*45，代表它有1303个样本和45个特征（包括经过独热编码而形成的虚拟特征）。这个矩阵再加上名叫'label'的向量，就可以被用于监督学习的各种分类算法了。

***



***
#### 4. Implementing Support Vector Machines algorithm
4.对客户信息矩阵中的信息进行支持向量机分析

First, a new support vector machine object is created. The “class_weight” parameter is set to be equal to a pre-defined dictionary dict in order to balance weights among different classes later. The “probability” parameter is set to be True to implement cross validation to create the probability model and enable probability estimates later. All other parameters are left to be their default value. Part of the client information (the first 1000 entries in this case) and their corresponding labels are parsed into the SVM object to fit the SVM model according to the given data.
首先，我们创建一个新的支持向量机对象，并将参数“class_weight”设置为一个之前自定义的字典以便调整各标签组之间的比重和将参数“probability”设置为真以使用交叉验证创建概率模型。所有其他参数都保留为默认值。客户信息中的前1000条客户信息和他们对应的标签被作为训练样本进行了支持向量机分析并创建契合的概率模型。


***
#### 4.Predict labels for test dataset and compute prediction accuracy
5.预测测试样本的标签并计算准确率

The remaining part of client information (666 entries) is used to test the accuracy of SVM model prediction. The corresponding entries in the information matrix are parsed into the predict_proba function of the SVM object. For each sample data entry, a vector that indicates its probability of belonging to each of the four labels is generated according to the probability model trained in step 3. The resulting output is a 658 by 4 matrix and stored into a new matrix proba. Then, two for loops are used to iterate through each entry and through each probability measure and assign each test sample to the label it has the highest probability of belonging to. The predictions are compared with the user label data and compute an accuracy of prediction. By optimising the weights for different label groups, we can achieve a prediction accuracy of approximately 81.38%.
剩余的666条客户信息（不包括标签）被用作测试支持向量机模型预测准确率的测试样本。对每一条客户信息运用支持向量机对象中的predict_proba函数都可以得到一个表示该客户属于每一种标签的对应概率。我们将对应概率最大的标签作为对该用户的标签预测，并与实际已知的用户标签进行对比。通过调整前一步中所述字典内的各标签对应比重，预测准确率最高可达81.38％。
