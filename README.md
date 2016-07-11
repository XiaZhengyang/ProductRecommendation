# Product Recommendation
**This is a recommender system based on collaborative filtering**

---

Approximately 100 entries of customer information were provided as raw data, and they were in JSON format.
原始数据中大约有100条客户信息。它们的格式是JSON。

The libraries needed include simplejson, numpy and scipy.
在python中，我们需要用到simplejson,numpy和scipy这三个库去~~解析这些数据~~。

At the preparation stage, load the json file and decode it.
在准备阶段，我们需要加载出这个JSON文件并对其进行解码。

#### 1. Construct the initial infoMatrix which does not include empty entries
1.建立最初的储存客户信息的矩阵,这个矩阵不包括空白的客户信息。

This is a m*k matrix, where m is the number of (valid) training examples and k is the number of features concerned. In our example, k is set to be 7 while m need not to be initialized upon the creation of this matrix. Instead, the matrix is rescaled to form a new row everytime a valid entry of data is found, and the personal particulars of the client on the newly found valid entry is put into the new row of the matrix. In the given example, this process will eventually yield a matrix of size 20 by 7. The seven attributes are age, gender, maritial status, education, number of dependent(s) and value of vehicle.

建立一个m乘k的矩阵，来储存用户的个人信息。其中，m是有效训练样本的数目，k是特征的数目。创建矩阵时，它的大小是1*k，随后，扫描输入的JSON文件，每发现一条有内容的用户信息，就把现有的矩阵增加一行，并把这个用户的信息存进新的行里面。在这个例子中，这一过程将会得到20乘7的矩阵，代表有20条有效用户信息，对每个用户都记录了7个属性（包括年龄、性别、婚姻状态、学历、月收入、受养人数目、车辆价值）。



***
#### 2. Find the theta which yields the minimun distortion
2.找出能使得K-means产生最小误差的theta向量。


***
#### 3. Perform K-means clustering to give a groupping
3.对客户信息矩阵中的信息进行K-means聚类分析，并得到一个分组方案以及每个组的重心的坐标。

This step shall be completed by the kmeans function from scipy library.


***
#### 4. Input a new entry of customer information, and find the cluster that it belongs to
4.输入一条新的客户信息，然后找到它所属的簇。（每个簇代表一种产品）

When the user inputs the seven attributes of a customer manually, the program shall calculate its Eucledian distance to all the cluster centroids, and give the index of the centroid that this new data point is closest to. The cluster that the new point belongs to denotes the type of product that she/he is most likely to buy.
每次输入一个新的客户的各项特征，程序将计算它离每个聚类中心的欧氏距离，并给出这个新数据应属的簇。这个新客户数据所属的簇代表她/他最可能购买的产品。



***

#### Issues to be solved: 待解决的问题：
The number of products (namely the number of clusters) is unknown.
暂时未知总共有多少种产品（总共有多少类）。


Which kind of product did each customer actually buy is unknown.
未知被分析的顾客实际上买了哪种产品。(因此无法评估是次训练的准确度)

How to quantify non-quantitative data (e.g. Occupation, Residential address, etc)
如何量化非定量的信息？（比如职业，家庭住址等等）




