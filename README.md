# Product Recommendation
**This is a recommender system based on collaborative filtering**

---

Approximately 100 entries of customer information were provided as raw data, and they were in JSON format.
原始数据中大约有100条客户信息。它们的格式是JSON。
The libraries needed include simplejson, numpy and scipy.
在python中，我们需要用到simplejson,numpy和scipy这三个库去解析这些数据。
At the preparation stage, load the json file and decode it.
在准备阶段，我们需要加载出这个JSON文件并对其进行解码。
#### 1. Delete all empty entries
1.删除所有空白的客户信息条目。

#### 2. Construct the initial infoMatrix
2.建立最初的储存客户信息的矩阵。

#### 3. Find the theta which yields the minimun distortion
3.找出能使得K-means产生最小误差的theta向量。

#### 4. Perform K-means clustering to give a groupping
对客户信息矩阵中的信息进行K-means聚类分析，并得到一个分组方案以及每个组的重心的坐标。

#### 5. Input a new entry of customer information, and find the cluster that it belongs to
输入一条新的客户信息，然后找到它所属的簇。（每个簇代表一种产品）

***
