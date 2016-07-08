# Product Recommendation
**This is a recommender system based on collaborative filtering**

---

Approximately 100 entries of customer information were provided as raw data, and they were in JSON format.
原始数据中大约有100条客户信息。它们的格式是JSON。

The libraries needed include simplejson, numpy and scipy.
在python中，我们需要用到simplejson,numpy和scipy这三个库去解析这些数据。

At the preparation stage, load the json file and decode it.
在准备阶段，我们需要加载出这个JSON文件并对其进行解码。

#### 1. Construct the initial infoMatrix which does not include empty entries
1.建立最初的储存客户信息的矩阵,这个矩阵不包括空白的客户信息。

This is a m*k matrix, where m is the number of (valid) training examples and k is the number of features concerned. In our example, k is set to be 7 while m need not to be initialized upon the creation of this matrix. Instead, the matrix is rescaled to form a new row everytime a valid entry of data is found, and the personal particulars of the client on the newly found valid entry is put into the new row of the matrix. In the given example, this process will eventually yield a matrix of size 20 by 7.

***
#### 2. Find the theta which yields the minimun distortion
2.找出能使得K-means产生最小误差的theta向量。


***
#### 3. Perform K-means clustering to give a groupping
3.对客户信息矩阵中的信息进行K-means聚类分析，并得到一个分组方案以及每个组的重心的坐标。


***
#### 4. Input a new entry of customer information, and find the cluster that it belongs to
4.输入一条新的客户信息，然后找到它所属的簇。（每个簇代表一种产品）

When the user inputs the seven attributes of a customer manually, the program shall calculate its Eucledian distance to all the cluster centroids, and give the index of the centroid that this new data point is closest to. The cluster that the new point belongs to denotes the type of product that she/he is most likely to buy.



***
