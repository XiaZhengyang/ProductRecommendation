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

This is a m*k matrix, where m is the number of (valid) training examples and k is the number of features concerned. In our example, k is set to be 7 while m need not to be initialized upon the creation of this matrix. Instead, the matrix is rescaled to form a new row everytime a valid entry of data is found, and the personal particulars of the client on the newly found valid entry is put into the new row of the matrix. In the given example, this process will eventually yield a matrix of size 20 by 7. The seven attributes are age, gender, maritial status, education, number of dependent(s) and value of vehicle.

建立一个m乘k的矩阵，来储存用户的个人信息。其中，m是有效训练样本的数目，k是特征的数目。创建矩阵时，它的大小是1*k，随后，扫描输入的JSON文件，每发现一条有内容的用户信息，就把现有的矩阵增加一行，并把这个用户的信息存进新的行里面。在这个例子中，这一过程将会得到20乘7的矩阵，代表有20条有效用户信息，对每个用户都记录了7个属性（包括年龄、性别、婚姻状态、学历、月收入、受养人数目、车辆价值）。



***
#### 2. Find the theta which yields the minimun distortion
2.找出能使得K-means产生最小误差的theta向量。

Theta is effectively a 1*7 matrix that assigns a different weighting to each feature. Different theta corresponds to different output (centroids and distortion value) of the k-means algorithm after combining with the information matrix. After every run of the k-means function, all theta values are simultaneously updated with the gradient descent algorithm. Theta is initialised to rescale all features to approximately 1 in order to make the function converge more efficiently. The smallest distortion value and its corresponding theta from 500 iterations are saved as the temporary optimum.
Theta是一个1*7的矩阵，包括了每一项特征对应的比重。不同的theta与训练样本结合后经过k-means算法会产生不同的簇类和误差。每运行一次k-means算法后，theta中的所有值都按照梯度下降法同时更新为新的值。为了提高梯度学习的效率，初始的theta值将所有特征值都大致放缩到1附近。在500次迭代中最小的distortion值和其对应的theta被保存为暂时的最佳结果。


***
#### 3. Implementing Support Vector Machines algorithm
3.对客户信息矩阵中的信息进行支持向量机分析

First, a new support vector machine object is created. The “class_weight” parameter is set to be equal to a pre-defined dictionary dict in order to balance weights among different classes later. The “probability” parameter is set to be True to implement cross validation to create the probability model and enable probability estimates later. All other parameters are left to be their default value. Part of the client information (the first 1000 entries in this case) and their corresponding labels are parsed into the SVM object to fit the SVM model according to the given data.
首先，我们创建一个新的支持向量机对象，并将参数“class_weight”设置为一个之前自定义的字典以便调整各标签组之间的比重和将参数“probability”设置为真以使用交叉验证创建概率模型。所有其他参数都保留为默认值。客户信息中的前1000条客户信息和他们对应的标签被作为训练样本进行了支持向量机分析并创建契合的概率模型。


***
#### 4.Predict labels for test dataset and compute prediction accuracy
4.预测测试样本的标签并计算准确率

The remaining part of client information (666 entries) is used to test the accuracy of SVM model prediction. The corresponding entries in the information matrix are parsed into the predict_proba function of the SVM object. For each sample data entry, a vector that indicates its probability of belonging to each of the four labels is generated according to the probability model trained in step 3. The resulting output is a 658 by 4 matrix and stored into a new matrix proba. Then, two for loops are used to iterate through each entry and through each probability measure and assign each test sample to the label it has the highest probability of belonging to. The predictions are compared with the user label data and compute an accuracy of prediction. By optimising the weights for different label groups, we can achieve a prediction accuracy of approximately 81.38%.
剩余的666条客户信息（不包括标签）被用作测试支持向量机模型预测准确率的测试样本。对每一条客户信息运用支持向量机对象中的predict_proba函数都可以得到一个表示该客户属于每一种标签的对应概率。我们将对应概率最大的标签作为对该用户的标签预测，并与实际已知的用户标签进行对比。通过调整前一步中所述字典内的各标签对应比重，预测准确率最高可达81.38％。
Which kind of product did each customer actually buy is unknown.
未知被分析的顾客实际上买了哪种产品。(因此无法评估是次训练的准确度)

How to quantify non-quantitative data (e.g. Occupation, Residential address, etc)
如何量化非定量的信息？（比如职业，家庭住址等等）
