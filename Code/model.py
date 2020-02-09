import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import csv



costs=[]
# add layer
def add_layer(inputs, in_size, out_size, activation_function=None):
   # add one more layer and return the output of this layer
   Weights = tf.Variable(tf.random_normal([in_size, out_size]))
   biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
   Wx_plus_b = tf.matmul(inputs, Weights) + biases
   if activation_function is None:
       outputs = Wx_plus_b
   else:
       outputs = activation_function(Wx_plus_b)
   return outputs


# Load all the data 
x_data = np.load('../Data/input data.npy')
x_data=x_data.astype(np.float)
#print('x_data range is:',np.unique(x_data[:,1]))
#x_data[:,1] = x_data[:,1]/106.0
#print('x_data range is:',np.shape(x_data))
#print('x_data range is:',np.unique(x_data[:,1]))
y_data = np.load('../Data/DRE data.npy')

# Seperate the training and test data
n_all_data=x_data.shape[0]
print('n_all_data is:',n_all_data)
n_train_data=int((1-0.2)*n_all_data)
print('n_train_data is:',n_train_data)
train_indices=np.random.choice(n_all_data,n_train_data,replace=False)
train_data, train_label=x_data[train_indices,...], y_data[train_indices,...]
test_data, test_label=x_data[np.delete(range(n_all_data),train_indices),...], y_data[np.delete(range(n_all_data),train_indices),...]
print('test_data shape is:', np.shape(test_data))
print('test_label shape is:', np.shape(test_label))


# define placeholder for inputs to network  
xs = tf.placeholder(tf.float32, [None, 3720])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer. transfer the nn to lr when set no hidden layer
l1 = add_layer(xs, 3720, 20, activation_function=tf.nn.relu)
#l2 = add_layer(l1, 40, 5, activation_function=tf.nn.relu)
# add output layer 
#prediction = add_layer(l1, 30, 1, activation_function=tf.nn.sigmoid)
prediction = add_layer(l1 , 20, 1, activation_function=tf.nn.sigmoid)

# define loss
# the error between prediciton and real data    
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction, labels = ys))

# choose the optimizer                 
# learning rate setting   
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)


# important step: initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# define epoch and training
for i in range(10):
   sess.run(train_step, feed_dict={xs: train_data, ys: train_label})
   cost = sess.run(loss, feed_dict={xs: train_data, ys: train_label})
   if i % 1 == 0:
       # to see the step improvement
       # print('cost after epoch %i: %f' % (i, cost))
       costs.append(cost)

#plt.plot(np.squeeze(costs))
#plt.show()

# define the threshold for the binary classification
Threshold = 0.3
y_pre_raw = sess.run(prediction, feed_dict={xs: test_data})

y_pre = y_pre_raw
y_pre[y_pre_raw>=Threshold] = 1
y_pre[y_pre_raw<Threshold] = 0

# count the number of TP,TN,FP,FN
count1=0
count2=0
count3=0
count4=0

for i in range(len(y_pre)):
    if y_pre[i]==1 and test_label[i]==1:
        count1=count1+1
#        print('right postive position is:',i)
    elif y_pre[i]==0 and test_label[i]==0:
        count2=count2+1
#        print('cont2 position is:',i)
    elif y_pre[i]==1 and test_label[i]==0:
        count3=count3+1
#        print('cont3 position is:',i)

    elif y_pre[i]==0 and test_label[i]==1:
        count4=count4+1
#        print('cont4 position is:',i)

# compute some statistic of the prediction
print('count1 is:',count1)
print('count2 is:',count2)
print('count3 is:',count3)
print('count4 is:',count4)
sensitivity=count1/(count1+count4)
print('sensitivity is',sensitivity)
Specificity=count2/(count2+count3)
print('Specificity is',Specificity)
print('y_pre mean is:', np.mean(y_pre))
print('test_label mean is:', np.mean(test_label))

print('y_pre range is:',np.unique(y_pre))
print('test_label range is:',np.unique(test_label))

# compute some evaluation metrics and print
correct_prediction = tf.equal(y_pre, ys)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
result = sess.run(accuracy, feed_dict={xs: test_data, ys: test_label})
print('accuracy result is:', result)

AUC = tf.metrics.auc(test_label, y_pre)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer()) # try commenting this line and you'll get the error
train_auc = sess.run(AUC)

#print(train_auc)

fpr, tpr, _ = roc_curve(test_label, y_pre)


AUC_ROC=roc_auc_score(test_label.flatten(), y_pre.flatten())

print('AUC_ROC is:',AUC_ROC)

precision, recall, _ = precision_recall_curve(test_label.flatten(), y_pre.flatten(),  pos_label=1)

AUC_prec_rec = auc(recall, precision)

print('AUC_PR is:',AUC_prec_rec)


####################################################
####################################################

filename = '../HK_epilepsy_data.csv'

# get the order the unique matrix
order_matrix = []

# used for disease code information storage .
ICD_code = []

# Construct the ICD9 CODE matrix containing all the disease in the datasets.
with open(filename) as f:
    reader=csv.reader(f)
    reader = list(reader)
    data_length = (np.array(reader)).shape[0]
#    print('shape of the reader is:', np.shape(reader))
    column = [[0 for i in range(data_length)] for j in range(16)]
#    print('shape of the column is:', np.shape(column))
    for row in reader:
        for k in range(16):
            diag=3*k+4
            column[k].append(row[diag])
#    print('the length of column10 is:', np.shape(column))
    ICD_code = column[1]+column[2]+column[3]+column[4]+column[5]+column[6]+column[7]+column[8]+column[9]+column[10]+column[11]+column[12]+column[13]+column[14]+column[15]   
#    print('shape of ICD is:', np.shape(ICD_code))
    order_matrix = np.unique(ICD_code)
    valueless_index = np.array(['','0','diag_cd_1','diag_cd_2','diag_cd_3','diag_cd_4','diag_cd_5','diag_cd_6','diag_cd_7','diag_cd_8','diag_cd_9',
    'diag_cd_10','diag_cd_11','diag_cd_12','diag_cd_13','diag_cd_14','diag_cd_15'])
    final_matrix = np.setdiff1d(order_matrix,valueless_index)
    print(final_matrix)
    final_matrix_shape = np.shape(final_matrix)
    print('Dimension of final_matrix is: ', final_matrix_shape)
#Random Forest
#####################################################################

columns = ['Gender', 'Age', 'Num_admin']

print(columns)
print(np.shape(columns))

feat_labels =  list(final_matrix)
#feat_labels = columns[0:]
feat_labels = columns+feat_labels
forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(train_data,train_label.ravel())

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print('shape of importances is: ', np.shape(importances))
print('shape of feat_labels is: ', np.shape(feat_labels))

for f in range(20):
    print("%2d) %-*s %f" % (f , 30, feat_labels[indices[f]], importances[indices[f]]))

y_pre_2 = forest.predict(test_data)
y_pre_2 = np.reshape(y_pre_2, (y_pre_2.shape[0],1))
print((y_pre_2.shape))

# statistic computation
count1=0
count2=0
count3=0
count4=0

for i in range(len(y_pre)):
    if y_pre_2[i]==1 and test_label[i]==1:
        count1=count1+1
#        print('right postive position is:',i)
    elif y_pre_2[i]==0 and test_label[i]==0:
        count2=count2+1
#        print('cont2 position is:',i)
    elif y_pre_2[i]==1 and test_label[i]==0:
        count3=count3+1
#        print('cont3 position is:',i)

    elif y_pre_2[i]==0 and test_label[i]==1:
        count4=count4+1
#        print('cont4 position is:',i)

print('count1 is:',count1)
print('count2 is:',count2) 
print('count3 is:',count3)
print('count4 is:',count4)
sensitivity=count1/(count1+count4)
print('sensitivity is',sensitivity)
Specificity=count2/(count2+count3)
print('Specificity is',Specificity)
print('y_pre mean is:', np.mean(y_pre_2))
print('test_label mean is:', np.mean(test_label))

# compute some metrics 

print('y_pre range is:',np.unique(y_pre_2))
print('test_label range is:',np.unique(test_label))

#y_pre_2 = tf.cast(y_pre_2,tf.float32)
y_pre_2 = y_pre_2.astype(np.float32)

correct_prediction = tf.equal(y_pre_2, ys)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
result = sess.run(accuracy, feed_dict={xs: test_data, ys: test_label})
print('accuracy result is:', result)

AUC = tf.metrics.auc(test_label, y_pre_2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer()) # try commenting this line and you'll get the error
train_auc = sess.run(AUC)

#print(train_auc)
print('test_label is',type(test_label))
print('y_pre_2 is',type(y_pre))
fpr, tpr, _ = roc_curve(test_label, y_pre_2)

AUC_ROC=roc_auc_score(test_label.flatten(), y_pre_2.flatten())

print('AUC_ROC is:',AUC_ROC)

precision, recall, _ = precision_recall_curve(test_label.flatten(), y_pre_2.flatten(),  pos_label=1)

AUC_prec_rec = auc(recall, precision)

print('AUC_PR is:',AUC_prec_rec)


saver = tf.train.Saver()  
  
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
    saver.save(sess, "../Model/model.ckpt")  

