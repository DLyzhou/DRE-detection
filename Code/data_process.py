# This is useda for the data processing before the model construction

import numpy as np
import csv
import time
import datetime
import pandas as pd

#define the path of data file
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
    
# re-open the file seems fast the data processing
# encode the patient disease information into vector format- using pandas
with open(filename) as f:
    f.seek(0,0)
    reader=csv.reader(f)
    header_row=next(reader)
    reader = list(reader)
#    print('data_list is: ',data_list)
    sex=np.zeros((1),dtype=int)
    age=np.zeros((1),dtype=int)
    patient_datas=[]
    patient_data=[]
    DREs=[] 
    num=0
    num_adm=0
    num_admin=[]   
    one_vector=[] 
    vector_result=[]
    for row in reader:
        num+=1
        print('now is round: ',num)
#        print('the reader point is: ',reader[num][0])
#        print('where the row is   : ',row[0])
        if row[1]=='':
            break
        if row[1]=='F':
            sex=0
        elif row[1]=='M': sex=1
        age= row[3]
        # adjust the ratio of age here
        age = str((int(age)//10)*0.1)
        vector = np.zeros((final_matrix_shape[0],),dtype=int)
        DRE=[int(row[139])]

        if row[6]=='':
            row[6]=row[5]
# aquire the date information for data filtering
        timestruct_ddm = time.strptime(row[6],'%m/%d/%Y')
        timestruct_dip = time.strptime(row[58],'%m/%d/%Y')
        timestamp_ddm = int(time.mktime(timestruct_ddm))
        timestamp_dip = int(time.mktime(timestruct_dip))

# only utilize the information recorded before first dipensing date (FDD)
        if timestamp_ddm<timestamp_dip:
            num_adm+=1
            exact_sex_age=[sex,age]
            patient_data=[sex,age]
            for i in range(7,50,3):
                for j in range(0, final_matrix_shape[0]):
                    if row[i]==final_matrix[j]:
                        vector[j] = 1

# the data after FDD contribute NONE to vector
# here we process the data using the judge on the pseudo_id
        else:
            vector=vector

        if num==1:
            vector_result=vector
#            print('vector_result is:', vector_result)
        elif num==data_length-1:
#            print('the final row is here')
            one_vector=vector_result|vector
            vector_result=np.zeros((final_matrix_shape[0],),dtype=int)
#            print('clear vector_result is:',vector_result )
            num_admin=[num_adm]
            num_adm=0
            patient_data = np.hstack((patient_data, num_admin))
            patient_data = np.hstack((patient_data, np.transpose(one_vector)))
            DREs.append(DRE)
            patient_datas.append(patient_data)
        elif row[0]==reader[num][0]:
            vector_result=vector_result|vector
#            print('vector_result is:', vector_result)           
        else:
            one_vector=vector_result|vector
            vector_result=np.zeros((final_matrix_shape[0],),dtype=int)
            num_admin=[num_adm]
            num_adm=0
#            print('#################################')
#            print('patient_data shape is:',np.shape(patient_data))
#            print('num_admin shape is:',np.shape(num_admin))
#            print('np.transpose(one_vector) shape is:',np.shape(np.transpose(one_vector)))
            patient_data = np.hstack((patient_data, num_admin))
            patient_data = np.hstack((patient_data, np.transpose(one_vector)))
#            print('patient_data shape is:',np.shape(patient_data))
#            print('patient_datas shape is:',np.shape(patient_datas))
            DREs.append(DRE)
#            print('shape of patient_data is: ', np.shape(patient_data))
            patient_datas.append(patient_data)
            patient_data = exact_sex_age
#            print('##################################')

# store the data for better visualization with pandas
    csv_file = pd.DataFrame(patient_datas[8000:8500])
    csv_file.to_csv('../Data/CSV_input.csv', encoding='utf-8', index=False)  
#    print('patient_datas are:', patient_datas[8])
    print('patient_datas shape is:',np.shape(patient_datas))
#    print(patient_datas)
#    print('DREs data is:', DREs[1])
    print('DREs data shape is:', np.shape(DREs))
    print('Example Patient_data 1 is:',patient_datas[0])


    np.save('../Data/input data.npy', patient_datas)
    np.save('../Data/DRE data.npy', DREs)

    
    




