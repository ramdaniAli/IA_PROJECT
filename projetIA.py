import pandas as pandora
import numpy as np
from sklearn.cluster import KMeans
import numpy as np


def get_data1():
    my_data = pandora.read_csv("general_data.csv")
    return my_data



def get_data():
    my_data = pandora.read_csv("general_data.csv")
    return my_data


data = get_data()
data_withoutyes = data.copy()


for x in data:
    print(x)



datas = data_withoutyes.where(data_withoutyes =="yes", 1, inplace=True)

# data1 = get_data1()
#
#
# print(data,data1)
# conditions_data = (data.Attrition =='Yes')&(data.Attrition.isna()==False)
#
#
#
# data = data[conditions_data]
# all_data = pandora.concat([data, data1], axis=1)
#
# print(all_data.T[0])
#
#
#
#
#
# kmeans = KMeans(n_clusters=2, random_state=0).fit((all_data))
# kmeans.labels_
#
# kmeans.predict
# kmeans.cluster_centers_
#




# print(data.T[0])







