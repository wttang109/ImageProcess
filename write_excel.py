# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 09:42:14 2018

@author: User
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_name = '降噪處理11.xlsx'
xl_workbook = pd.ExcelFile(file_name)  # Load the excel workbook
df = xl_workbook.parse("工作表1")  # Parse the sheet into a dataframe
g_150 = df['150lppi_good'].tolist()  # Cast the desired column into a python list
g_100 = df['100lppi_good'].tolist()
b_150 = df['150lppi_bad'].tolist()
b_100 = df['100lppi_bad'].tolist()

f_g_150 = np.fft.fft(g_150)
f_g_100 = np.fft.fft(g_100)
f_b_150 = np.fft.fft(b_150)
f_b_100 = np.fft.fft(b_100)

a_g_150 = np.abs(f_g_150)
a_g_100 = np.abs(f_g_100)
a_b_150 = np.abs(f_b_150)
a_b_100 = np.abs(f_b_100)

sam = 6468
step = 0.0423
sam_rate = 1/step
dist = sam_rate/sam
del_f = []
for i in range(0,sam):
    dist_list = dist*i
    del_f.append(dist_list)
'''
######### https://blog.csdn.net/u013250416/article/details/53189019
listk = ['del_frequency','150lppi_good','100lppi_good','150lppi_bad','100lppi_bad']
datas = {}
datas['del_frequency'] = del_f
datas['150lppi_good'] = a_g2_150
datas['100lppi_good'] = a_g2_100.tolist()
datas['150lppi_bad'] = a_bg_150.tolist()
datas['100lppi_bad'] = a_bg_100.tolist()

cols = pd.DataFrame(columns = listk)

for id in listk:
    cols[id] = datas[id]
cols.to_csv('降噪處理11.csv')
'''

def plot_f(subnum,y,value,picname,xname):
    plt.subplot(subnum)
    plt.plot(y,value)
############ https://blog.csdn.net/Running_J/article/details/52119336 ####
#    max_indx=np.argmax(ran)#max value index    
#    plt.plot(max_indx,ran[max_indx],'ks')
#    show_max='['+str(max_indx)+' '+str(ran[max_indx])+']'
#    plt.annotate(show_max,xytext=(max_indx,ran[max_indx]),xy=(max_indx,ran[max_indx]))
############ https://blog.csdn.net/Running_J/article/details/52119336 ####
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("FFT_{x}_未降噪_{y}".format(x=picname,y=xname),fontsize=20)
    plt.ylabel('abs',fontsize=20)
    my_x_ticks = np.arange(0, 12, 0.5)
    plt.xticks(my_x_ticks)
    plt.ylim((0, 12000))
############ https://blog.csdn.net/Running_J/article/details/52119336 ####################

plt.figure(figsize=(130,40), dpi=100, linewidth=0.9)
plot_f(411,del_f,a_g_150[:3234],file_name,'a_g_150')
plot_f(412,del_f,a_g_100[:3234],file_name,'a_g_100')
plot_f(413,del_f,a_b_150[:3234],file_name,'a_b_150')
plot_f(414,del_f,a_b_100[:3234],file_name,'a_b_100')
plt.savefig("FFT_{x}_未降噪.png".format(x=file_name))
plt.show()
