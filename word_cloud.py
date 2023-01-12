#-*- coding : utf-8 -*-
# coding: utf-8
import json
import re
import string
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

path = '/data2/wwy/caption/data/'

#中文分词，并根据词频制作每个类别的词云图。
def get_pic(path):
    t = ['bird2words','ImageEdit','spot']
    for i in range(len(t)):
        dic = {}
        filename = path + t[i] + '/train.json'
        with open(filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                jterm = json.loads(line.strip())
                if t[i] == 'bird2words':
                    words = jterm['description']
                else:
                    words = jterm['sentences'].split(' ')
                for word in words:
                    if word in dic:
                        dic[word] += 1
                    else:
                        dic.update({word:1})
        fin.close()

        # 生成对象
        wc = WordCloud(width=800, height=600, prefer_horizontal=1, background_color='white').generate_from_frequencies(dic)

        # 显示词云
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.show() 

        # 保存到文件
        wc.to_file('/data2/cwj/' + t[i] + '_wordcloud.png')
    return

get_pic(path)