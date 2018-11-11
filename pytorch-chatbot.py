# -*- coding: utf-8 -*-
# @Time    : 2018/10/9 21:59
# @Author  : seeledu
# @email   : seeledu@bug.moe
# @File    : pytorch-chatbot.py
# @Software: PyCharm
"""
利用pytorch写一个简单的chatbot
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
from torch.jit import trace,script
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

# 检验能否用gpu,不行的话设定为cpu模式.
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device("cuda" if USE_CUDA else "cpu")

# 下载语料
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)
def printLiunes(file, n= 10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLiunes(os.path.join(corpus,"movie_lines.txt"))
