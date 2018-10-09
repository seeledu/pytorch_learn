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
from torch.jit import trace
import torch.nn as nn
