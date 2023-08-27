# -*- coding: utf-8 -*-
# @File: yy_t.py
# @Author: yblir
# @Time: 2023/8/24 0:26
# @Explain: 
# ===========================================
import yaml

with open('config/config.yaml','r',encoding='utf-8') as f:
    data=yaml.safe_load(f)

print(data['param_dict'])