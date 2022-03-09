# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 16:11
# @Author  : Zeqi@@
# @FileName: utils.py.py
# @Software: PyCharm

import yaml
from pathlib import Path

def load_yaml(file_path: Path) -> dict():
    with open(file_path, "r") as f:
        try:
            data = yaml.safe_load(f)
            return data
        except yaml.YAMLError as exc:
            print("Yaml exception {}".format(exc))