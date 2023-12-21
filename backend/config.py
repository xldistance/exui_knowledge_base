import json
from pathlib import Path

# 获取当前文件的父目录路径
parent_path = Path(__file__).parent.parent



def set_config_dir(filename):
    file_path = parent_path / filename
    # 检查路径是否存在，如果不存在则创建它
    if not file_path.exists():
        file_path.mkdir()
    return file_path

# 创建exui_dialogue文件夹
dialogue_path = set_config_dir("exui_dialogue")

def config_filename(filename: str):
    # 返回连接后的完整文件路径
    return dialogue_path / filename

class GlobalState:

    def __init__(self):
        pass

    def load(self):

        file_path = config_filename("state.json")
        if file_path.exists():
            with open(file_path, "r") as f:
                r = json.load(f)
        else:
            r = {}


    def save(self):

        r = {}

        file_path = config_filename("state.json")
        r_json = json.dumps(r, indent = 4)
        with open(file_path, "w") as outfile:
            outfile.write(r_json)


global_state = GlobalState()
