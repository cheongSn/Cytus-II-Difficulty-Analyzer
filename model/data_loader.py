import pandas as pd
import os
import json
import zipfile


# 載入難度資料
def getDiffData():
    # QQ群載的
    df = pd.read_csv("real_difficulty.csv")
    # 去掉無資料的部分
    df = df.dropna(subset=["file_name"])
    return df


def getChartData():
    df = getDiffData()
    # 載入譜面資料
    for i in df.index:
        role_id = df.loc[i, "role_id"]

    # 全局變量，記住所有譜面資訊，有index就可以一起拿到
    available_chart = []
    real_diff_list = []
    all_song_name = []
    all_chart_json = []

    # CT2View會丟失頁面參數訊息，一般來說都直接忽略，但也可能會造成譜面難以使用
    # 但這裡只能放因為處理不當而壞掉的，不能單純放很難處理的
    dirty_data = ["ilka001_014+chaos", "neko002_001+glitch"]
    dirty_data = []

    for i in df.index:
        filename = df.loc[i, "file_name"] + ".json"
        if filename in dirty_data:
            continue
        if not os.path.exists("Cytus2Chart/" + filename):
            continue
        with open("Cytus2Chart/" + filename, newline="") as f:  # CT2View爬的
            chart_json = json.load(f)
        all_chart_json.append(chart_json)
        available_chart.append(filename)
        real_diff_list.append(df.loc[i, "diff"])
        all_song_name.append(df.loc[i, "song_name"])

    return real_diff_list, all_song_name, all_chart_json


if __name__ == "__main__":
    real_diff_list, all_song_name, all_chart_json = getChartData()
    print(c[0])


def getCytoidLevelChart(levelID):
    print("Loading Cytoidlevel:", levelID)
    if not levelID.endswith(".cytoidlevel"):
        levelID += ".cytoidlevel"
    # 指定 zip 檔路徑
    zip_path = levelID

    # 開啟 zip 檔
    with zipfile.ZipFile(zip_path, "r") as z:
        # 列出所有檔案
        # print("檔案清單:", z.namelist())

        # 讀取其中一個檔案內容（假設是文字檔）

        with z.open("level.json") as f:
            data = json.loads(f.read().decode("utf-8"))
            if len(data["charts"]) == 1:
                chart_diff_id = 0
            else:
                chart_diff_id = max(
                    range(len(data["charts"])),
                    key=lambda i: data["charts"][i]["difficulty"],
                )
                print("Multi level")
                print("Using Diff:", data["charts"][chart_diff_id]["difficulty"])
            # chart_diff_id = 0
            chart_meta = data["charts"][chart_diff_id]
            # print(chart_meta)
            if "storyboard" in chart_meta.keys():
                with z.open(chart_meta["storyboard"]["path"]) as sb:
                    data = json.loads(sb.read().decode("utf-8"))
                    if "chartBackup" in data.keys():
                        print("Has Backup")
                        return data["chartBackup"]

            with z.open(chart_meta["path"]) as cht:
                chart = json.loads(cht.read().decode("utf-8"))
                return chart
