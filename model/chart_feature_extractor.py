import matplotlib.pyplot as plt
import time
from collections import namedtuple
from utils import tick_to_seconds_precise
from RLhand_assignment import calculate_burst
import itertools
import heapq
from data_loader import getChartData
import numpy as np
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
import pandas as pd
import pickle


# 算每頁的分數，不同拍點有不同的加權計分方式
def get_page_score(note_list, page_list):
    ### PAGE_SCORE
    page_scores = defaultdict(float)
    for note in note_list:
        page_id = note["page_index"]
        type_ = note["type"]
        hold_tick = note.get("hold_tick", 0)
        tick = note.get("tick", 0)

        if type_ == 0:  # click
            page_scores[page_id] += 1
        elif type_ == 1:  # hold
            page_scores[page_id] += 1 + (hold_tick / 240) * 0.1
        elif type_ == 2:  # Lhold
            # 固定頭 1.5 分在起始頁
            page_scores[page_id] += 1.5
            start_tick = note["tick"]
            end_tick = start_tick + note["hold_tick"]
            while page_id < len(page_list):
                page = page_list[page_id]
                page_start = page["start_tick"]
                page_end = page["end_tick"]
                # 若頁面起點 >= LHold 結束，代表不再有交集，結束
                if page_start >= end_tick:
                    break
                # 計算與 LHold 的交集區間
                overlap_start = max(start_tick, page_start)
                overlap_end = min(end_tick, page_end)
                overlap_ticks = overlap_end - overlap_start
                # LHold時間計算分數
                page_scores[page_id] += (overlap_ticks / 240) * 0.1
                page_id += 1
        elif type_ == 3:  # DragH
            page_scores[page_id] += 0.5
        elif type_ == 4:  # DragC
            page_scores[page_id] += 0.2
        elif type_ == 5:  # Flick
            page_scores[page_id] += 1.5
        elif type_ == 6:  # CDragH
            page_scores[page_id] += 1
        elif type_ == 7:  # CDragC
            page_scores[page_id] += 0.2
    return page_scores


# 用公式算節奏複雜度
def rhythm_entropy(events, threshold_time=960, is_page=False, is_debug=False):
    """
    events: 節奏事件序列（可為音符時值列表）
    返回：節奏熵值（bit為單位）
    """
    events = events[events < threshold_time]
    events = events[events > 0]
    if len(events) == 0:
        return 0, 0
    events = np.round(events, 5)

    # 原本用後一個的兩倍是否等於前一個來偵測swing，但沒用
    # 後來改為偵測時值變化次數，不多
    swing_beat = 0
    for i in range(len(events) - 2):
        if events[i] != events[i + 1]:
            swing_beat += 1
        if events[i] != events[i + 2]:
            swing_beat += 0.2
    swing_beat_ratio = swing_beat / (len(events) - 1)

    # 每種時值出現次數
    unique, counts = np.unique(events, return_counts=True)

    # 同時過濾 unique 和 counts
    # count太小當成zure就不要了
    mask = (unique != 0) & (counts > counts.sum() * 0.01)
    unique = unique[mask]
    counts = counts[mask]

    probs = counts / counts.sum()

    ### 計算音符時值的 集中度，越不集中數值越高，表示節奏越複雜
    # 計算熵
    entropy = -np.sum(probs * np.log2(probs))
    if is_page:  # 應該沒用到
        entropy *= np.log2(len(events))

    # 這段觀察不同時值之間是否可以整除，作為複雜度參考，效果不好
    unique = np.concatenate([unique, [960]])
    new_counts = max(counts.sum() // 10, 2)
    counts = np.concatenate([counts, [new_counts]])

    probs = counts / counts.sum()
    ratio_penalty = 0
    total_weight = 0
    if len(unique) >= 2:
        for (i, a), (j, b) in itertools.combinations(enumerate(unique), 2):
            ratio = max(a, b) / min(a, b)
            ratio = round(ratio, 2)

            weight = probs[i] * probs[j]
            total_weight += weight

            # 整除（比值是整數）的情況，複雜度低
            if ratio.is_integer():
                continue

            # 2:3, 3:5, 5:6 等更難整除 → 複雜度較高
            ratio_penalty += weight  # 可以改成 weight * ratio 或更複雜公式

        ratio_penalty /= total_weight

    if is_debug:
        print(ratio_penalty, unique, counts)

    return entropy, ratio_penalty, swing_beat_ratio


def get_rhythm_entropy(note_list, page_scores, points, hands, page_list_length):
    # 這個函數只是把相關計算前處理封裝起來而已
    page_scores_sq = np.zeros(page_list_length)
    for page_index, score in page_scores.items():
        page_scores_sq[page_index] = score
    page_scores_mid = np.percentile(page_scores_sq, 75)

    tick_list = []
    R_tick_list = []
    L_tick_list = []

    for i, p in enumerate(points):
        if hands[i] == "R":
            R_tick_list.append(p.tick)
        else:
            L_tick_list.append(p.tick)
        tick_list.append(p.tick)

    tick_list = np.array(tick_list)
    tick_list = np.diff(tick_list)
    R_tick_list = np.array(R_tick_list)
    R_tick_list = np.diff(R_tick_list)
    L_tick_list = np.array(L_tick_list)
    L_tick_list = np.diff(L_tick_list)

    # 分多段計算，避免有些歌只有一半是跳拍被稀釋掉了
    bin_count = 9
    all_bins = np.round(np.linspace(0, len(tick_list) + 1, bin_count + 1)).astype(int)
    R_bins = np.round(np.linspace(0, len(R_tick_list) + 1, bin_count + 1)).astype(int)
    L_bins = np.round(np.linspace(0, len(L_tick_list) + 1, bin_count + 1)).astype(int)

    test_entropy_list = []
    test_ratio_penalty_list = []
    swing_beat_ratio_list = []

    for b in range(bin_count):
        all_diff = tick_list[all_bins[b] : all_bins[b + 1]]
        R_diff = R_tick_list[R_bins[b] : R_bins[b + 1]]
        L_diff = L_tick_list[L_bins[b] : L_bins[b + 1]]
        # 原本嘗試跟左右手分別的節奏合併後再放進去，但效果不佳
        # all_diff = np.concatenate([all_diff, R_diff, L_diff])
        # all_diff = np.concatenate([R_diff, L_diff])
        test_entropy, test_ratio_penalty, swing_beat_ratio = rhythm_entropy(
            all_diff, is_debug=False
        )
        test_entropy_list.append(test_entropy)
        test_ratio_penalty_list.append(test_ratio_penalty)
        swing_beat_ratio_list.append(swing_beat_ratio)

    # 取最難的80%位置
    test_entropy = np.percentile(test_entropy_list, 80)
    test_ratio_penalty = np.percentile(test_ratio_penalty_list, 80)
    swing_beat_ratio = np.mean(swing_beat_ratio_list)

    return test_entropy, test_ratio_penalty, swing_beat_ratio


# 土法煉鋼判斷複雜節奏，效果比前面那坨什麼熵的還好一點
def get_complex_score(tap_tick_list, page_list):

    n = len(tap_tick_list)
    if n == 0:
        return 0, 0
    m = len(page_list)
    complex_beat_count = 0
    l_ptr = 0  # 維護480 tick以內的範圍，不用每次都遍歷全部的tap_tick_list
    r_ptr = 0
    type_count = dict()
    cur_pid = 0  # 用來確定現在的note的page位置
    cur_page_start = 0
    for i in range(n):
        tick = tap_tick_list[i]
        while page_list[cur_pid]["end_tick"] < tick:
            cur_pid += 1
            cur_page_start = page_list[cur_pid]["start_tick"]
        while l_ptr < n and tap_tick_list[l_ptr] < tick - 480:
            l_ptr += 1
        while r_ptr < n and tap_tick_list[r_ptr] <= tick + 480:
            r_ptr += 1
        nearby = tap_tick_list[l_ptr:r_ptr]
        tick_shift = (cur_page_start - tick) % 480  # 強制從頁邊起算第一拍
        tick_shift = round(tick_shift / 10, 0) * 10 % 480

        if tick_shift == 0:
            continue

        elif tick_shift == 60:  ## 1/8
            type_count[0] = type_count.get(0, 0) + 1
            unit = 60
            if tick - unit in nearby and tick + unit in nearby:  # 32分交互
                complex_beat_count += 0.05
                continue
            if tick - unit in nearby:  # 拍頭偽雙壓
                complex_beat_count += 1
                continue
            complex_beat_count += 2.5

        elif tick_shift == 80:  ## 1/6
            type_count[1] = type_count.get(1, 0) + 1
            unit = 80
            if tick - unit in nearby and tick + unit in nearby:  # 24分交互
                complex_beat_count += 0.05
                continue
            if tick - unit in nearby:  # 拍頭偽雙壓
                complex_beat_count += 1
                continue
            complex_beat_count += 2.5

        elif tick_shift == 120:  ## 1/4
            type_count[2] = type_count.get(2, 0) + 1
            unit = 120
            if tick - unit in nearby and tick + unit in nearby:  # 16分交互
                continue
            if tick - unit in nearby or tick + unit in nearby:  # 二連點之類
                complex_beat_count += 0.5
                continue
            complex_beat_count += 2  # 純1/4，搞節奏心態

        elif tick_shift == 160:  ## 1/3
            type_count[3] = type_count.get(3, 0) + 1
            unit = 160
            if tick - unit in nearby and tick + unit in nearby:  # 12分交互
                complex_beat_count += 0.05
                continue
            if tick - 80 in nearby and tick + 80 in nearby:  # 24分交互
                complex_beat_count += 0.05
                continue
            if tick - 80 not in nearby and tick + 80 in nearby:  # 特別難搞的小swing節奏
                complex_beat_count += 3
                continue
            if tick - unit not in nearby and tick + unit in nearby:  # XOO XOO
                complex_beat_count += 1.5
                continue
            if (
                tick - unit in nearby and tick + unit not in nearby
            ):  # OOX OOX 可能沒這麼怪
                complex_beat_count += 0.8
                continue
            complex_beat_count += 1  # 其他單出十二分音的狀況

        elif tick_shift == 180:  ## 3/8
            type_count[4] = type_count.get(4, 0) + 1
            unit = 60
            if tick - unit in nearby and tick + unit in nearby:  # 32分交互
                complex_beat_count += 0.1
                continue
            complex_beat_count += 1  # 不是交互出現都很怪

        elif tick_shift == 240:  ## 1/2
            type_count[5] = type_count.get(5, 0) + 1
            unit = 240
            if tick - unit not in nearby and tick + unit not in nearby:  # 反半拍
                complex_beat_count += 0.1
                continue

        elif tick_shift == 300:  ## 5/8
            type_count[6] = type_count.get(6, 0) + 1
            unit = 60
            if tick - unit in nearby and tick + unit in nearby:  # 32分交互
                complex_beat_count += 0.1
                continue
            complex_beat_count += 1  # 不是交互出現都很怪

        elif tick_shift == 320:  ## 2/3
            type_count[7] = type_count.get(7, 0) + 1
            unit = 160
            if tick - unit not in nearby and tick + unit in nearby:  # 大swing
                complex_beat_count += 0.5
                continue
            if tick - 60 in nearby and tick + 60 not in nearby:  # 1/4 還能偽雙？
                complex_beat_count += 1.5
                continue
            # 原則上都出現過了

        elif tick_shift == 360:  ## 3/4
            type_count[8] = type_count.get(8, 0) + 1
            unit = 120
            if tick - unit in nearby and tick + unit in nearby:  # 16分交互
                continue
            if tick - unit in nearby or tick + unit in nearby:  # 二連點之類
                complex_beat_count += 0.5
                continue
            if tick - 360 in nearby and tick - 240 not in nearby:
                continue  # 有可能是332節奏的一部分
            complex_beat_count += 2

        elif tick_shift == 400:  ## 5/6
            type_count[9] = type_count.get(9, 0) + 1
            unit = 80
            if tick - unit in nearby and tick + unit in nearby:  # 24分交互
                complex_beat_count += 0.05
                continue
            if tick + unit in nearby:  # 小swing
                complex_beat_count += 3
                continue
            complex_beat_count += 2

        elif tick_shift == 420:  ## 7/8
            type_count[10] = type_count.get(10, 0) + 1
            unit = 60
            if tick - unit in nearby and tick + unit in nearby:  # 32分交互
                complex_beat_count += 0.05
                continue
            if tick + unit in nearby:  # 拍尾偽雙壓
                complex_beat_count += 1
                continue
            complex_beat_count += 2.5

        else:  # 持續性擺歪？
            type_count[11] = type_count.get(11, 0) + 1
            complex_beat_count += 1.5

    # print(type_count)
    return complex_beat_count, complex_beat_count / n


def get_all_feature(all_chart_json):
    is_test_time = False
    X = []
    for i_c, chart_json in enumerate(tqdm(all_chart_json, desc="Processing charts")):
        # for i_c in test_list:
        #   chart_json = all_chart_json[i_c]

        feature = []
        feature_name = []

        ### 一些list
        if is_test_time:
            start_time = time.time()
        # 因為轉換的關係所以不一定有sort好
        tempo_list = chart_json["tempo_list"]
        tempo_list = sorted(tempo_list, key=lambda x: x["tick"])
        page_list = chart_json["page_list"]
        page_list = sorted(page_list, key=lambda x: x["start_tick"])
        note_list = chart_json["note_list"]
        note_list = sorted(note_list, key=lambda x: x["tick"])
        if is_test_time:
            end_time = time.time()
            print("Sort list")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        if is_test_time:
            start_time = time.time()
        ### SONG_LENGTH
        last_tick = max([n["tick"] + n["hold_tick"] for n in chart_json["note_list"]])
        song_length = tick_to_seconds_precise(last_tick, tempo_list) - 200
        if song_length > 0:
            song_length = song_length**0.7  # 不太有歌超過3分鐘
        SONG_LENGTH = song_length + 200
        feature.append(SONG_LENGTH)
        feature_name.append("SONG_LENGTH")
        if is_test_time:
            end_time = time.time()
            print("SONG_LENGTH")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        if is_test_time:
            start_time = time.time()

        ### AVG_BPM
        # AVG_BPM = last_tick/8/SONG_LENGTH
        # feature.append(AVG_BPM)
        # feature_name.append("AVG_BPM")

        ### MAIN_BPM
        max_duration = 0
        for i in range(len(tempo_list)):
            start_tick = tempo_list[i]["tick"]
            value = tempo_list[i]["value"]  # μs/beat
            end_tick_segment = (
                last_tick if i == len(tempo_list) - 1 else tempo_list[i + 1]["tick"]
            )

            duration = (end_tick_segment - start_tick) * value

            if duration > max_duration:
                max_duration = duration
                MAIN_BPM = 60_000_000 / value  # 計算BPM
        feature.append(MAIN_BPM)
        feature_name.append("MAIN_BPM")
        if is_test_time:
            end_time = time.time()
            print("BPM")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        ### NOTE_TYPE_COUNT
        # 有一堆指標跟這個有相關，所以這部分效果相比之下沒這麼好
        # 如果測試資料把hard放進來，模型會用有沒有flick來作弊
        # Type 0: Click
        # Type 1: Hold
        # Type 2: L-Hold
        # Type 3: Drag-head
        # Type 4: Drag-child
        # Type 5: Flick
        # Type 6: CDrag-head
        # Type 7: CDrag-child
        type_counter = Counter(note["type"] for note in note_list)
        type_counts = [
            type_counter.get(i, 0) for i in [3, 4, 5, 6, 7]
        ]  # 沒有考慮drag的指標
        feature.extend(type_counts)
        # feature_name.extend(["Click", "Hold", "L-Hold", "Drag-head", "Drag-child", "Flick", "CDrag-head", "CDrag-child"])
        feature_name.extend(
            ["Drag-head", "Drag-child", "Flick", "CDrag-head", "CDrag-child"]
        )

        if is_test_time:
            start_time = time.time()
        # 算每頁加權分數
        page_scores = get_page_score(note_list, page_list)
        if is_test_time:
            end_time = time.time()
            print("Get Page Score")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        if is_test_time:
            start_time = time.time()
        # 每頁加權分數、用頁長考慮時間密度
        # 跟很多指標同質性太高所以刪掉
        # page_score_time_density = []
        # for page_index, score in page_scores.items():
        #   if score == 0:
        #     continue
        #   page = page_list[page_index]
        #   page_length = (page['end_tick'] - page['start_tick']) / 480
        #   if page_length < 0.01:
        #     continue
        #   page_score_time_density.append(score / page_length)
        # max_score = max(page_score_time_density)
        # median_score = np.median(page_score_time_density)
        # p75_score = np.percentile(page_score_time_density, 75)
        # p90_score = np.percentile(page_score_time_density, 90)

        # feature.append(max_score)
        # feature.append(median_score)
        # feature.append(p75_score)
        # feature.append(p90_score)
        # feature_name.append("page_time_max_score")
        # feature_name.append("page_time_median_score")
        # feature_name.append("page_time_p75_score")
        # feature_name.append("page_time_p90_score")

        # 不考慮頁長，只考慮視覺上的密度
        page_scores_list = list(page_scores.values())
        page_scores_list = [score for score in page_scores_list if score != 0]
        # max_score = max(page_scores_list)
        third_score = heapq.nlargest(3, page_scores_list)[
            2 if len(page_scores_list) >= 3 else -1
        ]
        # page_space_median_score = np.median(page_scores_list)
        page_space_p90_score = np.percentile(page_scores_list, 90)
        # page_space_p75_score = np.percentile(page_scores_list, 75)

        # feature.append(max_score)
        feature.append(third_score)
        # feature.append(page_space_median_score)
        # feature.append(page_space_p75_score)
        feature.append(page_space_p90_score)

        # feature_name.append("page_space_max_score")
        feature_name.append("page_space_third_score")
        # feature_name.append("page_space_median_score")
        # feature_name.append("page_space_p75_score")
        feature_name.append("page_space_p90_score")

        # 每頁耐力
        # 用數頁的最小值表示
        N = max(page_scores.keys()) + 1
        page_score_sequence = [page_scores.get(i, 0) for i in range(N)]
        window_sizes = [8, 16, 24, 32]
        results = {}
        for window_size in window_sizes:
            max_min = float("-inf")
            for i in range(len(page_score_sequence) - window_size + 1):
                window = page_score_sequence[i : i + window_size]
                local_min = min(window)
                if local_min > max_min:
                    max_min = local_min
            results[window_size] = max_min
        # feature.append(results[8])
        # feature.append(results[16])
        feature.append(results[24])
        # feature.append(results[32])
        # feature_name.extend(["page_endurance_8"])
        # feature_name.extend(["page_endurance_16"])
        feature_name.extend(["page_endurance_24"])
        # feature_name.extend(["page_endurance_32"])
        if is_test_time:
            end_time = time.time()
            print("Page Score Utility")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        ### 拍號、頁長
        # 效果不彰，很多地方都簡化掉了
        if is_test_time:
            start_time = time.time()
        page_ticks = []
        for page in page_list:
            ticks = page["end_tick"] - page["start_tick"]
            page_ticks.append(ticks)
        # 1. 統計 tick 出現次數
        counter = Counter(page_ticks)
        most_common = counter.most_common()
        # 第一常見 tick
        most_common_tick, most_common_count = most_common[0]
        if most_common_tick in [960, 480]:
            time_signatures_score = 0
        elif most_common_tick in [1920, 1440, 720]:
            time_signatures_score = 10
        elif most_common_tick < 1200:
            time_signatures_score = 15
        else:
            time_signatures_score = 25
        feature.append(time_signatures_score)
        feature_name.append("time_signatures_score")

        # 2. 第一常見佔比
        total = len(page_ticks)
        most_common_ratio = most_common_count / total
        # feature.append(most_common_ratio)
        # feature_name.append("most_common_tick_ratio")

        # 6. 變化次數（相鄰值不一樣就算變化）
        changes = sum(1 for i in range(1, total) if page_ticks[i] != page_ticks[i - 1])
        feature.append(changes)
        feature_name.append("page_tick_changes")

        # 7. 不同 tick 數值的種類數
        # unique_ticks_count = len(counter)
        # feature.append(unique_ticks_count)
        # feature_name.append("unique_ticks_count")
        if is_test_time:
            end_time = time.time()
            print("Time Signatures Score")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        ### 爆發度、三壓
        if is_test_time:
            start_time = time.time()
        burst_values, hands, points = calculate_burst(note_list, tempo_list)
        # 神奇的自動分配手順咚咚
        # 跟burst相關的指標表現都很好
        if is_test_time:
            end_time = time.time()
            print("Get Burst")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        if is_test_time:
            start_time = time.time()
        # print(len(burst_values), len(hands), len(points))
        THREE_HAND = hands.count("3+")
        feature.append(THREE_HAND)
        feature_name.append("THREE_HAND")  # 沒什麼用

        burst_array = np.array(burst_values)
        sorted_burst = np.sort(burst_array)[::-1]
        n = len(burst_values)
        max_val = sorted_burst[0]
        median_val = sorted_burst[int(0.5 * n)]
        fifth_val = sorted_burst[4]
        n = len(sorted_burst)
        p90_val = sorted_burst[int((1 - 0.9) * n)]
        p75_val = sorted_burst[int((1 - 0.75) * n)]
        burst_song_avg = np.sum(sorted_burst[int(0.1 * n) : int(0.9 * n)]) / SONG_LENGTH
        # feature.append(max_val)
        # feature.append(median_val)
        feature.append(fifth_val)
        feature.append(p90_val)
        # feature.append(p75_val)
        feature.append(burst_song_avg)
        # feature_name.append("burst_max")
        # feature_name.append("burst_median")
        feature_name.append("burst_fifth")
        feature_name.append("burst_p90")
        # feature_name.append("burst_p75")
        feature_name.append("burst_song_avg")

        if is_test_time:
            end_time = time.time()
            print("Burst Rank")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        if is_test_time:
            start_time = time.time()

        ## 用爆發值算耐力
        burst_series = pd.Series(burst_array)
        results = {}
        for window_size in [8, 100]:
            # 使用 rolling() 搭配 quantile(0.25)
            local_25 = burst_series.rolling(window_size).quantile(0.25)
            max_local_25 = local_25.max()
            results[window_size] = max_local_25

        feature.append(results[8])
        # feature.append(results[16])
        # feature.append(results[32])
        # feature.append(results[64])
        feature.append(results[100])  # 大約100 combo
        feature_name.append("burst_endurance_8")
        # feature_name.append("burst_endurance_16")
        # feature_name.append("burst_endurance_32")
        # feature_name.append("burst_endurance_64")
        feature_name.append("burst_endurance_100")
        if is_test_time:
            end_time = time.time()
            print("Burst Endurance")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        if is_test_time:
            start_time = time.time()

        # 考慮雙手同時出力可能比較難
        l_ptr = 0
        L_burst = 0
        R_burst = 0
        LR_low_burst_list = []
        window = 1
        for r_ptr, p in enumerate(points):
            if hands[r_ptr] == "L":
                L_burst += burst_values[r_ptr]
            elif hands[r_ptr] == "R":
                R_burst += burst_values[r_ptr]

            while l_ptr < r_ptr and points[l_ptr].t < p.t - window:
                if hands[l_ptr] == "L":
                    L_burst -= burst_values[l_ptr]
                elif hands[l_ptr] == "R":
                    R_burst -= burst_values[l_ptr]
                l_ptr += 1
            LR_low_burst_list.append(min(L_burst, R_burst))

        LR_low_burst_array = np.array(LR_low_burst_list)
        # burst_LR_low_75 = np.percentile(LR_low_burst_array, 75)
        # burst_LR_low_80 = np.percentile(LR_low_burst_array, 80)
        burst_LR_low_max = max(LR_low_burst_array[len(LR_low_burst_array) * 4 // 5 :])
        # feature.append(burst_LR_low_75)
        # feature.append(burst_LR_low_80)
        feature.append(burst_LR_low_max)
        # feature_name.append("burst_LR_low_75")
        # feature_name.append("burst_LR_low_80")
        feature_name.append("burst_LR_low_max")

        if is_test_time:
            end_time = time.time()
            print("Burst LR Low")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        if is_test_time:
            start_time = time.time()
        L_last = 0
        R_last = 0
        L_double_time = []
        R_double_time = []
        all_double_time = []
        double_click_threshold = 70 if most_common_tick < 840 else 140

        n = len(points)
        for id in range(n):
            if hands[id] == "L":
                if points[id].tick - points[L_last].tick < double_click_threshold:
                    L_double_time.append(points[id].t)
                    all_double_time.append(points[id].t)
                L_last = id
            elif hands[id] == "R":
                if points[id].tick - points[R_last].tick < double_click_threshold:
                    R_double_time.append(points[id].t)
                    all_double_time.append(points[id].t)
                R_last = id

        # window_sizes = [1, 1.5, 2, 2.5, 3]
        # results = {}
        # for window_size in window_sizes:
        #     w_start = 0
        #     results.setdefault(window_size, 0)
        #     for i in range(len(all_double_time)):
        #         while all_double_time[i] - window_size > all_double_time[w_start]:
        #             w_start += 1
        #         results[window_size] = max(results[window_size], i - w_start)

        # feature.append(results[1])
        # feature.append(results[1.5])
        # feature.append(results[2])
        # feature.append(results[2.5])
        # feature.append(results[3])
        # feature_name.append("double_click_1")
        # feature_name.append("double_click_1.5")
        # feature_name.append("double_click_2")
        # feature_name.append("double_click_2.5")
        # feature_name.append("double_click_3")
        feature.append(len(all_double_time))
        feature_name.append("double_count")
        if is_test_time:
            end_time = time.time()
            print("Double click")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        # 節奏相關指標
        if is_test_time:
            start_time = time.time()
        all_rhythm_entropy, rhythm_ratio_penalty, swing_beat_ratio = get_rhythm_entropy(
            note_list, page_scores, points, hands, len(page_list)
        )
        feature.append(all_rhythm_entropy)
        # feature.append(rhythm_ratio_penalty)
        feature.append(swing_beat_ratio)
        feature_name.append("all_rhythm_entropy")
        # feature_name.append("rhythm_ratio_penalty")
        feature_name.append("swing_beat_ratio")
        if is_test_time:
            end_time = time.time()
            print("Rhythm Entropy")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        if is_test_time:
            start_time = time.time()
        ### 節拍位置
        beat_type = []
        tap_tick_list = set()
        for note in note_list:
            if note["type"] == 7 or note["type"] == 4:
                continue
            tap_tick_list.add(note["tick"])
        tap_tick_list = list(tap_tick_list)
        tap_tick_list = sorted(tap_tick_list)

        complex_beat_count, complex_beat_ratio = get_complex_score(
            tap_tick_list, page_list
        )

        feature.append(complex_beat_count)
        # feature.append(complex_beat_ratio)
        feature_name.append("complex_beat_count")
        # feature_name.append("complex_beat_ratio")
        # print(f"{round(complex_beat_count, 2)}\t{round(complex_beat_ratio, 2)}")
        if is_test_time:
            end_time = time.time()
            print("Complex Beat")
            print(f"執行時間：{end_time - start_time:.6f} 秒")

        # print(feature)
        X.append(feature)
        # if i_c > -1:
        #   break

    X = np.array(X)
    return X, feature_name


def getXYdata(is_use_cache):
    if is_use_cache:
        with open("data_X_Y_feature.pkl", "rb") as f:
            X, Y, feature_name, all_song_name = pickle.load(f)
            return X, Y, feature_name, all_song_name

    real_diff_list, all_song_name, all_chart_json = getChartData()
    Y = np.array([real_diff_list]).reshape(-1)
    X, feature_name = get_all_feature(all_chart_json)

    with open("data_X_Y_feature.pkl", "wb") as f:
        pickle.dump((X, Y, feature_name, all_song_name), f)

    return X, Y, feature_name, all_song_name


if __name__ == "__main__":
    # X, Y, feature_name, all_song_name = getXYdata(False)
    X, Y, feature_name, all_song_name = getXYdata(True)
    # print(all_song_name)
