import matplotlib.pyplot as plt
import time
from collections import namedtuple
from data_loader import getChartData
from utils import tick_to_seconds_precise


### 給兩位置的時間座標，計算出力分數，t是秒
def compute_burst(
    x1, x2, t1, t2, hand, a=0.2, is_debug=False, is_post_process=False, type1=0
):
    dt = max(min(t2 - t1, 1), 0.001)  # 時間太長太短都忽略

    # 因為原本的手順測試運作兩好，後續只想更改burst最終計算方式，所以用is_post_process避免動到手順計算
    if is_post_process and type1 == 5:
        dt = max(dt - 0.05, 0.001)  # flick處理時間長
        if hand == "L":  # flick打完後位置偏移
            x2 -= 0.1
        elif hand == "R":
            x2 += 0.1

    dx = abs(x2 - x1)
    if is_post_process and type1 == 5:  # flick本身要位移
        dx += 0.1

    # 經驗公式
    time_burst = 1 / (dt**1.5)
    space_burst = (dx * 10) ** 1.5 / 10

    # 計算手順時讓左手在左、右手在右，但實際上算burst時可以不用管
    if not is_post_process:
        if hand == "L":
            hand_coe = 1 + (x2 * 10)
        elif hand == "R":
            hand_coe = 1 + ((1 - x2) * 10)
        else:
            hand_coe = 0
    else:
        hand_coe = 0

    if is_debug:
        output = [
            round(val, 2)
            for val in [
                space_burst,
                time_burst,
                (a + space_burst) * time_burst,
                hand_coe,
            ]
        ]
        # print(output)

    v = (a + space_burst) * time_burst + hand_coe
    if is_post_process:
        v = v**0.7  # 讓太大的值小一點
        if type1 == 5:  # 再幫flick多加點分數
            v = v + 0.5
    return (a + space_burst) * time_burst + hand_coe


def minimize_burst_with_path(points, a=0.2):
    from collections import defaultdict
    import math

    is_three_hand = False

    # 神奇的動態規劃，我也不是太懂，大都ChatGPT寫的
    MAX_STATES = 20
    n = len(points)
    dp = defaultdict(lambda: float("inf"))
    parent = dict()  # key: (l, r, i)，value: (prev_l, prev_r, 'L' or 'R')
    dp[(-1, -1, 0, 0)] = 0

    for i in range(n):
        new_dp = defaultdict(lambda: float("inf"))
        new_parent = dict()
        p_i = points[i]
        # print(len(dp))

        for (l, r, l_block, r_block), cost in dp.items():
            # 左手可用
            if i >= l_block:
                if l == -1:
                    left_burst = 0
                else:
                    p_l = points[l]
                    left_burst = compute_burst(
                        p_l.x, p_i.x, p_l.t, p_i.t, "L", a, type1=p_l.note_type
                    )
                new_cost = cost + left_burst
                key = (i, r, p_i.block_id, r_block)
                if new_cost < new_dp[key]:
                    new_dp[key] = new_cost
                    new_parent[(i, r, i)] = (l, r, "L")

            # 右手可用
            if i >= r_block:
                if r == -1:
                    right_burst = 0
                else:
                    p_r = points[r]
                    right_burst = compute_burst(
                        p_r.x, p_i.x, p_r.t, p_i.t, "R", a, type1=p_r.note_type
                    )
                new_cost = cost + right_burst
                key = (l, i, l_block, p_i.block_id)
                if new_cost < new_dp[key]:
                    new_dp[key] = new_cost
                    new_parent[(l, i, i)] = (l, r, "R")

            # 左右都不能用（3壓）
            if i < l_block and i < r_block:
                if not is_three_hand:
                    is_three_hand = True
                key = (l, r, l_block, r_block)
                if cost < new_dp[key]:  # 沒增加 burst
                    new_dp[key] = cost
                    new_parent[(l, r, i)] = (l, r, "3+")

        # 剪枝，讓DP快一點，大概不影響
        if len(new_dp) > MAX_STATES:
            sorted_states = sorted(new_dp.items(), key=lambda item: item[1])
            limited_states = sorted_states[:MAX_STATES]

            new_dp = defaultdict(lambda: float("inf"))
            for key, cost in limited_states:
                new_dp[key] = cost
        dp = new_dp
        parent.update(new_parent)

    # 找出最小總爆發值以及對應終點狀態
    min_cost = float("inf")
    final_state = None
    for (l, r, l_block, r_block), cost in dp.items():
        if cost < min_cost:
            min_cost = cost
            final_state = (l, r)

    # 回溯路徑
    hand_sequence = [""] * n
    cur_l, cur_r = final_state
    for i in reversed(range(n)):
        prev_l, prev_r, hand = parent[(cur_l, cur_r, i)]
        hand_sequence[i] = hand
        if hand == "L":
            cur_l = prev_l
        elif hand == "R":
            cur_r = prev_r

    return min_cost, hand_sequence
    # min_cost其實後來沒在管，反正要算burst序列
    # hand_sequence是R或L


def calculate_burst(note_list, tempo_list, is_debug=False):
    Point = namedtuple(
        "Point", ["idx", "t", "x", "block_id", "dur", "pid", "tick", "note_type"]
    )
    points = []
    # 除了drag child的note都視為click
    for note in note_list:
        if note["type"] == 7 or note["type"] == 4:
            continue
        t = tick_to_seconds_precise(note["tick"], tempo_list)
        if note["type"] == 1 or note["type"] == 2:  # 加上hold手不能動的時間
            hold_length = (
                tick_to_seconds_precise(note["tick"] + note["hold_tick"], tempo_list)
                - t
            )
            points.append(
                (
                    t,
                    note["x"],
                    hold_length,
                    note["page_index"],
                    note["tick"],
                    note["type"],
                )
            )
        else:
            points.append(
                (t, note["x"], 0, note["page_index"], note["tick"], note["type"])
            )

    N = len(points)
    # 做一點轉換
    new_points = []
    for i, (t, x, dur, pid, tick, note_type) in enumerate(points):
        if dur == 0:
            dur = 0.03  # 把所有click都設定最小間隔，可以偵測三壓
        t_block = t + dur
        i_cur = i + 1
        while i_cur < N and points[i_cur][0] <= t_block:
            i_cur += 1
        new_points.append(Point(i, t, x, i_cur, dur, pid, tick, note_type))

    points = new_points

    if is_debug:
        print(f"原物量：{len(note_list)}, 處理後物量：{len(points)}")

    # 這個函數是在抽取譜面特徵時的最大瓶頸
    start_time = time.time()
    cost, hands = minimize_burst_with_path(points)
    end_time = time.time()
    # print(f"執行時間：{end_time - start_time:.6f} 秒")

    burst_values = []
    prev_L = -1
    prev_R = -1
    # 重算一次burst序列
    for i, (p, hand) in enumerate(zip(points, hands)):
        if hand == "L":
            if prev_L == -1:
                burst = 0
            else:
                p_prev = points[prev_L]
                burst = compute_burst(
                    p_prev.x,
                    p.x,
                    p_prev.t,
                    p.t,
                    "L",
                    a=0.2,
                    is_debug=is_debug,
                    is_post_process=True,
                )
            prev_L = i
        elif hand == "R":
            if prev_R == -1:
                burst = 0
            else:
                p_prev = points[prev_R]
                burst = compute_burst(
                    p_prev.x,
                    p.x,
                    p_prev.t,
                    p.t,
                    "R",
                    a=0.2,
                    is_debug=is_debug,
                    is_post_process=True,
                )
            prev_R = i
        else:
            burst = 0  # 'error' 或 '3+' 設為0

        burst_values.append(burst)

    if is_debug:
        # 一些基本資訊
        print(f"執行時間：{end_time - start_time:.6f} 秒")
        print("最小爆發總和（最佳左右手分配）：", sum(burst_values))
        print("左右手分配：", hands)

        colors = {"L": "blue", "R": "red", "3+": "yellow"}
        # burst分布直方圖
        plt.figure(figsize=(8, 6))
        plt.hist(burst_values, bins=30, color="skyblue", edgecolor="black")
        plt.xlabel("burst_values")
        plt.ylabel("appear time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if is_debug and True:
        # 手順圖
        plt.figure(figsize=(8, 400))

        for i, (p, hand) in enumerate(zip(points, hands)):
            color = colors.get(hand, "gray")  # 預設為 gray，避免意外值
            label = hand if hands[:i].count(hand) == 0 else ""

            plt.scatter(p.x, p.t, color=color, s=100, label=label)
            plt.text(
                p.x,
                p.t,
                f" {i}",
                verticalalignment="center",
                horizontalalignment="center",
            )

            if p.dur > 0.031:
                plt.plot(
                    [p.x, p.x], [p.t, p.t + p.dur], color=color, linewidth=4, alpha=0.5
                )

        plt.legend()
        plt.grid(True)
        plt.xlabel("X 位置")
        plt.ylabel("時間（秒）")

        plt.xlabel("Position (X)")
        plt.ylabel("Time (t)")
        plt.gca().invert_yaxis()  # 時間向下遞增，視覺上從上到下順序
        plt.legend()
        plt.grid(True)
        plt.savefig("hand_assignment.png")

    return burst_values, hands, points


if __name__ == "__main__":
    real_diff_list, all_song_name, all_chart_json = getChartData()
    # 測試，理論上重新讀檔會慢一點，但少量測試應該沒差
    # with open("Cytus2Chart/" + "neko002_010+chaos.json", newline='') as f:
    # with open("Cytus2Chart/" + "xenon001_007+glitch.json", newline='') as f:
    # with open("Cytus2Chart/" + "neko001_061+chaos.json", newline='') as f:
    chart_json = all_chart_json[324]
    chart_json = all_chart_json[416]  # Lucid Traveler
    chart_json = all_chart_json[211]  # iL
    chart_json = all_chart_json[182]  # Halloween Party

    # chart_json = all_chart_json[443]
    note_list = chart_json["note_list"]
    note_list = sorted(note_list, key=lambda x: x["tick"])
    tempo_list = chart_json["tempo_list"]

    start_time = time.time()
    burst_values, hands, points = calculate_burst(note_list, tempo_list, is_debug=False)
    end_time = time.time()
    print(f"執行時間：{end_time - start_time:.6f} 秒")
