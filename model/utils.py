from wcwidth import wcswidth


# 幫助排版，免得全形字干擾字寬控制
def pad_display(text, width, align="left"):
    display_width = wcswidth(text)
    pad_len = max(0, width - display_width)

    if align == "left":
        return text + " " * pad_len
    elif align == "right":
        return " " * pad_len + text
    elif align == "center":
        left = pad_len // 2
        right = pad_len - left
        return " " * left + text + " " * right
    else:
        return text


# 用tick算時間(秒)
def tick_to_seconds_precise(tick, tempo_changes, ticks_per_beat=480):
    tempo_changes = sorted(tempo_changes, key=lambda x: x["tick"])

    current_time_microsec = 0
    for i in range(len(tempo_changes) - 1):
        start_tick = tempo_changes[i]["tick"]
        end_tick = tempo_changes[i + 1]["tick"]
        value = tempo_changes[i]["value"]  # μs/拍

        if tick < start_tick:
            # 輸入 tick 在目前區間之前
            break

        if tick < end_tick:
            # tick 落在此區間內
            delta_ticks = tick - start_tick
            beats = delta_ticks / ticks_per_beat
            current_time_microsec += beats * value
            return current_time_microsec / 1_000_000

        # tick 超過此區間，累積整段區間時間
        delta_ticks = end_tick - start_tick
        beats = delta_ticks / ticks_per_beat
        current_time_microsec += beats * value

    # 如果 tick 超過最後 tempo 節點，使用最後一段速度累積時間
    last_tick = tempo_changes[-1]["tick"]
    last_value = tempo_changes[-1]["value"]
    delta_ticks = tick - last_tick
    beats = delta_ticks / ticks_per_beat
    current_time_microsec += beats * last_value

    return current_time_microsec / 1_000_000
