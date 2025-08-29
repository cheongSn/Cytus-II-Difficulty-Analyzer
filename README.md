# Cytus-II-Difficulty-Analyzer
嘗試訓練Cytus II譜面難度自動標註的模型。訓練資料因版權問題不公開，有研究需求或建議請洽 Discord ID: error._.418

- Cytus II Chart from [CT2View](https://github.com/KiattipoomR/ct2view)
- Difficulty by [Team CN:DC](https://www.bilibili.com/opus/801126523008450581)

## Feature Extraction
- 考慮click hold flick，把拍點分配給左右手
- 考量單手處理兩個note的位移和時間，計算burst序列 (burst系列feature)
- 加權統計每頁note數量 (page_space系列feature)
- 土法煉鋼地統計複雜節奏出現次數 (complex_beat_count)

## Future Work
- 提高Feature可用性
    - 用比較簡單的方式計算節奏複雜度
    - 統計單頁拍點排列複雜度
    - drag形狀造成的難度
    - 把drag加入手續考慮 (建模一下手的移動方式懲罰以及drag換手懲罰，應該比較容易做到)
- 直接使用譜面畫面來train模型
    - chart priview generation
    - 單曲譜面太少，但譜面頁數很多，page encoder可能可行
    - page encoder loss = 單頁難度(額外標記?) + Contrastive Learning
    - page encoder + LSTM or Transformer?
- 擴充訓練集
    - cytoid qualified system ?
    - 難度精細度？
    - Easy、Hard難度

## Best result
- XGBRegressor
- Train size: 578  
- Test size: 65 
- TEST Mean Squared Error: 0.3162  
- TEST R² Score: 0.8531
- Feature selection
    - burst_p90
    - page_space_p90_score
    - page_space_third_score
    - burst_song_avg
    - complex_beat_count
    - Drag-child
    - burst_LR_low_max
    - double_count
    - SONG_LENGTH
    - burst_endurance_8
    - CDrag-head
    - MAIN_BPM
    - burst_fifth

| ID  | 曲名                                | 等級  | 預測   | 誤差   |
|-----|------------------------------------|------|-------|-------|
| 376 | DON'T STOP ROCKIN'                  | 12.8 | 11.72 | -1.08 |
| 141 | Break Through the Barrier           | 13.6 | 12.61 | -0.99 |
| 138 | dimensionalize nervous breakdo      | 13.8 | 12.88 | -0.92 |
| 589 | すゝめ☆クノイチの巻                 | 14.6 | 13.69 | -0.91 |
| 252 | II                                  | 15.0 | 14.16 | -0.84 |
| 450 | ANiMA                               | 16.0 | 15.20 | -0.80 |
| 576 | Doldrums                            | 14.2 | 13.41 | -0.79 |
| 324 | dynamo                              | 15.0 | 14.25 | -0.75 |
| 137 | Jazzy Glitch Machine                | 13.8 | 13.09 | -0.71 |
| 409 | #Interstellar_Believer              | 15.0 | 14.36 | -0.64 |
| 82  | Online                              | 13.4 | 12.78 | -0.62 |
| 208 | Occidens                            | 14.8 | 14.19 | -0.61 |
| 95  | Liberation <GLITCH>                 | 15.8 | 15.22 | -0.58 |
| 222 | Drifted Fragments                   | 12.8 | 12.33 | -0.47 |
| 621 | Pink Graduation                     | 11.0 | 10.55 | -0.45 |
| 213 | New Challenger Approaching          | 15.0 | 14.57 | -0.43 |
| 297 | DON'T LISTEN TO THIS WHILE DRI      | 13.6 | 13.22 | -0.38 |
| 76  | Praystation (HiTECH NINJA Remi      | 13.6 | 13.24 | -0.36 |
| 302 | Code Interceptor                    | 13.0 | 12.72 | -0.28 |
| 606 | Accelerator                         | 13.6 | 13.37 | -0.23 |
| 124 | Dasein                              | 13.0 | 12.77 | -0.23 |
| 414 | init()                              | 13.6 | 13.39 | -0.21 |
| 243 | syūten                              | 11.2 | 10.99 | -0.21 |
| 412 | Malicious Mischance                 | 14.0 | 13.81 | -0.19 |
| 177 | Lunar Mare                          | 14.6 | 14.42 | -0.18 |
| 176 | AssaultMare                         | 14.2 | 14.04 | -0.16 |
| 8   | Baptism of Fire (CliqTrack rem      | 12.0 | 11.91 | -0.09 |
| 375 | Caliburne ～Story of the Legend     | 15.2 | 15.12 | -0.08 |
| 104 | Ready to Take the Next Step         | 15.2 | 15.12 | -0.08 |
| 262 | Risoluto (VILA)                     | 13.2 | 13.14 | -0.06 |
| 250 | CHAOS //System Offline//            | 13.0 | 12.95 | -0.05 |
| 315 | But I Know                          | 10.2 | 10.20 | -0.00 |
| 6   | KANATA <GLITCH>                     | 11.8 | 11.80 | 0.00  |
| 462 | AIAIAI (feat. 中田ヤスタカ)         | 11.6 | 11.60 | 0.00  |
| 548 | LEVEL4                              | 13.0 | 13.01 | 0.01  |
| 403 | LAST Re;SØRT                        | 15.4 | 15.45 | 0.05  |
| 174 | Area184                             | 11.6 | 11.66 | 0.06  |
| 543 | Deus Ex Machina                     | 12.4 | 12.47 | 0.07  |
| 21  | Who Am I?                           | 10.8 | 10.90 | 0.10  |
| 257 | Blessing Reunion                    | 11.4 | 11.51 | 0.11  |
| 557 | hunted                              | 13.0 | 13.13 | 0.13  |
| 236 | Crimson Fate                        | 13.4 | 13.58 | 0.18  |
| 571 | Bass Music                          | 13.4 | 13.58 | 0.18  |
| 498 | Levolution                          | 11.8 | 12.00 | 0.20  |
| 232 | Still (Piano Version)               | 9.4  | 9.64  | 0.24  |
| 88  | UnNOT!CED                           | 13.0 | 13.28 | 0.28  |
| 530 | Gekkouka                            | 11.6 | 11.95 | 0.35  |
| 75  | Idol                                | 12.2 | 12.61 | 0.41  |
| 434 | 糸                                  | 12.0 | 12.44 | 0.44  |
| 431 | メルの黄昏                          | 12.4 | 12.85 | 0.45  |
| 342 | Lights of Muse                      | 14.2 | 14.66 | 0.46  |
| 624 | Blah!!                              | 13.8 | 14.30 | 0.50  |
| 197 | Halcyon                             | 13.4 | 13.99 | 0.59  |
| 500 | Oneiroi                             | 12.8 | 13.40 | 0.60  |
| 63  | Hard Landing                        | 10.8 | 11.40 | 0.60  |
| 510 | Fighting                            | 11.8 | 12.41 | 0.61  |
| 593 | Inari                               | 10.4 | 11.03 | 0.63  |
| 18  | New World                           | 10.0 | 10.65 | 0.65  |
| 513 | Phantom Razor                       | 12.6 | 13.31 | 0.71  |
| 72  | Better than your error system       | 13.2 | 13.94 | 0.74  |
| 117 | Claim the Game <GLITCH>             | 14.0 | 14.91 | 0.91  |
| 139 | cold                                | 11.4 | 12.31 | 0.91  |
| 636 | Starchaser <GLITCH>                 | 13.0 | 13.98 | 0.98  |
| 201 | AXION                               | 13.0 | 14.00 | 1.00  |
| 632 | LIT                                 | 12.6 | 14.19 | 1.59  |

