七個模型的最佳向量已回收；缺件不再是全向量。仍缺：

- 每個起點的完整初始向量、各起點最終向量（最佳向量除外）、逐迭代 objective/gradient/KKT 軌跡。
- 完成版 block-coordinate run 的確切 CLI、seed、外迴圈限制與原始套件版本／環境鎖。
- shared ν-q mapped refinement 的執行命令、步驟與中間向量；現存兩個 boundary 摘要值與最佳向量有小差異。
- 原始每銀河 quadratic/logdet 分量；本次計算只能標為 derived replay。

`command_log.txt` 第一段 --maxiter 400 --n-perturb 3 後另有 restart；不得把第一段命令當成完成版 run 的完整 provenance。源碼 seed 預設 42 不是已驗證 run seed。NFW top-two start ΔlogL≈29.116，不證全域最大值。沒有製造 CI、placeholder 或以 posterior median 替換 MLE。
