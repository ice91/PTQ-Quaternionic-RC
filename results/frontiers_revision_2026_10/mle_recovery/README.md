# REV26-P1A MLE provenance bundle

狀態：MLE_ARCHIVE_RECOVERED_PARTIAL；另有 MLE_REPLAY_DISCREPANCY（僅 shared ν-q boundary metadata 值）。七模型 likelihood replay 通過，並不等於 covariance approximation 或統計有效性通過。

`original_recovered/01_mle/` 是本次找到原檔的逐位元副本；全部 33 檔與 5a8dca57f0fc3484387c643b51e22c7e10d024a8 Git blob 及公開 tree 一致。`implementation_snapshot/` 和 `outer_evidence/` 保留來源與歷史報告。`derived_from_archive/replay_v1/` 是本次新評估，不是歷史結果。

完整來源路徑／SHA256／commit／provenance 在 SOURCE_MANIFEST.json 與 RECOVERY_INVENTORY.csv。SHA256SUMS 列出本 bundle 除自身以外所有檔案。參數與 multistart 缺件另有兩份 availability CSV。

固定點重播（不含 optimization）：

```sh
python3 -m venv /tmp/rev26-p1a-replay
/tmp/rev26-p1a-replay/bin/pip install -r derived_from_archive/replay_v1/environment_freeze.txt
PYTHONDONTWRITEBYTECODE=1 /tmp/rev26-p1a-replay/bin/python replay_fixed.py --out /tmp/rev26-p1a-replay-new-output
```

需從此 bundle 位置執行（腳本由位置找 implementation root），輸出目錄必須不存在。輸入為 original_recovered 內七個 binary theta 與 implementation dataset/sparc_tidy.csv；星系按名稱排序，半徑按 r 排序。來源快照可用於核驗，腳本會載入現行 production API，重播前應對照 SHA256SUMS／SOURCE_MANIFEST。

最小可重播 objective 與生成曲線所需 data/code/完整最佳向量均已具備。rotation_curve_points.csv 已保存全部模型所有觀測半徑的 model/data/residual/template 數值；代表圖選樣及渲染留待後續。完整重建歷史 multistart 過程仍缺上述 provenance，不能宣告全面 reproduction complete。

如日後另授權 re-estimation，必須寫入新的 reestimated/<run-id>/，先保存完整 starts、seed/RNG state、CLI、environment/data/code hashes、所有 final theta、逐步 objective、quad/logdet、bounds、termination/KKT 與 shared refinement metadata。不得覆寫舊目錄或把新估計標為 original recovery。本次不執行此階段。
