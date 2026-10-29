# REV26-P1B covariance gate evidence

總判定：COVARIANCE_CORRECTION_REQUIRED_BEFORE_REFIT。run_v1/GATE_VERDICTS.json分列五個verdict。原向量與來源在INPUT_AUTHORITY，source保全在SOURCE_HASHES_BEFORE及SOURCE_PRESERVATION_VALIDATION。

- PREDECLARED_DESIGN/OBSERVATION_CONTRACT：run前凍結；六真實＋三合成點、不選residual。
- run_v1/：全部FD、likelihood分解、C(theta)、matched identities、nuisance積分與四階收斂、parameter-point hashes及失敗紀錄。
- OPTIMIZER_CALL_CHAIN.json：來源AST與原callback固定點動態執行；不執行optimizer driver。
- EXTERNAL_SOURCES.json：SPARC primary source版本／定位與讀取失敗紀錄。

原實際執行script保存為run_gate_executed_v1.py，其SHA256與run_v1/execution_authority.json吻合。run_gate.py後續僅加--out路徑選項和目錄限制（數值body相同），可重播至本root內另一個不存在的目錄：

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/rev26-p1a-env/bin/python run_gate.py --out /Users/rocky/ptq-mond/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/covariance_gate/replay_new
```

臨時環境的精確pip freeze在run_v1/environment_freeze.txt，Python/platform在execution_authority；環境不存在時在獲准output root內新建venv並安裝freeze（本輪未改原環境）。prepare_gate.py與freeze_design.py拒絕覆寫已封存的authority/design；重播不需重新執行這兩支。analyze_gate.py驗證run_v1全部numerical資料並產生統計摘要；write_reports.py產生本輪docs。沒有optimization、production patch或新fit。

SHA256SUMS涵蓋所有本root檔案除自身；完整new-asset manifest在docs root。報告原reviewer comments缺失，待作者提供；不能把舊模擬review當本次外審。
