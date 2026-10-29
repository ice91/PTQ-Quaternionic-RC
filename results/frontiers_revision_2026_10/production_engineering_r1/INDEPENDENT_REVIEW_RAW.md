# REV26-P1F-R1 獨立唯讀源碼審查

審查代理 `/root/p1f_r1_independent_review`，依使用者明示要求。只新增本報告；未修改 canonical、candidate、資料、theta、歷史證據，沒有 optimizer／額外高成本數值呼叫。此稿時間點為 Gate C、regression 與 fixed-direction 數值工作進行中；不能把源碼同意當成未完成數值 gate 的 PASS。後續數值觀察另寫 addendum。

## 可獨立確認的結構

1. 原 model-law 函數的 AST 與 canonical 完全相同；新候選 `models.py` 全檔 bytes 與 P1F 一致，SHA256 `cba208085f4e5e6373fe3b1bd5b862e9be66a6d1412a6a7750c4c4b053bbc952`。
2. `_layout/_unpack` 為七 native 參數化保留 U slices、NFW lM slices、scalar eps/log10a0/lq/lnsigma。`catalog_velocity_at_distance` 只取同 galaxy 的 `U[gi]`、`lM[gi]`；NFW concentration 只依該 M200 及固定 c0/c_slope。其餘模型 mean 僅用同 galaxy 資料與 global scalars。條件 covariance/noise 沒有 pooled term，D/i 外部 kernels 亦不依另一 galaxy 的 local parameters，沒有發現 hidden cross-galaxy coupling。
3. Global sigma 是共享參數，但 given sigma 的 likelihood 仍按 galaxy marginal 相乘；其 objective derivative 正確為所有 galaxy derivative 之負和，不是將各 radius nuisance marginal factorize。
4. `DependencyGradient.evaluate` 先計 exact baseline，local 只重算 affected galaxy，global 重算全部；各 stencil 都 `theta.copy()`，故 `_unpack` 的 slice views 不會改原 baseline。回傳 SciPy gradient 為 native 坐標，exp/log10 transformations 已在重算 objective 內，自動包含該 native 數值差分的鏈式效果。負號及 central／inward one-sided 分母方向正確。
5. parameter-valued baseline cache 只存活於單一 evaluate call；不同 theta 不共用 mean／variance。每次 sample/source identity 都重檢，quadrature 僅重用既定 geometry nodes。無持久 parameter cache 失效問題。
6. 新 pure evaluator 契約檢查 theta shape/finite、model/layout/sample ID；optimizer objective 另強制原 hard bounds、不 clip；galaxy-local ID 拒絕負數、非整數、bool、未知字串；decoded-local 再查 U/lM/scalars 及資料 shape/finite。
7. runtime AST diagnostic wrapping 已換成明確 disabled registry，直接 covariance caller 與 WAIC/LOO/PPC/kappa/z diagnostics 在新候選拒絕，包含 explicit LEGACY request，屬安全 fail-closed。合法 per-gal marginal 及 descriptive residuals 留存，未創造 radius-level canonical predictive score。
8. warning telemetry 使用 category/message/file/line/model/version 聚合 count 與一個代表 hash，不保留每 event 大型字串清單；沒有全域 suppress RuntimeWarning，collector 外仍正常送 warnings，exceptions 回傳 False 繼續傳播。

## 已提出並處置的異議

- **verification receipt 未綁定 gradient/ledger 自己的程式碼：已修。** 原 `source_hashes` 只含五個 evaluator/model 檔，若之後改 ledger 邏輯可沿用舊 receipt。修補後加入 `dependency_gradient.py`；parent 說明正在執行的 Gate C 是舊 process，故保留其原 receipt，再以預存 AST hash 驗證 ledger/step/sample 邏輯未改，另加 sealed receipt 綁定第六檔。此 metadata bridge 必須明列而不能改寫原 evidence。
- **B1 gradient path 未有同等 dependency receipt 卻可請求：已修。** 現在 class 只接受 PRIMARY_B0，B1 pure likelihood 仍保留，沒有暗稱 B1 gradient 已驗證。
- **archived dense eps1e-8 FD roundoff：需分層解讀。** 全 objective subtraction 在分母1e-8時可放大 sum roundoff；local FD 改善取消但不能因此宣稱 original dense gradient 是真值。archived comparison 應使用預先記錄的 roundoff allowance；7個預定方向、兩步長、fine/coarse 是獨立精度 gate，不可拿 repeat=0 取代。

## 保留限制與建議

- warning handler 暫時覆寫 process-global `warnings.showwarning`；ContextVar 不使 Python warning machinery 成為 thread-safe。本輪 serial 執行可審查，尚未驗證 concurrent threaded likelihood calls；後續 driver/campaign 不應假裝已獲並行支援。
- representative warning hash 是 decoded pars＋gi；固定 H0/c0/c_slope、sample identity 與 order 可從外層 provenance追溯，但若將來允許多 config 混跑，宜把 config/order/sample hash 一起納入 warning point key。
- 任意 `FloatingPointError` 在 telemetry 被標記 `NUMERICAL_RESULT_CORRUPTION_CONFIRMED`；應解讀為該 numerical call 失敗，而不是已證明 warning 為根因。無錯誤的 finite outputs仍保留 `UNRESOLVED_WARNING_ROOT_CAUSE`，不能宣稱所有 warnings 無害。
- dependency 的數學結構可由 source證明，但 Gate C 的實證是原始 fixed theta、coarse96 全重算；primary仍192，selected fine192方向對照應另外報告。不能把此證據外推為全 hard-bound domain 的 rigorous integration 或 derivative error bound。
- operating-regime 的 B0/B1 floor interpretation、independent catalog kernels、missing D/i correlation與image-level uncertainties不由本工程修補消失。此任務不重新挑 target，也不產生 scientific MLE。
- full gradient可用 finite difference，但完整 native coordinate coverage／scale／bounds 的 tests 必須完成；有限7方向通過只支持下一個 bounded engineering smoke，不等同 ready for scientific MLE。

## 源碼層結論

未發現需要改變凍結 statistical target 的 fatal 科學缺陷。receipt 綁定與 B1 verification path 的兩個工程缺口已提出並修補。是否 `DEPENDENCY_GRADIENT`、`DIRECTIONAL_PRECISION` 與 `NEXT_DRIVER_SMOKE` 可 PASS／READY，需等待完整數值紀錄；本稿不預判。AI reviewer 同意不構成科學正確性證明。
