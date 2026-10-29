# REV26-P1F 獨立唯讀技術審查

審查代理：`/root/p1f_independent_integration_review`。受 Human 任務明示要求進行第二代理 critique；沒有 optimizer 呼叫，沒有修改 canonical、candidate、資料、theta 或先前證據。只新增本檔。AI agreement 不構成科學模型正確性證明。

本次審查時間點為完整 parity 完成、實際 optimizer smoke 準備啟動之時。下述對 driver 的判斷是源碼及已完成 deterministic evidence 的審查，**不是聲稱已看見接受 optimizer step**；最終實際 smoke 結果應由主報告另列。

## 已確認的實作與證據

- `marginal_likelihood.py` 採凍結 P1E segmented quantile；物理域 normalized D/i kernels、unit-square weights、sin(i)/sin(i0)、完整 conditional Gaussian normalization、B0/B1 floor 均與權威一致。完整 target 沒有 finite-tail cutoff，沒有第二次加入 propagated covariance。
- `models.catalog_velocity_at_distance` 確實同時縮放 radius 與 disk/bulge/gas 的 sqrt(s) amplitude；限制 NFW 的 halo 於改變的 physical radius 評估，而 M200 與 c(M) 固定。獨立 AST 比較確認所有原始 model-law 函數完全不變。
- 主要實際路徑是 CLI → `bounded_driver.run_bounded_smoke` → `mle.evaluate_loglike_components` → native `_layout/_unpack` → `galaxy_log_marginal` → actual model laws。optimizer scalar 為負 sum of galaxy log-marginals，沒有 prior density。公開 `fit_global`、`mle`、`experiments` 主要 likelihood 路徑均在 parity suite 以實際 import/call 驗證，非僅 AST。
- parity_v1：637 B0 +637 B1；2548 fine/coarse rows；728 native mappings；144 neighborhood rows；35 public full-sample call-path comparisons。max native parity 8.810729923425242e-13；mapping 9.094947017729282e-13；neighborhood parity 1.0231815394945443e-12。max fine/coarse per-galaxy 8.988752242089504e-6，max model sumabs 9.88420881853358e-6；repeat 0。沒有放寬凍結門檻。
- 新 primary 下的 radius-level WAIC/LOO/pointwise reporting 會 fail closed；合法輸出為 per-galaxy log-marginals 與明確標記的 descriptive residuals。舊 covariance 僅留 private/legacy；歷史 MCMC CLI、無預算 multistart driver 與舊結果 writer 明確 disabled。
- driver 對 input shape、finite、original hard bounds 每次檢查；完整目標 call 計數先遞增再執行，限制含 validation calls；每星系檢查 aggregate wall cap；190 optimizer calls/model 留9次 validation 空間。budget exception 不轉換成巨大 finite penalty，不自動 retry。callback 最多一 accepted iteration，SciPy `maxiter=1` 一致。
- 獨立重算 2025 個 preexisting 檔案 hashes 無變更；兩 Git HEAD 保持 outer 3eb302666da7eccca19c5843f729e98126d68977、implementation 5a8dca57f0fc3484387c643b51e22c7e10d024a8；protocol SHA256 3860858cfc991489dc5d5ee878236efa5591f43e0f11b74cb280b318beff76cc。
- 實際 NumPy/BLAS `matmul` warning 沒有被壓制。新增 warning_audit_v1 在91個真實 baryon cases 的相同192節點對 B0/B1 共182次直接 residual-squared 計算，最大 absolute logL 差8.526512829121202e-13；支持受測 warnings 沒有污染這些數值。**不表示 root cause 已證明，也不覆蓋未來所有 optimizer points。**

## 異議與處置建議

1. **中度、campaign 工程限制：dense finite differences 很昂貴。** q-family k=94 初始完整 gradient 預計95 calls；NFW k=183 預計184。190 optimizer cap 在 NFW 僅容許第一 gradient 再6 calls，可能無法完成第二 gradient／接受步。這是凍結預算的合理限制，不是科學 optimizer failure。若未 advance，必須 DRIVER_SMOKE=BUDGET_LIMITED；不得外推 full-campaign convergence time。最小後續為評估局部可分 likelihood 的 exact same-target finite differences／共享不變 galaxy terms，或另授權有明確成本的 engineering smoke；本輪不修改或 retry。

2. **中度、梯度證據限制：** `eps=1e-8` 是 integrated objective 的數值 forward differences，不是 analytic derivative。logL fine/coarse 通過與 identical theta repeat=0 均不能單獨證明梯度精度，尤其 bounds 與不同 native coordinate scales。應在報告明確保留此限制；真正 campaign 前需幾個預定參數、至少兩個合理 scaled FD steps 的方向導數及精度比較。不得把這變成 profile、尋優或新 statistical-target 選擇。

3. **低度、API 一致性：** `mle.objective_neg_loglike` 與 smoke driver 完整檢查 original hard bounds；`evaluate_loglike_components` 只查 shape/finite；`galaxy_loglike` 更是直接 `_unpack`，未統一 shape/bounds 檢查。當前 smoke 的外層 valid() 保護足夠，fixed-point authority 也應保留原始 theta 不暗改；但正式 production 整合應界定「純 evaluator 可評估 domain 外點」與「optimizer objective 必須 hard-bound」的接口，並對 per-gal evaluator 的多餘/缺少 theta 元素 fail closed。

4. **低度、legacy diagnostic 維護：** runtime AST wrapping 可阻斷直接 historical covariance caller，但不是長期穩定的 registration interface；explicit LEGACY 呼叫若進一步呼叫另一 guarded routine 且未傳版本，會再次被 primary-default guard 拒絕。這偏向安全 fail-closed 而非 primary 污染。不要宣稱所有 legacy workflows 仍兼容；正式整合時可以明確 registry 或完全 disabled。當前 primary WAIC/LOO 不可用的設計正確。

5. **低度、文檔殘留：** mle.py 模組 docstring 仍稱 production full-covariance Gaussian；marginal_likelihood.py 仍稱 frozen P1C laws reused read-only，但實際使用候選中原樣保留的 production laws。這不影響已測數值，但正式安裝前應修正文檔以免誤標 target。

6. **非 fatal、科學假設限制：** external independent Gaussian kernels 是 declared conditional nuisance distributions；不是 raw SPARC 獨立性證明。B0 為 nominal y0 additive floor，B1 是不同科學解讀，固定點 floor materiality 仍存在。新 target 不能消除 missing image-level/systematic/correlation information，也不能用本輪固定點或 smoke scores 宣稱新 MLE、q/scale、closure 或 IC ranking。

7. **非 fatal、數值證據範圍：** fine/coarse difference 是工程 refinement estimator，不是整個 hard-bound domain 的 uniform absolute error proof。smoke initial/terminal 必須保留完整精度結果；若 trajectory 出現 nonfinite、precision gate failure 或未解警告，應停止而不是調 target／tolerance。相同 full192 direct check 已足以支持目前 warning 的有限處置，仍需保留原始記錄與環境版本。

## 獨立結論

**沒有發現需要阻止目前已預先宣告 bounded smoke 的 fatal scientific 或 integration defect。** production/reference parity 的 PASS 有完整證據；source preservation 可獨立重算。是否可宣告 DRIVER_SMOKE PASS、是否可授權 seven-model campaign，必須依實際接受步、precision、wall/nfev 與上述 gradient/cost 限制作出分開判斷，不能從本審查的同意自動推出。

本審查不建議重開 statistical-target selection，不改 B0/B1 科學定義，不授權 canonical 安裝或任何 scientific refit。
