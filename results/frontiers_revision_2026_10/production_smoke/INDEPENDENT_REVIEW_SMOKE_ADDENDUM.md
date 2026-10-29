# 獨立唯讀補充：實際 optimizer-driver smoke

審查代理 `/root/p1f_independent_integration_review`。原始 `INDEPENDENT_REVIEW_RAW.md` 原樣保留。本補充只讀 actual result、完整 objective-call JSONL、smoke_analysis_v1 與原始 source manifest；沒有呼叫 optimizer、額外 likelihood 或修改候選。

**ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE。**

## 獨立重算的執行事實

- ptq-screen：199 full-sample calls =190 optimizer +9 validation；accepted iterations=0；wall 271.411084958秒。
- nfw1p：199 full-sample calls =190 optimizer +9 validation；accepted iterations=0；wall 274.571119333秒。
- 從兩份 objective_calls.jsonl 重新串流計數：aggregate call IDs 精確且唯一為1..398；每筆 scalar 與91 galaxy log-marginals 都 finite，且每筆 target 都是 `REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1`。沒有混入 Gaussian surrogate 或 B1 primary。
- 兩 terminal vectors 均逐元素等於 original initial vectors，符合無 accepted callback 時的凍結規則。沒有新 accepted engineering vector，更沒有 corrected MLE。
- 原記錄使用 `np.sum` 分解回算誤差0；獨立改用 `math.fsum` 重算所有398筆，最大差1.8189894035458565e-12，屬求和次序的浮點差。
- 兩模型初始／終點 precision checks 皆 PASS。q-family max per-gal fine/coarse 1.5049383961240892e-8、sumabs2.370220286707081e-8；NFW對應4.500698480569554e-8與7.031519722033863e-8；repeat0。但終點等於起點，**不是新 trial parameter neighborhood 的精度證明**。
- ptq-screen完整 initial95 calls 與 first trial95 calls 後，下一次呼叫於190 cap被攔截，尚無 accepted advance；NFW完成 initial184 calls，僅餘6 trial calls，未完成第二 gradient。兩者都是 `DRIVER_SMOKE_BUDGET_LIMIT`，不是 optimizer convergence failure。
- 實測 mean optimizer B0-only full91 objective：q-family1.3712924269947366秒、NFW1.3861478370842097秒。完整 initial gradient 實測128.840975379秒與254.181860296秒。這可用來估 objective/gradient 成本，不足預測接受步、iteration總數或收斂時間。
- 每模型捕捉1,291,386個 matmul warnings，均保留；先前182個相同192節點direct comparisons支持已測數值正確，但尚未證明 warning root cause，也沒有證明全trajectory梯度精度。
- 再次獨立重算2025個 preexisting files hashes：全部不變。

## 四項結論分開

- `PRODUCTION_PARITY=PASS`：完整 P1E 對照及實際 public-call evidence 成立，沒有發現新 fatal target defect。
- `DRIVER_SMOKE=BUDGET_LIMITED`：有真實 scipy optimizer callbacks、完整 initial finite-difference gradients、finite trial evaluations與硬 cap證據；**沒有 accepted step，故不得 PASS**。
- `NUMERICAL_OBJECTIVE=ACCEPTABLE_FOR_BOUNDED_MLE`：僅指凍結 target 在本輪受測 policy/points 的 likelihood 可評估、repeatable並通過宣告的 logL精度。此標籤不能解讀為 finite-difference gradient accuracy、全optimizer域精度或現有 driver 足以高效完成campaign。方向導數accuracy與gradient refinement尚未建立。
- `CAMPAIGN_READINESS=ENGINEERING_REPAIR_REQUIRED`：目前不能依此smoke直接宣告七模型campaign可以啟動；需先完成下一個有界工程步驟與Human審核。

## 最小後續

建議維持已選定 statistical target，另立一個有上限的 driver/gradient工程任務：重用 per-galaxy log-marginal的可分結構，對 local U／halo 擾動只重算受影響星系，對 global scale／q／sigma重算全樣本，保留完整相同192節點target與counts；對少量事前選定的global/local參數，核對至少兩個scaled finite-difference steps、hard-bound方向及coarse/fine方向導數。這是計算相同objective的工程設計，不能近似刪除likelihood項。

下一輪如要再做actual optimizer smoke，必須另有Human授權及新的固定預算／run-id；本輪沒有放寬cap或retry。正式production整合前仍應處理API validation一致性、legacy診斷registration與版本文檔。其餘B0/B1科學限制保留，不重開target選擇，不將本輪scores輸入科學表格。
