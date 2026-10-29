# REV26-P1F-R2 獨立 actual-smoke 數值補充

審查代理 `/root/p1f_r2_independent_driver_review`，時間 2026-10-28T23:42:26.524963Z。只讀JSON/JSONL、hash與source；沒有額外likelihood、gradient或optimizer呼叫，未改candidate/canonical或任何原始evidence。此稿補充原先pre-step rawreview，沒有改寫原稿的時間點。

## 實際 driver 與 accepted-step 獨立核實

- finaltested driver SHA256 `306bc0bc004ded9293ea5d41d6c546bcac49b5c271bef6b8df6312d037686f31`。23個testedcode檔及VERIFIED_INPUT_HASHES全部再次核實；六個scientific/gradient模組與R1相同。rawreview之後新增OBJECTIVE_PRECISION logging持久化屬紀錄功能，actual executed code有test/hash綁定，沒有target或derivative規則變動。
- 兩model各4次actual optimizer G批次、nfev=4、njev=4、nit=1；各一個moved callback。callback theta、returnedterminal、成功optimizerrequest hash與fun完全一致。
- 每model trial序列：INITIAL_REQUEST、兩個NOT_ACCEPTED_LINESEARCH_TRIAL、最後ACCEPTED_CALLBACK。沒有把trial當accepted，沒有額外starts或retry。
- SciPy success=False是maxiter=1停止，兩個projected native gradient infinity norms分別545.4470324933989與24.454350502352334；不構成stationarity或MLE convergence。

| 模型 | math.fsum獨立重建 F0-Fterminal | 10倍數值error threshold | improvement/threshold | 裁決 |
|---|---:|---:|---:|---|
| ptq-screen | 4.443505160975292 | 3.9610720713275367e-7 | 約1.12179e7 | ACCEPTED_AND_NUMERICALLY_VALIDATED |
| nfw1p | 2.120159661498292 | 1.3977977304991764e-6 | 約1.51679e6 | ACCEPTED_AND_NUMERICALLY_VALIDATED |

這些是工程objective decrease，不能當新correctedMLE或modelranking。math.fsum與原np.sum差只有約1e-12 roundoff。

## 初始／終點精度與梯度

四個audit全PASS；每個q點4方向、每個NFW點3方向，兩step×兩order，共56條direction/order/step records完整。candidate與independent fullsample FD、fine/coarse、兩step及noise都按凍結門檻通過；沒有單憑repeat0宣稱gradient正確。

| point | max per-gal refinement | sumabs refinement | repeat | max candidate scaled差 | 最差accuracy difference/limit |
|---|---:|---:|---:|---:|---:|
| q initial | 1.5049383961e-8 | 2.3702202867e-8 | 0 | 2.52243e-9 | .0273886 |
| q terminal | 7.5696888757e-9 | 1.5860891267e-8 | 0 | 8.52651e-10 | .0597600 |
| NFW initial | 4.5006984806e-8 | 7.0315197220e-8 | 0 | 3.41061e-9 | .110631 |
| NFW terminal | 4.8146802101e-8 | 6.9418058857e-8 | 0 | 1.31450e-8 | .112863 |

最大noise ratio4.8076681494e-5，低於.1 unresolved rule。所有claimed objective error=0，normalized pergal sum使用same np.sum decomposition error=0。bound-aware stencil與native units保留原規則，沒有事後縮步挑有利結果。

## Work 與 warning ledger

- JSONL事件ID1..16344唯一連續，逐筆galaxy ID/order/weights/nD×ni全部核對。
- actual=completed=16344gal marginals；118個complete requests，含10個gradient request（2preflight+8optimizer）與108個pure objective requests。
- quad density nodes=25,286,344,704，charged=completed。沒有pending lastcall或未保存failure。
- arithmetic proxy獨立math.fsum=141.4050246562196；stored141.40502465621523，約4.4e-12累加roundoff，不是額外工作。遠低600，actual遠低60000，aggregate205.470379042s低於1800。
- optimizer只用了56.6609081253proxy；initial/preflight與terminal精度約84.7441proxy也納入aggregate，沒有藏在validation標籤外。
- warnings總1,002,600、6keys，逐keycount相加吻合；exceptions=0，status為UNRESOLVED_WARNING_ROOT_CAUSE與FINITE_OUTPUT_ONLY_NOT_A_HARMLESSNESS_PROOF；每key有representative nativehash。R1舊missing telemetry沒有補造。
- source baseline2234檔獨立逐SHA256重查，0變更。dataset、originaltheta、P1A–R1、canonical與public assets未改。

## 保留意見與後續邊界

source-review列出的finite/refinement非全bounds誤差證明、warningrootcause未知、非thread-safe Python capture與硬體級crash-consistency限制仍保留。一步工程advance不能證NFW長程convergence、globaloptimum、q/scale/floor estimates、CI/profile或correctedIC。

兩個smoke結果足以支持**另授權 bounded P1G launch review**，前提是Human核准其七native starts/budgets、完整終點vectors、stationarity/KKT、terminalquadrature precision、failedstart報告、NFW獨立review及B1 sensitivity計畫。本輪沒有執行這些scientificfits。

支持分開裁決：DRIVER_WIRING=PASS；PTQ_SCREEN_STEP=ACCEPTED_AND_NUMERICALLY_VALIDATED；NFW_STEP=ACCEPTED_AND_NUMERICALLY_VALIDATED；SCIENTIFIC_CAMPAIGN_READINESS=READY_FOR_BOUNDED_P1G_AUTHORIZATION。這是依可重算紀錄的工程判斷，AI同意不是optimizerconvergence證明。
