# REV26-P1F-R1 獨立數值與授權範圍補充

審查者 `/root/p1f_r1_independent_review`。只讀既有 numerical records、source 與 hashes；本審查沒有呼叫 likelihood／gradient／optimizer 或修改 candidate。原 `INDEPENDENT_REVIEW_RAW.md` 保留其早期時間點，本檔補充已完成證據及處置。

## 已完成的獨立核對

- Gate C 原始與 sealed/current receipts保留。728個 local coordinates 各自 genuine full91 重算（coarse96），共67613 galaxy marginal evaluations；所有其他90星系變化最大0，local-vs-full最大0。實證是historical fixedpoints，不是將primary192改為96。source proof同時確認沒有hidden跨gal coupling。
- 當前6個source hashes與 `CURRENT_VERIFIED_DEPENDENCIES.json`逐一吻合，包含gradient/ledger自身。sole legacy closure AST change是 `fit_global.log_likelihood_per_galaxy` 完全disabled，其餘fit_global函數AST不變。B1 gradient未驗證路徑已reject，B1 pure evaluation仍在。
- 51個contract負控制PASS；pure／optimizer bounds分離、invalid native shapes/values/IDs、明確disabled registry與bad quadrature weights測試可讀。`models.py` bytes與P1F完全相同，所有canonical original law AST相同。
- 獨立用CSV parser重核2548 native fine/coarse rows、728 native mapping rows與144 neighborhood rows的數量、unique keys與finite values；全部符合。B0/B1各637，35個actual公開API路徑對照通過。max native parity8.810729923425242e-13、mapping9.094947017729282e-13、neighborhood1.0231815394945443e-12；fine/coarse max pergal8.988752242089504e-6、sample sumabs9.88420881853358e-6，維持原門檻。
- 7個預定方向×2 steps×2 orders=28 comparisons，無parity／accuracy／noise unresolved failure；actual9116 galaxy evals、64.2664012493 arithmetic-proxy fullsample equivalents、88.276265667s，未越96／300s上限。max scaled dep-vs-full1.9895196601282805e-8；fine/coarse最大7.953531167004257e-6；two-step最大0.0045229883133046656，對應NFW logM lower-bound one-sided方向，declared combined tolerance0.04088363850700804，最差error/limit0.1106307677。不能將它誤寫成所有absolute step differences都小於.001。max noise ratio4.807668149399694e-5，小於預定.1；兩model identical-theta repeat為0，但PASS不是只靠repeat。

## 全native gradient／成本與已保留的harness failure

| model | current galaxy evals | current proxy eq | current wall秒 | archived dense galaxy evals／eq／wall秒 |
|---|---:|---:|---:|---|
| PTQ-screen | 455 | 5 | 6.726141333 | 8645／95／128.840975379 |
| NFW-1p | 364 | 4 | 5.318691708 | 16744／184／254.181860296 |

上述是same1e-8 native-step comparator，重用P1F dense evidence，沒有重做95／184 full objectives。PTQ有1 baseline＋3 global全91 pass及91個local eval；NFW有1 baseline＋1 sigma全91 pass及182個local eval，所以semantic full91 passes為4／2，不能將base_calls1寫成total fullsample calls1。19／46倍proxy工作降低是明確arithmetic計數，19.16／47.79倍wall ratio是cross-run實測，不是固定未來保證。當前process RSS約148MB；未有同等archived dense peak RSS，不能發明完整memory speedup。

原生gradient與archived dense的最大差為7.958078644776378e-5／8.384404280570301e-5，在事先列出的whole-sum roundoff allowance內。這種tiny-FD比較只作分解一致性證據；7方向兩步長／orders才另外檢查精度，仍不證明全domain derivative error bound或gtol1e-6 stationarity。

`benchmark_v1` 在已完成NFW梯度後因signed-step comparator assertion中止：4個upper-bound U的archived signed h是負，new backward stencil依minus→base排序的分母為正；`(Fminus-Fbase)/(-h)=(Fbase-Fminus)/(+h)`。`ARCHIVED_BACKWARD_STENCIL_PROOF.json`逐一核對indices8/10/66/69的大小與backward_inward endpoints，故v2只修comparator符號比較而未修目標／梯度或放寬方向門檻。valid q v1與NFW v2結果合併，其失敗嘗試額外364gal／4eq成本也計入（gradient attempts總1183gal／13eq），沒有把失敗當成未執行。

**保留一項provenance缺口：** v1 failed-NFW的精確wall及當次warning count在assert前沒有persist，必須維持UNKNOWN，不能用v2冒充；現已保存原source／stderr／disposition，future harness改try/finally。此缺口不會否定完整成功梯度結果，但不得宣稱所有attempt telemetry完整。

## Warning與來源保護

- regression記錄492876 warnings，聚合成42 keys，count與category/message/source-line/model/version／representative hash保留，沒有存492876份重複字串。collector外warnings仍可見，exceptions繼續傳播；17warning後故意FPE的獨立控制已保存counts及exception。
- 8個預宣告代表點（兩model、兩galaxy、B0/B1）full192節點direct residual-squared versus fast sufficient-sum最大差1.9610979506978765e-12。只支持這些點的數值比對，root cause仍是 `UNRESOLVED_WARNING_ROOT_CAUSE`，無corruption confirmed證據，不宣稱所有warning無害。
- review提出的threading、hash及exception分類疑慮已作有限處置：collector要求serial Python main thread；num-validity只稱finite output，FPE記 `NUMERICAL_CALL_FAILED`而不宣稱warning造成corruption；固定config由manifest追溯。未驗證threaded parallel主目標。
- 本代理逐檔重hash2114個preexisting files，0 changed/missing；兩HEAD保持outer `3eb302666da7eccca19c5843f729e98126d68977`、implementation `5a8dca57f0fc3484387c643b51e22c7e10d024a8`。P1F原大型warning logs與P1D v1/v2沒有被改寫。

## 獨立裁決與下一範圍

- current candidate evaluator／bounds／diagnostic contracts：數值測試PASS；若以整個R1嘗試的provenance完整性作 `ENGINEERING_CONTRACT`，本代理保留 **CONDITIONAL**，僅因v1 comparator failure的past telemetry gap。此為主報告可明列並處置的審查保留，不要求重新進行728或統計target選擇。
- `DEPENDENCY_GRADIENT=PASS`：source／all-local full recomputation／same-native-step whole-gradient comparator均支持。
- `DIRECTIONAL_PRECISION=PASS`：限7個預宣告方向與兩步長／orders，不能當scientific stationarity證明。
- `NEXT_DRIVER_SMOKE=READY_FOR_AUTHORIZATION`：只支持下一個**工程**smoke，不是scientific MLE。

`NEXT_SMOKE_PROPOSAL.json`的兩個native historical starts、一次accepted iteration上限、L-BFGS-B jac=True、21 optimizer value-gradient batches/model、maxls20與600proxy eq／60000gal／1800s可容納完整bounded line-search嘗試及precision checks。這不保證一定accepted step；可因明確非budget的line-search／numerical failure停止。新driver必須實際用verified `scipy_value_and_gradient`與`collect_warnings`，不得直接用舊jac=None／event-list入口；R1舊driver現已入口disabled。initial/terminal精度、native decoding、hard bounds、每次work與exceptions都需保存。沒有授權本輪執行它，沒有canonical install、scientific refit、NFW multistart、MCMC、profiles或IC ranking。

AI審查同意不是科學正確性證明；Human Arbiter保留統計模型、production integration及fitting budget批准權。
