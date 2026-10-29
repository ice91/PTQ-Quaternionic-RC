# REV26-P1F-R2 獨立唯讀 driver/source 審查

審查代理 `/root/p1f_r2_independent_driver_review`，由Human任務明示要求。時間點 2026-10-28T23:37:19.672399Z：source已凍結、synthetic controls通過，但尚未讀取任何actual accepted-step結果。本稿不預判optimizer能否advance，也不把AI同意當科學證明。

新driver SHA256 `306bc0bc004ded9293ea5d41d6c546bcac49b5c271bef6b8df6312d037686f31`；CLI SHA256 `f34fa2b37c9396b80389e961308b4f70713bf9872117fa5719f6b048879a7a7f`；frozen protocol SHA256 `aefe3ebc8f0500ae3fd79eb39305da7e3025cc2b237d9df7dac0c7be3db4f6dc`。

## 獨立確認

- Receipt綁定models.py、fit_global.py、mle.py、marginal_likelihood.py、evaluation_contract.py、dependency_gradient.py六檔hash，全與R1逐byte相同。物理likelihood、D/i kernels、distance/template/NFW laws及192/96 nodes沒有改。
- New actual call path為新CLI → Runtime.optimize_once → scipy.optimize.minimize(fun,method=L-BFGS-B,jac=True,bounds=original) → Runtime.gradient → verified DependencyGradient.evaluate。fun同一批次返回負sum log-marginals與完整native g，沒有jac=None denseFD driver。
- Runtime.gradient建立expected evaluation sequence，逐次核對model/gi/order與baseline或真實FD endpoint native theta hash；source/sample/receipt由R1 engine再次驗，global all91、local僅verified affected IDs。theta.copy避免slice view改baseline，無跨theta parameter-valued cache。
- Hard guards在每gal前：600proxy equivalents、60000actualgal；每model21optimizer G batches；aggregate1800s；callback最多1 acceptediteration。已依review建議補successful G/pure/audit/summary後wallguard，防止最後gal開始未超時卻完成後超時還宣稱PASS。
- WorkLedger明確分charged attempted與completed，前一gal在下一charge時證明完成，最後gal只於successful request確認；failed lastgal不補造completed density。所有initial numerical preflight與terminal/gradient precision均走同一ledger、同一wall clock。
- 每fun入口持久化REQUESTED向量，例外前後保存；每galcheckpoint warning aggregate與currentbudget；try/finally、SIGTERM/SIGINT紀錄與finalsummary保留。CLI驗frozenprotocol、candidate testedcode、dataset、原theta、receipt、weights hashes。
- Actualaccepted證據必須是callback xk與某成功optimizerrequest hash完全一致、相對原theta實際移動；returned terminal要與acceptedcallback一致。不是僅用SciPy success或trial score。
- initial/terminal獨立pure192/96/repeat、完整pergal sum、bounds/native decode、同R1七方向兩step/fullFD比較；threshold不改。accepted objective decrease須超過10×宣告refinement+roundoff estimator，否則NUMERICALLY_AMBIGUOUS。gtol/oneiteration不代表MLEstationarity。
- 原legacy WAIC/LOO/radiusfactorization仍disabled。B1未進optimizer；不重開target選擇。

## 異議與限制及處置

1. **已處置：最後numerical call跨wallcap。** 補post-success wallguards；若極小overshoot仍須budgetlimited，不得以已完成numeric就PASS。
2. **已處置：cap阻止的新funrequest也要記錄。** fun入口先記向量及REQUESTED，failed attempt持久化；budget檢查前先確認此前成功gal完成。
3. **已處置：terminal precision失敗仍保留SciPy return。** failure summary留SciPy status/nit/nfev/fun與acceptedvectors，不把一個已發生callback隱藏掉。
4. **保留：failure最後一gal完成度有可能未知。** ledger已區別charged/complete，屬保守count而非編造quad執行數；結果報告不得把failed chargednodes稱已完成nodes。
5. **保留：warningrootcause未知。** 聚合不suppress、point provenance有decodedhash及actual FD nativehash；finite/outputcheck不是全域harmless證明，R1缺失telemetry不得補造。
6. **保留：refinement estimator非rigorous full-domain bound；scaledFD accuracy不支援直接宣稱gtol1e-6科學stationarity。** 只支持bounded工程advance的有限檢查。
7. **保留：durable紀錄依Python finally及atomicreplace/flush。** 正常SIGTERM/SIGINT可保存；不能保證SIGKILL/powerloss當下未flush資料。不得把此審查寫成硬體級crash-consistency證明。
8. **保留：source入口run(bundle)可被程式化呼叫，但已審核/授權入口是驗hash的新CLI。** 證據應記實際CLI，不能換另一entrypoint仍沿用已驗證說法。

## Source-review裁決

沒有發現需要阻止本次兩個預宣告bounded engineering smoke的fatal scientific或driver-wiring缺陷。只有當兩model初始preflight都通過後才可呼叫actualoptimizer；是否接受步、數值有效性及P1G授權須待實際evidence另判。未執行任何objective/gradient/optimizer，未修改candidate/canonical或任何先前證據；只新增本審查檔。
