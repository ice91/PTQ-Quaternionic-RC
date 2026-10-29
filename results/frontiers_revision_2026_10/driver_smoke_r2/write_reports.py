# -*- coding: utf-8 -*-
"""Read-only analysis -> new reports. No further optimization or likelihood calls."""
from pathlib import Path
import json,hashlib
B=Path(__file__).resolve().parent;I=B.parents[2];R=I.parent;D=R/'docs/frontiers_major_revision_2026_10/rev26_p1f_r2_driver_smoke';S=B/'execution_v1'
def ld(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,s):
 p=D/name
 if p.exists():raise FileExistsError(p)
 p.write_text(s)
def main():
 a=ld(B/'analysis_v1/VALIDATED_RESULTS.json');models=a['models'];q,n=models;budget=a['total_budget'];protocol=ld(B/'SMOKE_PROTOCOL.json');dec=ld(B/'DECISION.json')
 verdict='\n'.join('- `'+k+'='+v+'`' for k,v in dec['verdicts'].items())
 table='| model | genuine accepted iterations | optimizer G batches | ΔF (decrease) | 10× numerical threshold | optimizer seconds | outcome |\n|---|---:|---:|---:|---:|---:|---|\n'
 for r in models:table+=f"| {r['model']} | 1 | {r['optimizer_gradient_batches']} | {r['objective_improvement']:.12g} | {r['improvement_threshold']:.12g} | {r['optimizer_wall_seconds']:.6f} | {r['status']} |\n"
 pt='| model / point | max pergal192/96 | sumabs192/96 | repeat | candidate/fullFD max scaled difference | max gradient error/limit |\n|---|---:|---:|---:|---:|---:|\n'
 for m in ['ptq-screen','nfw1p']:
  for point in ['initial','terminal']:
   x=ld(S/(m.replace('-','_')+'_'+point+'_AUDIT.json'));p=x['objective_precision'];ds=x['directions'];pt+=f"| {m}/{point} | {p['max_pergal_refinement']:.10g} | {p['sumabs_refinement']:.10g} | {p['repeat_max']:.3g} | {max(z['candidate_difference'] for z in ds):.10g} | {max(c['difference']/c['limit'] for z in ds for c in z['checks']):.9g} |\n"
 put('00_EXECUTIVE_VERDICT.md',f'''# REV26-P1F-R2 執行裁決

{verdict}

{table}

兩run皆由實際SciPy L-BFGS-B jac=True完成一個moving callback；returned terminal=callback=同theta之已完成optimizer request，獨立重算支持objective decrease，非單靠success訊息。兩者SciPy success=False/status1，因maxiter=1正常停止；projected gradient infinity norm為545.4470及24.4544，不是stationarity／MLE證據。

整個數值流程（起點preflight、FD、optimizer、終點驗證全包含）16344 actual/completed galaxy marginals、141.405024656 arithmetic-proxy equivalents、25,286,344,704 conditional-density grid nodes、205.470379042秒，遠低於600/60000/1800上限。每model僅4/21 optimizer G batches、一次accepted iteration；沒有重試／追加start。

R1 fullparity由exact source/data identity重用，fresh startingvector與terminal audits通過。警告1,002,600events/6keys完整彙整，rootcause仍UNRESOLVED_WARNING_ROOT_CAUSE；不能據finite output稱全面無害。2234個既有檔案hash不變，canonical與歷史資料、提交稿、preprint/public assets皆未改。

可另授權bounded P1G scientific re-estimation，僅表示driver可推進凍結target。當前smoke terminal均ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE，不能作新headlineparams、MLE tables、IC或scientificranking。本輪沒有scientific fitting campaign或canonical installation。
''')
 put('01_DRIVER_INTEGRATION_AND_CALL_GRAPH.md','''# 新 driver 及實際呼叫鏈

R1 candidate逐byte複製，僅新增src/ptquat/repaired_smoke_driver.py及scripts/run_repaired_gradient_smoke.py；八個R1修補接口與所有physical/model/gradient函數不改。完整diff見integration_from_R1.diff與INTEGRATION_FILES.json。

```text
newCLI main -> frozenprotocol/code/inputhash + DRIVER_UNIT_GATE
 -> run -> bothmodels freshpreflight PASS
 -> Runtime.optimize_once
 -> scipy.optimize.minimize(fun,...method=L-BFGS-B,jac=True,originalbounds)
 -> fun: samepoint Runtime.gradient(theta,model,'optimizer',192)
 -> verified R1 DependencyGradient.evaluate
 -> native _layout/_unpack -> full physical B0 galaxy marginal
 -> global all91 / local affectedgal only
 -> samecall (F,nativegradient) tuple
 -> actual callback hash matches completedoptimizerpoint
 -> independent Runtime.pure + two-step/order fullrecomputedFD audit
```

完整nativegradientshape/order/finite、sourcehash/samplehash、B0version、hardbounds每G查核；不是jac=None，也沒有perlocal fullsample denseFD。counter預建exact native endpoint/affectedgal sequence與每次work_callback逐一核對，保存FD endpoint nativehash及vectors。no incompatible parameter cache，僅R1單call baseline與geometry cache。

回傳F=-sum91 galaxy logmarginals；沒有prior、regularization、linear propagated covariance或B1混入。complete Gaussian normalization仍在每D/i conditional density。合適decomposition是sum_g log marginal，不是單一Gaussian Q/logdet或radius-wise marginal product。

單元測試用明確synthetic minimize stub檢查jac=True/callback/wiring、caps與telemetry，沒有真正optimizer或gal likelihood；controls_v1/v2及SYNTHETIC_NAMESPACE明記不可誤當actualsmoke。actual只有execution_v1之兩run。舊P1F/R1 jac=None driver一直disabled且未呼叫。
''')
 put('02_PREFLIGHT_PARITY.md',f'''# Pre-smoke evidence與parity

開始前R1/P1E/P1A SHA256SUMS逐檔驗證，bootstrap檢查2234preexisting files。新CLI硬驗SMOKE_PROTOCOL SHA、TESTED_CODE_HASHES與VERIFIED_INPUT_HASHES（dataset、兩原theta、receipt、weights、protocol）。six scientific/gradient source與R1 exactidentity，receipt匹配。

故重用R1 B0/B1各637、728native mappings、144neighborhood及35actualpublicpaths的PASS；PARITY_REUSE明記recomputed paritygal calls=0。沒有假稱再次跑637campaign，沒有隱藏validationwork。pure/optimizer contract、WAIC/LOO/radius firewall與newdriver控件另測通過。

新數值preflight並未省略：兩個原始theta各計完整nativeG與獨立full91 F192/F96/repeat及同7方向的兩步/order全重算FD，reference對P1E archived pergal最大差q={ld(S/'PREFLIGHT_GATE.json')['models']['ptq-screen']['reference_pergal_max']:.17g}，NFW={ld(S/'PREFLIGHT_GATE.json')['models']['nfw1p']['reference_pergal_max']:.17g}。兩者全PASS才啟動第一次真正minimize。

所有freshpreflight計入同一600eq/60000gal/1800s aggregate ledger；不是以validation標籤排除。source/input hashing及synthetic控件無actualgal/optimizerwork，另記0。

{pt}
''')
 put('03_FROZEN_SMOKE_PROTOCOL.md',f'''# 凍結protocol與hardcaps

SMOKE_PROTOCOL.json SHA256 `{sha(B/'SMOKE_PROTOCOL.json')}`，早於任何actual likelihood/optimizer。CLI以此digest拒絕改動。

Target REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1；SPARC91/1782；fine segmentedquantile192、coarse96只precision；原D/i normalizedphysical kernels、B0 floor、model laws、distance/template/NFW與原bounds/counts不變。

只有ptq-screen與nfw1p，各一個P1A originalbestvector，無額外starts。L-BFGS-B jac=True,maxiter1,maxls20,maxcor10,ftol1e-12,gtol1e-6。梯度固定boundspan1e-4；central可行否則inward，沒有看到line-search後改步長。gtol只是軟體option。

Caps：21 optimizerG batches/model、600 arithmetic-proxy fullsampleequiv、60000 actualgal calls、1800secondsaggregate、最多一callback/model。charge前查hardcaps、每gal持久化work/warningcheckpoint，request完成及audit結束另查wall；最後一gal若超wall也不得PASS。任何cap不retry不擴大。

精度：pergal192/96<=1e-4、sample sumabs<=.01、samepoint pergalrepeat<=1e-12、claimedfun<=1e-8；7方向/兩steps1e-4,5e-5同R1門檻，scaledcandidate vsindependent abs2e-5+rel1e-6、兩order/兩step abs1e-3+rel1e-3、noise ratio<.1。

Genuine acceptedstep需callback actualadvance、exact completedrequest、獨立fine/coarse/repeat/pergal/decodedbounds/gradientaudit通過；ΔF須>10×(initialsumabs+terminalsumabs+16eps*sumabsellbothpoints)。這是engineering辨識門檻，不是統計顯著性。
''')
 for r,name in [(q,'04_PTQ_SCREEN_SMOKE.md'),(n,'05_NFW_SMOKE.md')]:
  model=r['model'];res=ld(S/(model.replace('-','_')+'_RESULT.json'));audit=ld(S/(model.replace('-','_')+'_terminal_AUDIT.json'))
  put(name,f'''# {model} actual optimizer smoke

ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE

`{r['status']}`。只有一個原歷史start，4個actual optimizerG batches（initialrequest+3line-searchtrial），2個notaccepted trial，最後trial由actualcallback接受且nativevector確實改變。callback/returnedterminal/完成request的hash完全相同。

SciPy nit1,nfev4,njev4,status1,successFalse，message STOP: TOTAL NO. of ITERATIONS REACHED LIMIT。未繼續optimize，不以successFalse解讀成scientific failure；沒有stationarity主張。projected gradient∞norm={r['projected_gradient_inf_norm']:.12g}，仍非gtol收斂。

獨立ΔF={r['objective_improvement']:.17g}；numericalestimate={r['numerical_error_estimate']:.17g}；10倍threshold={r['improvement_threshold']:.17g}，ratio={r['improvement_over_threshold']:.9g}。claimedfunerror0，decomposition0，repeat0。終點max pergal192/96={r['terminal_precision']['max_pergal_refinement']:.17g}、sumabs={r['terminal_precision']['sumabs_refinement']:.17g}；完整初始/終點預定gradientchecks全PASS。

optimizer部分{r['optimizer_actual_galaxy_calls']}gal calls、{r['optimizer_fullsample_equivalents']:.12g}eq、{r['optimizer_wall_seconds']:.9f}s；平均G {r['optimizer_mean_gradient_seconds']:.9f}s。含preflight與terminal精度的model總{r['all_model_actual_galaxy_calls']}gal/{r['all_model_equivalents']:.12g}eq；preflight{r['preflight_wall_seconds']:.9f}s、terminalprecision{r['terminal_precision_wall_seconds']:.9f}s。

exact initial/decoded/bounds在{model.replace('-','_')}_INITIAL.json；全部trialvectors/gradients/objectives及接受分類在*_TRIALS；callback在*_ACCEPTED_CALLBACKS；returnedvector/scipystatus在*_RESULT。資料全標engineering用途，不寫入scientificMLE tables或改原theta。
''')
 put('06_ACCEPTED_STEP_AND_PRECISION_AUDIT.md',f'''# Genuine step與數值精度

ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE

{table}

兩個callback都不是unacceptedtrial或mere successmessage。driver記錄callback時查exactsuccessfuloptimizerrequest，保存向量後returnedterminal再次hash比對；postprocessing逐列核對native shapes/bounds與完整ledger。

獨立recompute全部91 fine/coarse/repeat，並以fullsamplefinite differences核對同7預定global/local coords，兩step/twoorder共56方向估計、56個refinement/stepchecks及14candidate-gradient comparisons。所有原門檻保留；沒有以iteration結果改step或換galaxy。

{pt}

最大candidate/fullFD scaled差1.31450406116e-8；最差gradient precision difference/combinedlimit約0.112863；最大affected numerator noise ratio4.8076681494e-5，未觸發.1 unresolved。最大的兩步absolute scaled差約.004523，依abs+relative combinedlimit通過，不誤稱全部absolute差<.001。bounds向內FD保持原native units，explicitmarginal只按galaxy分解。

Improvement error使用sumabs191/96 estimator加roundoff，不是嚴格積分誤差定理。兩個ΔF遠超10倍工程辨識門檻，僅證明受測step下F降低可辨識；不建立bestfit、confidence、q/scale/floor estimates、NFWconvergence或modelranking。full-domain／stationarity尚須P1G獨立診斷。
'''.replace('sumabs191/96','sumabs192/96'))
 put('07_WORK_AND_WARNING_TELEMETRY.md',f'''# Work與warning provenance

ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE

全流程actual=completed={budget['actual_galaxy_calls']}gal；charged=completed density-gridnodes={budget['charged_density_nodes']}；arithmetic-proxy equivalents={budget['charged_arithmetic_proxy_equivalents']:.17g}；wall={budget['elapsed_wall_seconds']:.12f}s。118完整objective-bearing requests，含10完整G（2preflight+8optimizer）與108standalonefull91objectives。actualminimize呼叫只有兩次。

Per-gal costs按R1 GALAXY_WORK_WEIGHTS逐model/order/gal計算，考慮nD/ni/nobs與NFW工作，不假設每gal同成本；proxy不是精確CPUcost。hash與weightedledger全核對。math.fsum比serialsum差約4.4e-12是roundoff，並非預算違反。

optimizer q2886gal/31.86927524eq/45.503830s；NFW2231gal/24.79163289eq/35.770055s。加所有起終點驗證總141.405eq，沒有藏在validation後排除。每model G cap4/21、iteration1/1；沒有超cap或retry。

WarningCollector實際啟用，1,002,600events只6keys，保存category/message/file/line/model/B0version、count、firstUTC、decodedhash及對應actual FD nativehash。checkpoint每gal保存此前warning摘要；request/exception/finally與signalhandler保存未完成狀態。最後started與completed相等，沒有未明pending；所有requests COMPLETE、actual runtime exceptions0。

UNRESOLVED_WARNING_ROOT_CAUSE仍保留。finite outputs不是證明warning無害，R1的direct residual-squared audited points與本輪precision提供有限數值支持。R1首次failedNFW benchmark缺失exactwall/count原樣未知，沒有拿本輪補造。

CAP/exception單元測試為synthetic、不調用realoptimizer/likelihood；controls namespace與actualexecution分開。telemetry不是硬體斷電transaction guarantee；Python受控中斷／exceptions有try/finally與逐galcheckpoint。最後一個failedmarginal若無完成證據，chargedattempts與completed分開，絕不把其預估nodes當observedcomplete。
''')
 put('08_INDEPENDENT_REVIEW.md',f'''# 獨立read-only review

使用者要求第二代理 /root/p1f_r2_independent_driver_review；沒有額外optimizer/likelihood/gradient，沒有改candidate或canonical。AI同意不是optimizerconvergence證明。

Rawreview `{sha(B/'INDEPENDENT_REVIEW_RAW.md')}`，原檔INDEPENDENT_REVIEW_RAW.md；numericaladdendum `{sha(B/'INDEPENDENT_REVIEW_NUMERICAL_ADDENDUM.md')}`，原檔INDEPENDENT_REVIEW_NUMERICAL_ADDENDUM.md。source-review明示早於actualstep，沒有事後改成已看過結果。

已核對jac=True同point F/nativeG、六scientificmodule與R1 identity、noincompatiblecache、local/global正確依賴、原bounds、actualacceptedcallback與terminalrequest匹配。ledger1..16344及request1..118連續唯一、B0target一致；allcallbacks與4audits/56方向records通過。獨立math.fsum重建ΔFq4.443505160975292/NFW2.120159661498292，與主紀錄差roundoff。

處置與保留：

- 增加requestfinish/auditend的wallguard，避免最後gal跨界仍PASS；已於freeze前修並再測。
- CLI除了testedcode，亦查dataset/theta/protocol/receipt/weights hashes；實際入口正確。public run helper不是通用替代CLI的受測授權入口。
- charged/attempted與completed/nodes須分開，已實作並成功全部相等。
- 失敗/中斷durability需finally/checkpoints；有控制測試，仍不宣稱硬體級crash consistency。
- unresolvedwarning、B0/B1 interpretation、conditional externalkernels、全bounds域gradientprecision皆保留。
- projectedgradient545.447/24.454及maxiterstop清楚列出，不把acceptedstep稱stationaryMLE。

第二審查支持四裁決PASS/兩acceptedvalidated/READY_FOR_BOUNDED_P1G_AUTHORIZATION，只授權可供Human下一步審查，沒有本輪scientificfitting。
''')
 put('09_SCIENTIFIC_CAMPAIGN_READINESS.md','''# Bounded P1G readiness — inactive

READY_FOR_BOUNDED_P1G_AUTHORIZATION。兩代表模型真實driver已接受可辨識下降的一步，並不保證其他起點、全domain或NFWmultistart收斂。canonical installation仍未授權。

INACTIVE_P1G_PLAN.json是可審核提案，沒有執行：七native參數數目baryon92,MOND93,linearPTQ93,PTQnu93,PTQscreen94,MONDscreen94,NFW183；原91/1782sample、bounds/laws與uniformexplicitB0不改。新scientificdriver只擴大已驗callable的iterationbudget，不回用disabled舊driver。

每native三starts：一P1A ORIGINALvector（不是R2smoketerminal），另seed260601/260602固定nativebounds20..80%均勻構造；matchedfamilies以同physicalpoints映射，七native輸出完整保存。開始前先凍結構造與numericalstationarity/KKT acceptance，不按fit結果調整。

待Human審查硬預算：非NFW每start20000eq/8h/2500G；NFW每start40000eq/16h/5000G；最多三starts/model，總480000eq/192h上限。maxiter500,maxls20,同jac=True原nativeFDpolicy。這是cap提案，不是預測收斂需192h；任何cap停、failedstart保留、無自動替換或retry。NFWmultistart只會由新的P1G明示授權後執行，當前零。

每個start先過192/96numericalprecision，不過就BLOCKED_START不得換更有利start；endpoint保存完整theta、pergal/F/G/variance normalization、work/warnings、boundcontacts及獨立projectedgradient/KKT/兩step/order。小gtol/ftol或SciPysuccess不足證stationarity；不確定就PROVISIONAL_TERMINAL_NOT_VALIDATED_MLE。NFW需獨立convergence/multistart/halo-boundary review。matched PTQ/MOND每start/terminal物理點identity仍<=1e-8。

B1敏感度預先固定為全部七native endpoint×91的固定physicalpointpureB1評估；B1 optimization尚無verifiedgradient，不在此plan授權。不得看到B0ranking後才挑family做敏感度。

沒有CI/profiles/MCMC、manuscript/canonical/Git/public修改。IC只可在新validatedmaxima且另核准後計算，不能用R2scores；point-count BIC對within-gal correlations屬descriptive convention，不當effective-sample-size證明。
''')
 put('10_NEXT_EXECUTION_PROMPT.md','''# 唯一建議下一階段 prompt

```text
REV26-P1G — Bounded seven-native uniform-explicit B0 re-estimation

需Human另行明示授權本計畫才執行；R2未授權scientificfit或canonical installation。讀完整R2、R1、P1E及INACTIVE_P1G_PLAN.json，建立全新isolated scientificdriver/results/docs；過去evidence、data、theta、submitted/public、canonical唯讀。

保持REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1、91/1782、七native counts與originalbounds/laws、segmented192/96、原native dependencygradientpolicy。以R2已驗samepoint F/nativeG tuple接jac=True；開WarningCollector、source/receipt/data/workhash gates，先驗七modelcallpaths及matchedfamilyidentity。不得重開target或修改理論。

開始前凍結完整科學fitprotocol、KKT/projectedgradient與numericalderivativeuncertainty acceptance、start seeds與stop/failurepolicy。每model一P1A ORIGINALwarmstart加seed260601/260602兩個預定independentstarts（nativebounds中央20..80%，matchedfamilies同physicalpoints），保留七native完整輸出；禁止將R2terminal悄當scientificstart。

依INACTIVE_P1G_PLAN：非NFW每start<=20000eq/8h/2500G，NFW<=40000eq/16h/5000G，最多三starts/model、aggregate<=480000eq/192h；maxiter500/maxls20，任何cap停止且不retry、不更換失敗start。Human批准本P1G時才包含此明列NFWmultistart，當前不得先跑。

每start及terminal需192/96refinement與finitechecks，完整F/pergal/nativeG/bounds/step/warning/failureprovenance。小gtol/ftol或success不能取代獨立stationarity/KKT及兩步/order精度；不充分就標PROVISIONAL_TERMINAL_NOT_VALIDATED_MLE，不強作MLE表。NFW由read-only獨立review核對multistart、halo boundary與convergence；identity映射點誤差<=1e-8。

B1敏感度在看B0ranking前凍結，全部七nativeendpoint/91gal做fixedphysicalpureB1；不授權B1optimization。不得CI/profiles/MCMC、manuscript/canonical/public/Git/journal操作；不以smokescores算IC，point-countBIC若未來另核准只屬相關資料下descriptive convention。

回報每start數值與科學acceptance分開、全部failed/budgetlimitedcases、完整來源hash與cost；Human保留科學解讀、canonical與publication批准權。
```
''')
 put('SOURCE_PRESERVATION.md',f'''# Source preservation

outer root {R} HEAD3eb302666da7eccca19c5843f729e98126d68977；implementation是獨立root {I} HEAD5a8dca57f0fc3484387c643b51e22c7e10d024a8。AUTHORITY/GIT_AFTER保留worktrees、dirty/untracked；既有dirtyfiles完全不清理，無checkout/reset/stash/pull/commit/push/tag/release。

SOURCE_HASHES_BEFORE有2234preexistingfiles，開始前R1/P1E/P1A manifests逐hash核對，結尾再核。R1源碼、舊smoke/logs、缺失telemetry狀態、P1D failedv1與correctedv2、歷史MLE與data、submitted/preprint/public皆未改。

candidate源科學與gradient modules逐byte等於R1，僅新增newdriver/CLI。TESTED_CODE_HASHES、VERIFIED_INPUT_HASHES、receipt/weights/target/protocolhash、ENVIRONMENT與pipfreeze、integrationdiff、completeactualrequests/FDpoints/trials/accepted/terminal/precision與budget/warningfiles均保存。

actualwork全部execution_v1，onlytwoactualminimizecalls。controls_v1/v2為明標syntheticstubs，不是additionaloptimizerstarts。unitv2是新增postwallguard後再測，未跑actualfit；actualsmoke只有一次、無retry。所有optimizer資產標ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE。

數值counter始於freshinitialpreflight，所有相關precision計入同aggregatecaps；reuseR1parityexacthash明記0additionalgalcalls。沒有源碼或數值補丁在actualrun中途加入，testedhash與封存hash一致。

完整新資產僅在兩個R2roots，MANIFEST/SOURCE_MANIFEST/FAILURE_MANIFEST及SHA256SUMS給機器可讀身份。既有科學sources不覆寫，當前沒有scientificcampaign、NFWmultistart、CI/MCMC、manuscript/canonical或public/editorial行動。
''')
 print('created12reports',D)
if __name__=='__main__':main()
