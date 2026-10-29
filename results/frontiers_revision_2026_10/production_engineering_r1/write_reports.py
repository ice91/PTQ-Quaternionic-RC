# -*- coding: utf-8 -*-
"""Evidence reports only; no likelihood/optimizer calls."""
from pathlib import Path
import json,hashlib
B=Path(__file__).resolve().parent;I=B.parents[2];R=I.parent;D=R/'docs/frontiers_major_revision_2026_10/rev26_p1f_r1_engineering'
def ld(p):return json.loads(p.read_text())
def put(name,s):
 p=D/name
 if p.exists():raise FileExistsError(p)
 p.write_text(s)
def main():
 verdict=ld(B/'DECISION.json');dep=ld(B/'dependency_v1/VERIFIED_DEPENDENCIES.json');dg=ld(B/'directions_v1/DIRECTION_GATE.json');pg=ld(B/'parity_v1/PARITY_GATE.json');bench=ld(B/'BENCHMARK_SUMMARY.json');warn=ld(B/'warning_audit_v1/AUDIT.json');ct=ld(B/'contracts_v2/TESTS.json');checks=ld(B/'directions_v1/precision_checks.json');raw=ld(B/'directions_v1/all28_records.json');ledger=ld(B/'PARAMETER_DEPENDENCY_LEDGER.json')
 costtable='| native model | dense full91 passes | dependency full91 passes + local calls | galaxy calls | equivalents | seconds | work saving |\n|---|---:|---|---|---|---|---:|\n'
 for x in bench['models']:
  costtable+=f"| {x['model']} | {x['dense_fullsample_calls']} | {x['semantic_full91_passes']} + {x['local_extra_galaxy_evaluations']} | {x['dense_actual_galaxy_evaluations']} → {x['dependency_actual_galaxy_evaluations']} | {x['dense_fullsample_equivalents']} → {x['dependency_fullsample_equivalents']:.6g} | {x['dense_archived_wall_seconds']:.6f} → {x['dependency_wall_seconds']:.6f} | {x['work_speedup']:.1f}× |\n"
 dtable='| 模型/index/role | scheme | max dep/full scaled差 | max fine/coarse scaled差 | max兩步scaled差 | max noise ratio |\n|---|---|---:|---:|---:|---:|\n'
 for direction in ld(B/'DIRECTION_PROTOCOL.json')['directions']:
  m=direction['model'];j=direction['index'];rs=[x for x in raw if x['model']==m and x['index']==j];cs=[x for x in checks if x['model']==m and x['index']==j];role=rs[0]['name'];dtable+=f"| {m}/{j}/{role} | {rs[0]['scheme']} | {max(x['scaled_parity_difference'] for x in rs):.9g} | {max(x['scaled_difference'] for x in cs if x['comparison']=='fine_coarse'):.9g} | {max(x['scaled_difference'] for x in cs if x['comparison']=='two_steps'):.9g} | {max(x.get('noise_ratio',0) for x in cs):.9g} |\n"
 statuses='\n'.join(f'- `{k}={v}`' for k,v in verdict['verdicts'].items())
 put('00_EXECUTIVE_VERDICT.md',f'''# REV26-P1F-R1 工程裁決

{statuses}

純likelihood／硬界限objective／局部decoded／diagnostics接口已修復；同一physical B0 target、192/96 policy與所有物理模型不改。七模型全部728局部座標的全樣本重算通過，未受影響星系差0。B0/B1各637、728映射、144鄰域與35公開fullsample paths回歸通過。

{costtable}

七方向、兩步長、192/96共28比較通過，實際9116次galaxy marginal、64.2664012493 arithmetic-proxy fullsample equivalents、88.276265667s；未超96/300s。沒有optimizer、科學MLE、NFW starts、MCMC、profiles、IC或canonical installation。

ENGINEERING_CONTRACT保留CONDITIONAL，原因是首次NFW benchmark比較腳本在signed-step assertion失敗前未持久化exact wall/warning count；這些缺失沒有補造。符號等價已獨立核對，NFW成功補算與當前完整數值證據通過；future runner用try/finally、合成失敗telemetry負控制通過。此保留不等同likelihood defect。

可另授權新的、使用repaired gradient與aggregated warnings之bounded工程smoke，含restricted NFW，最多一次accepted step。舊P1F driver已disabled，不可直接重跑或稱它已驗新gradient。這不是READY_FOR_SCIENTIFIC_MLE；native FD精度與stationarity、跨整個boundsdomain有效性均未普遍證明。Human仍保留canonical integration及scientificfit批准權。
''')
 put('01_EVALUATOR_AND_BOUNDS_CONTRACT.md',f'''# Evaluator 與 bounds 契約

起點為P1F isolated candidate，非canonical。完整8檔diff為`integration_from_P1F.diff`，`INTEGRATION_FILES.json`記錄前後hash。新增evaluation_contract.py、warning_telemetry.py、dependency_gradient.py；修改mle.py、fit_global.py、marginal_likelihood.py、experiments.py、bounded_driver.py。models.py逐byte不變。

A 純scalar/component evaluation：必須七native model、非空且ID唯一sample、exact native theta shape與finite值。保留_layout/_unpack，純評估不以MLE hard bounds當prior，不clip。物理變換產生nonfinite／非法floor／資料形狀錯誤仍raise。

B optimizer objective：A加原始theta_bounds；bounds外直接ValueError，不clip、不penalty fallback。q/sigma在ln、MOND a0在log10、NFW M200在log10，gradient回native coordinate。沒有加prior或regularization。

C galaxy-local：mle.galaxy_loglike對fullnative theta先共用A，ID可exact name或非負integer；negative、bool、float、未知name及out-of-range拒絕。decoded API檢查U/lM shape、index、globals與sigma finite/positive，catalog arrays同長finite與合法D/i。單GalaxyData接口不自行推斷外部fullsample身份，sample/hash/ID mapping在上層受驗。

D diagnostics：experiments明確disabled registry取代runtime AST；public fit_global.log_likelihood_per_galaxy所有版本含LEGACY均disabled。合法的是per-galaxy log-marginals與標示descriptive residuals。沒有radius marginal factorization、WAIC/LOO ranking。MCMC CLI、舊multistart/writer及舊bounded driver入口停用。

`contracts_v2/TESTS.json` {len(ct['cases'])}個negative controls通過，另normalized kernels、Gaussian、no-double-count、units/laws等舊完整tests通過。純native validator允許U=1.6（僅示範pure/outsidebound差異），objective拒絕且原theta未clip。原參數數目92/93/93/93/94/94/183與bounds完全保留。

最後legacy-radius封閉在方向計算後完成；`LEGACY_CLOSURE_AST_PROOF.json`證明其他所有fit_global函數AST未改，舊source快照保存，新的CURRENT_VERIFIED_DEPENDENCIES receipt完整綁定最終source。方向數學結果沒有改寫；execution snapshot與metadata sealing鏈皆保留。
''')
 put('02_PARAMETER_DEPENDENCY_LEDGER.md',f'''# 精確參數依賴

`PARAMETER_DEPENDENCY_LEDGER.json`有742 native coordinates：728 local、14 global。`PARAMETER_DEPENDENCY_MATRIX.json`給每model之k×91 matrix；每列含index/name/model、native transform、bounds、affected IDs、cache invalidation與source理由。

source證據：_unpack回傳scalars與theta U/lM slices；catalog_velocity_at_distance只取U[gi]與lM[gi]；c(M200)只依該gal M200與固定c0/c_slope；沒有跨gal pooled mean、共同localprior或全樣本normalization。eps/log10a0/lnq與sigma為globals。kernel只依每gal catalog geometry，不依其他gal theta。

每個模型91個Ulocal，NFW額外91個halo local；每一座標依預宣告native span*1e-4向內擾動，真正重算全部91個coarse96 marginals，再單獨local重算及重建total。全部728座標、67613次gal evaluations、{dep['wall_seconds']:.6f}s。max unaffected=0，max local/full=0；沒有hidden coupling。完整728×91向量存`dependency_v1/full_vectors_all728.json`，不只摘要。

coarse96只為已宣告dependency verification，主objective仍192。獨立源碼indexing論證加7方向fine192 full comparator提供相互支持，不把coarse結果聲稱成全部參數域quadrature精度證明。

原receipt、第一次sealed與CURRENT receipt皆保留。獨立review要求綁定dependency_gradient.py本身並限B0 gradient，已修；ledger/step/sample函數AST在封存時一致。任何source、sample arrays、model/coverage不匹配均failclosed。B1純likelihood支持，但B1 gradient沒有冒用B0 dependency receipt。
''')
 put('03_DEPENDENCY_AWARE_GRADIENT.md',f'''# Dependency-aware native finite differences

F(theta)=-sum_g ell_g(theta)。local座標j只計其affected set A_j，故gradient=-d(sum_A ell)/dtheta_j；global A為全部91。這是同一gal marginal的代數分解，沒有按radius factorize或改conditional normalization。

DependencyGradient.evaluate先驗receipt/source/sample/version與原hardbounds；只在單次呼叫建立exacttheta baseline。每個stencil採theta.copy，只重算affected galaxies；global變動重算全部。geometry nodes可按原keys重用；沒有任何跨呼叫parameter mean/variance/marginal cache。sample/source每call核對。complete gradient依native index順序回傳，scipy_value_and_gradient給(F,full native array)，subset不得當SciPy完整jac。

兩步長為boundspan*(1e-4,5e-5)。可central時用central；否則向內單邊，不clip、不自動縮step。實際浮點endpoint之差作分母。upperbound backward表示(Fbase-Fminus)/(+h)，與P1F的(Fminus-Fbase)/(-h)相同。

原生完整gradient benchmark只在ptq-screen/NFW原始theta各算一次same-step1e-8，與已封存P1F完整dense gradient比；沒有再次做95/184個full objectives。max native difference為7.95808e-5／8.38440e-5，受fullsum subtraction roundoff影響，符合先宣告gamma91*2sumabs(logmarginals)/|h|，不拿這個寬roundoff bound宣稱實際derivative精度。真正精度證據另見7方向兩步／兩order。

{costtable}

本輪沒有minimize或任何optimizer呼叫。API可供下一個新driver接jac=True，但舊driverdisabled。generic ledger含七模型；fullnative gradient benchmark與方向accuracy只覆蓋上述兩歷史點，不外推所有七模型全bounds-domain或gtol=1e-6 stationarity。
''')
 put('04_FIXED_DIRECTION_PRECISION.md',f'''# 凍結方向與精度

PREDECLARED_CONTRACT與DIRECTION_PROTOCOL先於方向計算。7方向只依role、固定F574-1幾何、firstnative boundcontact與firsthalo決定，沒有看gradient sign/residual/fit improvement。qfamily涵蓋eps、lnq、lnsigma、localU；NFW涵蓋lnsigma、boundcontactU與local logM。

兩步長native boundspan*1e-4與5e-5，192/96同segmentation。dep/full agreement：scaled abs2e-5+rel1e-6；fine/coarse與兩步agreement：scaled abs1e-3+rel1e-3。scaled derivative=原native derivative×boundspan。noise檢查使用stencil上affected marginals的sumabs fine/coarse差／fine numerator；已證unchanged terms精確相消，不把共同fullobjective offset誤算成local derivative noise。ratio>=.1即unresolved，沒有放寬。

{dtable}

全部28 dep/full comparisons及28 precision checks通過。max scaled dep/full {dg['max_scaled_parity_difference']:.12g}；max fine/coarse {dg['max_scaled_fine_coarse']:.12g}；max two-step {dg['max_scaled_two_steps']:.12g}。最後者是NFW lower-bound halo one-sided方向，對應combined limit約0.040884，最差error/limit約0.11063；不是聲稱所有absolute差<.001。max noise ratio {dg['max_noise_ratio']:.12g}，無unresolved。finebaseline repeat每gal差0。

實際9116gal evaluations、64.2664012493成本proxy等價、88.276265667秒；上限96/300未觸發。work proxy逐gal按nD*nI*(2*nobs+10)+factor*nD*nobs，factor NFW100/其他30，normalize同model全91 fine192成本；coarse按實際節點。它是明示arithmetic proxy，不是假設每gal同成本，也不是精確CPU prediction；actual timing另列。weights、worklog及每pair raw結果完整保存。

數值refinement不是全bounds誤差定理；repeat0也不是gradient proof。這些證據支持下一工程smoke的有限差分政策，尚不足宣稱科學stationarity或uncertainty。
''')
 put('05_WARNING_TELEMETRY.md',f'''# Bounded warning telemetry

warning_telemetry採stream aggregation，每個(category, normalizedmessage, sourcefile,line,model,version)一key，保存exact count、首次UTC、代表decoded-point hash與validity狀態；max256 keys，溢出failclosed。沒有每event list。with collect_warnings顯式啟用；未啟用時Python預設warning仍可見，不globally suppress RuntimeWarning、不吞exceptions。

當前僅serial Python main thread，threaded capture會拒絕；ContextVar並不使process-global warnings.showwarning thread-safe。source/config/data由manifest綁定；point hash是decoded parameters/index，固定H0/c0/geometry在其他manifest，未宣稱hash單獨包含所有外部配置。

回歸492876個events只42keys；benchmark q32610、NFW26088 events有完整成功紀錄。class為UNRESOLVED_WARNING_ROOT_CAUSE，finite-output status不表示harmless。8個預宣告F574-1/IC4202×q/NFW×B0/B1完整192 direct residual-squared比較，max|delta logL|={warn['max_abs_delta']:.17g}；無已確認污染，但不能推論future全部theta。

合成控制17同類warning聚合為1key；另一負控制17warnings後deliberateFPE仍保存17counts/exception且向外傳。FPE標NUMERICAL_CALL_FAILED，不自動推成warning造成NUMERICAL_RESULT_CORRUPTION_CONFIRMED。

首次NFW benchmark harness在符號assert前未persist該attempt warning count及exactwall，明列UNKNOWN。P1F約1.29m/model完整eventlogs原樣保留；本輪缺口不以P1F或v2數值補造。future harness已加finally持久化warnings/exception，v1/v2執行source與stderr保留。此provenance限制是ENGINEERING_CONTRACT=CONDITIONAL的原因。
''')
 put('06_FULL_PARITY_RESULTS.md',f'''# 完整回歸

{pg['status']}。原P1F門檻完全不變。

- B0 637、B1 637，fine/coarse共2548 complete finite unique records，exactkeys一致。
- Native matched mappings728，max|delta|={pg['max_mapping_delta']:.17g}。
- Neighborhood/floor 144，maxcandidate/P1E delta={pg['max_neighborhood_parity']:.17g}。
- Native parity max|delta|={pg['max_native_parity']:.17g}。
- Max單gal refinement={pg['max_refinement_pergal']:.17g}，max model sumabs={pg['max_refinement_sumabs']:.17g}。
- 35個實際fullsample API比較、每model first/last local報告、descriptive residuals與predictivefirewall均通過。
- normalized physical kernels、zero-error conditional limit、full Gaussian normalization、distance/template/NFW/angle units、no covariance doublecount與原模型AST保持均通過。
- full91 NFW同theta repeat max=0，所有2114既有檔案再hash不變。回歸耗時{pg['elapsed_seconds']:.6f}s。

每個attempt/row與錯誤控制保存在parity_v1，沒有隱藏failedfit或更改source/reference來取得pass。dep/full objective一致，方向gradient比較依原凍結門檻通過。B1僅pure evaluation regression，未聲稱B1gradient或新fit。全部為原始歷史固定點，不是correctedMLE。
''')
 put('07_COST_AND_SPEEDUP.md',f'''# 成本與範圍

{costtable}

q的完整全91 passes是baseline1+global3=4，另91local；NFW是baseline1+sigma1=2，另182local。不可把base_calls=1誤寫為total fullsample passes=1。actual gal counts與proxy equivalents分列。

完整同step梯度成功工作總819gal、9equiv；加第一次NFW comparator harness失敗後已完成的364gal/4equiv，所有attempt總1183gal/13equiv。failedattempt exactwall未persist，不能把成功兩次wall總和當allattempt總wall。沒有花額外190fullcalls重做舊dense；dense成本沿用P1F samevector同step原始driver。

cross-run wall speedup約19.16×/47.79×，受warninglogging、allocator與執行環境差異影響，不是保證演算法固定倍率。物理依賴所節省的work則明確為19×/46×。成功benchmark RSS約148.4MB，direction process峰值180.3MB，沒有每warning-event記憶體累積；不將這個RSS當舊P1F精確peak比較。

arithmetic proxy及各gal nobs/nD/ni記在GALAXY_WORK_WEIGHTS；coarse96工作約fine的四分之一但依真實segments，並非每gal成本相同。對下一新central-gradient driver，在任何bound配置每batch最多q9eq、NFW7eq；這是kernel work上界，不是convergence時間預測。

新smoke提案600eq／60000gal／1800秒，兩model各21optimizer gradient-batches加最多6precision-batches的保守工作<=432eq、39312gal，容許maxls20完成第一iteration或明確非budget line-search失敗。未執行。
''')
 put('08_INDEPENDENT_REVIEW.md','''# 獨立唯讀審查

使用者明示要求的第二代理 /root/p1f_r1_independent_review，未修改candidate/canonical，未呼叫optimizer或額外高成本objectives。原始source-review與後續numerical addendum分檔保留；AI同意不是科學證明。

處置：

1. Receipt漏綁gradient module：修正，增加module hash、保留舊receipt與ledger/step/sample AST一致證據。
2. B1未驗dependency卻可要求gradient：修正，gradient限定B0；B1 purelikelihood仍支持。
3. Public LEGACY radius evaluator可繞過shape契約：完全disabled，原source快照及非相關functions AST一致證據保存。
4. InwardFD分母符號：獨立證明舊負h與新正denom+反向numerator等價；4個NFW上界indices與scheme逐筆核實，沒有改梯度公式／precision門檻。
5. Warning handler非threadsafe：限定serialmainthread，保留限制。point hash非全config，其他manifest補足provenance。
6. Tiny archived1e-8 fullsum cancellation：使用預先gamma91 roundoff comparator，精度另由七方向兩step/order支持。
7. Max兩step差0.004523需報combinedlimit0.040884，不誤稱都低於absolute.001。
8. Fullsample passes與base_calls區分：q4/NFW2加localcalls，對外報actualwork及proxy。
9. Failed benchmark telemetry缺口：接受ENGINEERING_CONTRACT=CONDITIONAL；不補造。futurefinally及exceptioncontrol完成，成功數值gate足以支持下一bounded工程smoke授權，非scienceMLE。
10. 舊driver仍是jac=None/eventlists：入口已disabled；下一階段必須新jac=True/collect_warnings driver，不聲稱舊CLI驗過repaired gradient。

統計target未重開，D/i Gaussian獨立與conditioning假設、B0/B1 scientific floor uncertainty、缺image-level/correlation資料等既有科學限制保留，不能由數值／AI審查消除。
''')
 put('09_NEXT_SMOKE_AUTHORIZATION_PACKET.md','''# 未啟動 driver-smoke 授權包

NEXT_SMOKE=READY_FOR_AUTHORIZATION，僅供Human審閱並另授權。canonical installation與scientific MLE仍未授權。

權威NEXT_SMOKE_PROPOSAL.json。新versioned隔離workspace，新driver使用DependencyGradient.scipy_value_and_gradient或同call返回(F,nativejac)，SciPy jac=True，並明啟collect_warnings。不得呼叫舊P1F CLI/driver。source/receipt/data/config/order/version hash必須核實，cache只允許exact matching point/policy，若新增cache先驗parity。

只ptq-screen及restrictedNFW，各一個原P1A recovered vector；不加starts、不profile。L-BFGS-B原nativebounds，maxiter1、maxls20、maxcor10、ftol1e-12、gtol1e-6（只是driver option，不宣稱目前gradient誤差足夠stationarity判定）；每model最多21optimizer gradient batches。

Primary segmented192，gradient固定boundspan1e-4、central可行否則inward，coarse96只作precision。initial/returned terminal核對objective192/96/repeat、native decode/bounds、完整pergal logmarginal sum、預定gradient precision，使用原門檻。不能因refinementfailure調target或step搜尋到有利結果。

最大600 arithmetic-proxy fullsample equivalents、60000 actualgal marginalcalls、1800s aggregate。最壞每batch q9/NFW7eq，21optimizer+最多6precision batches/model仍<=432eq與39312gal，預算足以覆蓋maxls20的一次iteration或明確非budgetABNORMAL終止，不保證接受步。必須同時hard enforce actual與equiv/wall、每gal guard，保存所有rejected trials、exceptions與stop。

每筆標ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE；complete initial/terminal native vectors、acceptedcallback、perrequestobjective/gradient/work/cost/warnings與precision都保存。若wall/工作上限到達仍budgetlimited且不自動retry。最多一次acceptedstep不是stationaryMLE，也不允許新headline params、IC、NFW multistart、sevenmodelcampaign、MCMC、profiles、canonical或public修改。
''')
 put('10_NEXT_EXECUTION_PROMPT.md','''# 唯一建議下一階段 prompt

```text
REV26-P1F-R2 — Repaired-gradient bounded optimizer-driver smoke

需Human另授權後才執行。在全新docs與implementation/results versioned roots，讀完整R1、P1E frozen target、NEXT_SMOKE_PROPOSAL.json與獨立審查。canonical、P1A–P1F/R1與歷史向量唯讀；不重開statistical target。

由R1 candidate另建新driver，明確jac=True讓同一call返回B0 F與native dependency-aware gradient，明啟bounded WarningCollector；禁止直接重跑已disabled的舊jac=None/eventlist driver。先驗所有source/receipt/data/version/geometry hashes、fullparity與所有caller；保留R1失敗metadata缺口，不補造。

只用原ptq-screen與restrictedNFW各一歷史vector，一start/model。固定192/96、boundspan1e-4 central或inwardgradient、原nativebounds；L-BFGS-B maxiter1,maxls20,maxcor10,ftol1e-12,gtol1e-6。最多21optimizer gradient batches/model、600 arithmetic-proxy fullsampleequiv、60000actualgal marginals、1800saggregate，含precisioncalls；不retry、不擴cap。

initial及任何returnedterminal核對192/96/repeat objective、完整native decode/bounds/pergal decomposition、凍結方向gradient refinement。每objective/gradient request保存actualcost、warnings、exceptions及acceptedcallback。任何非finite、source/precision/dependency failure立即停止，不調target/門檻來過關。若maxls結束未接受步，報明確非budgetdriverfailure；若budget到達如實budgetlimited。不要把低於gtol的數值訊息當已證stationarity。

全部輸出ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE。只能裁決driver能否在足額預算推進；不得scientificfit、sevenmodelcampaign、NFWmultistart、MCMC、profiles、IC、manuscript/canonical修改或Git/public/journal操作。Human保留後續全部批准權。
```
''')
 put('SOURCE_PRESERVATION.md',f'''# Source preservation

實際兩個獨立Git roots：{R} HEAD 3eb302666da7eccca19c5843f729e98126d68977；{I} HEAD 5a8dca57f0fc3484387c643b51e22c7e10d024a8。AUTHORITY/GIT_AFTER記錄完整dirty/untracked/worktrees。既有dirty文件不清理、不stash/reset/checkout/pull，沒有任何commit/push/tag/release。

SOURCE_HASHES_BEFORE記錄2114個既有檔案；P1F/P1E/P1A manifests開始前逐hash核實，INPUT_MANIFEST保留dataset與7個theta/path/hash、原source與reference。收尾再次逐file檢查。canonical、dataset、theta、submitted/preprint/public assets及P1A–P1F全部唯讀。P1D invalid v1與correctedv2均保留，未重做舊audit。

所有新資產只在兩個R1 roots，candidate是P1F複本；models.py與原physical law bodies完全不改。全套source/code/environment/pipfreeze、完整execution logs、所有數值cases與failure incidents皆保存；禁止optimizer的舊driver入口raise。沒有任何optimizer或fitting。direction、benchmark均FIXED_HISTORICAL_PARAMETER_POINT。

驗證鏈包含source-review前後receipt metadata、完全相同scientificfunction AST證明、legacyClosure前source snapshot與direction實際executedscript。benchmark_v1 signedstep assertion失敗保留；NFW attempt已做364gal/4eq，但未persist精確wall/warningcount，明列缺失；v2成功不假稱v1原件。futureharnessfinally修補及synthetic failure telemetry control完成。

最終MANIFEST、FAILURE_MANIFEST、SHA256SUMS與docs ARTIFACT_MANIFEST指向所有新檔；自身checksumfile不自我列入。Human保留canonical integration、newsmoke、scientificinference與publication批准權。
''')
 print('wrote12reports',D)
if __name__=='__main__':main()
