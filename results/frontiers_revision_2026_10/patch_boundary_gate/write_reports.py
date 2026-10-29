from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
B=Path(__file__).resolve().parent;I=B.parents[2];R=I.parent;D=R/'docs/frontiers_major_revision_2026_10/rev26_p1d_patch_boundary';P=B/'postprocessing_v2';proto=json.loads((B/'PREDECLARED_PROTOCOL.json').read_text());gate=json.loads((P/'FINAL_GATE.json').read_text());parity=json.loads((B/'parity_v1/PARITY_GATE.json').read_text());extra=json.loads((B/'neighborhood_patch_parity_v1/SUMMARY.json').read_text());run=json.loads((B/'boundary_v1/RUN_SUMMARY.json').read_text());edge=pd.read_csv(P/'seven_edge_on_summary.csv');decomp=pd.read_csv(P/'decomposition.csv');h=decomp[(decomp.point=='historical')&(decomp['T']==5.)];tier=pd.read_csv(B/'boundary_v1/tier1_all22.csv')
def js(p,x):
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write(name,text):
 p=D/name
 if p.exists():raise FileExistsError(p)
 p.write_text(text)
def table(df,cols,fmt=None):
 out='| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'
 for _,r in df.iterrows():out+='| '+' | '.join(format(float(r[c]),fmt[c]) if fmt and c in fmt else str(r[c]) for c in cols)+' |\n'
 return out
review=dict(agent='/root/p1d_independent_patch_review',role='independent second-agent read-only critique; no proof authority',actions='read-only code/diff/output checks and stdout-only Python; no writes/optimizer',recommendations={k:gate[k] for k in ['PATCH_PARITY','BOUNDARY_APPROXIMATION','FLOOR_SENSITIVITY','REFIT_LAUNCH']},confirmed=dict(source_calls=12,all637_finite_unique=True,mapped_points=364,NFW_independent_FD_cases=6,NFW_mass_range=[9,11,13],NFW_H0_variations='default/+5%;c0=8,c_slope=-0.13',NFW_max_relative_error=5.470334916578418e-10,all66_candidate_parity_pass=True,old_source_hashes_unchanged=1866),actual_defect=dict(issue="v1 cf.T/dd.T incorrectly uses pandas transpose instead of column ['T']",scope='v1 derived convergence linkage/neighborhood summaries; raw3300 numerical integrals unaffected',disposition='preserve failedv1 and source; regenerate postprocessing_v2 only; no target/contract or quadrature changes'),objections=[dict(severity='major_scientific_attribution',issue='seven edge-on inclination-only max0.043860,fullTaylor edge maxima are distance-dominated; do not call all7 inclination failures'),dict(severity='major_operating_regime',issue='wider-uncertainty selected controls material;11gal maxincl0.285845,fullTaylor0.311277,NL-A0.370923'),dict(severity='major_model_uncertainty',issue='B1-B0 max3.829767;B1T5-T4max0.011052 exceeds threshold'),dict(severity='coverage_limit',issue='22Tier1,9boundary integrals,13Tier1only;66one-sided/inward points no joint/optimizer domain'),dict(severity='verification_scope',issue='AST callback/covariance expressions are not complete diagnostics IO/sampling/fit-driver execution'),dict(severity='decomposition_limit',issue='inclination-only integrand retains D/i product, not unique interaction-free error decomposition')],parent_disposition='all objections retained; no production or statistical-target change to satisfy agent; supplementary66 parity within already predeclared points; fit launch requires Human statistical-target decision')
js(B/'INDEPENDENT_PATCH_REVIEW.json',review)
alt=dict(status='REVIEWABLE_PROPOSAL_ONLY_NOT_IMPLEMENTED',version='REV26-P1D-UNIFORM-EXPLICIT-PROPOSAL-v1',scope='apply same per-galaxy normalized likelihood definition to ALL91 galaxies and all7 parameterizations; no residual-based target switching',primary_candidate='B0 explicit physical-truncated D/i marginal in fixed y0',mean='mu=sin(i)/sin(i0)*v(s*r0,sqrt(s)*disk0,sqrt(s)*bulge0,sqrt(s)*gas0;theta)',conditional_B0='N(y0;mu,diag(e0^2+sigma_sys^2))',conditional_B1_sensitivity='N(y0;mu,diag(e0^2+(sigma_sys*sin(i)/sin(i0))^2))',prior='independent normalized Gaussian kernels D0/i0 with original widths,physical D>0 and0<i<pi/2; no extra fitted parameters orcos(i)prior',likelihood='product over91 of integral conditional*p_D*p_i dD di;equivalently sum galaxy log marginals',normalization='fullGaussian quadratic/logdet/nlog2pi and physical prior normalization in fixednominal velocity measure; no propagated covariance terms conditional',numerical_acceptance_pending='global algorithm/tail/quadrature policy must be predeclared and verified onall91 before production or optimization; refinement by numerical error,never fit improvement',parameter_counts_preserved={'baryon':92,'mond':93,'ptq':93,'ptq-nu':93,'ptq-screen':94,'mond-screen':94,'nfw1p':183},new_free_nuisance_parameters=0,no_hybrid_switch=True,unresolved=['Human choice surrogatevsphysical marginal','floor coordinate modeluncertainty','D/i independence/image-level limitations','physical-only prior tailnumerics beyondpresentT5','compute budget beforefitting'],current_patch='unchanged isolatedB0Gaussian candidate; this proposal is a separate target,not silently substituted')
js(B/'UNIFORM_ALTERNATIVE_SPECIFICATION.json',alt)
launch=dict(status='INACTIVE_REQUIRES_STATISTICAL_SPECIFICATION_DECISION',can_run=False,likelihood_candidate='REV26-P1D-B0-FULLD-GAUSS-CANDIDATE-v1',alternative='REV26-P1D-UNIFORM-EXPLICIT-PROPOSAL-v1,notimplemented',seven_models=alt['parameter_counts_preserved'],seeds_proposed_not_used=[42,104729,20261028],historical_best_as_start_only=True,no_start_vectors_generated=True,convergence_design='ifauthorized:fixeddata/target/environment;L-BFGS-B/scaledcoordinates and fixedblocks with fullC recomputation;function and stationarity gates;separateNFW reviewer-approved multistart budget',starts_record='completeinitial/finalvectors,RNGstate,allstartobjectives,termination/bounds/KKT/gradientdiagnostics;nooverwrites',KKT_design='scaledcoordinate projectedgradient infinity norm<=1e-4,interiorcentralgradient;one-sidedboundchecks;relativeobjective stability1e-8 over3acceptediterations;crossstart tolerance1e-4 logL fornonNFW;all gatesprovisional forauthorreview',uncertainties='afterconvergedauthorizedfit:profilepriorities q,acceleration,sigma_sys withboundary interpretation;noCIfromhistoricalmedians',B1_plan='predeclare allseven target-consistent sensitivity;neverchoosebyranking',information_criteria='computeonlyfromactuallyoptimizedvalues;ifauthorizedBIC_NPOINTS uses1782descriptiveconventionunderwithin-galaxycorrelation,noNefftheorem',new_run_id='reestimated/<likelihood-version>/<run-id>; preserveoriginal_recovered and01_mle',authorization_needed=['statisticaltarget/floor policy','new targetall91numericalvalidationifchosen','canonicalproductionpatch','fittingstartsandbudget','NFWconvergencepacket','uncertainty/profilecalculations','manuscriptreplacement'])
js(B/'INACTIVE_SEVEN_MODEL_LAUNCH_DESIGN.json',launch)
# Executive and complete mandatory reports.
write('00_EXECUTIVE_GATE.md','''# REV26-P1D controlled patch and operating-regime gate

| 判定 | 結果 |
|---|---|
| PATCH_PARITY | PASS |
| BOUNDARY_APPROXIMATION | MATERIAL_SENSITIVITY_REQUIRES_NEW_SPEC |
| FLOOR_SENSITIVITY | MATERIAL_MODEL_UNCERTAINTY |
| REFIT_LAUNCH | REQUIRES_STATISTICAL_SPECIFICATION_DECISION |

隔離candidate在canonical來源不變的前提下完成完整D covariance修正，且計算parity通過；但整個預宣告operating domain存在Taylor／floor／tail敏感度，不建議直接啟動七模型重估。此判定不表示七個edge-on銀河皆有重大傾角錯誤；分解結果顯示其單獨inclination effect較小，而較寬uncertainty案例超出本輪事先訂定旗標。

637個reference fixed historical points：max|ΔlogL|2.8421709430404007e-13、max相對C1.876397487901259e-15。364個native映射max|ΔlogL|1.1368683772161603e-13。所有7組standalone parity tests通過；66個預宣告鄰域點另補candidate parity，max|ΔlogL|5.684341886080802e-14、max相對C1.428647150412206e-15，原callback對endpoint差0。未呼叫optimizer。

Tier1涵蓋全部22個near-boundary銀河；Tier2全部7個i0=90銀河×7模型；Tier3另2個boundary與2個interior幾何control。共11銀河77個歷史參數點、66個參數擾動、3300個積分值；1188組quadrature convergence全通過，max128→256差7.495443071547925e-6。使用101105664 node-density evaluations，低於150000000預算，約8秒；沒有加測有利案例或降低誤差。

真實edge7在七模型間的max|inclination-only nonlinear−matched-linear|0.043860，低於0.1旗標；IC4202/UGC11455的fullTaylor差0.141584/0.122917主要為distance remainder。全11銀河maxinclination effect0.285845（F574-1 baryon），maxfullTaylor0.311277、maxNL−A0.370923；maxfloor差3.829767（F568-3 NFW）。B1 tail T5−T4最大0.011052亦超過0.01預宣告旗標，不能宣稱尾端皆可略。

已保留一項真正軟體失敗：v1後處理誤把欄位T寫成pandas `.T`，造成neighborhood NaN與convergence linkage錯誤。3300筆raw積分不受影響；原檔／原script保留，新postprocessing_v2只重建derived summaries並嚴格finite/count驗證，沒有重跑quadrature或改契約。獨立代理也發現並核對此修復。

本輪交付可審查patch，另提出對全部91銀河一致定義的explicit-nuisance B0/B1替代規格，沒有實作或混用target。作者須先選定統計目標／floor政策，再決定canonical patch與新數值fitting授權。原資料、theta、稿件、歷史science outputs、P1A–P1C與canonical source均保留。全部新結果是固定點，沒有correctedMLE、IC新排名或新參數估計。
''')
write('01_ISOLATED_PATCH_DIFF_AND_CALL_GRAPH.md','''# Isolated patch、逐檔用途與caller coverage

candidate位於numerical `isolated_candidate/`；統一diff為`production_candidate.diff`，likelihood identifier `REV26-P1D-B0-FULLD-GAUSS-CANDIDATE-v1`。它是全新local copy，沒有.git mutation或canonical checkout；原production可逐位元比對。

| 修改檔 | 最小必要目的 |
|---|---|
| src/ptquat/models.py | 新增model_distance_jacobian；非halo族v/(2D)，NFW保留固定M200/c(M)下halo radius response；所有既有model law AST body完全不變 |
| src/ptquat/likelihood.py | build_covariance要求full-JD keyword，替換radius-only項；Ji在fixed y0 model-mean measure按rad；nominal-floor B0、完整Gaussian與既有regularization程式保留 |
| src/ptquat/fit_global.py | 4個實際C caller都傳full JD：global logL、diagonal pointwise diagnostic、full per-galaxy diagnostic與run後處理 |
| src/ptquat/mle.py | single-galaxy likelihood傳full JD；global ML objective沿fit_global；不修改optimizer、priors或bounds |
| src/ptquat/experiments.py | 7處實際direct covariance callers有新API依賴；_make_vfun factory及coverage自建vfun供給full-JD callback，無留下舊covariance |
| tests/test_covariance.py | 原測試是實際舊API consumer；替合成homogeneous-distance案例明示完整JD，使新required-keyword契約可審核 |

額外experiments/test修改由rg與AST找到確切依賴，不是順便重構。PATCH_CALL_GRAPH.json列出全部12處source calls；不包含definition本身或standalone test。required keyword使漏改caller明確TypeError，不默認radius-only fallback。舊positional v_fun_for_grad暫留作signature continuity但不再用於距離項；_num_grad_v_wrt_r仍是舊diagnostic helper，未被新C調用。

均未改sample/masks、data loader、stellar M/L、七native layouts、q/acceleration domains、learned sigma_sys、NFW concentration freedom或prior densities。對ptq-split的既有非本稿caller保留homogeneous distance支援，不把它新增為第八個reported fitted model。

實際optimizer callbacks以source AST直接固定點執行：neg_g/neg_g2→neg_full→objective_neg_loglike→fit_global.log_likelihood→C；neg_u與NFW neg_um→galaxy_loglike→C。沒有執行outeroptimizer driver。experiments `_make_vfun`與7個C expression、fit run後處理expression以AST執行；不宣稱已跑完整diagnostics IO、sampling或fit driver。

舊pointwise diagonal diagnostic仍明示其原用途，不將它當full multivariateGaussian；full likelihood和logdet路徑均核驗。外部未列出的API consumers未被本輪掃描覆蓋，新required keyword將fail closed，後續canonical整合時應作migration review。
''')
write('02_REFERENCE_PARITY_RESULTS.md',f'''# Patch/reference parity：先通過才執行boundary

容許誤差在PREDECLARED_PROTOCOL事先凍結：每銀河logL abs1e-8、C relative norm1e-9、mapped logL abs1e-8、Jacobian relative1e-6或near-zero absolute1e-5。FD step1e-3/1e-4/1e-5，angles用rad。本輪沒有為了pass更動P1C reference或contract。

| 檢查 | 結果 |
|---|---|
| 637個原theta每銀河full Gaussian/C/SPD | PASS；maxabslogL={parity['max_logL_abs']:.9g}、maxrelativeC={parity['max_C_relative']:.9g} |
| 364個native matched mappings | PASS；maxabslogL={parity['max_mapped_logL_abs']:.9g} |
| global/full-pergal/pointwise-diag/objective | 全7模型逐caller核對reference；沒有以diag冒充full |
| optimizer actual nested callbacks | 直接AST執行globals/finalpolish/Υ/NFW callback，與endpoint一致；optimizer0次 |
| experiments/postprocess dependencies | 全7×8 covariance expression/factory組合通過；無artifactIO或sampling |
| JD/Ji/NFW/units/minimalbaryon | 126個FD比較與near-zero規則、Mpc/kpc和rad/deg、最小variance16例皆通過 |
| parameter-dependent covariance | 基本sigma擾動及supplemental原66鄰域點完整覆蓋scale/q/Υ/sigma/halo |
| source/evidenceimmutability | PASS |

7組standalone tests、0 parity failures；完整results/failures與logs位於parity_v1。Gaussian residual quadratic、logdet、nlog2π分解逐列記錄；所有tested C SPD，jitter未需觸發。原regularization branch未改，也未聲稱域外全不需jitter。

獨立review建議補強scale/q/Υ/halo變動後caller parity，因此僅使用既已預宣告及完成的66個點，重建native向量並核對hash，以candidate galaxy logL/capturedC和actual sourceAST callback核驗。neighborhood_patch_parity_v1：maxlogL={extra['max_logL_abs']:.9g}、relativeC={extra['max_C_relative']:.9g}、callbackdelta0。沒有新增case、integral或parameter search。

初始parity通過後才開始boundary；隔離patch在此後未因邊界結果而改動以保住歷史排名。數值parity不等於B0 Gaussian surrogate已等同physical-truncated marginal。
''')
coverage=tier[['galaxy','edge_on','i_deg','i_err_deg']].copy();coverage['full_integral']=coverage.galaxy.isin(proto['integration_galaxies'])
write('03_BOUNDARY_TEST_PROTOCOL.md','''# 凍結選樣、分層覆蓋與預算

PREDECLARED_PROTOCOL.json在patch parity/boundary數值比較前建立，SHA256 `8c815a14d5512b4290f1aa212d30992d12cfb6aade443ee0f802e879aaad638b`。沒有使用residual、fit改善或model ranking選樣。

Tier1全部22個min(i0,90−i0)≤3ei銀河；Tier2其中全部7個i0=90；Tier3額外boundary最大ei的F574-1、最近boundary的NGC4010，以及interior最近60deg的D631-7和interior最大ei的F568-3。tie依galaxy name，不重抽case。

'''+table(coverage,['galaxy','edge_on','i_deg','i_err_deg','full_integral'],{'i_deg':'.1f','i_err_deg':'.1f'})+'''
Tier1給physical Gaussian prior mass、截斷後mean/variance、T5 projection mean/variance、一階Ji approximation等低成本diagnostic。7個edge-on在22清單內重疊，不重算為另7個。22中9個有七模型full integral；13個僅Tier1，共91個galaxy/model nonlinear combinations未測，詳all22_model_coverage.csv。另兩interior×7=14個control組合，所以historical integrals77點。

四個比較：A為corrected B0 Gaussian；B為同prior線性conditional mean積分；C為完整nonlinear B0；D為完整nonlinear B1。另預宣告inclination-only nonlinear integrand：保留sin(i)但D mean只線性化；它包含D/i product，不是唯一interaction-free誤差分解。fullNL−ionly是distance nonlinear remainder；linear−A隔離truncation/normal-Gaussian surrogate差。

D/i獨立Gaussian kernels使用名義errors，角度轉rad；physical D>0、0<i<π/2，與±T交集逐軸正規化。不加cos(i)prior、不重複加入propagated covariance。T=3/4/5，每軸64/128/256 Gauss-Legendre，logsumexp/fullGaussian normalization；T5不是精確無限tail。convergence absΔlogL≤1e-3，tailT5−T4旗標0.01。

所有77點跑全tails/orders；3個geometry-selected鄰域銀河IC4202（edge最小ei）、UGC07151（edge最大ei）、F574-1，另66個原bounds內固定perturbations只跑T5/128/256。各logL-effect旗標0.1，皆為本輪保守diagnostic criteria，不是普遍科學安全閾值。

最多1200秒、150million conditional-density node evaluations、order不超過256，不開adaptive新case。本輪101105664次，約8秒；1188組convergence全部通過最大7.50e-6，0integrationexceptions。沒有給edge7設ei=0，也沒刪掉任何原資料。
''')
# Full49 rows stored independently for easy author review.
edge_rows=h[h.galaxy.isin(proto['tier2_edge_on'])].copy();edge_rows.to_csv(B/'postprocessing_v2/seven_edge_on_all49.csv',index=False)
write('04_REAL_EDGE_ON_RESULTS.md','''# 七個真實edge-on：全部七模型、沒有以synthetic替代

下表為各銀河七模型的最大absolute logL difference。完整49列含有符號effects存postprocessing_v2/seven_edge_on_all49.csv；來源均為固定原theta，不是correctedMLE。

'''+table(edge,['galaxy','models','inclination_nonlinear_minus_linear_max_abs','distance_remainder_nonlinear_minus_ionly_max_abs','full_nonlinear_minus_linear_max_abs','floor_B1_minus_B0_max_abs'],{c:'.6f' for c in ['inclination_nonlinear_minus_linear_max_abs','distance_remainder_nonlinear_minus_ionly_max_abs','full_nonlinear_minus_linear_max_abs','floor_B1_minus_B0_max_abs']})+'''
七個真實edge-on的inclination-only最大0.0438596，低於本輪0.1flag。IC4202和UGC11455的fullTaylor差0.141584、0.122917主要是distance remainder（分別最大0.139430、0.121587），不能說成傾角邊界造成全部重大錯誤。truncation linear−A在七者max2.44e-6，原因是一階Ji接近零，但非線性sin仍存在；不能由此推斷任何ei/θ都可忽略。

更寬uncertainty的預選boundary F574-1（i65±10deg）baryon，inclination-only−linear=−0.285845、fullNL−linear=−0.311277、linear−A=−0.059647、NL−A=−0.370923，超過凍結0.1旗標。不是只挑edge7的漂亮結果；interiorcontrol也全保留，D631-7某模型fullNL−linear可約0.168089，不能把所有Taylor差歸為邊界傾角。

BOUNDARY_APPROXIMATION=MATERIAL_SENSITIVITY_REQUIRES_NEW_SPEC適用整個已宣告operating-regime審查，而非「七個edge-on皆有重大傾角偏差」。P1C的synthetic i90只是前階段stress test，本輪不用它代替任何真實銀河。

本輪未證實91銀河／所有θ的uniform bound。13個只Tier1的boundary銀河及其91個model combinations明列未做nonlinear integration；不靠篩選／縮小inclination errors取得pass。
''')
write('05_B0_B1_FLOOR_SENSITIVITY.md','''# B0/B1、tail及替代規格

B0在fixed nominal y0座標有constant additive sigma_sys，B1在intrinsic circular velocity座標給sigma_sys，再投影b(i)；兩者都包含e0²且完整normalized conditional Gaussian，不另加D/i covariance。B0為作者預先核准candidate，不因其ranking或數值接近歷史結果而選。

全11個historical galaxy points（77model組合）max|B1−B0|=3.829766865，發生F568-3 NFW，signed差−3.829766865。edge7內maxfloor差僅0.026135；因此FLOOR_SENSITIVITY=MATERIAL_MODEL_UNCERTAINTY不是來自把edge案例不當外推。

T5−T4 maxB0=0.002362831、maxB1=0.011051993、maxlinear=0.002154571；worst B1為D631-7 linearPTQ，超過0.01預宣告tailflag。1188個128→256收斂皆通過（最大7.495443e-6），所以這裡tail敏感度不是quadrature不收斂。有限tail與physicalprior形狀仍有假設依賴，不加跑T6或任意調resolution以求pass。

**獨立、可供審查的統一替代：**UNIFORM_ALTERNATIVE_SPECIFICATION.json定義對所有91銀河與全部7parameterizations使用同一explicit nuisance likelihood：

L_g(θ)=∫physical N[y0; sin(i)/sin(i0) vm(D;θ), S_B0] p_D(D)p_i(i) dDdi。

距離模板／半徑依完整catalog變換；kernel以各自physicaldomain正規化。B1作同樣scope的floor sensitivity，conditional只measurement+floor，沒有propagation double counting。不再挑某些銀河用Gaussian、某些用explicit依residual改target。此proposal保留原k與bounds，D/i只integrate，不新增fitted nuisance parameters。

替代尚未實作／全91數值驗證／採納或擬合；physical-only prior integration與tail/全域error-budget必須另預宣告並核驗。它能避免一階sin近似，不能憑空解決D/i correlations、image-level warp或floor座標唯一性。作者也可明確選擇將B0Gaussian當自立surrogate並接受已見限制，但不能繼續宣稱它在全樣本等同physical nonlinear marginal。

沒有在本輪patch中偷換target、對銀河混用不相容定義、按有利ranking選floor，或調bounds/qualitycuts。
''')
write('06_PARAMETER_NEIGHBORHOOD_SENSITIVITY.md','''# 66個預宣告鄰域固定點，沒有profile/search

geometry-only選IC4202、UGC07151、F574-1；每個原模型theta依scope擾動：acceleration×1.01、q×1.01、sigma_sys×1.02、該銀河Υ在bounds內+0.01否則−0.01、NFW logM在bounds內+0.02dex否則−0.02。沒有joint perturbations、沒有挑likelihood改善方向；U/halo方向規則只看bounds。

66個native向量的coordinate old/new、bounds與float64 bytes hashes在boundary_v1/parameter_points.json。全部valid；T5／128/256積分，不另外生成起點或跑optimizer。源資料/原theta不變，這些標FIXED_PARAMETER_PERTURBATION，不叫correctedMLE。

用同一θ點的nonlinear-minus-linear／floor effects再與historical baseline比較，避免把raw logL改變誤稱approximation instability。最大effect變動：fullTaylor0.0970960、inclination-only0.0200468、floor0.0333439、NL−A0.0970960；全在本輪0.1 neighborhood-change flag內，但最大值接近旗標，不是domain穩健性證明。worst為IC4202 NFW halo+0.02dex，主要是distance非線性而非inclination。

補充candidate/parity核驗全部既有66點與actual sourceAST callback：logL差max5.68e-14、C差max1.43e-15、callbackendpoint差0，沒有新增integration。這是數值caller更新證據，不代表已預測corrected optimizer trajectory。

**曾發生的後處理失敗：**v1 cf.T/dd.T取到DataFrame transpose，v1 neighborhood_effect_changes.csv為NaN，convergence linkage有vacuous .all()。raw3300積分／獨立convergence數值沒有變動。原v1與run_boundary.py保留；postprocessing_v2.py從raw重新分組，明示['T']、nonempty4constructs與66finite/unique assertions，0target/contract change、0重新積分。V1_DERIVED_FAILURE.json與獨立review記錄完整。正式解讀一律使用postprocessing_v2，而非錯誤v1 summaries。

未測其餘parametercoordinates、負向非boundsperturbations、聯合變動、其他銀河鄰域、全optimizerdomain；不能聲稱歷史optimum附近就是corrected optimum。
''')
write('07_INDEPENDENT_PATCH_REVIEW.md','''# Independent read-only critique

第二代理 `/root/p1d_independent_patch_review` 在patch/parity完成後獨立讀diff、canonical與reference，並檢查boundary/raw與v2。未改檔、未呼叫optimizer；同模型第二代理不是跨模型共識，也不是proof authority。完整評語與處置存INDEPENDENT_PATCH_REVIEW.json。

確認：12sourcecallers皆explicitfullJD、original law bodies不變、requiredkeywordfailclosed、637finite/unique和364mapping通過、1866份舊sourcehash不變。另以mass9/11/13、H0default/+5%、c0=8/cslope−0.13的6個NFWcontrol核對fullrescalingFD，maxrelativeerror5.47e-10。

代理確實發現v1 `.T`後處理錯誤，主代理summary檢查也發現NaN。雙方核對raw3300finite/unique、1188個convergence真正全pass，原numerical分解column與raw差≤5.68e-14；v2保留錯誤並只修derivedmetadata與66neighbor比較，未修改patch／reference／contract以配合review。補66candidateparity是原既有點的verification，不是新增敏感度campaign。

保留異議：

- 七個edge-on inclination-only小於0.1，fullTaylor edge大值主要由distance，不能過度歸因inclination。
- 寬uncertainty預選案例有materialTaylor／floor差，B1tailflag亦超限；不應直接啟動fit。
- 66個one-sided/inward鄰域點是有界證據，未涵蓋joint變動或correctedoptimizerdomain。
- 22Tier1、9boundaryfullintegral、13Tier1only須精確報告；ASTexpressions不是整個diagnosticsIO、sampling或fitdriver測試。
- B0surrogate不等於physical marginal；inclination-onlyintegrand保留D/i product，分解非唯一。

雙方建議PASS／MATERIAL_SENSITIVITY_REQUIRES_NEW_SPEC／MATERIAL_MODEL_UNCERTAINTY／REQUIRES_STATISTICAL_SPECIFICATION_DECISION。無未解計算parity分歧；統計target/floor/operatingregime仍由作者裁定。代理同意不使其成為科學正確性的證明。
''')
write('08_SEVEN_MODEL_REFIT_LAUNCH_PACKET.md','''# Inactive seven-model launch design：尚不可以執行

operating-regime gate未通過無限制適用性，REFIT_LAUNCH=REQUIRES_STATISTICAL_SPECIFICATION_DECISION。此packet只保存後續review所需條件，沒有啟動fitting，也沒有生成任何NFW/startvectors。machine-readable設計見INACTIVE_SEVEN_MODEL_LAUNCH_DESIGN.json，can_run=false。

必要前置：作者選定統計target（B0Gaussian surrogate及限制，或統一explicit marginal）、B0/B1策略、physical/tail數值接受條件；新target如被採用須先完成全91reference與candidatevalidation。作者另授權canonical整合、starts/預算與optimizerworkflow，才可進入新run-id。

| parameterization | k | 保留內容 |
|---|---|---|
| baryon |92|91Υ+sigma_sys|
| MOND simple-ν |93|log10a0+91Υ+sigma_sys|
| linear PTQ |93|wide epsilon+91Υ+sigma_sys|
| PTQ-ν |93|matched epsilon+91Υ+sigma_sys|
| PTQ-screen |94|matched epsilon+lnq+91Υ+sigma_sys|
| MOND-screen |94|log10a0+lnq+91Υ+sigma_sys|
| restricted NFW-1p |183|91Υ+91logM200+sigma_sys;c(M)固定|

原bounds：Υ[.05,1.5]、q[.25,8]、sigma_sys[.001,200]km/s、logM[9,13]；νfamilies a0[5e-11,2e-10]SI對映epsilon，linearPTQ保留[1e-6,3.999]。不加MLprior、不把boundarycontact扣掉k。

候選likelihood版本REV26-P1D-B0-FULLD-GAUSS-CANDIDATE-v1；如果作者另選explicit，必須新版本，不把target切換藏在同一run。每run記錄data/code/environment、fullnativeinitialtheta、CLI、RNGstate、allstartterminalvectors、iterationobjective、quad/logdet/norm、bounds、termination與stationarity。歷史best只作可能start，不保證correctedoptimum；暫擬獨立seed42/104729/20261028僅供未來protocol審查，尚未使用、未抽樣生成starts。

未來optimizer候選可採scaled native coordinates／L-BFGS-B與原block更新，但functionstability、scaled projectedgradient／boundKKT、跨start objective一致性必須均檢查。暫擬stationarity infinitynorm1e-4、連續3acceptediteration relativeobjective1e-8、nonNFW crossstart1e-4logL供作者review，不當成已完成收斂證據。NFW另核准multistart／alternativeoptimizer workpacket，原跨start約29logL問題不能靠本輪likelihoodparity宣告解決。

維持七parameterization逐一report；MOND/PTQ配對可共用physicalfamily計算但保留native映射與matchedobjectivechecks，不以sharing掩蓋失敗。優先profile q、scale、sigma_sys與boundary影響；任何uncertainty/profile/MCMC需另授權，不從歷史posterior冒出新CI。B1敏感度須對一致target預先排程，不挑較佳ranking。

只有真正maximize與收斂後才能報對應AIC/BIC；BIC_NPOINTS以1782為within-galaxy correlations下的descriptive convention，不是已建立Neffcriterion。本輪固定點不計correctedICrankings。新估計寫reestimated/<likelihood-version>/<run-id>，不能覆寫original_recovered、舊01_mle或scienceassets。
''')
write('09_UNRESOLVED_STATISTICAL_ASSUMPTIONS.md','''# 未解假設與操作限制

1. B0 nominal-floor是作者核准candidate，B1intrinsic-floor仍是合理敏感度；max3.83logL差意味模型不確定性，不能從fixedpointlikelihood選定唯一真實噪音規格。
2. IndependentGaussian D/i kernels是declaredapproximation；rawcrosscorrelation與image-level uncertainties未取得。explicitmarginal能處理非線性，但不會自動修復錯誤／缺漏measurementmodel。
3. Fixedy0、名義去投影velocity/error、完整catalogsqrtD模板與physicalr變換一致；globalinclination adjustment不等於重新fitvelocityfields／warps／imagingdeprojection。沒有新資料權威支持後者。
4. Physicalpriortruncation、finiteTtail、Taylorlinearization、floorcoordinate與quadratureerror已分別記錄。orderconverged不等於tail可略，更不等於Gaussian surrogate是exactmarginal。
5. 本輪0.1logL/0.01tail旗標為先驗declareddiagnostics，非普遍顯著性或科學無害界線。materialflag要求review，而不是已證所有modelranking改變。
6. 全22boundary只9做fullintegral，其餘13只Tier1；其91modelcombinations未測。兩個interiorcontrol不提供91或domainuniformbound。
7. Smallone-sided/inwardperturbations未涵蓋jointneighborhood或correctedoptimizertrajectory；maxeffectchange0.097096接近flag，不足以保證optimizer域穩定。
8. 計算API/callerparity通過，但未跑fitdriver、sampling或diagnosticIO整體pipeline；legacyoutside-repoAPI需requiredkeywordmigration。canonical仍舊target，isolatedcandidate未安裝。
9. v1postprocessingdefect已透明保留並用v2derived修復；原scienceinputs和rawintegrals不變。正式結論不引用NaNv1neighbor或vacuousconvergenceflags。
10. 新best-fit q/epsilon/a0/sigma_sys、confidenceinterval、AIC/BICranking全部UNKNOWN_PENDING_REFIT。作者先決定統計target／sensitivities，再授權codeintegration與fittingbudget；本輪到此停止，不開延伸campaign。
''')
write('10_NEXT_EXECUTION_PROMPT.md','''# 最小下一個可執行任務（先由Human Arbiter批准）

REV26-P1E — Uniform Likelihood Target Decision and Isolated Numerical Acceptance; NO FIT.

讀P1D全部reports、postprocessing_v2/FINAL_GATE、production_candidate.diff、UNIFORM_ALTERNATIVE_SPECIFICATION、INDEPENDENT_PATCH_REVIEW及SHA256SUMS。canonicalsource/舊results/稿件保持不動；不重做archive或Symmetryaudit。

1. 作者先選定是否保留B0Gaussian作自立surrogate並承認已見適用限制，或採對全部91galaxy一致定義的physical-truncatedexplicitB0marginal；B1保留相同scope sensitivity，不按residual或改善方向混用likelihood。
2. 如批准統一explicitreference，在全新root/run-id實作normalizedconditional×physicalprior的二維deterministicintegral，先凍結tail／numericalacceptance／budget。選一個可供全部91與全部七parameterizations一致使用的numericalpolicy，refinement只能依integrationerror，不能依ranking。驗證fixedy0Jacobian、conditionalfloor、noD/idoublecounting、NFWresponse與nativefamilyidentity。
3. 先localtests，再以七原theta作637固定點接受檢查；清楚記錄新target、finite-tail誤差及所有失敗。B1tailflag尚超0.01，要事先決定tolerance／physical-onlyintegrationpolicy，不能隱藏或事後擴大門檻。本步不算newMLE或ICrankings。
4. 比較P1Dsurrogate及uniformtarget，交付reviewableproductionpatchproposal與inactive seven-model launchconditions。若作者僅保留surrogate，也要寫清楚哪些latentnuisanceaccuracy claims被放棄、B1／operatingregime如何報告，不宣稱exactmarginalization。

STOP：不執行optimizer、MCMC、NFWstarts、profiles／CI、newfitparameters、稿件數值替換、canonicalproductionedit（除非另明確批准）、Gitcommit/push/tag/release或public/journalwrite。結果交作者決定下一次production與七模型擬合授權，不自行開始延伸sensitivitycampaign。
''')
(B/'README.md').write_text('''# P1D isolated patch and boundary audit

Verdicts: PATCH_PARITY PASS; BOUNDARY_APPROXIMATION MATERIAL_SENSITIVITY_REQUIRES_NEW_SPEC; FLOOR_SENSITIVITY MATERIAL_MODEL_UNCERTAINTY; REFIT_LAUNCH REQUIRES_STATISTICAL_SPECIFICATION_DECISION.

Canonical source remains unchanged. isolated_candidate/src/ptquat contains candidate source; production_candidate.diff is complete six-file patch incl verified experiments/test API consumers. PATCH_CALL_GRAPH maps12 direct source C callers. PREDECLARED_PROTOCOL was frozen before computations.

parity_v1:637 fixed points,364 mappings,all paths/tests. neighborhood_patch_parity_v1:the existing66 fixed perturbations plus actual sourceAST callback checks, no new integrations. boundary_v1:3300 raw integrals,825 attempted grids,22 diagnostic galaxies,all7real edge-on×7models,4geometry controls,66 neighborhood points. All results fixed historical or declared perturbations, not correctedMLE.

IMPORTANT: boundary_v1/neighborhood_effect_changes.csv is a PRESERVED FAILED DERIVED ARTIFACT (NaN due pandas .T column bug); v1 decomposition convergence flags also used incorrect linkage. Raw integrals/convergence numerical values were unaffected. Use postprocessing_v2 ONLY for decomposition/convergence/neighbor/decision summaries. postprocess_v2.py rebuilds from raw without new quadrature or contract change; V1_DERIVED_FAILURE.json records the failure.

Run order: bootstrap.py -> freeze_protocol.py -> build_patch.py -> test_patch.py -> run_boundary.py -> postprocess_v2.py -> verify_neighborhood_patch.py -> write_reports.py -> finalize.py. All output roots/run IDs reject overwrites; scripts are preserved, including failed original postprocessor. For exact reproduction copy only scripts plus candidate/spec/protocol authority to a NEW sibling run-id at same depth, preserving absolute original-input paths; do not copy completed output folders. Do not rerun original prepare/freeze against already frozen files. Use temporary environment in parity_v1/environment_freeze.txt; command prefix PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/rev26-p1a-env/bin/python. No optimization or canonical installation.

Complete logs: parity_execution.log, boundary_execution.log, postprocessing_v2.log, neighborhood_patch_parity.log; execution metadata/authority/spec hashes are machine-readable. INDEPENDENT_PATCH_REVIEW records separate read-only critique and the actual derived failure. UNIFORM_ALTERNATIVE_SPECIFICATION is a proposal, not implemented target; INACTIVE_SEVEN_MODEL_LAUNCH_DESIGN can_run=false.

Allnew assets/hashes are indexed in docs ASSET_MANIFEST; each root has SHA256SUMS. Source preservation validation covers all1866 preexisting files. Never use fixed-point values as optimized maxima or IC rankings.
''')
js(B/'EXECUTION_LOG.json',dict(stages=[dict(stage='A',command='bootstrap.py',prior_files=1866),dict(stage='protocol',command='freeze_protocol.py',frozen_before_comparisons=True),dict(stage='isolated_patch',command='build_patch.py',canonical_writes=0),dict(stage='parity',command='test_patch.py',console='parity_execution.log',status='PASS'),dict(stage='bounded_sensitivity',command='run_boundary.py',console='boundary_execution.log',raw_integrals=3300,status='raw numerical complete;derived postprocess bug retained'),dict(stage='derived_repair',command='postprocess_v2.py',console='postprocessing_v2.log',new_quadrature=0,status='PASS'),dict(stage='existing66_supplementaryparity',command='verify_neighborhood_patch.py',console='neighborhood_patch_parity.log',new_cases=0,new_quadrature=0,status='PASS')],environment='parity_v1/environment.json and environment_freeze.txt',independent_review='read-only second agent; review not proofauthority',optimization=False))
print('reports/alternative/inactive launch design written')
