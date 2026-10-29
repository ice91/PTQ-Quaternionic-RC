from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;I=B.parents[2];R=I.parent;D=R/'docs/frontiers_major_revision_2026_10/rev26_p1e_uniform_target';A=B/'acceptance_v1';P=I/'results/frontiers_revision_2026_10/patch_boundary_gate'
s=json.loads((A/'ACCEPTANCE_SUMMARY.json').read_text());policy=json.loads((A/'NUMERICAL_POLICY.json').read_text());local=json.loads((B/'local_v1/tests.json').read_text());val=json.loads((B/'verification_v1/VALIDATION.json').read_text());protocol=json.loads((B/'PREDECLARED_PROTOCOL.json').read_text());f=pd.read_csv(A/'full_sample_all1274.csv');tot=pd.read_csv(A/'totals.csv');floors=pd.read_csv(A/'B1_B0_full637.csv');cross=pd.read_csv(A/'P1D_finite_T5_crosscheck.csv');pilot=pd.read_csv(A/'pilot_quantile_convergence.csv');linear=pd.read_csv(A/'same_physical_prior_linear_pilot.csv')
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
b0=tot[tot.floor=='B0'].copy();b1=tot[tot.floor=='B1'].set_index('model');b0['B1_minus_B0']=b0.apply(lambda r:b1.loc[r.model].explicit_logL-r.explicit_logL,axis=1);maxfloor=floors.loc[floors.B1_minus_B0.abs().idxmax()];full0=f[f.floor=='B0'];maxsur=full0.loc[full0.explicit_minus_surrogate_B0.abs().idxmax()];edge=full0[full0.i_deg==90].groupby('galaxy').explicit_minus_surrogate_B0.agg(['min','max']).reset_index();near=f[(f.floor=='B0')&f.physical_boundary_3sigma].galaxy.nunique();gauss=pd.read_csv(B/'verification_v1/Gaussian_only_cost_comparator.csv')
review=dict(agent='/root/p1e_independent_target_review',mode='independent second-agent read-only source/outputs/numerical critique; no proofauthority',recommendation='UNIFORM_EXPLICIT_B0_READY_FOR_FIT_REVIEW',checks=dict(fast_vs_direct_conditional_integrals=48,fast_vs_direct_max_logL=3.154809746774845e-12,all91_node_geometry_physical=True,full1274_unique_finite_complete=True,native_mapping728_pass=True,neighborhood72points144rows_pass=True,source_hashes_checked=1957,quantile_fallback_predeclared=True),objections=[dict(severity='numerical_limit',issue='successive refinement estimates not rigorous bounds,notuniformoptimizerdomain guarantee',disposition='explicit reports;recheckprecision atauthorizednewoptima/importantbounds,notnewmethodsresearch'),dict(severity='major_scientific_uncertainty',issue='fullsamplefloor sensitivity max15.626114,linearPTQsample+28.166984; notnumericalerror orreasonchoosefloor',disposition='B0 chosenbeforevalues;B1 separatelyidentifiedsensitivity'),dict(severity='surrogate_limit',issue='B0explicit-minus-surrogate pergal approx-3.45089..+6.75363',disposition='noequivalencetocorrectedGaussian claimed;fixedpointsnotMLE'),dict(severity='assumption_limit',issue='conditionalnuisanceGaussian/independence/flatpriorconditioning/catalogi interpretation notrawempiricalfacts',disposition='declared targetassumptions remain'),dict(severity='cost_reporting',issue='2.53..2.59sec isfull91B0+B1combined,notB0-onlybenchmark; near6Bbudget',disposition='useconservativecombinedcostupperbound,noextraexplicitintegrations'),dict(severity='boundary_limit',issue='zero_uncertaintyati90meansclosurelimit,notordinaryopen-domainpriorpoint',disposition='documentdegenerateGaussianlimit;realerrorsneverzeroed')],numerical_warnings='NumPymatmulwarningsretained;48directcomparisons andfinitechecks noobservedcorruption',extra_density_nodes_approx=9800000,no_files_modified=True,no_optimizer=True,parent_response='allobjections retained;noalgorithm/target/tolerancechange requestedor made;chosenA isreviewreadinessonly')
js(B/'INDEPENDENT_REVIEW.json',review)
# Unique final target recommendation, distinct from Gaussiancandidate identifier.
decision=dict(recommendation='UNIFORM_EXPLICIT_B0_READY_FOR_FIT_REVIEW',target_identifier='REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1',primary='uniformphysical-domainexplicitB0in fixednominaly0',B1_role='separate intrinsic-floor model sensitivity under identicalphysicalnuisancedistributions',computation_valid=True,numerical_goals_pass=True,statistical_assumptions_established_by_data=False,uniform_over_optimizer_domain_proven=False,production_integrated=False,fitting_performed=False,ready_for='Humanapprovalofproductionintegrationreviewandrealoptimizerdriversmoketest;thenboundedsevenparameterizationcampaignreview',not_permission_to_fit=True,no_fitted_estimates_or_IC_rankings=True,metrics=s,source_preservation='all1957baselinefilesverified unchanged;finalvalidationpending',limitations=[x['issue'] for x in review['objections']],B0_choice_reason='Humanapprovedmeasurementcoordinate interpretation beforevalues;notfavorableranking')
js(B/'TARGET_DECISION.json',decision)
# Preserve specific failed numericalpolicy pilot cases, syntax incident and budgets without suppression.
pfail=pilot[abs(pilot.delta512_256)>1e-4].copy();pfail.to_csv(B/'UNSEGMENTED_QUANTILE_ACCEPTANCE_FAILURES.csv',index=False)
# Inactive integration/fitting design; no starts constructed.
plan=dict(status='INACTIVE_HUMAN_APPROVAL_REQUIRED',target_identifier=decision['target_identifier'],canonical_integration=False,optimizer_calls_authorized_in_P1E=0,new_likelihood_module_proposal='src/ptquat/marginal_likelihood.py;unit-square physicalpriornodes,normalized conditionalB0/B1,cache,budget/accuracychecks',files_to_review=['src/ptquat/likelihood.py','src/ptquat/models.py','src/ptquat/fit_global.py','src/ptquat/mle.py','src/ptquat/experiments.py'],primary_objective='sum91 logmarginals,notGaussianpropagatedC;nohybridgalaxyswitch',Gaussian_role='diagnostic surrogateonly,distinctversion',native_parameter_counts={'baryon':92,'mond':93,'ptq':93,'ptq-nu':93,'ptq-screen':94,'mond-screen':94,'nfw1p':183},bounds='frozenoriginalparameter_domain_manifest;noextraD/i fittedparameters;noMLprioronphysicaltheta',must_before_MLE=['canonical/isolatedintegrationreview','637B0/B1fixedpointparity728nativeidentity','trueoptimizerdriversmoketest ONLYwhenexplicitlyauthorized','predeclaredmultistartcostbudgetandcompleteallstarttheta/provenance','bound/KKTconvergencediagnostics','separateNFWoptimizationreview','B1refit-sensitivityplan','precisionchecksatsmokepoints/authorizedterminalfits'],starts='historicalbestpossiblewarmstartonly;independentseededstarts needseparatebudget;nonegenerated',future_paths='reestimated/REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1/<run-id>',B1_identifier='REV26-P1E-UNIFORM-EXPLICIT-B1-PHYSICAL-v1',information_criteria='onlyfromactuallyoptimizedlikelihoodafternewfit;BIC_NPOINTS1782descriptiveunderwithin-galaxycorrelation,noNeffproof',cost_upper_bound_seconds_per_full91_B0_call=2.60,hypothetical_4000_full91_calls_hours_upper_estimate=4000*2.6/3600,cost_scope='observedB0+B1combinedruntimeusedconservatively;notoptimizerbenchmarkorpromise',no_open_ended_new_methods_stage=True)
js(B/'INACTIVE_INTEGRATION_AND_REFIT_PLAN.json',plan)
write('00_EXECUTIVE_DECISION.md',f'''# Final bounded pre-fit likelihood-selection gate

**唯一建議：UNIFORM_EXPLICIT_B0_READY_FOR_FIT_REVIEW。**

主target選為對全部91galaxies、全部七nativeparameterizations一致定義的physical-domain explicitly marginalized B0；新identifier `REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1`。B0依作者事前核准的fixednominaly0 floor意義選定，不使用有利likelihood或ranking選擇。B1仍為同scope、另識別的科學敏感度。

物理域D>0、0<i<π/2，external Gaussian kernels在各物理域正規化；uniformquantile積分沒有finiteTtargetcutoff。未分段quantile pilot的max512−256差{policy['quantile_pilot_max_last_delta']:.9g}未達1e-4，依凍結規則統一切到唯一預定fallback：physicalCDF-knot分段quantile，全銀河固定每piece192階，96階作精度比較；不是逐銀河混用target，也沒有第三方法或延長研究。

B0/B1各637個fixedhistoricalpoints全部finite/unique、無省略；728個native mappings（兩floor各364）最大|ΔlogL|{s['max_native_mapping_delta']:.6g}。72鄰域／近hardfloor界限點、144floor結果全通過。七組localcorrectness先PASS才進全sample。

工程目標：最大pergalrefinement {s['max_pergal_refinement']:.6g}≤1e-4；每模型sumabsrefinement max{s['max_model_sumabs_refinement']:.6g}≤0.01；modeldifference conservative refinement estimate max{s['max_model_difference_error_estimate']:.6g}≤0.02。這些是refinement估計，非嚴格誤差定理或整個optimizer域的uniformbound。

主程序約{s['elapsed_seconds']:.2f}秒，maxRSS約{s['max_process_RSS_bytes']/1e6:.2f}MB，{s['density_node_evaluations']:,}node-densityevaluations低於凍結6B預算。單模型全91同時計B0+B1約2.53–2.59秒；以其作B0-only的保守成本上界，不冒稱獨立B0-onlybenchmark。全91NFW重複兩次最大roundoff{s['repeat_max_abs']:.6g}，固定qprobesexactrepeat0。

B0−Gaussian surrogate的單銀河差約−3.45089至+6.75363；B1−B0單銀河最大+15.626114（baryon UGC09133），科學模型不確定性遠大於integration refinement，不能用它改選有利floor。全部值仍是原theta固定點，沒有correctedMLE、newq/scale/sigma_sys、CI或ICrankings。

這表示可以請HumanArbiter批准productionintegrationreview及真正optimizerdriversmoketest，再審核有界七模型fittingbudget；本輪未整合canonical、未呼叫任何optimizer。下一階段是工程整合與明確啟動審查，不再開放式探索新積分方法。
''')
write('01_FROZEN_TARGET_AND_NUMERICAL_PROTOCOL.md','''# Frozen target、接受門檻與成本契約

PREDECLARED_PROTOCOL.json在P1E任何numericalcomparison前凍結，SHA256 `cce98964d0cdaf1b388b3e815285e6df8ef6655e56f308964bcb9b22beb43d81`。FROZEN_STATISTICAL_TARGET.json記錄target；INPUT_MANIFEST與SOURCE_HASHES_BEFORE記錄data、原theta、P1C reference、P1D候選及舊evidence。Canonicalrepo與outerrepo分別核對，不假設共用Git歷史。

Physical target：fixed y0 density，mu=sin(i)/sin(i0) vm(s r0,sqrt(s)templates;θ)，B0 S=diag(e0²+sigma²)，B1 S(i)=diag[e0²+b(i)²sigma²]。conditional只含measurement＋floor；不加propagatedD/iC。保留原law、M/L、NFWfixedM/c(M)、91/1782sample、sevenlayouts、bounds/k；D/i integrated而非新增fittedparameters。

External uncertaintykernels被用作在catalog測量條件下的normalized conditional nuisance distributions。其解讀須假設Gaussian外部measurementkernel、指定nominalwidth及locallyflat D/i prior／conditioning慣例，再限制physicaldomains；不加cos(i)orientationprior。independence、Gaussian shape與imageleveltransform未被rawSPARC資料獨立證實；沒有D/i相關資料。

Localcorrectness absnormalization1e-10、logL1e-8、nativeidentity1e-8、repeat1e-12；數值acceptance pergalrefinement1e-4、sample signed及sumabsrefinement各0.01、兩模型difference用各自sumabsrefinement之和估計，門檻0.02。後者不是統計顯著性或BICeffectivesamplesize判準。

Pilot由geometry／既有case凍結：IC4202、UGC07151、F574-1、NGC4010、D631-7、F568-3、UGC03205；七模型兩floors。先uniformquantile64/128/256/512，任一pilot最後差超1e-4則全域改唯一segmentedfallback，每piece24/48/96/192；完整0..1概率域保留。所有attempts與失敗policycases保留，不降低precisiongoal、不按likelihood好壞挑method。

fullsample先localPASS，後B0/B1全637，以同一固定最高order和倒數order比較；無逐galaxy target切換。72鄰域點：P1D原66加IC4202的MOND/MOND-screen/NFW sigma=0.001001、F574-1相同三模型sigma=199.999；其餘theta不變。q smoothness為UGC07151 MOND-screen的相對offset−2e-4/−1e-4/0/1e-4/2e-4，固定probe不是profile/search。每個probe重複、另兩次全91NFWrepeat，沒有線搜尋或估計parametererror。

Budget：最多兩個methodfamilies、1200秒、6Bnode-density、512MiB memory；不超uniform512或piece192；失敗保留，不開第三方法或新case。實際主程序約43秒／5.978148224Bnodes，local與獨立小規模核驗合計仍低於6B；後續artifactverification和Gaussian-onlytiming沒有新增explicitnodes。
''')
write('02_UNIFORM_EXPLICIT_REFERENCE.md',r'''# Uniform explicit reference implementation

`explicit_reference.py`是新隔離module；只讀重用已核驗P1C model-law/reference資料結構，沒有匯入productionfitter或修改任何P1A–P1D檔案。新code專注physicalpriorquantile與marginalGaussian積分，非重寫PTQ理論。

對標準normal kernel的physical bounds a,b，Z=Φ(b)−Φ(a)。u∈(0,1)映為z=Φ⁻¹[Φ(a)+uZ]。數值實作同時計CDF與survival q=Φ(−b)+(1−u)Z，upperhalf用−Φ⁻¹(q)，避免1−CDF在upperendpoint的消去。權重是unitprobability空間的Gauss-Legendreweights；Jacobian等價於normaldensity與原measure相消，不額外漏乘或重乘sigmaD/sigmai。

D物理域a=−D0/sigmaD,b=∞；i域a=−i0/sigmai,b=(π/2−i0)/sigmai。i0=90deg仍為半normal支撐，不把ei設零。所有實際nodes嚴格D>0、0<i<π/2；endpoint／nonfinite／invalidweight／nonpositivevariance failclosed，未對probabilities任意clip後當成正確target。

分段quantile只把同一unitinterval在normalizedphysicalCDF(z=−6,−4,−2,0,2,4,6)切piece，仍保留0和1兩端的完整概率質量。這些knots是數值分割，不是Tcutoff；每piece固定192階，外部geometry決定piece數，不依模型residual／排名／θ選order。所有七parameterizations、91銀河及fixedprobes使用同policy。

D/i nodes／weights只依cataloggeometry，lru_cache重用；parameter改變時modelvelocity、sigma-dependentvariance與完整logdensity重算。B0/B1共用mean但有不同conditionalvariance，B1determinant隨i變。主積分使用logsumexp／logaddexp累加，conditionalnormalization為sumlog(2πvariance)，沒有propagatedD/iCdoublecounting。

為節省記憶體，conditionalquadratic利用exact weighted sufficient sums：sum y²/w−2b sum yv(D)/w+b²sum v(D)²/w，B1之w隨i變；128個Dnodes一block，避免nD×ni×npoints的大tensor。這是代數恆等式，不是忽略radius／殘差項。負quadratic若超64machine-epsilon尺度的roundoffbound立即fail；finiteassertions先於aggregation，沒有用max(NaN)掩蓋錯誤。

可選linearmean只為matchedphysical-prior診斷，primary永遠full nonlinearmean。零nuisanceerror支援pointmassconditional-limit測試；如nominali90且error→0，應稱physicalclosurelimit，非openphysicaldomain的普通prior點。真實資料errors沒有被改為零。
''')
lt=pd.DataFrame([dict(test=x['test'],status=x['status']) for x in local])
write('03_NORMALIZATION_AND_QUADRATURE_TESTS.md',f'''# Correctness、failed pilot與refinement evidence

'''+table(lt,['test','status'])+f'''
七組localcorrectness先PASS：physicalpriorconstantintegration、extremeprobability/endpoints、zeroerrorGaussianlimit、fixedy0/Jacobian、fullsqrtD/NFWresponse/radianunits、B0/B1variance與Gaussiannormalization/no-doublecount、nativeidentity/repeatability與immutability。invalidprobability0/1/NaN、nonpositiveconditionalvariance明確failclosed。所有testattempts/log保留。

獨立第二代理將48個真實／合成fastsufficient-sum結果與directconditionalgrid比較，最大|ΔlogL|3.154809746774845e-12；全部finite。NumPy matmul曾輸出divide/overflow/invalidwarnings，原console保留；finitechecks、directcomparison及完整outputs未見數值污染，未為warnings換target或放寬門檻。

未分段quantilepilot最高order512仍有max512−256差{policy['quantile_pilot_max_last_delta']:.9g}，大於1e-4；失敗case詳UNSEGMENTED_QUANTILE_ACCEPTANCE_FAILURES.csv。這是預先容許的numericalpolicyfailure，不是physicaltarget改變。依凍結switchrule全域segmentedquantile；沒有第三方法。

選定192/96全sample：maxpergalrefinement={s['max_pergal_refinement']:.9g}，maxmodel sumabs={s['max_model_sumabs_refinement']:.9g}，maxabs signedtotal={s['max_model_signed_refinement']:.9g}，兩模型differenceerror估計max={s['max_model_difference_error_estimate']:.9g}。B0/B1共1274rows無一超1e-4，model totals均在0.01內。各個refinementvalue、sumabs與pairwiseestimates皆保存，不只列成功例。

這些是相鄰order比較的工程誤差估計，不是rigorousabsoluteintegralerrorbound；亦未證全optimizerdomain uniformprecision。粗精order共享相同target與物理域，沒有tail-modelcutoff誤差混入此refinement比較。後續已授權fit應在真實smokepoints／terminalcandidates核對相同numericalacceptance，而非宣稱本輪已覆蓋未知optimum。

原driver在任何numericalrun前曾有一個syntax typo（else199.999），exit1；failedsource/log另存v0，修正空白後才開始acceptance_v1，無既有results覆寫或契約變更。這項失敗與quantilepolicyfailure都在incidentmanifest中，不宣稱所有attempts從未失敗。
''')
write('04_FULL_SAMPLE_FIXED_POINT_RESULTS.md',f'''# Full91×7 physical explicit acceptance

所有結果均標FIXED_HISTORICAL_PARAMETER_POINT，不是CORRECTED_MLE。B0和B1各637rows，91個銀河、七個原nativevectors全部覆蓋；無缺件／duplicate／nonfinite／dropcase。無新的q、epsilon、a0、sigma_sys、CI、closure或AIC/BICrankings。

下表是原theta下的explicitB0總值、對P1CcorrectedGaussiansurrogate差、B1−B0差與refinement估計；排列沿原模型order，不代表新ranking。

'''+table(b0,['model','explicit_logL','explicit_minus_surrogate','B1_minus_B0','sum_abs_refinement'],{'explicit_logL':'.9f','explicit_minus_surrogate':'.9f','B1_minus_B0':'.9f','sum_abs_refinement':'.6g'})+f'''
728個native matched comparisons是兩floors各364：重新映射nativeepsilon/log10a0、freshdecode後評估；max|ΔlogL|={s['max_native_mapping_delta']:.6g}。原MOND/PTQ-ν各自歷史theta的總值可有差，不是同一physicalpoint；不能拿它反駁matchedidentity。sharednu-q亦維持同family，不把七parameterizations偷偷減為較少reportedmodels。

full_sample_all1274.csv保存fine/coarse logL、refinement、surrogate差、D/i geometry、physicalpriorZ、權重、node數、floor、Υ/halo boundarymetadata與cost。728mapping、14sampletotals與modeldifferenceerror估計另有CSV。sampleerror同時列signed與sumabs，避免正負抵銷造成虛假pass；保守modeldifference估計是兩個modelsumabs相加。

B0pergal explicit−surrogate約−3.45089至+6.75363；最大abs在linearPTQ/IC2574。這些差異不只是numericalroundoff，也不是參數移動方向或新optimum預測。historicalobjective／Gaussian版本與此explicittarget的估計不可混用；真正newMLE與IC必須另經獲准maximize後才建立。
''')
write('05_B0_B1_AND_BOUNDARY_SENSITIVITY.md',f'''# Fullscope floor、physicalboundary與近似差異

B1使用與B0完全相同的physicalprior／quantilepolicy／full91scope，不是挑選有利subset。最大單銀河B1−B0={float(maxfloor.B1_minus_B0):.9f}，發生{maxfloor.galaxy}/{maxfloor.model}；全樣本floor差在表04逐模型完整列出。linearPTQ sample約+28.166984，MOND約−0.608126，sharedq約−0.858248，NFW約−1.442411；這是模型假設敏感度，不是integrationerror，不能因B1較大便換primary。

全部22個P1Dnear-boundary銀河、七個nominali90、較寬error案例與interiorcontrols均已包含1274rows，不再有P1D的13個Tier1-only空白。七realedge在B0physicalexplicit−Gaussian的七模型signed範圍如下：

'''+table(edge,['galaxy','min','max'],{'min':'.6f','max':'.6f'})+'''
這個表是總target差，不能全部稱為inclinationeffect。P1D correctedpostprocessing_v2已顯示edge7較大差常是distance餘項，inclination-onlymax約0.04386；較寬error F574-1則有約0.28585 inclinationeffect。此處重用既有有界分解，不重跑campaign，不使用invalidv1summaries。

same_physical_prior_linear_pilot.csv對七個事前pilot、七模型兩floor另積linearconditionalmean，與fullnonlinear比較；linear與nonlinear使用相同physicaltruncation，因此可條件式辨別Taylor差。linearphysical−Gaussian亦含physicaltruncationeffects；B1variance仍隨i，該linear比較只線性化mean，不宣稱已把所有變異非線性抽掉。

P1D finiteT5 crosscheck使用correctedv2中同11個historicalgalaxies/all7/floors，共154rows。fullphysical−finiteT5最大abs：B0約2.17e-5、B1約1.72e-4（worstD631-7 linearPTQ B1）。這不是parityfailure：兩者先前finite-tailprior與目前physical-onlyprior定義不同；quantile積分直接保留完整target，而非把T5當無限域。

physicalpriorZ保存D與i各自normalization；i0=90有Zi≈0.5，非線性sin在half-domain真正積分，不把ei歸零或以surrogate替換。各層分解依所選中間model而定，D/i interaction使causalpartition非唯一。rawD/icorrelations與image-levelwarps仍未知；explicitnumericalaccuracy不能證明外部kernel解讀唯一正確。
''')
write('06_OBJECTIVE_COST_AND_STABILITY.md',f'''# Fixedobjective成本、repeatability與實作取捨

主acceptance含pilot兩method、全部B0/B1、native mappings、physical-linear診斷、72鄰域點與repeat：{s['density_node_evaluations']:,}node-density、約{s['elapsed_seconds']:.3f}秒，maxRSS{s['max_process_RSS_bytes']/1e6:.3f}MB，低於凍結1200秒／6B／512MiB。local測試約0.1M與獨立read-only額外約9.8Mnodes合計仍低於6B；接近上限後未追加largeintegration。artifactverification與Gaussian-onlycomparator沒有新增explicitnodes。

單模型全91同時計算B0+B1約2.53–2.59秒，兩次完整NFWrepeat約2.59秒，max總值差{s['repeat_max_abs']:.6g}（reductionroundoff），各fixedqprobe exactrepeat0。這是B0+B1combinedmeasure；不冒稱直接量到B0-onlyruntime。以2.60秒當B0-only保守cost上界，在同hardware/thread/cachepolicy下可供後續預算review。

獨立GaussianB0reference-only comparator全91約{val['Gaussian_min_seconds']:.6f}…{val['Gaussian_max_seconds']:.6f}秒（21個固定timing）。explicit兩floor合計確實較昂貴，約數百倍，但絕對秒級；不是用Gaussian忽略materialcontribution來省成本。hypothetical4000次完整objective可用約2.89小時／model的保守CPUwallestimate作budget起點，七模型約20小時；這不是實際optimizerbenchmark或收斂承諾。blocklocalcalls／futuregradient／hardware會改成本，NFWbudget另審。

節點／weights由外部geometrycache，所有θ用同segmented192policy；不依residual/排名自動切order，避免數值objectivepolicy跳換。128Dnodeblocks、exactweighted-sums與BLAS減少3Dtensor；mean、sigma-dependentvariance與normalization每次重算，沒有凍結parameter-dependentlikelihood。

Neighborhood：P1D原66點＋六個hardfloornearbound點=72，B0/B1共144rows，maxfine−coarse={s['max_neighborhood_refinement']:.6g}，無失敗。full-bounddomain／jointperturbations未全面覆蓋。UGC07151 MOND-screen五個q固定offset呈平滑有限increment且repeat0，僅作數值stability，不是profile、Fishererror、bestq方向或optimizertrajectory。

先前driver syntaxattempt與uniformquantilepilot失敗保留；這不改數學target。mainnumericalexceptions0、fullrowsfinite。下一階段應做真正optimizer-driver wiring/smoke及已批准terminalpointprecisionchecks；本輪沒有optimizer、NFWstarts、line-search、MCMC或profiles。
''')
write('07_INDEPENDENT_REVIEW.md','''# Second-agent read-only adversarial review

在isolatedimplementation/localtests完成後，第二代理 `/root/p1e_independent_target_review` 作code與數值read-onlycritique，沒有改檔／optimizer／MCMC；不是跨模型proof或AI科學權威。machine-readable原意與disposition在INDEPENDENT_REVIEW.json。

獨立確認：quantileCDF/survival、physicalnormalization、unitweights、fixedy0與B0/B1完整Gaussian一致；沒有finiteTtargetcutoff或propagatedCdoublecount。48個fastsum/directgrid比較max3.15e-12、91個nodegeometry皆physical；1274fullrows／728mapped／144neighborhood完整finiteunique，總和與refinement重算僅CSVroundtrip差；1957份sourcehash不變。下界floor三模型另以segmented48/96/192查核穩定。額外independentdensitynodes約9.8M，總預算仍守住。

保留異議與處置：

- refinement只是工程估計，不是嚴格absolutebound／optimizer域uniformproof；後續真實smoke/terminalcandidates須核對精度，不因此開新methodsprogram。
- B1最大單銀河15.626114、linearPTQsample+28.166984，是重大floor模型不確定性；B0依事前coordinatepolicy選定，不按結果換target。
- explicit與Gaussian單銀河差可達−3.45089…+6.75363，不能把surrogate當exactmarginal。
- 外部conditionalkernels的Gaussian/independence/flatpriorconditioning／catalogglobalinclination是declaredassumptions；rawcorrelation與imagelevelsystematics仍未知。
- 2.53–2.59秒是B0+B1合計，不能冒稱B0-onlybenchmark；主nodes近6B，停止額外大積分。
- zeroerror在i90只可說closure/conditional-limit，不是openphysicaldomain內的普通priorpoint；本輪真實errors未改。

雙方均建議唯一A：UNIFORM_EXPLICIT_B0_READY_FOR_FIT_REVIEW。沒有要求改target／tolerance去取得pass；獨立同意不等於科學模型被唯一證實。productionintegration與真正fitting仍由HumanArbiter另授權。
''')
write('08_TARGET_DECISION_AND_LIMITATIONS.md','''# Final statistical target recommendation

**UNIFORM_EXPLICIT_B0_READY_FOR_FIT_REVIEW。**

Computationalvalidity：同一完整physicaltarget、fixedy0measure、完整D/template/NFWtransform、normalizedconditionalGaussian與externalnuisanceprior均明確；localtests、full637×2、nativeidentities、neighborhood與repeatability通過。

Numericalacceptance：globallyselectedsegmentedquantile192/96達原工程門檻，sampleaccumulation及modeldifferenceerror估計分開列；未放寬門檻、未混用target、未把finiteT視為physicaldomain。

Scientificmateriality：explicit修正與floor差異可大，必須重新估計才能知道新bestfits／IC；B0的measurement-coordinate意義清楚且事前批准，但B1仍是科學模型不確定性，不能由fixedpointlikelihood唯一淘汰。原MLE、CI、q、scale、sigma_sys、closure與ranking都未由本輪建立。

Statisticalassumptions：externalGaussianerrors、independence、locallyflat D/i conditioning與globalcatalogtransform均宣告，沒有rawcorrelation／imagelevel reconstruction可驗證。Explicitintegral比Gaussianlinearization更完整地實作已選定假設，不代表完整觀測truth或PTQtheoryvalidation。

Practicality：秒級full91jointfloorobjective及有界memory支持提交boundedMLEreview；比Gaussiansurrogate昂貴，但未在凍結accuracy/costbudget失敗，因此不選Gaussianfallback作primary。P1DGaussiancandidate保留為diagnostic／approximatecomparison，識別版本分開。

這是FINAL bounded pre-fittingtargetgate，不建議再開第四種結果或無限敏感度研究。下一步是Human批准uniformexplicitproductionintegration、實際driver-smoke與有界sevenmodelbudget；不是自動啟動fitting。本輪沒有canonicalintegration或optimizer。Fulloptimizerdomain未被證明precisionuniform，將精度驗證作後續工程acceptance而非新的statisticaltarget選擇。
''')
write('09_INACTIVE_PRODUCTION_AND_REFIT_PLAN.md','''# Inactive uniform explicit integration and refit design

新primaryID：REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1。B1 sensitivity：REV26-P1E-UNIFORM-EXPLICIT-B1-PHYSICAL-v1。不可沿用P1DGaussian-surrogateID；不能某galaxy用explicit、另一些用Gaussian或依start切target。

最小production整合proposal（本輪未做）：新增src/ptquat/marginal_likelihood.py封裝physicalquantilenodes／cache、normalizedconditional、B0/B1與完整pergalmarginal；models.py供完整Dscaledmean（原law不變）；likelihood.py保留conditionalGaussian與Gaussiandiagnostic但primary不再使用propagatedC；fit_global.py、mle.py所有primaryglobal/pergalobjective改為sumlogmarginals。experiments.py的modelcomparison/callers需明確likelihoodversion；不能把nonfactorizedmarginal偷拆成pointwisependentWAIC，舊diagproxy不得冒作同一target。未驗證的caller應failclosed。

整合acceptance：canonical改動先review，對本輪1274fixedrows／728nativecomparisons與newtargetID核對；conditionalmean/floor/normalization、physicalgeometry、parameter-dependentvariance不能cache成舊θ值；metadata保存externalkernel／measure／orderpolicy。P1Dpatch只作Gaussiandiagnostic保留，不混進primaryexplicitobjective。

原七parameterizations及k維持92/93/93/93/94/94/183；原Υ、scale、q、sigma_sys、halo bounds不變，D/i不新增fittedparameters。MLobjective是已declaredmarginalconditional likelihood，不額外加入θprior。pairfamilies可共用physicalkernel計算但七nativeoutputs都要report並映射核對。

開始實際campaign前，Human另批准真正optimizer-driver smoke：完整callchain/normalization、gradient或boundstationarity可觀測、logging、timing與numericalprecision在smokepoints可追查；此task不能在P1E執行。之後才核准predeclaredindependentstarts/seeds、multistartbudget、stoppingcriteria及failurehandling。歷史best只可能warmstart，不保證correctedoptimum；所有startinitial/finalvectors、objectivecomponents、RNGstate、termination/bounds/KKT/gradient、target/data/code/environment/version皆保存。

NFW維持restricted1p/c(M)固定，需另有收斂與multistartreview，不因本輪numericalmarginalpass宣稱globaloptimum。B1另identifier下做一致的sevenmodel sensitivity，不能選更好ranking；uncertainty/profile優先q/scale/sigma_sys且須另授權。不做MCMC預設。

新估計使用reestimated/<new-likelihood-ID>/<run-id>，不改original_recovered、舊01_mle或P1A–P1E。新AIC/BIC只能由真正optimizerconverged值建立；BIC_NPOINTS=1782是within-galaxycorrelations下descriptiveconvention，不是已建立Neff。Manuscript數值替換／publicwrites需再由作者批准。

成本：2.60秒/full91jointfloor作保守primary上界，4000次objective約2.89小時/model供預算review，非optimizer承諾；finite-differencegradients與localblocks／NFW將改cost。先smoke得到實際drivercost再鎖定campaignbudget，不開放式搜尋integration方法。
''')
write('10_NEXT_EXECUTION_PROMPT.md','''# 一個有界下一任務：REV26-P1F Uniform Explicit Production Integration Review and Driver Smoke

僅在HumanArbiter批准後執行。採用P1E唯一target建議UNIFORM_EXPLICIT_B0_READY_FOR_FIT_REVIEW；讀P1E全部reports、TARGET_DECISION、explicit_reference、protocol、fullacceptanceoutputs與SHA256SUMS。停止再開likelihood-selection研究。

1. 在新isolatedcandidate／run-id整合uniformphysicalexplicitB0 primary，使用新likelihoodID；B1另版本sensitivity，P1DGaussian僅diagnostic，不按galaxy/start混用target。保留全部原data/laws/bounds/k；canonical只有作者明確批准才可改。
2. 核對全637×2fixedpoint與728nativeidentity、physicalpriors／fixedy0／fullDtemplates/NFW、conditionalvariance/normalization，所有actualglobal/pergal/diagnosticcallers要標明target，不把marginal拆成假independentpointwise。source/artifacts保全，rejectoverwrite。
3. 若作者同時明確授權optimizer-driver smoke，事先鎖定至多兩個代表parameterizations、各至多一個start與一個iteration、嚴格evaluation/runtimebudget；標為wiring/driver-smoke，不當成MLE或newscience結果。未有此授權則只交patch/test與smokeplan，不能呼叫optimizer。使用固定uniformnodepolicy並在smokepoints核對refinement；這是工程acceptance，不重選target。
4. 交付可審閱productiondiff、所有smoke／parity/logging／cost/failureevidence及inactive七parameterizationcampaignbudget，NFWconvergence另審。只有作者另批准campaign、starts/seeds、B1sensitivity、bound/KKT/uncertainty範圍後才開始真正MLE。

不做fullrefit、MCMC、NFWmultistart、profiles/CI、newICranking、稿件table替換、Symmetry/理論audit、Gitcommit/push/tag/release或public/journalcommunication。結果交HumanArbiter決定正式integration與有界sevenmodel重估；保留最終科學及publication批准。
''')
(B/'README.md').write_text('''# REV26-P1E uniform explicit target numerical acceptance

Unique recommendation: UNIFORM_EXPLICIT_B0_READY_FOR_FIT_REVIEW. New target ID REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1; B1 separate alternativefloor under identicalphysicalprior. No fit/canonicalchange/manuscriptreplacement.

PREDECLARED_PROTOCOL/FROZEN_STATISTICAL_TARGET precede allnumeric work. explicit_reference.py implementsphysical-truncatednormalquantile andexactnormalizedconditionalGaussian; it reusesfrozenP1Claws read-only. All91/all7 useone globallyselected segmentedquantile192,96refinement; fullprobabilitydomain, nofiniteTtargetcutoff. Ordinaryquantilepilot failure ispreserved, nothidden.

Run order:bootstrap.py -> test_local.py -> run_acceptance.py -> verify_and_cost.py -> write_reports.py -> finalize.py. Everyoutputrun-ID refusesoverwrite. Forreproductioncopyonlysource/protocol/inputauthoritytoNEWsiblingrun-IDatsamedepthandrunexistingtemporaryPythonenvironmentfromlocal_v1/environment_freeze; sourcepaths pointtoimmutableprojectinputs. Do notcopycompletedoutputfolders or rerunoriginalbootstrap/freeze. Commandprefix PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/rev26-p1a-env/bin/python.

Complete logs local_execution.log/acceptance_execution.log/verification_execution.log. Firstpre-executiondriver syntaxerror preserved asrun_acceptance_syntax_failure_v0.txt andacceptance_syntax_failure_v0.log; fixedonlywhitespacebeforeacceptance_v1, no scientificcontractchange.

acceptance_v1 savesallpilotorders/methods,1274fullrows(B0/B1each637),14totals,728nativecomparisons,144neighborhoodrows,physical-linear/pastfiniteTcrosschecks,smoothness/repeats/costs,attempts andexceptions. Numericalerrorrows areestimates, notrigorousbounds. Fixedhistoricalpoints areNOTcorrectedMLE ornewAIC/BICrankings.

verification_v1 recomputesfinite/unique/count/sum/errorchecks fromsavedoutputs andtimesGaussian-onlyreference(noextraexplicitnodes). INDEPENDENT_REVIEW retainsseparateagentcritique. TARGET_DECISION andINACTIVE_INTEGRATION_AND_REFIT_PLAN distinguishreviewreadinessfromauthorization. Source/inputhashesandtwoSHA256SUMS/ASSET_MANIFEST preserveallpriorfailedP1Doutputs; onlypostprocessing_v2used.
''')
js(B/'EXECUTION_LOG.json',dict(sequence=[dict(stage='bootstrap',command='/usr/bin/python3 bootstrap.py',status='PASS',prior_files=1957),dict(stage='localcorrectness',command='test_local.py',console='local_execution.log',status='PASS'),dict(stage='driverpreexecution',command='run_acceptance.py initialsyntaxattempt',console='acceptance_syntax_failure_v0.log',status='SyntaxError before any numericcomputation;failedsourcepreserved'),dict(stage='acceptance',command='run_acceptance.py correctedwhitespace',console='acceptance_execution.log',status='PASS;quantilepilotaccuracyfailuregloballyswitchedtopredeclaredfallback'),dict(stage='artifactverification/Gaussiancost',command='verify_and_cost.py',console='verification_execution.log',extra_explicit_integrations=0,status='PASS')],thread_environment='PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1',python_environment='local_v1/environment.json + environment_freeze.txt',independent_review='read-only secondagent;nodecostapprox9.8M',main_node_cost=s['density_node_evaluations'],combined_local_main_independent_node_cost_conservative_upper=5990000000,no_optimizer=True))
print('11reports plus machine targetdecision written')
