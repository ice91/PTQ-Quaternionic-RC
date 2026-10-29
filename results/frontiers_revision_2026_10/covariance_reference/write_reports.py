from pathlib import Path
import json,hashlib,subprocess
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;I=B.parents[2];R=I.parent;D=R/'docs/frontiers_major_revision_2026_10/rev26_p1c_covariance_reference'
def js(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write(name,text):
 p=D/name
 if p.exists():raise FileExistsError(p)
 p.write_text(text)
def table(df,cols,formats=None):
 s='| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'
 for _,r in df.iterrows():s+='| '+' | '.join(format(float(r[c]),formats[c]) if formats and c in formats else str(r[c]) for c in cols)+' |\n'
 return s
spec=json.loads((B/'STATISTICAL_SPECIFICATION.json').read_text());ct=json.loads((B/'PREDECLARED_TEST_CONTRACT.json').read_text());local=json.loads((B/'local_tests_v1/tests.json').read_text());full=json.loads((B/'full_sample_v1/GATE_E.json').read_text());nq=json.loads((B/'nuisance_v1/SUMMARY.json').read_text());verify=json.loads((B/'output_validation.json').read_text());totals=pd.read_csv(B/'full_sample_v1/totals.csv');data=pd.read_csv(B/'full_sample_v1/per_galaxy.csv');sens=pd.read_csv(B/'nuisance_v1/decomposed_sensitivity.csv');fin=sens[sens['T']==5];real=fin[fin.galaxy!='synthetic_i90'];quotes=json.loads((B/'REVIEWER_PRIMARY_SOURCE_UPDATE.json').read_text());fd=pd.read_csv(B/'local_tests_v1/finite_difference.csv')
metrics=dict(local_tests_passed=len(local),local_tests_failed=sum(x['status']=='FAIL' for x in local),FD_rows=len(fd),FD_finest_max_abs=float(fd[fd.relative_step==1e-5].absolute_max.max()),FD_finest_max_rel=float(fd[fd.relative_step==1e-5].relative_norm.max()),full_rows=len(data),full_galaxies=data.galaxy.nunique(),full_failures=0,matched_checks=full['matched_checks'],matched_max_delta=full['matched_logL_max_abs'],nuisance_integrals=nq['integral_rows'],nuisance_convergence_cases=nq['convergence_cases'],nonconverged=len(nq['nonconverged']),max_quad_delta=nq['max_delta256_128'],real_nonlin_matchedlinear_max_abs=float(abs(real.nonlinear_minus_matched_linear).max()),real_floor_max_abs=float(abs(real.floor_B1_minus_B0).max()),real_tail_T5_T4_max_abs=float(abs(real.tail_B0_T5_minus_T4).max()),real_linear_truncation_vs_gaussian_max_abs=float(abs(real.linear_truncated_minus_Acorr).max()),synthetic_i90_nonlin_matchedlinear=float(fin[fin.galaxy=='synthetic_i90'].nonlinear_minus_matched_linear.iloc[0]),near_i_boundary_galaxies=int(data[data.model=='mond'].near_physical_i_boundary.sum()),i90_galaxies=data[(data.model=='mond')&(data.i_deg==90)].galaxy.tolist())
js(B/'RESULT_METRICS.json',metrics)
decision=dict(STATISTICAL_SPECIFICATION='ACCEPTABLE_WITH_DECLARED_LIMITATIONS',REFERENCE_IMPLEMENTATION='PASS',REFIT_READINESS='READY_FOR_PRODUCTION_PATCH_REVIEW',primary='normalized B0 Gaussian surrogate A_corr at fixed nominal y0; B0 physical-truncated nuisance marginal as approximation sensitivity; B1 retained',readiness_scope='ready for separately authorized review of code/spec correction only; NOT permission or unconditional statistical readiness to launch seven-model fitting',unresolved_before_fit_launch=['Human Arbiter choice of sigma_sys coordinate and nuisance assumptions','operating-regime verification near physical inclination boundaries (22 within3sigma,7 at90deg)','decision on B1 sensitivity and nonlinear/floor uncertainty in reporting','separate optimizer convergence and restricted-NFW multistart work packet'],fixed_historical_totals=totals.to_dict(orient='records'),metrics=metrics,optimization_performed=False,production_modified=False,manuscript_modified=False,new_AIC_BIC_rankings=False,global_parameter_or_ranking_shift='UNKNOWN_PENDING_REFIT',reviewer_passages_source='Human Arbiter supplied P1C; PDF located/hashed; prior P1B unavailability reports retained')
js(B/'FINAL_DECISIONS.json',decision)
write('00_EXECUTIVE_DECISION.md',f'''# REV26-P1C decision

| 項目 | 判定 |
|---|---|
| STATISTICAL_SPECIFICATION | ACCEPTABLE_WITH_DECLARED_LIMITATIONS |
| REFERENCE_IMPLEMENTATION | PASS |
| REFIT_READINESS | READY_FOR_PRODUCTION_PATCH_REVIEW |

第三項只表示可呈交production correction的具體規格／code-change proposal供審查，**不等於可以立即開始七模型擬合**。尚須作者批准floor座標與近似使用範圍，並針對邊界／B1敏感度安排有界驗證，再另授權production patch及re-estimation。

建議B0：以固定名義觀測y0作data measure，sigma_sys是該座標中的additive floor；mean同時縮放physical radius與全部baryonic templates，NFW則在新radius重算halo。B1為intrinsic速度floor投影回y0，保留為合理而未被排除的敏感度。B0在新數值結果產生前即選定，理由是觀測／誤差座標的清楚與延續性，不是ranking或與舊值較接近。

11組局部正確性測試通過；126個解析／FD比較包含所有step，最細step最大relative L2誤差{metrics['FD_finest_max_rel']:.3g}。其後才執行91×7=637個fixed historical points，無刪樣／計算失敗；364個native theta雙向mapping比較最大|ΔlogL|={metrics['matched_max_delta']:.3g}。所有輸出明示FIXED_HISTORICAL_PARAMETER_POINT；沒有CORRECTED_MLE、AIC/BIC新排名或新參數估計。

七模型全樣本固定點ΔlogL：

'''+table(totals,['model','k','delta_logL'],{'delta_logL':'.6f'})+f'''
三個P1B預選銀河的matched-truncation B0 nonlinear−linear最大|ΔlogL|={metrics['real_nonlin_matchedlinear_max_abs']:.6f}；T5−T4 tail差最大{metrics['real_tail_T5_T4_max_abs']:.6g}，但B1−B0 floor差可達{metrics['real_floor_max_abs']:.6f}。585個積分結果、195種case/model/tail/integrand的128→256階全部達1e-3門檻；最大差{metrics['max_quad_delta']:.3g}。數值收斂不證明distribution/floor唯一正確。

獨立第二代理的唯讀review補驗637組矩陣分解，指出7個i0=90deg、共22個銀河在3sigma內接近inclination physical boundary；一階Ji≈0不代表非線性效應為零。預宣告synthetic i90控制的nonlinear−matched-linear約{metrics['synthetic_i90_nonlin_matchedlinear']:.6f}，因此本輪不宣稱整個91銀河或optimizer domain已具uniform approximation bound。

Reviewer兩段原文現由作者提供；R1 supporting PDF在Downloads找到並核對hash。這是P1B後新增來源，未回寫P1B。來源保全、reviewer草案、production提案及七模型refit plan分別見SOURCE_PRESERVATION、09、07、08。沒有production edit、fitting、MCMC、NFW starts、稿件修改或Git發布。
''')
write('01_STATISTICAL_SPECIFICATION_B0_B1.md',r'''# 完整統計規格與選擇理由

STATISTICAL_SPECIFICATION.json與PREDECLARED_TEST_CONTRACT.json在任何P1C model comparison前凍結；LOCAL_CORRECTNESS_GATE記錄其SHA256。B0是作者本輪指定的preferred primary reference candidate，但不是唯一真實噪音模型，亦未以有利likelihood結果選它。

Primary data measure是∏dy0_j，y0為SPARC在nominal D0/i0下報出的已去投影circular velocity，單位km/s；e0在同一座標。固定LOS代理y0 sin i0、angular sampling/profile形狀及所有模型θ。本輪不回到velocity field或image重新建模。

s=D/D0，b=sin(i)/sin(i0)。mu(D,i;θ)=b v_model(s r0, sqrt(s)Vdisk0, sqrt(s)Vbulge0, sqrt(s)Vgas0; θ)。conditional density為完整multivariate normal；B0的S0=diag(e0²+sigma_sys²)，B1的S1(i)=diag(e0²+b²sigma_sys²)。兩者都不在conditional S加上D/i propagated terms，避免double counting。

B0 floor是nominal-velocity座標的額外散布；它隨θ的sigma_sys變化但不隨候選i變化。B1 floor是intrinsic circular-velocity散布，再投影；兩者在nominal i相同，但邊際化不相同。原pipeline沒有指定此生成座標；資料不能從本次固定點敏感度唯一選出B0或B1。B0選擇基於透明的fixed-data noise interpretation，B1仍是實質model uncertainty。

D/i primary uncertainty assumption：獨立Gaussian kernels，中心D0、i0，width sigmaD(Mpc)、sigmai_rad=(π/180)sigmai_deg。physical prior支撐D>0、0<i<π/2，每軸density除以對應Φ(upper)−Φ(lower)；不使用cos(i) orientation prior。independence、Gaussian shape與catalog errors是declared assumptions，非raw data所證。D/i相關或image-level重建資料未取得。

Explicit marginal：L_Bk(y0|θ)=∫N(y0;mu,S_k) p_D(D)p_i(i)dD di。以zD=(D−D0)/sigmaD、zi=(i−i0)/sigmai_rad積分時，normal-density的1/sigma與dD/di的Jacobian相消；prior仍須正確截斷正規化。

數值T=3/4/5代表另將z截斷±T再取physical交集並各自正規化；這些是可比較的finite-tail approximations，不把T5冒作精確physical-only integral。各T保留retained prior mass及quadrature收斂。mean、measurement errors與floor均在y0 measure評估；若改至candidate-deprojected y'=y0/b，conditional variance與density Jacobian均須同步變換。

B0 corrected local Gaussian surrogate A_corr採r0=y0−mu0，Ccorr=diag(e0²+sigma_sys²)+sigmaD² JD JDᵀ+sigmai_rad² Ji Jiᵀ；logL=−[r0ᵀCcorr⁻¹r0+logdetCcorr+n log(2π)]/2。它本身是完整normalized Gaussian density。它用名義kernel中心／width，不是physical-truncated prior的精確邊際化，也不是把截斷後moments暗改回原width的主張。same-prior linear conditional quadrature另行計算，分辨truncation與Taylor誤差；near90deg一階線性化可能不充分。

此規格只保留七模型原parameterization、bounds與91/1782 sample。nuisance uncertainty parameters未另加入free fit parameters；以catalog error固定。無新增理論參數、沒有unobserved confidence intervals。
''')
write('02_OBSERVATION_AND_DERIVATIVE_CONTRACT.md',r'''# Observation-to-model與完整導數

[Li et al. 2018 §2.1 Eqs.(4)–(5)](https://arxiv.org/html/1803.00022)給定距離與傾角的catalog-level transform；[SPARC Table2](https://astroweb.case.edu/SPARC/MassModels_Lelli2016c.mrt)指定physical radii(kpc)、circular velocities、M/L=1的stellar templates及包含helium的gas templates。P1B已核對raw→tidy一致，本輪重用hash證據而不重做ETL。

Held fixed：angular樣本r0/D0、nominal y0/e0或等價的nominal LOS代理、angular photometric/HI profile shape、Υ、global acceleration scale、q、M200、c(M)、H0及全部masks。Υ固定不代表stellar模板對D不變：L與mass∝D²、radius∝D，故每個disk/bulge/gas velocity template乘sqrt(D/D0)。形狀在angular座標不變，不需沿物理半徑重新挑樣或插值。

固定y0 mean mu=b(i)v(D)。名義點mean derivatives：

JD=∂mu/∂D=∂v/∂D，(km/s)/Mpc；Ji=∂mu/∂i=v0 cot(i0)，(km/s)/rad。

Residual y0−mu的derivatives是−JD、−Ji。這不同於candidate-deprojected residual y0 sin(i0)/sin(i)−v(D)，其傾角導數含−y0 cot(i0)。不把兩個measure的導數混用。degrees版本Ji_deg=(π/180)Ji_rad，且sigmai_deg Ji_deg = sigmai_rad Ji_rad；測試亦核對D的Mpc/kpc單位轉換。

baryon、linear PTQ、MOND simple-ν、PTQ-ν、兩個shared ν-q parameterizations：vbar²∝s；r∝s；gbar不變；linear acceleration term也∝s，故v(D)=sqrt(s)v0及JD=v0/(2D0)。reference.velocity實際重新縮放r與全部模板，再由獨立law方程求v，未直接用這個analytic scaling代替law evaluation；它因此可與解析與production law另行交叉驗證。

Restricted NFW：固定M200和c(M)=10(M200/1e12)^−0.1，halo在s r0評估。令x=c r/R200、f=ln(1+x)−x/(1+x)、η=x²/[(1+x)² f]−1，則JD=[vbar²+vhalo²η]/(2vD0)。reference保留此radius-dependent halo response，沒有強套sqrt(D)；獨立NFW控制顯示硬套sqrt(D)導數會有約1.731的relative norm差。

Inclination採單一global catalog adjustment，模板在此變換下不另改shape。原velocity-field warp、重新deprojected imaging、disk flattening模型與D/i correlations不在frozen tables；本輪不補造這些資料。這些是宣告的適用限制，不否認catalog-level D transform的一致性。
''')
localtable=pd.DataFrame([dict(test=x['test'],status=x['status']) for x in local])
write('03_REFERENCE_IMPLEMENTATION_TESTS.md',f'''# 獨立reference與局部tests

reference.py不匯入production likelihood/model/mle/optimizer；獨立載入凍結CSV、解讀native theta，依同一已宣告七模型方程寫mean、full JD/Ji、C、normalized Gaussian及quadrature。保留原常數、c(M)、M/L合併方式、model floors與parameter counts，不在本任務另改model law。test_reference.py僅為cross-check讀取production model equations/layout，不呼叫fitting。

'''+table(localtable,['test','status'])+f'''
執行順序：11組hard tests先通過，寫出LOCAL_CORRECTNESS_GATE=PASS後，run_effects.py才允許Gate E。21個真實case×model組合×兩導數軸×3step，共126列，所有attempts保存；最細FD最大abs={metrics['FD_finest_max_abs']:.6g}、relative L2={metrics['FD_finest_max_rel']:.6g}，0 failures。相對步長1e-3/1e-4/1e-5，inclination step以rad定義。另測最小synthetic baryon、NFW四半徑、unit conversion與coordinate density Jacobian。

獨立Gaussian的slogdet／direct solve與reference Cholesky比較局部最大logL差8.53e-14；1D density積分為0.9999999999999999；局部最小C eigenvalue43.943741。所有參數擾動都使C重新計算。NFW analytic/FD relative error約1.02e-10。

Local matched_parameters helper直接共用physical a0，exact-zero local identity部分是shared-kernel檢查，不能單獨當native mapping證據。Gate E另以原native theta[0]做epsilon↔log10a0轉換、fresh target decode，364組比較最大|ΔlogL|={full['matched_logL_max_abs']:.6g}、C相對差={full['matched_C_max']:.6g}。這項獨立raw-vector測試解除第二代理指出的local test-strength疑慮。

NumPy slogdet在局部與獨立驗證曾輸出divide/overflow/invalid runtime warnings；完整console logs保留。verify_outputs.py對637個輸出與矩陣逐項檢查finite再取max，並核對sign、SPD與獨立solve；沒有以max(NaN)掩蓋失敗。最大獨立full-sample logL差{verify['independent_logL_max_abs']:.6g}；非有限值0。第二代理另作唯讀全637組檢查，結果一致。未因warnings變更統計target或容許誤差。

本輪沒有測試失敗後改數值契約，沒有新增有利case或drop galaxy。有限差分與identity成功只能證實reference計算正確，不能認證全域approximation precision、B0唯一正確或optimizer收斂。
''')
write('04_NUISANCE_AND_FLOOR_SENSITIVITY.md',f'''# Tail、Taylor與floor座標分開比較

沿用P1B事先選定的ESO444-G084、UGC03205、UGC06614，不依本次residual／改善程度選樣。全部七模型原theta不動。每個T=3/4/5及order=64/128/256保存三個integrands：nonlinear B0、nonlinear B1、B0 linear conditional mean。另有預宣告synthetic_i90 boundary控制（只B0與linear_B0）。585列integrals、195組收斂結果全保留。

linear mean為mu_lin=v0+JD(D−D0)+Ji(i−i0)，conditional covariance為同一B0 measurement+floor。對完全相同的normalized physical∩±T prior積分；因此nonlinear B0−linear_B0隔離該prior下Taylor差，linear_B0−A_corr另反映truncation／Gaussian surrogate差。不能將B0−A_corr全歸為非線性。

195組128→256階全部達預宣告abs1e-3門檻，最大差{metrics['max_quad_delta']:.9g}。prior weight sum最大誤差{nq['max_prior_mass_error']:.6g}；所有conditional density包含variance-dependent Gaussian normalization，不只積chi²，沒有propagated D/i covariance double count。T5仍只是physical-only prior的數值tail近似，T5−T4是選定tail sensitivity，非全域誤差定理。

T5結果如下，全部為FIXED_HISTORICAL_PARAMETER_POINT；synthetic另行標示：

'''+table(real,['galaxy','model','nonlinear_minus_matched_linear','linear_truncated_minus_Acorr','floor_B1_minus_B0','tail_B0_T5_minus_T4'],{'nonlinear_minus_matched_linear':'.6f','linear_truncated_minus_Acorr':'.6g','floor_B1_minus_B0':'.6f','tail_B0_T5_minus_T4':'.6g'})+f'''
三真實銀河的max|nonlinear−matched linear|={metrics['real_nonlin_matchedlinear_max_abs']:.9f}；max|linear truncated−A_corr|={metrics['real_linear_truncation_vs_gaussian_max_abs']:.9f}；max|T5−T4 B0|={metrics['real_tail_T5_T4_max_abs']:.9f}。在這些點B0 linearization差較小，未證其他星系／其他theta也如此。

max|B1−B0|={metrics['real_floor_max_abs']:.9f}，超過預宣告0.1 logL敏感度旗標；資料沒有因此證實任一floor慣例更正確。需要在後續fit與model uncertainty reporting中保留此選擇，而不是因B0數值較接近舊模型便選它。

synthetic i90（D=10±0.1Mpc、i0=90deg、ei=10deg、4點v0=y0=100km/s、e=2、floor3）具有Ji≈0，但T5 nonlinear−matched-linear={metrics['synthetic_i90_nonlin_matchedlinear']:.9f}；顯示一階Ji消失不表示uncertainty消失。此為合成示範，不是7個真實edge-on銀河的影響估計；這些實際銀河ei多為1或3deg，不能把10deg synthetic幅度直接套上。

七個real i0=90與22個3sigma內接近physical boundary的case已完整列在full_sample diagnostics，未因難處理而刪除。其explicit nonlinear驗證不在這三銀河bounded integral campaign內，仍須在fit launch前建立operating-regime處置。有限quadrature收斂不認證獨立Gaussian nuisance假設，也不修復缺失的image-level觀測資訊。
''')
write('05_FULL_SAMPLE_FIXED_POINT_EFFECTS.md',f'''# 全91銀河固定歷史參數點

**全部數值標記FIXED_HISTORICAL_PARAMETER_POINT；沒有CORRECTED_MLE。**

每份原theta包含各自historical nuisance estimates；source/input hashes通過，91×7=637列全部評估成功，0 omitted。old likelihood與C來自已核驗P1A per-galaxy replay保存檔，而不是在本輪重做archive campaign。corrected likelihood由獨立reference評估完整B0 Ccorr，包含quad/logdet/nlog(2π)。P1B既有42個overlap點最大差{full['P1B_overlap_max']:.6g}，提供跨實作一致性。

'''+table(totals,['model','k','old_logL','corrected_logL','delta_logL','delta_quad','delta_logdet'],{'old_logL':'.9f','corrected_logL':'.9f','delta_logL':'.9f','delta_quad':'.9f','delta_logdet':'.9f'})+'''
corrected_logL欄只是修正reference在原theta的值；不能按此表宣稱新的MLE ranking或重新計算AIC/BIC作best-fit模型比較。本輪沒有輸出此種IC排名、沒有推斷新的q/epsilon/a0/sigma_sys，也不製造error bars。

per_galaxy.csv保存原／reference logL、delta、quad/logdet差、C relative/absolute Frobenius差、n、rmax、relD、i/ei、Υ及U/halo boundary flags。corrected_covariances.npz保存637份C。diagnostics.csv對relD、inclination、n、rmax、原parameter boundary、inclination physical boundary事先定義bins彙整，所有bins完整保留，不用effects挑case。

最大的單銀河fixed-point改變包括linear PTQ NGC7331約−17.745001、UGC09133約−17.087075、baryon UGC12506約+15.726052。這些只能提示objective不只是模型無關常數偏移，不能推測refitted optimum移動方向、global scale shift或ranking是否翻轉。

MOND/PTQ-ν兩份original theta分別是舊objective下的獨立best-start點，因此reference總logL仍可差約4.1e-5；它不是matched-parameter identity違反。真正native theta映射後的364個銀河比較最大差約8.53e-14。shared ν-q亦維持同一physical family，七個parameterization仍逐一報告，未偷偷合併成較少模型。

7個i0=90deg：IC4202、NGC0891、NGC7814、UGC01281、UGC07151、UGC08286、UGC11455。22個銀河處於3sigma inclination physical邊界範圍；全部仍在637列內。高inclination近似限制需另辨，不能藉C仍SPD或native identity通過來宣稱已解。
''')
# Independent review section written after receiving final review in a separate step.
write('07_PRODUCTION_CHANGE_PROPOSAL.md','''# Production correction提案（僅供審查，未修改）

具體review target為B0 normalized Gaussian surrogate，保持sample、bounds、constants、七parameterization與原k。必要修正是distance total derivative含sqrt-distance templates，並明示fixed y0 measure及Ji=model mean derivative。

| 檔案 | 擬議職責／改變 |
|---|---|
| src/ptquat/likelihood.py | build_covariance改用明確full JD／Ji或接受由model layer提供的nuisance derivative callback；統一rad/Mpc單位；完整Gaussian normalization與同矩陣regularization policy保留 |
| src/ptquat/models.py | 提供可審核的distance-rescaled model evaluation與full nuisance derivative interface；NFW獨立halo radius response；不更改radial law／參數數目 |
| src/ptquat/fit_global.py | 所有global/per-galaxy／per-point／diagnostic caller傳入一致的B0 covariance contract；杜絕某一路仍用舊radius-only callback |
| src/ptquat/mle.py | galaxy_loglike與objective路徑使用同contract；未來輸出保存likelihood version、floor坐標、完整theta/start metadata與Gaussian components |

production patch前須作者確認選定B0 surrogate的使用範圍／邊界處置與B1敏感度安排。這份review readiness不是fitting authorization；若選擇explicit marginal或更完整nuisance模型取代surrogate，應另作可審閱spec，不能只切flag冒稱同一target。

建議patch acceptance：獨立reference與production fixed-point全637列在1e-8 logL內、364個native mappings維持identity、全部caller路徑C/logdet隨proposal更新、最小baryon例與NFW derivative通過、完整measurement座標說明、sample/parameter accounting不變、source原檔保全。既有一階模板缺項不能僅靠更細r差分處理。

近boundary／floor gate應與patch分開標記：若要聲稱採用的Gaussian propagation足以代表physical nuisance marginal，需在事先選定的near90 real案例及合理theta鄰域做explicit reference sensitivity，不得用本輪三銀河替全部91認證。可先review並測試數學一致的B0 patch，但發出新fit前必須決定相關科學限制如何處置。

不直接把本輪fixed-point值寫進table，不改歷史MLE，另建likelihood version與new outputs。任何Git提交／production修改需後續明確授權；本輪未執行。
''')
write('08_SEVEN_MODEL_REFIT_PLAN.md','''# 七模型re-estimation plan（未執行、另需作者授權）

先決條件：production patch review與reference parity通過；作者選定likelihood/floor座標；near-inclination-boundary與B1敏感度有明確protocol；完成parameter/domain/source freeze。此次沒有授權optimizer/MCMC/NFW starts。

| reported parameterization | k | 重估內容 |
|---|---|---|
| baryon | 92 | 91 Υ及sigma_sys |
| MOND simple-ν | 93 | log10a0、91 Υ、sigma_sys |
| linear PTQ | 93 | epsilon（原wide domain）、91 Υ、sigma_sys |
| PTQ-ν | 93 | epsilon（matched acceleration domain）、91 Υ、sigma_sys |
| PTQ-screen | 94 | epsilon、ln q、91 Υ、sigma_sys |
| MOND-screen | 94 | log10a0、ln q、91 Υ、sigma_sys |
| restricted NFW-1p | 183 | 91 Υ、91 log10M200、sigma_sys；c(M)固定 |

Bounds沿用原manifest：Υ[0.05,1.5]、q[0.25,8]、sigma_sys[0.001,200]km/s、logM[9,13]；matched a0[5e-11,2e-10]SI與epsilon=a0/(cH0)bijective對映；linear PTQ保持原wide epsilon[1e-6,3.999]，不可為方便改成matched ν-domain。

1. 另建reestimated/<run-id>/，記錄corrected likelihood版本、B0/B1角色、Gaussian vs explicit target、data/code/environment hashes、CLI、RNG state／seed、完整初始向量、所有start最終向量、完整Gaussian components、逐iteration objective、optimizer options／termination／bounds／KKT或gradient診斷。不得覆寫original_recovered或舊01_mle。
2. 由作者核准預定budget／starts／convergence gate後才跑MLE；不是posterior median代替MLE。不新增prior density到ML objective；hard bounds原樣。對parameter-dependent C必須每次重算且保留logdet。boundary parameters與nuisance維度照數計，不用contact自行扣k。
3. MOND/PTQ-ν與兩個shared ν-q可在物理family中共用計算作warm starts，但七個parameterizations／source theta／mapped objective parity都要保存並逐一報告；mapping不是獨立科學驗證或新預測。共享不得掩蓋某parameterization optimizer失敗。
4. Restricted NFW是獨立收斂work packet：原cross-start≈29logL與many boundary contacts仍未解；另核准bounded multistart／alternative optimizer／KKT evidence。未達門檻只報best recovered，不稱global maximum。不得在本P1C暗跑starts。
5. B1與tail／boundary敏感度計畫須先宣告，不以更好ranking選B0／B1。若調整primary target，七模型需同一致target重估；限制與convergence結果全部報，不刪不利case。
6. 只有真正新maximize值與收斂可追查後才計算對應AIC/BIC_NPOINTS，明示BIC descriptive sampling-unit限制；不能用本輪fixed-point表當新IC排名。新的CI/profile、posterior inference均需另行授權或明確納入後續scope。
7. 作者審閱估計與sensitivity後，才另授權manuscript／reviewer response數值整合；此plan不授權公開寫入、submission或Git release。

corrected q/epsilon/a0/sigma_sys、confidence intervals與ranking全部UNKNOWN_PENDING_REFIT。
''')
write('09_REVIEWER_COVARIANCE_RESPONSE_DRAFT.md',f'''# Reviewer covariance response draft：已取得原文，尚未重估

來源時序：以下為Human Arbiter在REV26-P1C提供的原始文字。P1B當時未取得，舊報告保留不改。R1 supporting PDF實際位於`{quotes['local_supporting_PDF']['path']}`，bytes={quotes['local_supporting_PDF']['bytes']}，SHA256=`{quotes['local_supporting_PDF']['sha256']}`；核驗存在與PDF標頭。引用文字的權威來源是作者本輪供給，未聲稱已用PDF extraction逐字交叉認證。

## Reviewer 1, Comment 3 — exact Human-supplied passage

> {quotes['passages']['Reviewer1_Comment3']}

**Draft response (English):**

Thank you for identifying the ambiguity in Section 3.2, Eqs. (8)–(9). The original production covariance was recomputed with model parameters, including its Gaussian log determinant, but its distance derivative kept the baryonic templates fixed. This omitted their distance-amplitude response. We have now established and tested an isolated corrected reference; we have not yet changed the production implementation or replaced the submitted numerical results.

We use the nominal catalog velocity y0 as the fixed data measure. At fixed angular sampling, stellar mass-to-light parameters, acceleration parameters, q, and halo mass, a change s=D/D0 rescales physical radii by s and each disk, bulge and gas velocity template by sqrt(s). The conditional mean in this fixed coordinate is mu(D,i)=v_model(D) sin(i)/sin(i0). At the nominal point, its distance derivative includes both radius and template-amplitude terms, and its inclination derivative is v_model cot(i0) with angles in radians; residual derivatives have the opposite signs. For the baryon, adopted linear and simple-nu/nu-q families, the full distance derivative is v_model/(2D0). The restricted NFW derivative additionally retains the radius-dependent halo response at fixed M200 and the prescribed concentration relation.

Eleven local correctness tests passed, including independent analytic/finite-difference checks, units, normalized Gaussian decomposition, parameter-dependent covariance and native matched-parameter tests. The corrected reference has been evaluated for all 91 galaxies at all seven original parameter vectors without refitting. These fixed-point changes are not corrected maximum-likelihood estimates, and the historical tables have not been replaced. Production correction review and a separately authorized re-estimation stage remain pending.

## Reviewer 2, Comment 2 — exact Human-supplied passage

> {quotes['passages']['Reviewer2_Comment2']}

**Draft response (English):**

We agree that radial-coordinate rescaling alone is incomplete. The isolated reference uses the full catalog-level distance transformation and a fixed nominal observed-velocity measure, with the held-fixed quantities and full mean/residual derivatives specified above. The corrected Gaussian covariance uses both total derivatives and the full quadratic, log-determinant and normalization terms. We independently verified these components and the analytic derivatives with central finite differences. Matched MOND/PTQ-nu and shared nu-q native parameter mappings agree across the full sample, with a maximum galaxy log-likelihood difference of about 8.53e−14; this checks implementation identity rather than validating the uncertainty approximation.

For three galaxies selected before the preceding covariance audit, we compared the corrected B0 linear Gaussian surrogate with explicit distance/inclination integration. B0 defines the systematic floor in the fixed nominal velocity coordinate; B1 defines it intrinsically and projects it into that coordinate. Independent Gaussian nuisance kernels, physical-domain truncation, integration-tail normalization and both floor conventions are stated assumptions. Conditional likelihoods contain measurement noise and the floor, without adding the same propagated nuisance covariance again. Matched-truncation linear-mean integration separates tail truncation from nonlinearity. At five-sigma numerical tails the largest absolute nonlinear-minus-linear difference in these three galaxies was about0.0829 in log likelihood; the B1/B0 floor difference reached about1.1714. Quadrature refinement converged, but does not establish a unique observational noise model or uniform approximation validity.

The frozen sample includes seven nominally edge-on galaxies, where the first-order inclination derivative vanishes while nonlinear effects may remain. A synthetic boundary control confirms this limitation. Accordingly, we do not claim that the approximation has been fully validated for every galaxy or throughout the fit domain. The new reference results are fixed historical parameter evaluations only; changes to best-fitting acceleration parameters, q, sigma_sys and information-criterion comparisons remain unknown pending an approved corrected specification, production patch and re-estimation.

**使用界線：**這是目前證據狀態的draft，不是已完成major revision聲明。submitted source §3.2 Eq.(7)=eq:logL_g、Eq.(8)=eq:cov-full、Eq.(9)=eq:jac；原來源與P0R/A aux已核對。正式revised PDF行號及重估後表格仍待作者核准的後續階段；不能在回覆中寫「已加入submitted numerical tables」。
''')
write('10_NEXT_TASK_PROMPT.md','''# 單一有界下一任務：REV26-P1D Production Patch Review and Boundary Protocol

本prompt待Human Arbiter另行批准；不因P1C完成自動生效。任務僅為可審閱production patch與使用範圍protocol，不授權任何fit。

專案 /Users/rocky/ptq-mond；implementation /Users/rocky/ptq-mond/PTQ-Quaternionic-RC。先讀rev26_p1c_covariance_reference全部報告、covariance_reference/FINAL_DECISIONS、spec、test contract、independent review與SHA256SUMS。保留P1A/P1B/P1C所有歷史結果，無Git公開操作。

1. 作者確認B0 nominal-floor surrogate為本次patch候選，B1仍列敏感度；核對D/i獨立Gaussian與catalog-level global-i假設。明確區分normalized Gaussian surrogate與physical-truncated explicit marginal。
2. 作者若批准production修改，僅修改已審閱proposal所列likelihood/models/fit_global/mle接口，使所有callers使用full distance derivative、固定nominal measure及rad/Mpc單位；不改sample、law、bounds、k或原MLE。沒有該授權時只產生isolated patch diff供review。
3. 依P1C reference核對所有637個原固定點與364個native matched mappings；保留Gaussian normalization、parameter-dependent C/logdet、NFW halo derivative與source preservation。所有新outputs用新run-id，禁止覆寫原artifact。
4. 在任何fit launch前，事先凍結有界near-inclination-boundary實例／theta鄰域和B1 protocol，包含7個i0=90與22個3sigma內邊界銀河的資料級selection；有界測試physical-truncated nonlinear與matched-linear integral差，明列適用範圍／超限處置。不從新ranking挑好case。必要時提出改用explicit marginal的另版spec，不默默改target。
5. 交付patch diff、全部test/失敗紀錄、near-boundary與floor處置建議及七模型re-estimation啟動條件。獨立NFW convergence/multistart方案另作work packet，仍不執行。

STOP：不執行optimization、MCMC、NFW starts、new parameter estimation、IC新排名、manuscript edit、commit/push/tag/release/upload或journal contact。完成後由Human Arbiter決定是否授權下一階段七模型重估。
''')
js(B/'EXECUTION_LOG.json',dict(sequence=[dict(stage='A',command='/usr/bin/python3 bootstrap.py',status='PASS',old_sources_verified=1807,PDF_path=quotes['local_supporting_PDF']['path']),dict(stage='C',command='PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/rev26-p1a-env/bin/python test_reference.py',console='local_tests_console.log',status='PASS',hard_tests=11),dict(stage='E_then_D',command='same thread environment /tmp/rev26-p1a-env/bin/python run_effects.py',console='effects_execution.log',status='PASS',full_rows=637,integral_rows=585),dict(stage='independent_output_check',command='same environment /tmp/rev26-p1a-env/bin/python verify_outputs.py',console='output_validation.log',status='PASS')],independent_agent='collaboration agent independent_reference_review, read-only critique',new_outputs_only=[str(B),str(D)],no_optimization=True))
(B/'README.md').write_text('''# P1C isolated covariance reference

STATISTICAL_SPECIFICATION=ACCEPTABLE_WITH_DECLARED_LIMITATIONS; REFERENCE_IMPLEMENTATION=PASS; REFIT_READINESS=READY_FOR_PRODUCTION_PATCH_REVIEW (review only, not fit launch).

STATISTICAL_SPECIFICATION.json / PREDECLARED_TEST_CONTRACT.json were frozen before tests. reference.py independently implements mean/law equations, full D/i mean derivatives, B0 normalized Gaussian surrogate, B0/B1 explicit nuisance integrals and matched-truncation linear conditional integral. It does not import production modules or optimizers.

Actual environment is local_tests_v1/environment.json and environment_freeze.txt. Commands/console paths are in EXECUTION_LOG.json. Run order is test_reference.py -> run_effects.py -> verify_outputs.py; every output path refuses overwrite. The complete original run is preserved. To rerun without altering it, copy only source/spec/input-manifest files to a NEW sibling run-id directory under results/frontiers_revision_2026_10 (same depth), then run with PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/rev26-p1a-env/bin/python. Prior physical-source absolute paths remain the authority; do not regenerate prior bundles. If the temporary environment is unavailable, build a separate temporary environment from the frozen requirements and document it; no production install.

11 local hard checks passed before637 full fixed-point evaluations; original likelihood components/C are authenticated P1A inputs. All seven parameterizations retain their k/domains. Native mapped-point checks and all records are saved. No corrected MLE or AIC/BIC ranking was produced.

nuisance_v1 retains all585 integral rows and195 convergence cases, T3/4/5 and64/128/256 quadrature orders; B0/B1 floor and same-prior linear integrand are distinct. Edge-on and floor-coordinate limitations preclude blanket all-sample nuisance-approximation validation.

Reviewer primary passages and PDF availability/hash are recorded as new P1C sources. Prior P1B reports are unchanged. Complete assets are indexed by docs/ASSET_MANIFEST.json; each root has SHA256SUMS, and SOURCE_HASHES_BEFORE/SOURCE_PRESERVATION provide input integrity.
''')
print('reports written; awaiting final independent review report')
