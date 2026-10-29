from pathlib import Path
import json,hashlib,sys
import pandas as pd
B=Path(__file__).resolve().parent;I=B.parents[2];R=I.parent;D=R/'docs/frontiers_major_revision_2026_10/rev26_p1b_covariance_gate';V=B/'run_v1'
G=json.loads((V/'GATE_VERDICTS.json').read_text());S=G['statistics'];design=json.loads((B/'PREDECLARED_DESIGN.json').read_text());e=pd.read_csv(V/'covariance_effects.csv');f=pd.read_csv(V/'finite_difference_summary.csv');u=pd.read_csv(V/'covariance_updates.csv');q=pd.read_csv(V/'explicit_nuisance.csv');conv=pd.read_csv(V/'quadrature_convergence.csv')
def write(n,s):(D/n).write_text(s)
def table(df,cols,formats=None):
 s='| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'
 for _,r in df.iterrows():s+='| '+' | '.join((format(float(r[c]),formats[c]) if formats and c in formats else str(r[c])) for c in cols)+' |\n'
 return s
sources=[dict(url='https://arxiv.org/html/1803.00022',version='1803.00022v2, 2018-07-03',locator='Sec2.1 Eqs4–5 and Sec2.2',role='catalog distance/template scaling, velocity/error deprojection, explicit positivity and global inclination approximation'),dict(url='https://astroweb.case.edu/SPARC/MassModels_Lelli2016c.mrt',locator='Table2 header notes1–3',role='radius kpc, circular velocities, random error excludes inclination, gas1.33 helium, stellar templates M/L1'),dict(url='https://astroweb.case.edu/SPARC/',locator='Basic SPARC data; photometric and mass-model product descriptions',role='distinguish raw photometry from inclination-corrected mass-model products'),dict(url='https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/AJ/152/157?format=html&tex=true',role='attempted upstream catalog ReadMe',status='web open timeout; not relied upon as successfully fetched')]
(B/'EXTERNAL_SOURCES.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False)+'\n')
verdict_table='| 驗證層級 | 判定 | 證據 |\n|---|---|---|\n'
for k,v in G['verdicts'].items():verdict_table+=f"| {k} | {v['status']} | {v['evidence']} |\n"
write('00_GATE_EXECUTIVE_VERDICT.md','''# REV26-P1B covariance gate

**唯一 overall verdict：COVARIANCE_CORRECTION_REQUIRED_BEFORE_REFIT。**

production 會更新自己所定義的 C(theta) 與 logdet；但距離差分固定 baryonic templates，漏掉標準 SPARC 的幅度縮放。這是科學依賴缺項，不是 roundoff，也不被歷史 objective replay 或 matched-family identity 的成功抵銷。

'''+verdict_table+f'''
六個預選真實銀河×七模型，加三個明示合成 baryon 控制；405項FD摘要，最細步長135項全數通過解析導數比對，最大相對norm誤差 {S['finest_fd_analytic_max_rel_norm']:.3g}。最粗步長45項傾角差分未過門檻，全部保留。168個固定callback測試、108個matched點、3銀河×7模型×2 floor慣例×4 quadrature解析度，無執行例外；303份保存C均SPD，沒有jitter。

只補完整D導數、保留原model-based傾角項時，42個真實固定點ΔlogL範圍 {S['distance_only_real_min_delta_logL']:.6f} 至 {S['distance_only_real_max_delta_logL']:.6f}，最大相對Frobenius C差 {S['distance_only_real_max_C_relative']:.6f}。這些不是全樣本校正後maxima。

顯式nuisance積分的42種case/model/floor組合在64→128階差皆低於1e-3（最大 {S['quadrature_max_128_64']:.6g}）；數值收斂不代表其截斷Gaussian、獨立D/i或floor座標假設已由資料認證。原A與B0固定點差可達約−2.83916 logL。完整D線性參考與B0在本次三銀河的差至多0.083372；B1 floor慣例可至1.087491。不能外推全樣本。

原MLE表可保留為凍結歷史objective的可重播紀錄；不得當作已通過完整D/i驗證或校正後有效的MLE／AIC／BIC。production correction有必要，但本輪未改code。任何新fitting均須作者另行授權，且先核准統計specification與隔離reference驗證；七個模型均受D缺項影響，若採修正規格，七者都需重新估計。全局scale、sigma_sys、q與ranking影響為UNKNOWN_PENDING_REFIT。

原始Reviewer1 C3／Reviewer2 C2文字本機未取得，已向作者請求；07是證據草案，無冒造quote。資料級global D/i慣例可查，但影像級inclination重新建模與D/i correlation仍缺來源。完整source保全見SOURCE_PRESERVATION.md；本次沒有refit、MCMC、稿件／release修改、公開寫入或Git狀態重設。
''')
write('01_OBSERVATION_TRANSFORM_AND_JACOBIAN_LEDGER.md',r'''# 固定觀測量、座標與導數契約

`fetch_vizier.py` 讀 J/AJ/152/157；`preprocess.py` 將 table2 Rad/Vobs/e_Vobs/Vgas/Vdisk/Vbulge 直接改欄名，合併table1 D/eD/i/ei並切樣。raw→tidy的1782列速度、誤差、模板、名義D全部差0，table1幾何四欄亦差0（RAW_TIDY_LINK、NOMINAL_GEOMETRY_LINK）。不是重新下載或重製frozen data。

[SPARC原始Table2說明](https://astroweb.case.edu/SPARC/MassModels_Lelli2016c.mrt)指定R為kpc、Vobs為circular velocity、e_Vobs不含inclination systematic；Vdisk/Vbul以[3.6] M/L=1給定，Vgas已含1.33 helium。這些是名義D0/i0下的產品，不是LOS速度或角半徑。production再用單一Υ乘disk²+bulge²，加gas²；保留原定義。

[Li et al. 2018 §2.1 Eqs.(4)–(5)](https://arxiv.org/html/1803.00022)提供本輪採用的catalog-level transform。下列推導是將該transform代入本專案固定模型所得，不是聲稱原文驗證本likelihood。

令s=D/D0，a(i)=sin(i0)/sin(i)，b(i)=1/a(i)。固定angular samples α=r0/D0、nominal LOS投影y0 sin(i0)及其誤差、角向photometric/HI profile形狀、Υ、acceleration scale、q、halo mass/c(M)、H0；不重新擬合。

| 量 | D擾動 | i擾動 | 單位／範圍 |
|---|---|---|---|
| physical sample radius | r=s r0 | global-catalog approximation固定 | kpc；角向profile形狀不變 |
| disk/bulge/gas模板 | Vk=sqrt(s) Vk0 | 標準catalog rescaling不另改模板 | km/s；形狀在r/s座標相同 |
| stellar/gas mass | flux保持；M∝D² | 未重做影像deprojection | Υ固定；氣體conversion不變 |
| y_obs | 對D不變 | y=a y0 | 已deprojected km/s |
| e_obs | 對D不變 | e=a e0 | km/s；與y同變換 |
| model vm | 重新算r及全部模板 | catalog convention下不變 | 原production law；不改參數 |

r_trans(D,i;theta)=a(i)y0−vm(s;theta)。因此名義點：

J_D_full=−∂vm/∂D，單位(km/s)/Mpc；J_i_full=−y0 cot(i0)，i用rad，單位(km/s)/rad。資料誤差的導數de/di=−e0 cot(i0)也必須在顯式conditional density變換中計入；只改residual、不改density/error會定義另一個likelihood。

對baryon、linear PTQ、simple-ν兩種參數化、shared ν-q兩種參數化，vm(s)=sqrt(s)vm0：vbar²∝s、r∝s、gbar=vbar²/r不變，故J_D_full=−vm0/(2D0)。這是固定angular profile下的total derivative。production以固定模板的vfun(r±0.01kpc)求partial derivative；它不是完整rotation curve的total distance derivative。baryon vfun根本不依r，production JD=0，遺漏立即可見。

NFW固定M200/c時，令x=c r/R200、f=ln(1+x)−x/(1+x)、η=x²/[(1+x)²f]−1，則∂vm/∂D=[vbar²+vhalo² η]/(2 vm D)。NFW也有同一baryon amplitude缺項。本次解析式與FD逐點核對，未改production模型。

production儲存JD^p=+(∂vm/∂r)_templates-fixed r/D、Ji^p=+vm cot(i)。報告與CSV保留其原始意義；為比較r=y−vm，CSV的production_residual_oriented明示取負號。獨立D/i的單一outer product不受該整體符號反轉影響，但這不修復缺失幅度，也不掩蓋∂r/∂i用y或vm的差別；未建立correlated D/i的cross terms。

**傾角的兩種殘差必須區分。** 在固定primary觀測座標y0中，r_nom=y0−b(i)vm(D)，所以J_i_nom=−vm cot(i0)，與production model-based項符號對齊後一致。r_nom=b(i)r_trans，故名義點兩導數相差cot(i0)r0；它們不是相同residual的導數，不能以overall sign解釋。Ji用vm可作固定觀測生成模型的線性化；不能宣稱它就是∂[a y0−vm]/∂i。完整density必須固定measure。Gate5在y0座標積分，包含投影後variance與normalization，避免任意資料Jacobian因子。

**BLOCKED_BY_CONVENTION子項：** 原始HI velocity field、warped ring inclination、影像重建disk thickness/flattening與inclination改變後的mass-template重新推導不在凍結table中；沒有替它們編造導數。[SPARC產品頁](https://astroweb.case.edu/SPARC/)明確區分未修正photometry與inclination-corrected mass-model產品。本次只能驗證有文獻依據的global catalog transform，不能宣稱完整觀測重建通過。D/i covariance與sigma_sys的生成座標也未由原pipeline指定；後者在Gate5以兩種明示conditional慣例比較。這些限制不消除已可獨立確定的distance-template缺項。
''')
selection=pd.DataFrame(design['selected'])
write('02_CASE_SELECTION_AND_PREDECLARED_TOLERANCES.md',f'''# 事先凍結的有界設計

PREDECLARED_DESIGN.json在任何本輪model comparison前建立；run_v1/execution_authority.json記錄其SHA256：`{hashlib.sha256((B/'PREDECLARED_DESIGN.json').read_bytes()).hexdigest()}`。OBSERVATION_CONTRACT.json也在run前凍結。population表保留91銀河幾何與資料斜率；依序從未選者按極值選，tie按galaxy name。未讀取fit residual選樣；slope是positive observed v/r的log OLS斜率，僅供選樣。

'''+table(selection,['galaxy','selection_rule','i_deg','relD','i_err_deg','n','observed_log_slope'],{'relD':'.6f','observed_log_slope':'.6f'})+'''
固定模型向量：七份original_recovered/01_mle/*/theta_mle.npy；INPUT_AUTHORITY.json記錄path/hash，tested_parameter_points.json記錄每個點完整vector bytes hash及變動coordinates。nominal原向量、bound contact與共享q摘要差異均不調整。

三合成控制：synthetic_flat/rising/falling，baryon only；β=0,0.5,−0.25，i=35/60/85deg；8點r=1…8kpc、D=10±1Mpc、ei=3deg、vd=80r^β、Υ=1、sigma_sys=3、e=2、y=vd+3km/s。全部標synthetic_reference，不是歷史MLE。

距離FD：hD/D0=1e-3,1e-4,1e-5；傾角FD：hi=相同比例×max(i0_rad,0.5)。數值導數通過規則預宣告為abs_max≤1e-5或relative L2 norm≤1e-6；不同量綱的raw absolute值分軸保存，不能混為統計精度。objective比較abs1e-8，C relative Frobenius1e-9，mapped logL abs1e-7。後者較P1A門檻寬但本輪實際誤差仍在1e-11內；沒有事後改門檻。

科學影響標記：|ΔlogL|≥1或C相對norm≥0.01，只是篩選需說明的固定點變化，不是universal安全界線。sigma/scale/q各×1.01，選定Υ在bounds內+0.01否則−0.01；所有點驗證在原hard bounds內。沒有找最佳點。

quadrature預選前三個galaxy：ESO444-G084、UGC03205、UGC06614；對七模型與兩floor慣例跑16/32/64/128階；比較最後64→128的logL差≤1e-3。相同模型/資料點全集保留，budget最多1e6 node-model-floor evaluations、15分鐘、不加新case、不提高至128以上。實際42個integral×21760 nodes=913920 conditional density評估，run約2.92秒。source-AST callback評估不呼叫任何optimizer driver。

405項FD包含45個coarse-step數值未通過；並非刪去失敗後選出好案例。所有135個最細步長組合通過。計算例外0；run_v1/failures.json為空陣列，數值門檻失敗仍在完整CSV。
''')
derr=f[(f.relative_step==1e-5)&(f.axis=='D')&(f.label=='historical')].groupby('model').production_analytic_rel_norm.agg(['min','max']).reset_index()
examples=e[(e.label=='historical')&(e.specification=='full_D_fixedframe')&(e.case.isin(['UGC03205','NGC2403']))]
write('03_FINITE_DIFFERENCE_RESULTS.md',f'''# 導數與固定點objective影響

catalog contract內的解析full residual與中心FD在最細步長均一致：135/135通過，最大relative norm誤差{S['finest_fd_analytic_max_rel_norm']:.6g}。45個粗傾角步長超過原門檻；兩個更細步長的結果全保留於finite_difference_summary.csv與finite_difference_points.csv。這是數值收斂，不能替production缺項背書。

production distance residual-oriented JD對解析full JD的relative norm差（六真實銀河範圍）：

'''+table(derr,['model','min','max'],{'min':'.6f','max':'.6f'})+'''
production Ji若被解讀為r_trans的導數，relative norm差最高1.211513；若在固定y0座標以model mean線性化，與−vm cot(i)對齊後最大roundoff約1.18e-16。不能把前者一律宣稱為intrinsically wrong inclination covariance；真正可確定的FAIL為distance amplitude缺失及manuscript/code對完整residual導數的涵義未釐清。

下表只把JD替成完整D total derivative，保留production模型inclination項及名義residual／measurement/floor：

'''+table(examples,['case','model','delta_logL','covariance_relative_frobenius'],{'delta_logL':'.6f','covariance_relative_frobenius':'.6f'})+f'''
全部42點ΔlogL為{S['distance_only_real_min_delta_logL']:.9f}…{S['distance_only_real_max_delta_logL']:.9f}；沒有求全樣本corrected optimum或生成新MLE table。full_D_transformed另把Ji改成−y cot(i)，是另一種residual propagation spec，不等同Gate5固定measure的marginal likelihood；其42點ΔlogL約−1.815…+5.665，不能混用來宣稱校正值。

每個spec的quad/logdet/normalization/logL在covariance_effects.csv；303份C在covariances.npz，全部SPD且無jitter。本輪矩陣反解使用獨立scipy cho_solve，production以其原Cholesky計算；不是只比兩次同一wrapper。

最小失敗分量是預宣告synthetic_flat的第一點：D0=10Mpc、vd0=80km/s、Υ=1，vm(D)=80sqrt(D/10)。full ∂r/∂D=−4(km/s)/Mpc，production=0；sigmaD=1時漏掉16(km/s)^2的distance variance。MINIMAL_FAILING_CASE.json保存此既有控制的子分量。半徑差分再精細也無法恢復固定模板下不存在的依賴。

資料、向量、原likelihood皆未修正；新C只是本輪diagnostic reference。尚未覆蓋91銀河的全量effects，不外推其總和。
''')
write('04_COVARIANCE_UPDATE_AND_LOGDET_AUDIT.md',f'''# optimizer callback、C(theta)與logdet

實際production call chain（OPTIMIZER_CALL_CHAIN.json保留source hash）：

- mle.py:336 `_block_coordinate_mle` 的 `neg_g`／`neg_g2` → `neg_full` → `objective_neg_loglike` → fit_global.py:151 `log_likelihood` → likelihood.py:19 `build_covariance` → likelihood.py:58 `gaussian_loglike`。
- 每銀河Υ：`neg_u` → mle.py:143 `galaxy_loglike` → 重新build C與完整Gaussian。
- NFW每銀河：`neg_um` → 同一`galaxy_loglike`；固定halo，只改Υ作本輪測試。

本輪由AST擷取原nested callback，保留原body與預設gi，以預宣告inputs直接呼叫。沒有執行`_block_coordinate_mle`或scipy optimizer，沒有讓callback更新theta。globals callback每次建91個C；local callback每次1個C。動態攔截build_covariance回傳值，與獨立重建矩陣／線代相比。final-polish neg_g2 body經AST核驗與neg_g相同。這證明實際來源callback路徑的即時計算，未驗證optimizer全程schedule／global convergence。

168個測試點：callback objective與production endpoint最大差{S['callback_max_abs_delta']:.3g}；captured C相對差≤{S['captured_C_max_rel']:.6g}；單銀河獨立分解−production≤{S['independent_galaxy_max_abs_delta']:.6g}。所有非原始點的C均變化，logdet也非凍結：scale Δlogdet約0.000612…0.011732，q約0.001506…0.025330，Υ約−0.001871…0.019640，sigma_sys約0.042996…1.434654。逐列quadratic/logdet/nlog(2π)全部保存於covariance_updates.csv。

production有對稱化與失敗才增加jitter的分支；本次實際矩陣全部SPD，無regularization增量，所以獨立與production使用同一有效矩陣。未測超出本輪points的非SPD保底分支，不能保證任何外域都不需jitter。

108個matched-point比較包含兩組family的雙向mapping與same physical參數擾動：max relative C={S['matched_C_max_rel']:.6g}，max|ΔlogL|={S['matched_logL_max_abs']:.6g}。這裡不比較各自獨立maxima。所有bounds通過。PASS僅針對frozen implemented C的更新與identity；不認證其JD科學正確性。
''')
qhigh=q[q.order==128];qtable=qhigh[qhigh.construction=='B0_nominal_floor']
write('05_EXPLICIT_NUISANCE_SENSITIVITY.md',r'''# 有界顯式D/i積分

只用預宣告的前三個真實銀河，七個原始MLE固定向量；不調整D/i、Υ或global parameters。全部168列（3×7×2×4）保留在explicit_nuisance.csv，42組convergence結果另存quadrature_convergence.csv。

A為原production propagated-covariance Gaussian。另列A_fullD_fixedframe：完整distance total derivative搭配model-based inclination、原nominal measurement/floor；它是診斷參考，不是改過的production。A_full_transformed與固定觀測生成模型的measure不同，未將其稱作同一marginal likelihood。

B在固定觀測y0座標有完整normalization：

mu(D,i)=b(i)vm(D)，b=sin(i)/sin(i0)，r_nom=y0−mu。

B0 conditional covariance為diag(e0²+sigma_sys²)：floor定義在nominal deprojected速度座標，不隨候選i變。B1為diag[e0²+b(i)² sigma_sys²]：floor定義在intrinsic circular速度座標，再投影。兩者measurement variance都是e0²，符合固定y0 measure；換至candidate deprojected座標時，measurement errors為a(i)e0，並須包含density Jacobian，得到同一固定座標結果。B0/B1沒有另加入propagated D或i covariance，沒有double counting。

D與i uncertainty model：獨立standard Gaussian zD=(D−D0)/eD、zi=(i−i0)/ei，measure dzD dzi，截斷±3sigma並與D>0、0<i<π/2取交集，normalization各為Φ(upper)−Φ(lower)。i以rad計；不加cos(i) orientation prior。不宣稱這就是raw likelihood或被資料證實的prior；只是一個有來源支持global transform下的明示conditional sensitivity。對本次三銀河，±3sigma本身均在物理域內，physical truncation不再縮小區間。

每軸16/32/64/128階Gauss-Legendre，logsumexp累加Gaussian conditional×normalized prior weights；D與i的integration Jacobian已被z變數及其Gaussian density消去，不漏eD/ei因子。所有點完整conditional logdet=n點variance之log和；不是只積chi-square。

''' + table(qtable,['case','model','delta_B_minus_A','delta_B_minus_fullD'],{'delta_B_minus_A':'.6f','delta_B_minus_fullD':'.6f'})+f'''
42種case/model/floor在64→128均達1e-3 criterion，最大差{S['quadrature_max_128_64']:.9g}。32→64最大差0.119249，16→32最大差0.741933；低解析度不能獨自當成答案。prior quadrature weight總和最大偏差{S['prior_weight_max_error']:.6g}。

B0−A全範圍{S['B0_minus_A_min']:.9f}…{S['B0_minus_A_max']:.9f}。B0對完整D線性參考的max|difference|為{S['B0_minus_fullD_max_abs']:.9f}，在這三個銀河較小；此差同時包含非線性及±3sigma prior truncation，不能全部歸為Taylor誤差。B1對同一參考可達{S['B1_minus_fullD_max_abs']:.9f}，顯示floor座標有實質假設依賴。未跑更廣tails或其他D/i correlation；未提出全樣本approximation bound。

**判讀：**frozen A不能視為已建立之完整D/i線性邊際化。距離缺項已確定；在B0這個特定conditional model下，補完整D後的線性近似在本次三銀河差較小，但B1、未知correlation、warped geometry、測量error生成模型與全sample仍未驗證。沒有把B視為唯一真值，也沒有把積分收斂當作觀測規格的正確性。
''')
write('06_SCIENTIFIC_MATERIALITY_AND_REFIT_GATE.md','''# 科學影響與重估門檻

| 類別 | 本輪可建立的事實 | 不可推論 |
|---|---|---|
| roundoff | replay/獨立Gaussian/identity誤差約1e-11或更小 | 不抵銷distance缺項 |
| metadata | shared ν-q boundary值約1e-8差保留 | 不調theta以追summary |
|數值FD誤差|最細step相對norm約1.04e-9；粗step45項不通過|粗step失敗不等物理失敗；production差不隨step消失|
| likelihood實質差 | UGC03205的MOND固定點只補D變約−2.807829，q-family約−2.827114 | 不是校正後maxima／不是ranking翻轉證據 |
|統計規格|B0/B1、truncated Gaussian與D/i independence都明示|未證唯一nuisance測量模型或全樣本近似bound|
|不完整propagation|baryonic template amplitude缺失，可由analytic及synthetic精確重現|不能繼續宣稱frozen JD涵蓋全部distance uncertainty|

scale、sigma_sys、q及每銀河Υ均會與改變的C及logdet共同影響objective；本次固定點差異及parameter-dependent covariance測試表明不能先驗當作常數offset。但未求校正objective梯度／profile／optimum，因此global acceleration、sigma_sys、q與model ranking的量化影響全部UNKNOWN_PENDING_REFIT。不得用本次ΔlogL直接移動原best-fit值。

若k和資料樣本固定，某個相同參數點的−2ΔlogL只提供IC objective部分的算術敏感度；例如|ΔlogL|≈2.827可對應約5.654單銀河objective變化。它不是重新maximize後AIC/BIC差，也未處理correlated-data BIC解釋（S04不在本輪）。不能把六銀河外推到91或只調表格數字。

**原表處置：**歷史MLE/AIC/BIC表及資料一律保留；可在內部暫作已凍結objective的來源基準。不能將其暫留解讀為「完整covariance gate通過後可直接沿用投稿結論」。production likelihood correction是重估前必要工作，尚須Human Arbiter核准。

**最小修正規格提案（本輪不實作）：**先選定fixed primary data measure（建議明示nominal y0），完整mean mu(D,i)=vm[sqrt-distance-scaled templates, distance-scaled radius]sin(i)/sin(i0)，獨立或有證據的joint D/i uncertainty，明示sigma_sys所在速度座標。若選B0並作Gaussian first-order approximation，C=diag(e0²+sigma²)+eD²(∂mu/∂D)(∂mu/∂D)^T+ei_rad²(∂mu/∂i)(∂mu/∂i)^T；∂mu/∂D必含模板幅度與halo radius，∂mu/∂i=vm cot(i0)。不把y-based propagated error與model mean邊際化混為一式。若採B1或explicit integration，須定義conditional covariance、normalization及near-bound treatment，另驗近似精度。

任何採用修正D規格的後續比較，baryon、MOND simple-ν、linear PTQ、PTQ-ν、PTQ-screen、MOND-screen及restricted NFW-1p七模型都需重新估計，保留原parameter accounting與同一physical domain。PTQ/MOND兩組family仍可由相同transformation維持identity；identity不能替statistical validity背書。

作者核准下一個corrected-model task後，先隔離reference spec與bounded read-only驗證，不動production；通過後再另行授權production change及全部必要re-estimation。NFW convergence warning仍獨立存在，本輪不做additional starts。
''')
write('07_REVIEWER_RESPONSE_EVIDENCE.md','''# 暫定 reviewer response evidence（非最終回覆）

**Reviewer來源阻擋：**本機只有P0R register與舊內部模擬review，沒有本次external decision letter／逐字comments。R1 covariance comment3、R2 covariance comment2的編號來自作者本輪指示；沒有證據可quote或聲稱accurately paraphrase其actual wording。已以非阻斷問題向作者索取原文。本報告不能完成該項文字核對；以下是待原文對照的provisional evidence，不冒充reviewer原句。

實際稿件定位：submitted flat source §3.2 “Full within-galaxy covariance”；Eq.(7) label eq:logL_g為完整Gaussian，Eq.(8) eq:cov-full為C，Eq.(9) eq:jac為JD/Ji。方程號由未改的P0R/A candidate aux核對，與submitted同公式序列；原始TeX lines267–297。此處僅引用，未改稿。

依作者本輪任務及S01–S03而做的**內部議題摘要，非reviewer paraphrase**：完整observation-to-residual導數是否齊全、C(theta)/logdet是否在proposal更新，以及propagation與explicit nuisance的差異是否有界。

## Reviewer1 — covariance comment3（待核對原文）

Provisional evidence response (English):

We audited the implementation underlying Section 3.2, Eqs. (7)–(9), at the recovered historical parameter vectors without refitting. The catalog-level distance transformation rescales both the physical radius and all baryonic velocity templates. The frozen radius finite difference keeps those templates fixed, and therefore omits part of the distance response. Analytic derivatives and central finite differences converge, but the implemented distance Jacobian does not equal that full derivative. In UGC03205, replacing only this distance term changes the fixed-point MOND galaxy log likelihood by approximately −2.808. The frozen estimates remain historical outputs of the implemented objective, not validated estimates for a corrected covariance. We have not changed the likelihood or manuscript, and corrected-specification validation and any re-estimation remain pending author authorization.

## Reviewer2 — covariance comment2（待核對原文）

Provisional evidence response (English):

The production optimizer callbacks recompute the implemented parameter-dependent covariance and Gaussian log determinant. At 168 prescribed callback evaluations the independently reconstructed galaxy likelihoods agreed within 6e−13; matched PTQ/MOND covariance and likelihood identities also held. These checks do not establish the correctness of the distance propagation. Bounded explicit distance/inclination integration for three preselected galaxies converged numerically, but comparison with the frozen covariance revealed fixed-point differences up to about 2.84 in log likelihood. The integration assumes a specified independent, truncated Gaussian uncertainty model; the treatment of the systematic floor introduces a further conditional choice. We therefore cannot claim that the nuisance approximation has been fully validated or that the published parameter estimates and information-criterion rankings are unaffected. Their impact is unknown pending an approved corrected specification and refitting stage.

作者取得原文後需逐項確認兩comment真正範圍、補上其quote／準確paraphrase與正式revised PDF行號，不能直接將本草案寄出。沒有宣稱「fully validated」、新fit、production correction已完成或實證問題已全部解決。
''')
write('08_NEXT_TASK_PROMPT.md','''# 下一個最小安全任務（待Human Arbiter批准，現在不執行）

REV26-P1C — Corrected covariance specification and isolated reference gate; NO FITTING.

你是統計再現工程代理。專案 /Users/rocky/ptq-mond；implementation /Users/rocky/ptq-mond/PTQ-Quaternionic-RC。先讀rev26_p1b_covariance_gate全部報告與numerical covariance_gate/PREDECLARED_DESIGN、OBSERVATION_CONTRACT、GATE_VERDICTS、MINIMAL_FAILING_CASE、INPUT_AUTHORITY及SHA256SUMS。上一階段overall verdict為COVARIANCE_CORRECTION_REQUIRED_BEFORE_REFIT；七個原MLE保持原樣，不再做archive discovery或Symmetry science。

本prompt只有在作者批准此corrected-model task後生效。它不授權production edit或任何optimization。

1. 與作者確認primary觀測measure、sigma_sys是nominal/LOS/intrinsic哪個座標、D/i的measurement distribution、獨立性／joint covariance，以及是否只採catalog-level global inclination。不能以原production未記載的假設冒作歷史事實。取得external reviewer原文，補真實requests對照，但不聯絡reviewers。
2. 把所選統計規格寫成完整normalized likelihood，明示資料變數變換Jacobian、e_obs transformation、distance模板sqrt scaling、physical radius scaling、所有Υ固定測試與restricted NFW c(M)。若採fixed-frame Gaussian propagation，先列∂mu/∂D、∂mu/∂i與noise model，分開linear truncation errors與measurement assumptions。
3. 只在新的 results/frontiers_revision_2026_10/covariance_reference/ 建隔離reference implementation及唯讀tests；production source、data、舊results與稿件保持原樣。預先凍結更廣但有界的uncertainty／tail／floor敏感度及數值門檻；先重現最小baryon缺項例，再驗完整D residual、matched identities、Gaussian normalization與explicit nuisance convergence。不要沿用y-based Ji去替代fixed-measure mean derivative而不說明。
4. 對原七模型固定向量做有界（必要時先作者核准全91銀河）reference effect audit，清楚標示fixed-point objective，不產生新MLE、CI或更新AIC/BIC表。若用Gaussian prior與truncated quadrature比較，加入相同truncation下的linear conditional reference，以分離tail和Taylor effects；該設計須先凍結。
5. 交付chosen specification、blocked conventions、reference code/data/source hashes、全部failure/成功cases、covariance/logdet分量與approximation limits。提出下一階段production patch／re-estimation的具體reviewable方案，保持七模型原parameter counts与family matching，說明NFW需要獨立收斂方案。不得自動執行。

STOP：不跑global refit、NFW starts、MCMC，不改production，不改稿，不commit/push/tag/release/upload。只有Human Arbiter可另行批准production修正與新擬合。所有新估計（如日後授權）放新reestimated/run-id，不能標original_recovered。
''')
(B/'README.md').write_text('''# REV26-P1B covariance gate evidence

總判定：COVARIANCE_CORRECTION_REQUIRED_BEFORE_REFIT。run_v1/GATE_VERDICTS.json分列五個verdict。原向量與來源在INPUT_AUTHORITY，source保全在SOURCE_HASHES_BEFORE及SOURCE_PRESERVATION_VALIDATION。

- PREDECLARED_DESIGN/OBSERVATION_CONTRACT：run前凍結；六真實＋三合成點、不選residual。
- run_v1/：全部FD、likelihood分解、C(theta)、matched identities、nuisance積分與四階收斂、parameter-point hashes及失敗紀錄。
- OPTIMIZER_CALL_CHAIN.json：來源AST與原callback固定點動態執行；不執行optimizer driver。
- EXTERNAL_SOURCES.json：SPARC primary source版本／定位與讀取失敗紀錄。

原實際執行script保存為run_gate_executed_v1.py，其SHA256與run_v1/execution_authority.json吻合。run_gate.py後續僅加--out路徑選項和目錄限制（數值body相同），可重播至本root內另一個不存在的目錄：

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/rev26-p1a-env/bin/python run_gate.py --out /Users/rocky/ptq-mond/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/covariance_gate/replay_new
```

臨時環境的精確pip freeze在run_v1/environment_freeze.txt，Python/platform在execution_authority；環境不存在時在獲准output root內新建venv並安裝freeze（本輪未改原環境）。prepare_gate.py與freeze_design.py拒絕覆寫已封存的authority/design；重播不需重新執行這兩支。analyze_gate.py驗證run_v1全部numerical資料並產生統計摘要；write_reports.py產生本輪docs。沒有optimization、production patch或新fit。

SHA256SUMS涵蓋所有本root檔案除自身；完整new-asset manifest在docs root。報告原reviewer comments缺失，待作者提供；不能把舊模擬review當本次外審。
''')
print('9 reports and numerical README written')
