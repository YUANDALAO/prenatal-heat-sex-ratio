"""Extra quantities for figures. (1) EXPLORATORY: H5 GEST effect on neonatal/infant death by climate tertile.
(2) Sex-specific GEST effects (female = GEST, male = GEST + GxM) from the pre-specified hot-tertile H5 model."""
import os, numpy as np, pandas as pd
from scipy import stats
from analysis_primary import fe_ols, DATA, RES
cols=['survey','v001','cc','b3','male','primary','clim_tertile','age_m','b5','b7','T2M_anom_GEST','T2M_anom_POST']
B=pd.read_parquet(os.path.join(DATA,'births.parquet'),columns=cols); B=B[B.primary].copy()
B['clus']=B.survey+'_'+B.v001.astype(int).astype(str); B['ctym']=B.cc+'_'+B.b3.astype(int).astype(str)
B['nnd']=((B.b5==0)&(B.b7==0)).astype(float); B['imr']=((B.b5==0)&(B.b7<12)).astype(float)
B['GxM']=B.T2M_anom_GEST*B.male; B['PxM']=B.T2M_anom_POST*B.male
X=['T2M_anom_GEST','GxM','male','T2M_anom_POST','PxM']; rows=[]
for out,minage in [('nnd',1),('imr',12)]:
    for t in range(3):
        D=B[(B.clim_tertile==t)&(B.age_m>=minage)]
        r,V,b=fe_ols(D,out,X,['clus','ctym'],'clus')
        g=r.set_index('term').loc['T2M_anom_GEST']
        rows.append(dict(outcome=out,group=f'tertile {t}',b=g.b_pp,lo=g.lo_pp,hi=g.hi_pp,p=g.p,rate=D[out].mean()*1000,n=len(D)))
        if t==2:   # sex-specific effects from the pre-specified model
            for sex,w in [('female',np.array([1,0,0,0,0.])),('male',np.array([1,1,0,0,0.]))]:
                est=w@b*100; se=np.sqrt(w@V@w)*100
                rows.append(dict(outcome=out,group=f'hot, {sex}',b=est,lo=est-1.96*se,hi=est+1.96*se,p=2*stats.norm.sf(abs(est/se)),rate=D.loc[D.male==(sex=='male'),out].mean()*1000,n=int((D.male==(sex=='male')).sum())))
O=pd.DataFrame(rows); O.to_csv(os.path.join(RES,'fig_data_h5.csv'),index=False); print(O.round(3).to_string(index=False))
# map data: births per grid cell and tertile (primary sample) -- cell centroids only, no cluster coordinates
cols=['cell','clim_tertile','primary']; C=pd.read_parquet(os.path.join(DATA,'births.parquet'),columns=cols)
C=C[C.primary].groupby(['cell','clim_tertile']).size().rename('births').reset_index(); C.to_csv(os.path.join(RES,'fig_map_cells.csv'),index=False)
print('cells',len(C))
