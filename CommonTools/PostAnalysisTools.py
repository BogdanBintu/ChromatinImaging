#Bogdan Bintu
#Copyright Presidents and Fellows of Harvard College, 2017.

import sys,os,glob
import numpy as np
from scipy import spatial
import cPickle as pickle
import matplotlib.pylab as plt
## Plotting tools
def fig_no_axis(**kwargs):
    """Get figure with no right/upper axis and which allows for saving text in pdf and larger font."""
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.size']=22
    fig1, ax1 = plt.subplots(facecolor='white',**kwargs)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    return fig1

def to_maps(zxys):
    from scipy.spatial.distance import pdist,squareform
    mats=np.array(map(squareform,map(pdist,zxys)))
    return mats
    
def nan_corr_coef(x_,y_):
    x=np.ravel(x_)
    y=np.ravel(y_)
    keep=(np.isinf(x)==False)&(np.isinf(x)==False)&(np.isnan(x)==False)&(np.isnan(y)==False)
    x=x[keep]
    y=y[keep]
    return np.corrcoef([x,y])[0,1]

def corr_coef(x_,y_,print_err=False):
    x=np.ravel(x_)
    y=np.ravel(y_)
    keep=(np.abs(x)!=np.inf)&(np.abs(y)!=np.inf)&(np.isnan(x)==False)&(np.isnan(y)==False)
    x=x[keep]
    y=y[keep]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    if print_err:
        model = sm.OLS(y,A)
        result = model.fit()
        return np.corrcoef([x,y])[0,1],c,m,result.bse
    return np.corrcoef([x,y])[0,1],c,m

## Compartment analysis tools


def pca_components(im_cor):
    """returns the evals, evecs sorted by relevance"""
    from scipy import linalg as la
    data = im_cor.copy()
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    #evecs_red = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return  evals, evecs#, np.dot(evecs_red.T, data.T).T
def get_AB_boundaries(im_cor,evec,sz_min = 1,plt_val=False):
    vec = np.dot(evec,im_cor)
    vec_ = np.array(vec)
    vec_[vec_==0]=10.**(-6)
    def get_bds_sign(vec_s):
        val_prev = vec_s[0]
        bds_ = []
        for pos,val in enumerate(vec_s):
            if val!=val_prev:
                bds_.append(pos)
                val_prev = val
        return np.array(bds_)

    vec_s = np.sign(vec_)
    bds_ = get_bds_sign(vec_s)
    vec_ss = vec_s.copy()
    bds_ext = np.concatenate([[0],bds_,[len(vec_s)]])
    for i in range(len(bds_ext)-1):
        if bds_ext[i+1]-bds_ext[i]<sz_min:
            vec_ss[bds_ext[i]:bds_ext[i+1]]=0    
    
    first_val = vec_ss[vec_ss!=0][0]
    vec_ss_ = []
    for vvec in vec_ss:
        if vvec==0:
            if len(vec_ss_)>0:
                vec_ss_.append(vec_ss_[-1])
            else:
                vec_ss_.append(first_val)
        else:
            vec_ss_.append(vvec)
    bds = get_bds_sign(vec_ss_)
    bds_score = []
    bds_ext = np.concatenate([[0],bds,[len(vec)]])
    for i in range(len(bds)):
        lpca = np.median(vec[bds_ext[i]:bds_ext[i+1]])
        rpca = np.median(vec[bds_ext[i+1]:bds_ext[i+2]])
        #print lpca,rpca
        bds_score.append(np.abs(lpca-rpca))
    if plt_val:
        plt.figure()
        plt.title('A/B pca 1 projection')
        plt.plot(vec,'ro-')
        plt.plot(bds,vec[bds],'go')
        plt.show()
    return bds,bds_score
def get_gen_pos_gen_vals(mats):
    """Given list of single cell distance matrices, find the population-average median, then group based on genomic distance and compute medians across groups.
    Perform fit in log space 
    """
    im_dist = np.nanmedian(mats,0)
    gen_pos,gen_vals = get_norm(im_dist,func=np.nanmedian)
    ro,c,m=corr_coef(np.log(gen_pos),np.log(gen_vals))
    gen_vals = np.exp(c)*gen_pos**m
    return gen_pos,gen_vals
def nan_corrcoef(x,y):
    x_ = np.array(x)
    y_ = np.array(y)
    keep = (np.isinf(x_)==False)&(np.isinf(y_)==False)&(np.isnan(x_)==False)&(np.isnan(y_)==False)
    if np.sum(keep)>2:
        return np.corrcoef(x_[keep],y_[keep])[0,1]
    return 0
def cor_mat(im_log):
    im_log = np.array(im_log)
    im_cor = np.zeros(im_log.shape)
    for i in range(len(im_cor)):
        for j in range(i+1):
            im_cor[i,j]=nan_corrcoef(im_log[i],im_log[j])
            im_cor[j,i]=im_cor[i,j]
    return im_cor
def get_cor_matrix(mat,gen_pos=None,gen_vals=None,plt_val=True):
    mat_ = np.array(mat)
    
    if plt_val:
        plt.figure()
        plt.title('original distance matrix')
        plt.imshow(-mat_,interpolation='nearest',cmap='seismic')
        plt.colorbar()
        plt.show()
    mat_[range(len(mat_)),range(len(mat_))]=np.nan
    
    #normalize for polymer effect
    if gen_pos is not None:
        mat_ = perform_norm(mat_,gen_pos,gen_vals)
        #mat_ = np.log(mat_)
    else:
        #mat_ = np.log(mat_)
        pass
    mat_[np.isinf(mat_)]=np.nan
    if plt_val:
        plt.figure()
        plt.title('distance normalized matrix')
        plt.imshow(-mat_,interpolation='nearest',cmap='seismic')
        plt.colorbar()
        plt.show()  

    #compute correlation matrix
    mat_ = cor_mat(mat_)

    if plt_val:
        plt.figure()
        plt.title('correlation matrix')
        plt.imshow(mat_,interpolation='nearest',cmap='seismic')
        plt.colorbar()
        plt.show()
    return mat_

## Tad insulation tools defined as in Nagano et al., Nature, 2017

## For STORM
from scipy.spatial.distance import cdist
def overlap_metric(mlist_cluster1,mlist_cluster2,dist=200,error_tol=0.05,num_cutoff=1,kdtree=False,norm_tag='mean'):
    """Given two lists of mlists compute the overlap fraction"""
    xyz1 = np.array(cluster_to_xyz(mlist_cluster1,hlim=0,nmin=0,z_cut_off=500)).T
    xyz2 = np.array(cluster_to_xyz(mlist_cluster2,hlim=0,nmin=0,z_cut_off=500)).T
    if len(xyz1)==0 or len(xyz2)==0:
        return np.nan
    else:
        if not kdtree:
            dist_mat = cdist(xyz1,xyz2)
            th = dist_mat<dist
            fr1 = np.sum(th,axis=0)
            fr1 = np.sum(fr1>=num_cutoff)/float(len(fr1))
            fr2 = np.sum(th,axis=1)
            fr2 = np.sum(fr2>=num_cutoff)/float(len(fr2))
            return np.mean([fr1,fr2])
        else:
            tree = spatial.KDTree(xyz1)
            neighs = tree.query_ball_point(xyz2,dist,eps=error_tol)
            norm = float(len(xyz1)+len(xyz2))/2.
            frac_overlap = np.sum(np.array(map(len,neighs))>=num_cutoff)/norm
            return frac_overlap
            
def spotfiles_to_map(save_spots,metric_dic={'name':'dist'},reversed_=False,dim=3):
    maps=[]
    for save_spot in save_spots:
        map_=spotfile_to_map(save_spot,metric_dic=metric_dic,reversed_=reversed_,dim=dim)
        if map_ is not None:
            maps.append(map_)
    return maps
def spotfiles_to_center(save_spots,func=np.median,reversed_=False):
    maps=[]
    for save_spot in save_spots:
        map_=spotfile_to_center(save_spot,func=func,reversed_=reversed_)
        if map_ is not None:
            maps.append(map_)
    return np.array(maps)
def spotfile_to_center(save_spot,func=np.median,reversed_=False):
    dic = pickle.load(open(save_spot,'rb'))
    if dic.get('finished',False):
        visited = dic.get('visited',[])
        center_dic = {}
        cluster_dic = {}
        for e in visited:
            cluster_dic[e]=[]
            if dic.has_key(e):
                if type(dic[e]) is dict:
                    if dic[e].has_key('mlist_clusters'):
                        if dic[e].get('mlist_clusters',None) is not None:
                            cluster_dic[e] = dic[e]['mlist_clusters']
            center_dic[e] = cluster_to_center(cluster_dic[e],func=func)
        reg_dic = non_redundant_regions(visited,reversed_=reversed_)
        reg_keys = np.sort(reg_dic.keys())
        #print reg_dic
        return [center_dic[reg_dic[reg_keys[i]]] for i in range(len(reg_keys))]
    return None