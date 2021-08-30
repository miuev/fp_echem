import numpy as np
from scipy.integrate import odeint

def equilibrium(a_h,dG,T):
    K = np.exp(-dG/(8.6173E-5*T))
    theta = (a_h*K)/(a_h*K+1)
    return theta

def dcov_du_setup(a_h,km,stoichs,red = True):
       
    def dcov_du(cov,t):
        
        result = []
                         
        for i in np.arange(int(stoichs.shape[-1])):
            mass_trans = km*(stoichs[0,i]*equilibrium(a_h,cov[0]-stoichs[1,i],298.15)-cov[i+1])
            result.append(mass_trans)

        result = np.array(result)
        
        return result
    
    return dcov_du


def intercalation(U,comp_i,km,a_h,stoichs,area_ratio):
    
    neg_0 = comp_i
    neg_t = U
    
    dcov_du_neg = dcov_du_setup(a_h,km,stoichs,red = True)
    neg_cov = odeint(dcov_du_neg, neg_0, neg_t)
    neg_dU = abs(neg_cov[1,0]-neg_cov[0,0])
    
    neg_int = -(96485000)*area_ratio*np.diff(neg_cov[:,1:].sum(axis=1))/neg_dU
    
    return neg_int

# saving for bi-directional intercalation sweeps
# def intercalation(U_o,U_s,U_f,v,res,comp_i,km,a_h,stoichs,area_ratio):
    
#     span_neg =  U_o - U_s 
#     span_pos = -U_s + U_f
    
#     neg_0 = [U_o]
#     for i in np.arange(int(len(comp_i))):
#         neg_0.append(comp_i[i])
#     neg_0 = np.array(neg_0)

#     neg_t = np.linspace(0,int(np.round(span_neg/v)),int(np.round(res*span_neg/v)))
    
#     dcov_du_neg = dcov_du_setup(a_h,v,km,stoichs,red = True)
#     print(dcov_du_neg)
#     neg_cov = odeint(dcov_du_neg, neg_0, neg_t)
#     neg_dU = abs(neg_cov[1,0]-neg_cov[0,0])
    
#     pos_0 = [U_s]
#     for i in np.arange(int(len(comp_i))):
#         pos_0.append(neg_cov[-1,i+1])
#     pos_0 = np.array(pos_0)

#     pos_t = np.linspace(0,int(np.round(span_pos/v)),int(np.round(res*span_pos/v)))
    
#     dcov_du_pos = dcov_du_setup(a_h,v,km,stoichs,red = False)
    
#     pos_cov = odeint(dcov_du_pos, pos_0, pos_t)
#     pos_dU = abs(pos_cov[1,0]-pos_cov[0,0])
    
#     neg_pot = neg_cov[:,0][:-1]
#     pos_pot = pos_cov[:,0][:-1]

#     neg_int = -(96485000)*area_ratio*np.diff(neg_cov[:,1:].sum(axis=1))/neg_dU
#     pos_int = -(96485000)*area_ratio*np.diff(pos_cov[:,1:].sum(axis=1))/pos_dU
    
#     return neg_pot, neg_int, pos_pot, pos_int
