import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
import matplotlib.pyplot as plt
from matplotlib import cm

def ecdf(data1,data2,x,y): #generates empirical cdf for two specific values
    #only works for marginal distributions between 0 and 1
    sample=np.column_stack((data1,data2))
    c=0
    for i in range(0,len(data1)):
        if sample[i,0]<=x and sample[i,1]<=y:
            c=c+1
    return c/len(data1)
    #sum(1 for obs in sample if sample[:,[0]]<=x and sample[:,[1]]<=y)

def tot_ecdf(data1,data2,nbuckets): #generates entire empirical cdf
    #only works for marginal distributions between 0 and 1
    ecdf_list=[]
    ecdf_dict={}
    for x in range(0,nbuckets+1):
        x=x/nbuckets
        for y in range(0,nbuckets+1):
            y=y/nbuckets
            cdf=ecdf(data1,data2,x,y)
            ecdf_list.append([x,y,cdf])
            ecdf_dict[(x,y)]=cdf
    return np.array(ecdf_list), ecdf_dict


def epdf(data1,data2,lboundx,uboundx,lboundy,uboundy): #generates empirical pdf for two specific values
    #only works for marginal distributions between 0 and 1
    sample=np.column_stack((data1,data2))
    s=0
    for n in range(0,len(data1)): #loop through all sample elements
        if lboundx<= sample[n,0] <= uboundx and lboundy <= sample[n,1] <= uboundy:
            s=s+1
    return s/len(data1)
    

def tot_epdf(data1,data2,nbuckets): #generates entire empirical pdf
    #only works for marginal distributions between 0 and 1
    epdf_list=[]
    epdf_dict={}
    for i in range(0,nbuckets): #we cover possible variable values in [0,1]
        uboundx=(i+1)/nbuckets-0.0000001 #open interval to avoid double counting
        if i==nbuckets-1:
                uboundx=(i+1)/nbuckets  #include last interval upper bound
        lboundx=i/nbuckets
        for j in range(0,nbuckets):
            uboundy=(j+1)/nbuckets-0.0000001 #open interval to avoid double counting
            if j==nbuckets-1:
                uboundy=(j+1)/nbuckets #include last interval upper bound
            lboundy=j/nbuckets
            pdf=epdf(data1,data2,lboundx,uboundx,lboundy,uboundy)
            #epdf_list.append([[lboundx,uboundx],[lboundy,uboundy],cdf])
            epdf_list.append([lboundx,lboundy,pdf])
            epdf_dict[(lboundx,uboundx,lboundy,uboundy)]=pdf
    return np.array(epdf_list), epdf_dict


#EMPIRICAL TAIL DEPENDENCE
def tail(data1,data2,lboundx,uboundx,lboundy,uboundy): #generates empirical pdf for a given interval
    #only works for marginal distributions between 0 and 1
    sample=np.column_stack((data1,data2))
    s=0
    for n in range(0,len(data1)): #loop through all sample elements
        if lboundx<= sample[n,0] <= uboundx and lboundy <= sample[n,1] <= uboundy:
            s=s+1
    return s
    


#final_array_cdf, final_dict_cdf = tot_ecdf(u1,u2,nbuckets)
#final_array_pdf, final_dict_pdf = tot_epdf(u1,u2,nbuckets)


# #ECDF PLOT
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(final_array_cdf[:,0],final_array_cdf[:,1],final_array_cdf[:,2])
# ax.set_title("Empirical joint cdf")
# ax.set_xlabel("u1")
# ax.set_ylabel("u2")
# ax.set_zlabel("ecdf")
# plt.show()


# #EPDF PLOT 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(final_array_pdf[:,0],final_array_pdf[:,1],final_array_pdf[:,2], c=final_array_pdf[:,2])
# ax.set_title("Empirical joint pdf")
# ax.set_xlabel("u1")
# ax.set_ylabel("u2")
# ax.set_zlabel("epdf")
# plt.show()


# #EDF PLOT 2D
# plt.scatter(u1,u2,2, marker='o', linewidths=1)
# plt.title('Uniform marginals scatterplot')
# plt.show()

# plt.hist2d(u1,u2,nbuckets)
# plt.title('Uniform marginals 2D histogram')
# plt.show()




