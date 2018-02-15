
# coding: utf-8

# In[339]:


import pandas as pd
import numpy as np
import rasterio
import time
import random


# In[360]:


import os
import pdb

import numpy as np
import pandas as pd
from scipy.optimize import leastsq


def model_fourier(params, agdd, n_harm):
    """
    Fourier model
    :param params:
    :param agdd:
    :param n_harm:
    :return:
    """
    integration_time = len(agdd)
    t = np.arange(1, integration_time + 1)
    result = t*.0 + params[0]
    w = 1

    for i in range(1, n_harm * 4, 4):
        result = result + params[i] * np.cos(2.0 * np.pi * w * t / integration_time + params[i+1])                  + params[i+2]*np.sin(2.0 * np.pi * w * t / integration_time + params[i+3])
        w += 1

    return result


def mismatch_function(params, func_phenology, ndvi, agdd):
    """
    The NDVI/Phenology model mismatch function
    :param params:
    :param func_phenology:
    :param ndvi:
    :param agdd:
    :param years:
    :return:
    """
    # output stores the predictions
    output = []

    oot = ndvi - func_phenology(params, agdd, n_harm=8)

    [output.append(x) for x in oot]

    return np.array(output).squeeze()


def do_fourier(ndvi, gdd, n_harm=8, init_params=None):
    """
    :param ndvi:
    :param gdd:
    :param n_harm:
    :param init_params:
    :return:
    """
    n_params = 1 + n_harm * 4

    if init_params is None:
        init_params = [.25, ] * n_params
        (xsol, mesg) = leastsq(mismatch_function, init_params, args=(model_fourier, ndvi, gdd), maxfev=1000000)
        model_fitted = model_fourier(xsol, gdd, n_harm)

    return model_fitted


def get_PTD(df,gl,gu,sl,su):
    """
    Get phenological transition dates (greenup, senescence)
    :param df:
    :return:
    """
    # Input dataframe has an index comprised of day of year and remaining columns signify NDVI
    # Linearly interpolate dataframe columns to fill in missing values
    print(df['LAI_2015_corn'].tolist())
    df = df.apply(pd.Series.interpolate)
    print(df['LAI_2015_corn'].tolist())
    first_index=df['LAI_2015_corn'].first_valid_index()
    print(first_index)
    if(first_index>14):
        #print("error preventing")
        df.ix[14:first_index,'LAI_2015_corn']=200
        df=df.ix[14:,]
        #print(len(df.index))
    else:
        df=df.ix[first_index:,]
        #print(len(df.index))
    
    # Now compute mean of all columns and get the smoothened NDVI
    arr_smooth = do_fourier(df.mean(axis=1), [8.0] * len(df))
    print(arr_smooth)
    if(first_index>14):
        plt.plot(list(range(14-1+len(arr_smooth))),np.append([np.NaN]*(14-1),arr_smooth))
    else:
        plt.plot(list(range(first_index-1+len(arr_smooth))),np.append([np.NaN]*(first_index-1),arr_smooth))
    plt.xlabel("DOY")
    plt.ylabel("LAI")
    plt.title("LAI smoothened time series plot")
    plt.show()
    
    if(first_index>14):
        return np.append([0]*(14-1),arr_smooth)
    else:
        return np.append([0]*(first_index-1),arr_smooth)


# In[341]:


df_insert = pd.DataFrame({'LAI_2015_corn': [np.NaN]*7,
                    'LAI_2015_soy': [np.NaN]*7})
def dataframe_extraction(r1,r2,row,col):
    """This function returns a pandas dataframe that holds the NDVI values for 1-365 doys of 2015 and 2016 years.
    inputs: r1, r2: the NDVI Rasterstacks for 2015 and 2016 years. Each raster has 365 layers one corresponding to a doy of that year.
            row, col: the row and column of the raster pixel we would like to get the dataframe for """
    id=np.array(range(1,1+r1.shape[0]))
    data_frame = (pd.DataFrame({'LAI_2015_corn':r1[0:r1.shape[0]+1:1,row,col],
                          'LAI_2015_soy':r2[0:r1.shape[0]+1:1,row,col]},index=id))
    return data_frame
    df=pd.DataFrame()
    for i in range(len(data_frame)):
        df=pd.concat([df,data_frame.iloc[[i]],df_insert])
    df.index=range(1,369)
    return df[:365]


# In[342]:


from multiprocessing import Pool
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
def initialize_rasters(path1,path2):
    raster1=rasterio.open(path1)
    tot_cols=raster1.width
    tot_rows=raster1.height
    a=raster1.read()
    print("Tot rows: ",tot_rows," Tot cols: ",tot_cols)
    raster2=rasterio.open(path2)
    b=raster2.read()
    return (a,b,tot_rows,tot_cols)
def ritvik_fn(df,gl,gu,sl,su):
    #return (random.randint(1,50),random.randint(120,365))
    return get_PTD(df,gl,gu,sl,su)

def myfunction(index,rasters):
    a=rasters[0]
    b=rasters[1]
    tot_cols=a.shape[2]
    tot_rows=a.shape[1]
    #print("Tot rows: ",tot_rows," Tot cols: ",tot_cols)
    row=int(index/tot_cols)
    col=index-(tot_cols*row)
    #print("row: ",row," col: ",col)
    df=dataframe_extraction(a,b,row,col)
    #df.loc[57]['NDVI_2015']=None
    df[df<=200]=None
    
    lai_2015_corn=df['LAI_2015_corn'].tolist()
    clean=[x for i,x in enumerate(lai_2015_corn) if str(x) != 'nan']
    dicti={x:i for i,x in enumerate(lai_2015_corn) if str(x) != 'nan'}
    clean = [max(0, min(x, 10000)) for x in clean]
    
    if (len(clean)>0):
        maxi=int(max(clean))
        print(clean)
        plt.plot([dicti[x] for x in clean],clean)
        #plt.plot(list(range(len(clean))),clean)
        plt.xlabel("DOY")
        plt.ylabel("LAI_2015_corn")
        plt.title("LAI time series plot")
        plt.show()
        #print(df.loc[50:80]['NDVI_2015'])
        ##return ritvik_fn(pd.Series.to_frame(df['NDVI_2015']))
        return ritvik_fn(pd.Series.to_frame(df.loc[0:365]['LAI_2015_corn']),10,60,135,196)
    else:
        return [0]*46


# In[343]:


src=rasterio.open('LAI_2015_corn_NA.tif')
a,b,tot_rows,tot_cols=initialize_rasters('LAI_2015_corn.tif','LAI_2015_soy.tif')
y=[(a,b)]*tot_rows*tot_cols
ind=range(tot_rows*tot_cols)
l=list()
l=list(list(zip(ind,y))[:])
lai_new=np.zeros_like(a)


# In[361]:


#191150,561100
test=myfunction(989152,(a,b))


# In[358]:


test


# In[279]:


index=561100
row=int(index/tot_cols)
col=index-(tot_cols*row)
lai_new[:,row,col]=test
print(lai_new[:,row,col])


# In[ ]:


start=time.time()
with Pool(processes=3) as pool:
    ind_start=988650+500
    ind_end=ind_start+100
    lai_values=pool.starmap(myfunction,l[ind_start:ind_end])
    pool.close()
    pool.join()
end=time.time()
print(end-start)
print(lai_values.shape)


# In[ ]:


start=time.time()
profile=src.profile
profile.update(count=46)
print(profile)
for j in range(len(lai_values)):
    for index in list(range(ind_start,ind_end)):
        row=int(index/tot_cols)
        col=index-(tot_cols*row)
        lai_new[:,row,col]=lai_values[index-ind_start]
    
    with rasterio.open('LAI_2015_corn_smoothed.tif', 'w', **profile) as dst:
        dst.write(lai_new.astype(rasterio.uint16), 1)
print(time.time()-start)

