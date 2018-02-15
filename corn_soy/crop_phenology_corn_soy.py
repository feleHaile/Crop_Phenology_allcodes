
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import rasterio
import time
import random,sys,getopt


# In[4]:

df_insert = pd.DataFrame({'NDVI_corn_2015': [np.NaN]*7,
                    'NDVI_soy_2015': [np.NaN]*7})
def dataframe_extraction(r1,r2,row,col):
    """This function returns a pandas dataframe that holds the NDVI values for 1-365 doys of 2015 and 2016 years.
    inputs: r1, r2: the NDVI Rasterstacks for 2015 and 2016 years. Each raster has 365 layers one corresponding to a doy of that year.
            row, col: the row and column of the raster pixel we would like to get the dataframe for """
    id=np.array(range(1,1+r1.shape[0]))
    data_frame = (pd.DataFrame({'NDVI_corn_2015':r1[0:r1.shape[0]+1:1,row,col],
                          'NDVI_soy_2015':r2[0:r1.shape[0]+1:1,row,col]},index=id))
    df=pd.DataFrame()
    for i in range(len(data_frame)):
        df=pd.concat([df,data_frame.iloc[[i]],df_insert])
    df.index=range(1,1+len(df.index))
    return df[:313]


# In[5]:

"""with rasterio.open('/Users/koutilya/Downloads/MOD09A1.h10v04.brdf_corrected/MOD09A1.A2015113.h10v04.brdf_product.02.01.tif') as src:
    print("CRS: ", src.crs)
    print("Band Count: ",src.count)   #Band count
    print("Indexes: ", src.indexes)
    print("Raster width:", src.width)
    print("Raster height:", src.height)
    print("DTypes of the Raster: ", src.dtypes)
    print("Extent of the Raster: ", src.bounds)
    print("Transformation from points to XYZ: ", src.transform)
    print("sample coordinate of the left uppermost point: ", src.transform * (0, 0))#row=0 column=0
    a=src.read()
    print("Total Count: ", (a.size))
    print(a.shape)
    print(a[0,1200,1200])
src=rasterio.open('MOD09A1.A2015113.h10v04.brdf_product.02.01.tif')
a=src.read()
print(a.shape)
k=rasterio.open('MOD09A1.A2016113.h10v04.brdf_product.02.01.tif')
b=k.read()
print(dataframe_extraction(a,b,1200,1200))"""


# In[71]:

import os
import pdb
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


def get_PTD(df):
    """
    Get phenological transition dates (greenup, senescence)
    :param df:
    :return:
    """
    # Input dataframe has an index comprised of day of year and remaining columns signify NDVI
    # Linearly interpolate dataframe columns to fill in missing values
    df = df.apply(pd.Series.interpolate)
    
    # Now compute mean of all columns and get the smoothened NDVI
    arr_smooth = do_fourier(df.mean(axis=1), [8.0] * len(df))
    
    # For all other crops and regions, take differential
    # To get doy_green, find the last occurrence of the max differential
    diff_green = np.diff(arr_smooth[:365 + 1])
    doy_green = np.where(diff_green == diff_green.max())[0][-1]
    doy_senesc = np.diff(arr_smooth[:365 + 1]).argmin()

    return doy_green, doy_senesc


# In[103]:

#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
#import matplotlib.pyplot as plt
#%matplotlib inline
def initialize_rasters(path1,path2):
    raster1=rasterio.open(path1)
    tot_cols=raster1.width
    tot_rows=raster1.height
    a=raster1.read()
    print("Tot rows: ",tot_rows," Tot cols: ",tot_cols)
    raster2=rasterio.open(path2)
    b=raster2.read()
    if(b.shape[0]!=a.shape[0]):
        print("determine what doy you are missing!!")
        t1=b[0:6] #determine what doy you are missing!! in this case its DOY49 thus insert as 7th layer
        t2=b[6:b.shape[0]]
        p=b[0]*0
        p=p.reshape(1,tot_rows,tot_cols)

        tp=np.append(t1,p,axis=0)
        b=np.append(tp,t2,axis=0)
        #print(b.shape)
    return (a,b,tot_rows,tot_cols)
def ritvik_fn(df):
    #return (random.randint(1,50),random.randint(120,365))
    return get_PTD(df)

def isNaN(num):
    return ((num != num) or num==None)

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
    df[df<=0]=None
    
    first_valid_indices=df.apply(lambda col: col.first_valid_index())
    for col in df.columns:
        if(not isNaN(first_valid_indices[col])):
            df.loc[1][col]=df.loc[first_valid_indices[col]][col]

    pairs=list()
    for col in df.columns:
        ndvi=df[col].tolist()
        clean=[x for x in ndvi if str(x) != 'nan']
        clean = [max(0, min(x, 10000)) for x in clean]
        if(len(clean)>0):
            maxi=int(max(clean))
            if(maxi>2000):
                g,s=ritvik_fn(pd.Series.to_frame(df[col]))
            else:
                g=0
                s=0
        else:
            g=0
            s=0
        pairs.append((g,s))
     
    """if (len(clean_soy)>0 || len(clean_corn)>0):
        plt.plot(list(range(len(clean_soy))),clean_soy)
        plt.xlabel("DOY")
        plt.ylabel("NDVI")
        plt.title("NDVI time series plot")
        plt.show()
        return ritvik_fn(pd.Series.to_frame(df['NDVI_soy_2015']))
    else:
        return (0,0)"""

    return pairs


# The doy interval we need to consider for the dataframe passed into ritvik_fn changes from crop to crop and ste to state. For eg, for winter wheat in Kansas, greep up happens in 1:60 doys and sennescence in 125:200 doys.

# In[8]:

#a,b,tot_rows,tot_cols=initialize_rasters("A2015_177_ndvi_480m.tif","A2015_185_ndvi_480m.tif")
def main(argv):
    opts,args=getopt.getopt(argv,"hp:m:r:s:",["processes=","mode=","run=","side="])
    for opt,arg in opts:
        if opt=="-h":
            print("python crop_phenology.py -p 16 -m 100 -r s -s 1 ")
            print("Give the number of processes into -p switch and the number of pixels to run as -m switch and the way to run as -r switch and which half of the raster as --side")
            print("python crop_phenology.py -p 16 -m 0 -r s|p  -s 1|2   -> m=0 implies run on all pixels and r=s implies run sequentially where as r=p implies running paralelly")
            sys.exit()
        elif opt in ("-p","--processes"):
            num_proc=int(arg)
        elif opt in ("-m","--mode"):
            index_input=int(arg)
            if index_input==0:
                ind_start=0
            else:
                ind_start=1150139
        elif opt in ("-r","--run"):
            run=arg
        elif opt in ("-s","--side"):
            side=int(arg)    

    src=rasterio.open('NDVI_480m_stack_soy.tif')
    a,b,tot_rows,tot_cols=initialize_rasters('NDVI_480m_stack_corn.tif','NDVI_480m_stack_soy.tif')
    #src=rasterio.open('NDVI_2015_nebraska.tif')
    #a,b,tot_rows,tot_cols=initialize_rasters("NDVI_2015_nebraska.tif","NDVI_2016_nebraska.tif")
    y=[(a,b)]*tot_rows*tot_cols
    ind=range(tot_rows*tot_cols)
    l=list()
    l=list(list(zip(ind,y))[:])
    greenup=np.zeros(shape=a[1].shape)
    sen=greenup

    #plant=a[1]*0
    #har=a[1]*0
    #plant=plant.astype('int32')
    #har=har.astype('int32')

    start=time.time()
    if index_input==0:
            if(side==1):
                ind_start=0
                ind_end=int(len(l)/2)
            elif(side==2):
                ind_start=int(len(l)/2)
                ind_end=len(l)
            #print(ind_end)
    else:
            ind_end=ind_start+index_input

    print("ind_start: ",ind_start," ind_end: ",ind_end)
    if(run=="p"):
        with Pool(processes=num_proc) as pool:
            #ind_start=988800
            #ind_end=ind_start+1000
            pairs_crops=pool.starmap(myfunction,l[ind_start:ind_end])
            pool.close()
            pool.join()
    elif(run=="s"):
        for i in range(ind_start,ind_end):
            pairs_crops.append(myfunction(l[i][0],l[i][1]))
    end=time.time()
    print(end-start)
    #print(pairs)


    profile=src.profile
    profile.update(count=1)
    print(profile)
    for j in range(len(pairs_crops[1])):
        pairs=[pairs_crops[i][j] for i in range(len(pairs_crops))]
        greenup=np.zeros(shape=a[1].shape)
        sen=np.zeros(shape=a[1].shape)
        for index in list(range(ind_start,ind_end)):
            row=int(index/tot_cols)
            col=index-(tot_cols*row)
            greenup[row][col]=pairs[index-ind_start][0]
            #plant[row][col]=pairs[index-ind_start][0]-15
            #print("Row: ",row," Col: ",col," ",greenup[row][col])
            sen[row][col]=pairs[index-ind_start][1]
            #har[row][col]=pairs[index-ind_start][1]+45
        #np.clip(plant, 1, 365, out=plant)
        #np.clip(har, 1, 365, out=har)
        #plant=plant.astype('uint32')
        #har=har.astype('uint32')
        
        with rasterio.open('greenup_'+str(j)+'.tif', 'w', **profile) as dst:
            dst.write(greenup.astype(rasterio.float64), 1)
        with rasterio.open('sen_'+str(j)+'.tif', 'w', **profile) as dst:
            dst.write(sen.astype(rasterio.float64), 1)


if __name__=="__main__":
    main(sys.argv[1:])



