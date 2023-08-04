#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import time
import shutil
import re
import math
import pickle
import glob

#function to create a xarray from DARTS results 
def ModelOut(m):
    re_time_data = re.compile('(?P<origin>\w*?)[\s:]*(?P<name>[\w\s]+) \(?(?P<unit>[\w\/]+)\)?')

    time = np.array(m.physics.engine.time_data['time'])
    data_arrays = []
    origins = set()
    ds = xr.Dataset()

    for k, v in m.physics.engine.time_data.items():
        if re_time_data.match(k):
            origin, name, unit = re_time_data.match(k).groups()
            #substitute spaces with underscores in all names
            name = name.replace(' ', '_')
            origin = origin.replace(' ', '_')
            
            ds = ds.merge({name:
                        xr.DataArray(
                            data=np.array(m.physics.engine.time_data[k]).reshape(1, -1) if origin else np.array(m.physics.engine.time_data[k]), 
                            coords={'origin': [origin], 'time': time} if origin else {'time': time},
                            dims=('origin', 'time') if origin else ('time'),
                            attrs={'unit': unit}
                        )
            })
    return ds

#calculate the reference to compare models
def media_function(time_range, time_data, data_type):
    media =[]
    i= 1
    while i<=(len(time_range)-1):
        s=[]
        for j,tempo in enumerate(time_data['time']):
                if time_range[i-1]<time_data['time'][j]<=time_range[i]:
                    s.append(time_data[data_type][j])
                    np.mean(s)
        media.append(np.mean(s))
        i=i+1
    return media


# %%
#Creat a function to delete all the files in the folder data/simutaltions/it0 and data/simutaltions/it1 and data/simutaltions/it2 and data/simutaltions/it3
def deleteOutFiles():
    #delete all files in the folder data/simutaltions/it0
    path = 'data/simulations/it0'
    files = os.listdir(path)
    for file in files:
        os.remove(os.path.join(path, file))

    #delete all files in the folder data/simutaltions/it1
    path = 'data/simulations/it1'
    files = os.listdir(path)
    for file in files:
        os.remove(os.path.join(path, file))

    #delete all files in the folder data/simutaltions/it2
    path = 'data/simulations/it2'
    files = os.listdir(path)
    for file in files:
        os.remove(os.path.join(path, file))

    #delete all files in the folder data/simutaltions/it3
    path = 'data/simulations/it3'
    files = os.listdir(path)
    for file in files:
        os.remove(os.path.join(path, file))
    #delete all files in the folder data/simutaltions/it3
    path = 'data/simulations/it4'
    files = os.listdir(path)
    for file in files:
        os.remove(os.path.join(path, file))
    #delete all files in the folder data/simutaltions/it3
    path = 'data/simulations/it5'
    files = os.listdir(path)
    for file in files:
        os.remove(os.path.join(path, file))

def MultipliNegatives(dObs):
   for i in range(len(dObs)):
       if dObs[i] < 0:
           dObs[i] = abs(dObs[i])
   return dObs

#-----ESDMA FUNCTIONS-----

# %%
# Rotate coordinates and flattens the matrix to an array
def CalcHL(x0, x1, L, theta):
    cosT = np.cos(theta)
    sinT = np.sin(theta)
    dx = x1[0] - x0[0]
    dy = x1[1] - x0[1]

    dxRot = np.array([[cosT, -sinT], [sinT, cosT]]) @ np.array([[dx], [dy]])
    dxFlat = dxRot.flatten()

    return np.sqrt((dxFlat[0]/L[0])**2 + (dxFlat[1]/L[1])**2)

# Calc covariance between two gridblocks
def SphereFunction(x0, x1, L, theta, sigmaPr2):
    hl = CalcHL(x0, x1, L, theta)

    if (hl > 1):
        return 0
    
    return sigmaPr2 * (1.0 - 3.0/2.0*hl + (hl**3)/2)

def GaspariCohnFunction(x0, x1, L, theta):
    hl = CalcHL(x0, x1, L, theta)

    if (hl < 1):
        return -(hl**5)/4. + (hl**4)/2. + (hl**3)*5./8. - (hl**2)*5./3. + 1.
    if (hl >= 1 and hl < 2):
        return (hl**5)/12. - (hl**4)/2. + (hl**3)*5./8. + (hl**2)*5./3. - hl*5 + 4 - (1/hl)*2./3.
    
    return 0

# convert index numeration to I J index
def IndexToIJ(index, ni, nj):
    return ((index % ni) + 1, (index // ni) + 1)

# Convert i J numeration to index
def IJToIndex(i,j,ni,nj):
    return (i-1) + (j-1)*ni

def BuildPermCovMatrix(Ni, Nj, L, theta, sigmaPr2):
    Nmatrix = Ni * Nj
    Cm = np.empty([Nmatrix, Nmatrix])
    for index0 in range(Nmatrix):
        I0 = IndexToIJ(index0,Ni,Nj)
        for index1 in range(Nmatrix):
            I1 = IndexToIJ(index1,Ni,Nj)
            Cm[index0, index1] = SphereFunction(I0, I1, L, theta, sigmaPr2)
    return Cm

# Builds the localization matrix. wellPos is a list with tuples, 
# each corresponding with the position of the data
def BuildLocalizationMatrix(Ni, Nj, wellPos, L, theta):
    Npos = Ni * Nj
    Nd = len(wellPos)
    Rmd = np.ones([Npos,Nd])
    for i in range(Npos):
        # Get the index of the cell
        Im = IndexToIJ(i, Ni, Nj)
        
        for j in range(Nd):
            Iw = wellPos[j]

            Rmd[i, j] = GaspariCohnFunction(Im, Iw, L, theta)
    return Rmd

def PlotModelRealization(m, Ni, Nj, title, axis, vmin=None, vmax=None):
    return PlotMatrix(m.reshape((Ni,Nj),order='F').T, title, axis, vmin, vmax)

def PlotMatrix(matrix, title, axis, vmin=None, vmax=None):
    axis.set_title(title)
    return axis.imshow(matrix, cmap='RdYlGn_r', vmin=vmin, vmax=vmax, aspect='auto')

def RunModels(destDir, MGrid, MScalar, l, final_time, time_step):
    for j,mGridColumn in enumerate(MGrid.T):
        job_file = os.path.join(destDir,f'data_model{j}')
        #grid = f'grid_{job_file}'
        print(job_file)

        mGridColumn = pd.DataFrame(mGridColumn)
        mGridColumn.to_pickle(f'{job_file}_grid.pkl')
        
        print(f'Grid {job_file} saved')

        with open(job_file, 'w+') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f'#SBATCH --job-name=it{l}_{j}\n')
            fh.writelines("#SBATCH --time=2:00:00\n")
            fh.writelines("#SBATCH --mem=1G\n")
            fh.writelines("#SBATCH --partition=compute\n")
            fh.writelines("#SBATCH --account=research-ceg-gse\n")
            #fh.writelines("#SBATCH --cpus-per-task=16\n")            
            fh.writelines(f'#SBATCH -e {job_file}_err\n')
            fh.writelines(f'#SBATCH -o {job_file}_out\n')
            #fh.writelines("module load 2022r2\n")
            fh.writelines(f'a={int(j)}\n')
            fh.writelines(f'b={str(destDir)}\n')
            fh.writelines(f'c={str(final_time)}\n')
            fh.writelines(f'd={str(time_step)}\n')
            fh.writelines(f'python main_darts_compact_media.py $a $b $c $d\n')

            #fh.writelines("Rscript $HOME/project/LizardLips/run.R %s potato shiabato\n" %lizard_data)
            
            
        os.system(f'sbatch {job_file}') #{str(j)} {str(destDir)} {str(time_range[-1])} ') 

def check_job(filename):
    while not os.path.exists(filename):
        time.sleep(2)
        print('waiting for job last to finish')
    return filename         

# %%
#Read the result from the model
def ReadModels(destDir, columnsNameList, Nd, Ne,time_range,time_step):
    D = np.empty([Nd, Ne])
    for i in range(Ne):
        while not os.path.exists(f'{destDir}/data_model'+str(i)+'.pkl'):
            time.sleep(2)
            print('waiting to READ job: '+str(i))
        print(f'reading model result: {i}')
        dataSet = pd.read_pickle(f'{destDir}/data_model'+str(i)+'.pkl') 
        dataModel=[]
        model_value=[]
        d_models=[]
        dataModel=dataSet[dataSet['time'].isin(time_range+time_step)]
        model_value = dataModel[columnsNameList]
        #drop hearder and concatenate all the data
        model_value=np.array(model_value)
        d_models = model_value.T.flatten()

        d_models = MultipliNegatives(d_models)   
        print(f'd_models shape: {d_models.shape}')     
           
        D[:,i] = d_models 

    return D


##%%
#ReadModels(destDir, columnsNameList, Nd, Ne)

# %% [markdown]
# Functions to process and update models
# Finds the truncation number
def FindTruncationNumber(Sigma, csi):
    temp = 0
    i = 0
    svSum = np.sum(Sigma)
    stopValue = svSum * csi
    for sv in np.nditer(Sigma):
        if (temp >= stopValue):
            break
        temp += sv
        i += 1
    return i

def CentralizeMatrix(M):
    meanMatrix = np.mean(M, axis=1)
    return M - meanMatrix[:,np.newaxis]

# Psi = X9 in (12.23)
def UpdateModelLocalized(M, Psi, R, DobsD):
    DeltaM = CentralizeMatrix(M)
   
    K = DeltaM @ Psi
    Kloc = R * K
    return M + Kloc @ DobsD

def UpdateModel(M, Psi, DobsD):
    DeltaM = CentralizeMatrix(M)

    X10 = Psi @ DobsD
    return M + DeltaM @ X10

def calcDataMismatchObjectiveFunction(dObs, D, CeInv):
    Ne = D.shape[1]
    Nd = D.shape[0]

    Od = np.empty(Ne)
    for i in range(Ne):
        dObsD = dObs - D[:,i].reshape(Nd,1)
        Od[i] = (dObsD.T) @ (CeInv[:,np.newaxis] * dObsD)/2
    return Od

# Replaces the pattern with the value in array cosrresponding its position.
# Only 1 group per line for now...
def ReplacePattern(matchobj, array):
    return f'{array[int(matchobj.group(1))]:.2f}' 

