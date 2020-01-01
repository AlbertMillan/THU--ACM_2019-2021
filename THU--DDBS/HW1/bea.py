import numpy as np
import pylab
import scipy.cluster.hierarchy as sch

#Create matrices
n = 7
# AA = np.zeros([n,n])
# AA = np.array([[42,0,37,5],[0,82,7,75],[37,7,44,0],[5,75,0,8]])
AA = np.array([ [30,30,30,30,30,30,30],
                [30,30,30,30,30,30,30],
                [30,30,30,30,30,30,30],
                [30,30,30,60,30,30,60],
                [30,30,30,30,30,30,30],
                [30,30,30,30,30,30,30],
                [30,30,30,60,30,30,60]
     ])
CA = np.zeros([n,n])

#Create global arrays and variables
array = []
maxIndex = []
maxIndex2 = []
index = 0

#Assign values to initial matrix
# AA[0,0] = 45; AA[0,1] = 0; AA[0,2] = 45; AA[0,3] = 0
# AA[1,0] = 0; AA[1,1] = 80; AA[1,2] = 5; AA[1,3] = 75
# AA[2,0] = 45; AA[2,1] = 5; AA[2,2] = 53; AA[2,3] = 3
# AA[3,0] = 0; AA[3,1] = 75; AA[3,2] = 3; AA[3,3] = 78

#Calculate global bond energy
def cont(ai, ak, aj):
    if ak == aj:
        return 2 * bond(ai, ak) + 2 * bond(ak, aj)
    else:
        return 2 * bond(ai, ak) + 2 * bond(ak, aj) - 2 * bond(ai, aj)

#Calculate bond energy for 2 columns
def bond(ax, ay):
    result = 0
    if ax<0 or ay<0:
        return 0
    if ax == index:
        for i in range(0,n):
            result += array[i] * CA[i,ay]
        return result
    if ay == index:
        for i in range(0,n):
            result += array[i] * CA[i,ax]
        return result
    for i in range(0,n):
        result += CA[i,ax] * CA[i,ay]
    return result
    
def maxCont(res):
    maxValue = 0
    for i in range(len(res)):
        if res[i][3]>maxValue:
            maxValue = res[i][3]
            maxindex = res[i][2]
    return maxindex

def BEA(AA):
    #Add two columns to new matrix
    for i in range(0,n):
        global CA
        CA[i,0] = AA[i,0]
        CA[i,1] = AA[i,1]
    global index
    index = 2
    #Create array for storing the results
    s = [0,0,0,0]
    #Calculate bond energy and insert new column, where
    #it generates most general bond energy
    while (index<=n-1):
        results = []
        global array
        array = AA[:,index]
        i = 0
        while (i<=index):
            result = cont(i-1,index,i)
            s[0]=i-1
            s[1]=index
            s[2]=i
            s[3]=result
            v=[]
            v=s[:]
            results.append(v)
            i+=1
        CA = np.insert(CA, maxCont(results), np.array((array)), 1)
        maxIndex.append(maxCont(results))
        maxIndex2.append(index)
        CA = np.delete(CA, (-1), 1)
        CA[:,[0, 1]] = CA[:,[1, 0]]
        index+=1
    return CA
    
BEA(AA)
print(CA)
#Rearrange the rows according to the order of columns
for i in range(len(maxIndex)):
    CA[[maxIndex[i], maxIndex2[i]],:] = CA[[maxIndex2[i], maxIndex[i]],:]
print(CA)
