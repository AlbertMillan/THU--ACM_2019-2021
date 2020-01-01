# !python3
import time

def factorial(maxNum, minNum):
    fact = 1
    for i in range(maxNum,minNum,-1):
        fact *= i
    return fact

def getAvailable(visited,element):
    count = 0

    for i in range(element-1,0,-1):
        if i not in visited:
            count += 1

    return count

def binarySearch(arr,low,high,totalPerms, factTerm):

    while low <= high:
        mid = low + (high-low) // 2

        nPerms = mid * factTerm

        if totalPerms > nPerms:
            low = mid + 1
            index = mid
            oldPerms = nPerms
        elif totalPerms < nPerms:
            high = mid - 1
        else:
            low = mid + 1
            index = mid
            oldPerms = nPerms

    return index, oldPerms


def permutation(totalPerms,n):
    factTerm = _FACT__INIT
    oldFact = 1
    res = []
    values = [x for x in range(1, n+1)]


    for i in range(len(perm)):
        factTerm = int(factTerm / oldFact)

        oldPerms = 0
        newPerms = 0
        term = 0
        index = 0

        index, oldPerms = binarySearch(values,0,len(values)-1,totalPerms,factTerm)
                    
        totalPerms = totalPerms - oldPerms
        res.append(values.pop(index))
        oldFact = _FACT__INDEX - i

    print(*res)



# Process Input
inputLine = input()
n,a = inputLine.strip().split(' ')
n = int(n)
a = int(a)

inputLine = input()
perm = list(map(int,inputLine.strip().split(' ')))

# Timer
start = time.process_time_ns()

# Tracking Variables
_FACT__INDEX = n - 1
_FACT__INIT = factorial(n-1,1)

visited = set({})
factTerm = _FACT__INIT
oldFact = 1
prevPerms = 0

# Operations
# 1. Count the number of preceding permutations of the given number
for i in range(len(perm)):
    element = perm[i]

    valid = getAvailable(visited,element)
    factTerm = int(factTerm / oldFact)    
    prevPerms += valid * factTerm

    visited.add(element)

    oldFact = _FACT__INDEX - i

totalPerms = prevPerms + a


if a == 0:
    print(*perm)
else:
    # 2. Generate Perms
    permutation(totalPerms,n)

end = time.process_time_ns()
print('Execution Time (ms): '+ str((end-start) / 1000))