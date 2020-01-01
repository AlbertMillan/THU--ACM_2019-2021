#!python3

"""
    Input:
        3  -2  2
        -2  5  -1

        First Line: length of array | lower bound | upper bound
        Second Line: Array values

    Output:
        3

    Example:
        Given nums = [-2, 5, -1], lower = -2, upper = 2,
        Return 3.
        The three ranges are [0, 0], [2, 2], [0, 2] and their respective sums are: -2, -1, 2.

The following implementation runs in O(n log n)
"""
import math
import time

def P(nums):
    sums = list([0, nums[0]])

    for i in range(2,len(nums)+1):
        sums.append(sums[i-1] + nums[i-1])

    return sums


# Merge-Sort Combine
def mergeArr(sums, left, delim, right):
    idxL = 0
    idxR = 0

    L = [sums[x] for x in range(left,delim)]
    R = [sums[x] for x in range(delim,right)]


    for i in range(left,right):
        if (left + idxL  < delim) and (delim + idxR < right):
            if L[idxL] <= R[idxR]:
                el = L[idxL]
                idxL += 1
            else:
                el = R[idxR]
                idxR += 1

            sums[i] = el

        elif (left + idxL  >= delim):
            sums[i] = R[idxR]
            idxR += 1

        else:
            sums[i] = L[idxL]
            idxL += 1

    return sums


def merge(sums,left,delim,right):
    count = 0
    j, k = delim, delim

    for i in range(left,delim):
        while k < right and (sums[k] - sums[i]) <= u:
            k += 1
        while j < right and (sums[j] - sums[i]) < l:
            j += 1
        
        count += k - j

    return count


def rangeSum(sums,left,right):
    if right - left <= 1:
        return 0

    count = 0
    # delim = math.floor( (left+right) / 2 )

    if left < right:
        delim = math.floor( (left+right) / 2.0 )
        count += rangeSum(sums, left, delim)
        count += rangeSum(sums, delim, right)
        count += merge(sums, left, delim, right)

        # Sort array
        sums = mergeArr(sums, left, delim, right)

    return count


if __name__ == "__main__":

    inputLine = input()
    n, l, u = map(int,inputLine.strip().split(' '))

    inputLine = input()
    nums = list(map(int,inputLine.strip().split(' ')))
    
    sums = P(nums)

    print(rangeSum(sums,0,n+1))

    # end = time.process_time_ns()

    # print('Execution Time (ms): '+ str((end-start) / 10**9))
    