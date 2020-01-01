
import math

_MAX = 10000000


def print_neatly(n,M,words):
    # words.insert(0,'')
    c = [0 for i in range(n+1)]
    
    for j in range(1,n+1):

        c[j] = _MAX
        i = math.floor(max(1,j-M/2+1))

        while i <= j:

            sup = M - len(words[i-1])

            for k in range(i,j):
                sup = sup - len(words[k]) - 1
            
            if sup < 0:
                lc = _MAX
            elif ( (j-1) == (n-1) ) and sup>0:
                lc = 0
            else:
                lc = sup**3
            
            if (c[i-1] + lc < c[j]):
                c[j] = c[i-1]+lc

            i += 1

    return c[n]




if __name__ == "__main__":
    
    inputLine = input()
    n, M = map(int,inputLine.strip().split(' '))

    inputLine = input()
    words = inputLine.strip().split(' ')

    print(print_neatly(n,M,words))


    # word like first as the the complete