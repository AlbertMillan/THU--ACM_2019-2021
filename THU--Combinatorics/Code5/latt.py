import time

TOP = 0
BOTTOM = 1

RIGHT = 1
LEFT = 0
DOWN = 0
UP = 1

def permsComp(r,m,s):
    # r = abs(x2-x1) 
    # m = abs(y2-y1)
    n = r + m 

    # Compute Combination using Pascal's Triangle
    # s = r - (x2 - y2)
    s = (n-s if s > ((n+1)/2) else s )
    r = (n-r if r > ((n+1)/2) else r )
    p = 1000000007

    C = [0 for i in range(r+1)]

    # Top of Pascal's Triangle
    C[0] = 1

    # Construct Pascal's Triangle
    for i in range(1, n+1):

        for j in range(min(i, r), 0, -1):
            
            C[j] = (C[j] + C[j-1]) % p 

        # print(*C, sep = ", ")

    # Retrieve at corresponding position
    return C[r] - C[s]
    # return (1 / (r + xDistDel) ) * C[r] 


def allPerms(x1,y1,x2,y2):
    r = abs(x2-x1) 
    n = ( r + abs(y2-y1) )

    # Compute Combination using Pascal's Triangle
    
    r = (n-r if r > ((n+1)/2) else r )
    p = 1000000007

    C = [0 for i in range(r+1)]

    # Top of Pascal's Triangle
    C[0] = 1

    # Construct Pascal's Triangle
    for i in range(1, n+1):

        for j in range(min(i, r), 0, -1):
            
            C[j] = (C[j] + C[j-1]) % p 

        # print(*C, sep = ", ")

    # Retrieve at corresponding position
    return C[r] 


def direction(hDis,vDis):
    hDir = 0
    vDir = 0
    if hDis < 0:
        hDir = LEFT
    else:
        hDir = RIGHT
    if vDis < 0:
        vDir = DOWN
    else:
        vDir = UP

    return hDir,vDir

def classify(x,y):
    if y > x:
        return TOP
    else:
        return BOTTOM


# ***
# * origin:    (x1,y1)
# * target:    (x2,y2)
# * delimiter: y = x
# ***
def lattice(x1,y1,x2,y2):
    
    # Case where origin or target points are touching the delimiter
    if y1 == x1 or y2 == x2:
        return 0

    # Determine position of the points.
    # Return 0 if crossing is required.
    side1 = classify(x1,y1)
    side2 = classify(x2,y2)

    if side1 != side2:
        return 0

    # Pre-processing step:
    # 1. Two points, no way to touch line
    # 2. Delimiter cuts rectangle, but both points are in the same side of the graph.  
    hDis = x2 - x1
    vDis = y2 - y1

    hDir, vDir = direction(hDis,vDis)

    r = abs(hDis)
    m = abs(vDis)

    yVal = 0
    limit = 0
    s = 0

    if hDir == LEFT and vDir == DOWN:
        # Biggest y value is at x1
        if side1 == BOTTOM:
            yVal = y1
            limit = x2
            s = m - (x2 - y2)
        else:
            yVal = y2
            limit = x1
            s = m - (x2 - y2)

    elif hDir == RIGHT and vDir == UP:
        # Biggest y value is at x1
        if side1 == BOTTOM:
            yVal = y2
            limit = x1
            s = r - (x2 - y2)
        else:
            yVal = y1
            limit = x2
            s = r - (x2 - y2)


    if side1 == BOTTOM:
        if yVal > limit: 
            # return permsComp(x1,y1,x2,y2)
            return permsComp(r,m,s)
        else:
            return allPerms(x1,y1,x2,y2)
    
    else: # Maybe I have to project this as the line cuts in the bottom side
        if yVal < limit: 
            return permsComp(r,m,s)
        else:
            return allPerms(x1,y1,x2,y2)
        
    return 0

    

if __name__ == "__main__":

    inputLine = input()
    x1,y1,x2,y2 = map(int,inputLine.strip().split(' '))


    # start = time.process_time_ns()

    print("{0}".format( int(lattice(x1,y1,x2,y2 ) )) )

    # end = time.process_time_ns()

    # print('Execution Time (ms): '+ str((end-start) / 10**9))
