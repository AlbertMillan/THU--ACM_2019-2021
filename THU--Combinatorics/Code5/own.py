TOP = 0
BOTTOM = 1

RIGHT = 1
LEFT = 0
DOWN = 0
UP = 1

FALSE = -1

def direction(x1,y1,x2,y2):
    hDis = x2 - x1
    vDis = y2 - y1

    r = abs(hDis)
    m = abs(vDis)

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

    return hDir,vDir, r,m


def classify(x,y):
    if y > x:
        return TOP
    else:
        return BOTTOM


def allPerms(r,n, s=FALSE):

    if s != FALSE:
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

    delete = 0
    if s != FALSE:
        delete = C[s]

    # Retrieve at corresponding position
    return (C[r] - delete) % p


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

    hDir, vDir, r, m = direction(x1,y1,x2,y2)

    if (hDir == RIGHT and vDir == DOWN) or (hDir == LEFT and vDir == UP):
        return allPerms( r, r+m )

    elif side1 == BOTTOM:
        yVal = 0
        limit = 0
        s = 0
        if hDir == LEFT and vDir == DOWN:
            yVal = y1
            limit = x2
            s = r - (x1 - y1)
        else:
            yVal = y2
            limit = x1
            s = r - (x2 - y2)
        
        if yVal >= limit:
            return allPerms( r, r+m, s)
            # return permsComp()
        else:
            return allPerms( r, r+m )

    else:
        yVal = 0
        limit = 0
        if hDir == LEFT and vDir == DOWN:
            yVal = y2
            limit = x1
            s = r - (abs(x2 - y2))
        else:
            yVal = y1
            limit = x2
            s = r - (abs(x1 - y1))
        
        if yVal <= limit:
            # return -1
            return allPerms( r, r+m, s)
        else:
            return allPerms( r, r+m )



if __name__ == "__main__":

    inputLine = input()
    x1,y1,x2,y2 = map(int,inputLine.strip().split(' '))

    print("{0}".format( lattice(x1,y1,x2,y2 ) ) )