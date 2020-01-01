# Main
inputLine = input()
n, r = inputLine.split(' ')
n = int(n)
r = int(r)
r = (n-r if r > ((n+1)/2) else r )
p = 1000000007

C = [0 for i in range(r+1)]

# Top of Pascal's Triangle
C[0] = 1

# Construct Pascal's Triangle
for i in range(1, n+1):

    for j in range(min(i, r), 0, -1):
        
        C[j] = (C[j] + C[j-1]) % p 

    print(*C, sep = ", ")

# Retrieve at corresponding position
print(C[r]) 
