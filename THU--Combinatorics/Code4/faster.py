

def partition(n,k):

    for i in range(1,n+1):
        for j in range(1,k+1):
            if (i-j) < 0:
                arr.append(table[i][j-1])
                break

            val = table[i-j][j]
            if val != 0:
                table[i][j] = (table[i][j-1] + val) % 1000000007
            else:
                table[i][j] = (table[i][j-1] + arr[i-j]) % 1000000007

    return table[n][k]


if __name__ == "__main__":

    inputLine = input()
    n, k = map(int,inputLine.strip().split(' '))

    table = [ [0] * (k+1) for i in range(n+1)]
    arr = list([0])

    for i in range(1,k+1):
        table[0][i] = 1

    start = time.process_time_ns()

    print("{0}".format( (partition(n,k) ) ))

    end = time.process_time_ns()

    print('Execution Time (ms): '+ str((end-start) / 10**9))
