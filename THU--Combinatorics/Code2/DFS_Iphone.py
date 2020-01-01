import math

CENTER = 5
SUM = 10


count = 0

def callback(curr):
    # Add current to the stack
    stack.append(curr)
    count = 0
    # valid = False

    # Loops over all the elements it can access
    if len(stack) == (l-1):
        if b in dictionary[curr-1]:
            stack.append(b)
            # print(stack)
            stack.pop()
            stack.pop()
            return 1
        else:

            if ((curr + b) in jumpSet and curr != 5 and b != 5):
                jumped = (curr + b) / 2

                # element checked
                if jumped in stack:
                    stack.append(b)
                    # print(stack)
                    stack.pop()
                    stack.pop()
                    return 1
                    # valid = True
                    

    else:
        # Check Jumping Possibilities
        extra = []

        if CENTER in stack and curr is not CENTER:
            extra.append(SUM - curr)

        if curr in corners:
            # Check row
            row = math.floor((curr-1) / 3)
            col = (curr-1) % 3

            if row == 0:
                if (curr+3) in stack:
                    extra.append(curr + 2*3)
            else:
                if (curr-3) in stack:
                    extra.append(curr - 2*3)

            if col == 0:
                if (curr+1) in stack:
                    extra.append(curr + 2)
            else:
                if (curr-1) in stack:
                    extra.append(curr - 2)


            # Check Column
        for i in dictionary[curr-1]:

            if i not in mySet and i not in stack:
                count += callback(i)

        for j in extra:
            if j not in mySet and j not in stack:
                count += callback(j)

    if len(stack) != 1:
        stack.pop()

    return count



inputLine = input()
a, b, l = inputLine.split(' ')
a = int(a)
b = int(b)
l = int(l)


dictionary = [
    [2,4,5,6,8],
    [1,3,4,5,6,7,9],
    [2,4,5,6,8],
    [1,2,3,5,7,8,9],
    [1,2,3,4,6,7,8,9],
    [1,2,3,5,7,8,9],
    [2,4,5,6,8],
    [1,3,4,5,6,7,9],
    [2,4,5,6,8]
]

stack = []
mySet = set([a,b])

corners = set([1,3,7,9])
middle = set([2,4,6,8])
jumpSet = set([10,4,8,16,12])

print(callback(a))