
def Table(n):
    myfile = ""
    for i in range(1,11):
        myfile += f"{n} x {i} = {n*i}\n"
    with open(f"myfiles_{n}.txt", "w") as file:
        file.write(myfile)
for i in range(2,41):
    Table(i)











    