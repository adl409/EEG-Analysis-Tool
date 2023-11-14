import os
import pandas as pd

base = "../250ms"
entries = os.listdir(base)

# Going through each of the time intervals
def parse(hand):
    folders = ["Van250Tot0", "Van500Tot250", "Van750Tot500", "Van1000Tot750", "Van1250Tot1000", "Van1500Tot1250",
                   "Van1750Tot1500", "Van2000Tot1750", "Van2250Tot2000", "Van2500Tot2250", "Van2750Tot2500", "Van3000Tot2750"]
    

    dataFrames = {}

    for folder in folders:
        allFrames = []
        peoples = os.listdir(os.path.join(base, folder))
        for person in peoples:
            numOfFiles = len(os.listdir(os.path.join(base,folder,person,hand)))
            for i in range(1, numOfFiles):
                filename = "../250ms/" + str(folder) + "/" + str(person) + "/" + hand + "/P"+ str(person) + hand[0] + str(i) + ".csv"

                tempdf = pd.read_csv(filename, header=None)
                allFrames.append(tempdf)

        if len(allFrames) > 0: 
            finaldf = pd.concat(allFrames, axis=1)
        else:
            finaldf = pd.DataFrame()

        dataFrames[folder] = finaldf

        print(folder)
        print(finaldf)

    return dataFrames
# with open("cat.csv", 'w') as f:

#     for i, entry in enumerate(entries):
#         print(entry)
#         f.write(entry + ", ")
#         list = [0] * 22
#         person = os.listdir(os.path.join(base, entry))
#         for j, num in enumerate(person):
#             hands = os.listdir(os.path.join(base, entry, num))
#             for hand in hands:
#                 values = os.listdir(os.path.join(base, entry, num, hand))

#             list[int(num) - 1] = len(values)
#         f.write(str(list)[1:-1] + "\n")

def main():
    dictionary = {}

    dictionary["Left"] = parse("Left")
    dictionary["Right"] = parse("Right")

    print(dictionary)

main()
