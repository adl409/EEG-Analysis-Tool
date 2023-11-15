import os
import pandas as pd

base = "../250ms"
entries = os.listdir(base)

# Going through each of the time intervals
def parse(hand):

    #Specifying all the folders that contains the CSV files
    folders = ["Van250Tot0", "Van500Tot250", "Van750Tot500", "Van1000Tot750", "Van1250Tot1000", "Van1500Tot1250",
                   "Van1750Tot1500", "Van2000Tot1750", "Van2250Tot2000", "Van2500Tot2250", "Van2750Tot2500", "Van3000Tot2750"]
    
    # Dictionary that is going to store all the array of data frames from the folders
    dataFrames = {}

    # From all of the folders
    for folder in folders:

        # Array that's going to stores all the data frames in that specific folder
        allFrames = []

        # Parsing though all the files in the folder
        peoples = os.listdir(os.path.join(base, folder))
        for person in peoples:
            numOfFiles = len(os.listdir(os.path.join(base,folder,person,hand)))
            for i in range(1, numOfFiles):
                filename = "../250ms/" + str(folder) + "/" + str(person) + "/" + hand + "/P"+ str(person) + hand[0] + str(i) + ".csv"

                # Creating a single data frame from a single CSV file
                finaldf = pd.read_csv(filename, header=None)
                finaldf = finaldf.to_numpy().transpose()
                allFrames.append(finaldf)

        dataFrames[folder] = allFrames

    return dataFrames

def main():
    dictionary = {}

    dictionary["Left"] = parse("Left")
    dictionary["Right"] = parse("Right")

    print(dictionary)

main()
