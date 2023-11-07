import pandas as pd

class Experiment:
    
    def __init__(self, personNum, expNum, intervals, result):
        
        # Set of three data points allows correct file set to be selected from files
        # result of experiment
        self.resultShort = result.upper()
        if(self.resultShort == 'R'):
            self.resultLong = "Right"
        else:
            self.resultLong = "Left"
        # Experiment participant number
        self.personNum = personNum
        #Experiment number
        self.expNum = expNum
        
        # intervals is number of intervals from 0 to be used in trial
        self.intervals = intervals

        # Data frame containing all eeg data for experiment
        self.df = self.createDF()

        self.data = [self.df, self.resultShort]

    # Creates dataframe for selected experiment and interval
    def createDF(self):

        folders = self.intervalToFolders()
        
        for folder in folders:
            filename = folder+"/"+str(self.personNum)+"/"+self.resultLong+"/P"+str(self.personNum)+self.resultShort+str(self.expNum)+".csv"
            tempdf = pd.read_csv(filename)
            print('\n\n', filename, '\n')
            print(tempdf, "\n")

        return tempdf

    # Gives directory names for chosen number of intervals. 
    def intervalToFolders(self):
        
        folders = ["Van250Tot0", "Van500Tot250", "Van750Tot500", "Van1000Tot750", "Van1250Tot1000", "Van1500Tot1250",
                   "Van1750Tot1500", "Van2000Tot1750", "Van2250Tot2000", "Van2500Tot2250", "Van2750Tot2500", "Van3000Tot2750"]
        
        return folders[0:self.intervals]

    


def main():

    # df = pd.read_csv('')

    test = Experiment(2, 2, 4, "r")

    test.createDF()

    return 0


main()