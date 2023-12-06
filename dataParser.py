import pandas as pd
import numpy as np
import time

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
        
        # interval is a value that determines how many time interval files are used. For example, an interval of 3 would result in data being collected from the 0-250, 250-500, & 500-750ms ranges. 
        self.intervals = intervals

        # Data frame containing all eeg data for experiment
        self.df = self.createDF()

        self.data = [self.df, self.resultShort]

    # Creates dataframe for selected experiment and interval
    def createDF(self):

        folder = self.intervalToFolders()
        allFrames = []
        
        
        
        filename = folder+"/"+str(self.personNum)+"/"+self.resultLong+"/P"+str(self.personNum)+self.resultShort+str(self.expNum)+".csv"
        
        # Try-Except block allows dataframes to be created only for files that exist 
        try:
            tempdf = pd.read_csv(filename, header=None)
            allFrames.append(tempdf)
            #print('Adding: ', filename, '....\n')
        
        except:
            return pd.DataFrame()
        
        # Returns empty df if no files exist
        # This is filtered out later by getParticipantsExperiments()
        if len(allFrames) > 0: 
            finaldf = pd.concat(allFrames, axis=1)
        else:
            finaldf = pd.DataFrame()

        return finaldf
        

    # Gives directory names for chosen number of intervals. 
    def intervalToFolders(self):
        
        folders = ["Van250Tot0", "Van500Tot250", "Van750Tot500", "Van1000Tot750", "Van1250Tot1000", "Van1500Tot1250",
                   "Van1750Tot1500", "Van2000Tot1750", "Van2250Tot2000", "Van2500Tot2250", "Van2750Tot2500", "Van3000Tot2750"]
        
        return folders[self.intervals - 1]


def getParticipantsExperiments(particpantNumber, interval):
    # create instances of experiment using partNum until failure
    # Repeat for both R and L
    # Return all experiments in list
    # return value is a list of "Data" objects for given participant
    # Data object is list of form [npArray, Result]

    final = []
    
    # Collecting data
    # a range of 60 was collected arbitrarily because it is known that no participant had more than 60 trials on either side

    desiredColumnNumber = 65

    # Collecting all right data
    for i in range(60):
        temp = Experiment(particpantNumber, i, interval, "r")
        if(temp.df.empty == False and temp.df.shape[1] == desiredColumnNumber):

            # Hacky method of converting df to np array without having to modify former code.
            # Will result in getAllData Returning a list containing numpy arrays
            temp.data[0] = temp.data[0].to_numpy()
                        
            final.append(temp.data)
    
    # Collecting all left data
    for i in range(60):
        temp = Experiment(particpantNumber, i, interval, "l")
        if(temp.df.empty == False and temp.df.shape[1] == desiredColumnNumber):
            
            # Hacky method of converting df to np array without having to modify former code.
            # Will result in getAllData Returning a list containing numpy arrays
            temp.data[0] = temp.data[0].to_numpy()
            
            final.append(temp.data)
    
    data = []
    labels = []

    if(len(final) != 0):
        #modifying data format to match tensorflow format
        #data output is 3d numpy array
        # label output is list of labels
        # r is labeled as 0, l is labeled as 1
        for each in final:
            data.append(each[0])
            if(each[1] == 'R'):
                labels.append(0)
            elif(each[1] == 'L'):
                labels.append(1)

        # making training data 3d array
        data = np.dstack(data)
        # reordering training data to [count, x, y]
        data = np.transpose(data, (2,0,1))
        labels = np.array(labels)

        print("Shape of Data: ", data.shape, "\n")

        return (data, labels)
    
    else:
        return ([], [])


# Collects data from all participants based on chosen interval
# interval is a value that determines how many time interval files are used. For example, an interval of 3 would result in data being collected for the 0-250, 250-500, & 500-750ms ranges. 

def getAllData(interval):
    allData = np.array([])
    allLabels = np.array([])

    for i in range(30):
        data, labels = getParticipantsExperiments(i, interval)
        
        if len(data) > 0:
            print("Data for participant:", i, "has been collected. Shape:", data.shape, " \n\n")
            if len(allData) == 0:
                allData = data
                allLabels = labels
            else:
                #np.dstack((allData, data))
                allData = np.concatenate((allData, data), axis=0)
                allLabels = np.append(labels, allLabels)

    print("Combined data shape: ", allData.shape, '\n')
    
    return allData, allLabels


# for testing purposes
#def main():

#
#    #test = Experiment(2, 2, 12, "r")
#
#    #print(test.createDF())
#
#    #print(Experiment(2, 2, 12, "r").df)
##
#     data = getParticipantsExperiments(2, 12)
##
#    #print(len(data))
##
#     i = 0
#     for item in data:
#         print(item, i)
#    #    i += 1
#
#    start = time.time()
#
 #   print(len(getAllData(12)))
#
 #   print("RUNTIME: ", time.time() - start)
#
 #   return 0
#
#
#main()