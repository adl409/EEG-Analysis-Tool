from test_model import makeIndividualModel


def main():
    
    individuals = [10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  2,  20,  21,  22,  4, 5,  6,  7,  8,  9]
    
    for i in range(12):
        for individual in individuals:
            print("Modelling individual: ", individual, "Experiment: ", i+1, '\n')
            try:
                makeIndividualModel(i + 1, individual)
            except:
                print("Failure on individual: ", individual, "Experiment: ", i+1, '\n' )
main()