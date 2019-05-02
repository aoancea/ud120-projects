#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    # print predictions
    # print net_worths

    # squares = [(ages[ii][0], net_worths[ii][0], ((predictions[ii] - net_worths[ii]) ** 2)[0]) for ii in range(0, len(predictions))]
    # squares.sort(key=lambda tup: tup[2])
    # squares = squares[0: int(len(squares) * 0.9)]
    
    # cleaned_data = squares

    cleaned_data = [(ages[ii][0], net_worths[ii][0], ((predictions[ii] - net_worths[ii]) ** 2)[0]) for ii in range(0, len(predictions))]
    cleaned_data.sort(key=lambda tup: tup[2])
    cleaned_data = cleaned_data[0: int(len(cleaned_data) * 0.9)]

    return cleaned_data