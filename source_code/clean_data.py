import pandas as pd
import numpy as np

def string_to_list(string):
    string = string.replace('\n', '')
    lst = list(string)
    # normalize the data and make it zero centered
    lst = [int(x) / 9 - 0.5 for x in lst]
    return str(lst).replace("'", "")

if __name__ == "__main__":
    data = pd.read_csv("sudoku_orig.csv")
    data = data.sample(n=10000)
    data = data.apply(lambda x: x.apply(string_to_list))
    data.to_csv("sudoku_cleaned.csv", index=False)