import pickle
import sys

try:
    data_list = pickle.load(open("dataset/EXP/raw/GRAPHSAT.pkl", "rb"))
    print("Loaded", len(data_list), "objects.")
    if data_list:
        print("First object type:", type(data_list[0]))
        keys = list(data_list[0].__dict__.keys())
        print("Attributes:", keys)
except Exception as e:
    print("Error:", e)
