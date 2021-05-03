import pickle

with open("encodings3.pickle", "rb") as f1:
    dct1 = pickle.load(f1)

with open("encodings7.pickle", "rb") as f2:
    dct2 = pickle.load(f2)
    
encods = dct1["encodings"]
names = dct1["names"]

for en in dct2["encodings"]:
    encods.append(en)
for nm in dct2["names"]:
    names.append(nm)
    
alldct = {"encodings": encods, "names": names}

with open("all_encodings.pickle", "wb") as nf:
    nf.write(pickle.dumps(alldct))

