import jiwer 

f = open("temp.txt", "r")
predicted = f.read().splitlines()
# print(l)
f.close()

f = open("gt_temp.txt", "r")
gt = f.read().splitlines()
f.close()

w = jiwer.cer(gt, predicted)*100
print(w)
