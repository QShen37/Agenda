from MLT.threshold_predictor import ThresholdPredictor

predictor = ThresholdPredictor("MLT/model.pt")

beta, gamma = predictor.predict("Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.\nA. True, True \nB. False, False\nC. True, False\nD. False, True ")

print(beta)
print(gamma)