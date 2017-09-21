
"""optimal_w3(Gamma, Beta, loss_aversion, B0, gm, delta, first_guess, w2)"""
"""
This file find the right b for a given optimal w3 (that is, the share of stock holdings) and orther asset weights
"""
import nf2009 as nf
    
def b_search():
    target_w = 0.31
    # guess value
    b = 0.05
    w = round(nf.optimal_w3(Gamma = 2.5, B0 = b, W2 = 0), 2)

    while True:
        print (w)
        if abs(w-target_w) >= 0.05:
            if w > target_w:
                print ("b is big increased")
                b += 0.025
                w = round(nf.optimal_w3(Gamma = 2.5, B0 = b, W2 = 0.5), 2)
            elif w < target_w:
                print ("b is big reduced")
                b -= 0.025
                w = round(nf.optimal_w3(Gamma = 2.5, B0 = b, W2 = 0.5), 2)
        elif 0.02 <= abs(w-target_w) <0.05:
            if w > target_w:
                print ("b is middle increased")
                b += 0.01
                w = round(nf.optimal_w3(Gamma = 2.5, B0 = b, W2 = 0.5), 2)
            elif w < target_w:
                print ("b is middle reduced")
                b -= 0.01
                w = round(nf.optimal_w3(Gamma = 2.5, B0 = b, W2 = 0.5), 2)
            
        elif 0.0 <= abs(w-target_w) <0.02:
            if w > target_w:
                print ("b is increased")
                b += 0.001 
                w = round(nf.optimal_w3(Gamma = 2.5, B0 = b, W2 = 0.5), 2)
            elif w < target_w:
                print ("b is reduced")
                b -= 0.001
                w = round(nf.optimal_w3(Gamma = 2.5, B0 = b, W2 = 0.5), 2)
            elif w == target_w:
                print (b)
                break
    return round(b,2)
print (b_search())
