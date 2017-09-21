"""optimal_w3(Gamma, Beta, loss_aversion, B0, gm, delta, first_guess, w2)"""
"""
This file find the right b for a given optimal w3 (that is, the share of stock holdings) and orther asset weights
"""
import nf2009 as nf
def b_search():
    target_w = 0.1
    # guess value
    b = 0.05
    gamma, la, w2 = 4.0, 2.5, 0.3
    w = nf.optimal_w3(gamma, 0.98, la, b, 1.0, 1.0, 0.025, w2)
    while True:
        if abs(w - target_w) >= 0.05:
            if w < target_w:
                print "b is big increased"
                b += 0.025
                w = nf.optimal_w3(gamma, 0.98, la, b, 1.0, 1.0, 0.025, w2)
            elif w > target_w:
                print "b is big reduced"
                b -= 0.025
                w = nf.optimal_w3(gamma, 0.98, la, b, 1.0, 1.0, 0.025, w2)
        elif 0.02 <= abs(w - target_w) < 0.05:
            if w < target_w:
                print "b is middle increased"
                b += 0.01
                w = nf.optimal_w3(gamma, 0.98, la, b, 1.0, 1.0, 0.025, w2)
            elif w > target_w:
                print "b is middle reduced"
                b -= 0.01
                w = nf.optimal_w3(gamma, 0.98, la, b, 1.0, 1.0, 0.025, w2)
        else:
            if w < target_w:
                print "b is increased"
                b += 0.001
                w = nf.optimal_w3(gamma, 0.98, la, b, 1.0, 1.0, 0.025, w2)
            elif w > target_w:
                print "b is reduced"
                b -= 0.001
                w = nf.optimal_w3(gamma, 0.98, la, b, 1.0, 1.0, 0.025, w2)
            elif w == target_w:
                print b
                break
    return b
print b_search()
