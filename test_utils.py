import util
import numpy as np

import tensorflow as tf




def test_p1():
    fname = 'data_part1.team_000.csv'
    res = util.load(fname)
    print list(res.keys())
    return

def p1_strat():
    data = util.load_insample()

    #return = SC(t)/SC(t-1) -1
    SC = util.extract_SC(data)
    RCC = SC[1:] / SC[:-1] - 1
    print 'RCC shape ', RCC.shape
    #print 'RCC ', RCC[:10, :10]
    avgRCC = np.cumsum(RCC, axis=0) / (1 + np.arange(RCC.shape[0])[:, None])
    #return avgRCC
    #print 'avgRCC shape ', avgRCC.shape
   
    # try geometric mean
    #cumprodRCC = np.cumprod(RCC, axis=0)
    #powers = 1. / (np.asarray(1 + np.arange(RCC.shape[0]), dtype=float))
    #geoavgRCC = np.power(cumprodRCC, powers[:, None])
    #print 'geoavgRCC shape ', geoavgRCC.shape

    avrRCC = np.mean(RCC, axis=1, dtype=float)
    #print 'avrRCC', avrRCC[:10]
    cumavrRCC = np.cumprod(1 + avrRCC)
    powers = 1./(np.asarray(1 + np.arange(RCC.shape[0])))
    #print 'powers ', powers[:10]
    avrRCC = np.power(cumavrRCC, powers) - 1.
    #print 'avrRCC shape ', avrRCC.shape
    #print 'avrRCC', avrRCC[:10]


    

    W1 = np.zeros((SC.shape[0], SC.shape[1])) 
    # 0...T-2
    W2 = (-1/RCC.shape[1]) * (RCC[:-1] - avrRCC[:-1][:, None])
    W1[2:] = W2 # 2...T
    #print 'W2 raw: ', W2.shape, 'W1 shape: ', W1.shape
    #print 'First 10 stocks W1', W1[:10, :10]
    #print 'sum', np.sum(W2[0])

    RP = np.divide(np.sum(W2 * RCC[1:], axis=1), np.sum(np.abs(W2), axis=1))
    #print 'RP shape ', RP.shape
    #print 'RP ', RP[:10]

    return RCC, avrRCC


def p2_strat():
    data = util.load_insample()
    SO = util.extract_SO(data)
    SC = util.extract_SC(data)
    SH = util.extract_SH(data)
    SL = util.extract_SL(data)
    TVL = util.extract_TVL(data)

    RCC, avrRCC = p1_strat()
    # 1...T
    RCO = SO[1:] / SC[:-1] - 1
    cumRCO = np.cumprod(1 + np.mean(RCO, axis=1))
    powers = 1. / np.asarray(1 + np.arange(RCO.shape[0]))
    avrRCO = np.power(cumRCO, powers) - 1
    # 0...T
    ROC = SC / SO - 1
    cumROC = np.cumprod(1 + np.mean(ROC, axis=1))
    powers = 1. / np.asarray(1 + np.arange(ROC.shape[0]))
    avrROC = np.power(cumROC, powers) - 1
    # 1...T
    ROO = SO[1:] / SO[:-1] - 1
    cumROO = np.cumprod(1 + np.mean(ROO, axis=1))
    powers = 1. / np.asarray(1 + np.arange(ROO.shape[0]))
    avrROO = np.power(cumROO, powers) - 1
    # 0...T
    RVP = (1 / (4 * np.log(2))) * (np.log(SH) - np.log(SL))**2
    
    avrTVL = np.zeros(TVL.shape)
    avrRVP = np.zeros(RVP.shape)
    powers = 1. / np.asarray(1 + np.arange(TVL.shape[0]))
    avrTVL[:200, :] = np.cumsum(TVL[:200, :], axis=0)
    avrRVP[:200, :] = np.cumprod(RVP[:200, :], axis=0)
   
    print TVL.shape, RVP.shape
    for i in np.arange(200, TVL.shape[0]):
        avrTVL[i, :] = (avrTVL[i-1, :] - TVL[i-200, :] + TVL[i, :])
        avrRVP[i, :] = np.multiply(np.divide(avrRVP[i-1, :], RVP[i-200, :]), RVP[i, :])
    powers[200:] = 1. / 200
    #avrTVL = np.power(avrTVL, powers[:, None])
    avrTVL = np.multiply(avrTVL, powers[:, None])
    avrRVP = np.power(avrRVP, powers[:, None])


    #print 'RCO ', RCO[:10, :10]
    #print 'avrRCO ', avrRCO[:10]
    #print 'ROC ', ROC[:10, :10]
    #print 'avrROC ', avrROC[:10]
    #print 'ROO ', ROO[:10, :10]
    #print 'avrROO ', avrROO[:10]

    #print 'RVP ', RVP[:10, :10]
    #print 'avrTVL ', RVP[:10, :10]
    #print 'avrRVP ', RVP[:10, :10]

    return RCO, avrRCO, ROC, avrROC, ROO, avrROO, TVL, avrTVL, RVP, avrRVP

def p2_opt():
    RCC, avrRCC = p1_strat()
    RCO, avrRCO, ROC, avrROC, ROO, avrROO, TVL, avrTVL, RVP, avrRVP = p2_strat()
    print 'Shapes'
    print RCC.shape, avrRCC.shape
    print RCO.shape, avrRCO.shape
    print ROC.shape, avrROC.shape
    print ROO.shape, avrROO.shape
    print TVL.shape, avrTVL.shape
    print RVP.shape, avrRVP.shape

    B = np.zeros((12, RCC.shape[0], RCC.shape[1]))
    B[0, 1:] = (RCC[:-2, :] - avrRCC[:-1][:, None])/RCC.shape[1]
    B[1] = (ROO[:, :] - avrROO[:, None])/ROO.shape[1]
    B[2, 1:] = (ROC[:-1, :] - avrROC[:-1][:, None])/ROO.shape[1]
    B[3] = (RCO[:, :] - avrRCO[:, None])/RCO.shape[1]
    B[4, 1:] = (TVL[:-1, :] / avrTVL[:-1, :]) * (RCC[:-1, :] - avrRCC[:-1][:, None])/RCC.shape[1]
     
    #a_coef = np.random.randn(12)
    #W2 = a_coef * B
    #a_coef = tf.Variable(tf.float32, shape=[12])
    #W2 = tf.placeholder(tf.float32, shape=[RCC.shape[0] - 1, RCC.shape[1]])

    



    





def save_p1():
    RCC, avrRCC = p1_strat()
    RCO, avrRCO, ROC, avrROC, ROO, avrROO, TVL, avrTVL, RVP, avrRVP = p2_strat()
    np.savetxt('RCC.csv', RCC, fmt='%10.5f', delimiter=',')
    np.savetxt('avrRCC.csv', avrRCC, fmt='%10.5f', delimiter=',')
    np.savetxt('RCO.csv', RCO, fmt='%10.5f', delimiter=',')
    np.savetxt('avrRCO.csv', avrRCO, fmt='%10.5f', delimiter=',')
    np.savetxt('ROC.csv', ROC, fmt='%10.5f', delimiter=',')
    np.savetxt('avrROC.csv', avrROC, fmt='%10.5f', delimiter=',')
    np.savetxt('ROO.csv', ROO, fmt='%10.5f', delimiter=',')
    np.savetxt('avrROO.csv', avrROO, fmt='%10.5f', delimiter=',')
    np.savetxt('TVL.csv', TVL, fmt='%10.5f', delimiter=',')
    np.savetxt('avrTVL.csv', avrTVL, fmt='%10.5f', delimiter=',')
    np.savetxt('RVP.csv', RVP, fmt='%10.5f', delimiter=',')
    np.savetxt('avrRVP.csv', avrRVP, fmt='%10.5f', delimiter=',')
    print 'Done saving P1 files'
    return
 


if __name__ == '__main__':
    #test_p1()
    save_p1()
    #p2_strat()
    #p2_opt()
