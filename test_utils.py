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
    RVP = (1 / (4 * np.log(2))) * (np.log(SH[1:]) - np.log(SL[1:]))**2
    
    avrTVL = np.zeros(TVL.shape)
    avrRVP = np.zeros(RVP.shape)
    powers = 1. / np.asarray(1 + np.arange(TVL.shape[0]))
    avrTVL[:200, :] = np.cumsum(TVL[:200, :], axis=0)
    avrRVP[:200, :] = np.cumsum(RVP[:200, :], axis=0)
    #avrRVP[:200, :] = np.cumprod(RVP[:200, :], axis=0)
   
    print TVL.shape, RVP.shape
    for i in np.arange(200, TVL.shape[0]):
        avrTVL[i, :] = (avrTVL[i-1, :] - TVL[i-200, :] + TVL[i, :])
        if i < RVP.shape[0]:
            avrRVP[i, :] = (avrRVP[i-1, :] - RVP[i-200, :] + RVP[i, :])
        #avrRVP[i, :] = np.multiply(np.divide(avrRVP[i-1, :], RVP[i-200, :]), RVP[i, :])
    powers[200:] = 1. / 200
    #avrTVL = np.power(avrTVL, powers[:, None])
    avrTVL = np.multiply(avrTVL, powers[:, None])
    avrRVP = np.multiply(avrRVP, powers[:-1, None])
    #avrRVP = np.power(avrRVP, powers[:, None])


    #print 'RCO ', RCO[:10, :10]
    #print 'avrRCO ', avrRCO[:10]
    #print 'ROC ', ROC[:10, :10]
    #print 'avrROC ', avrROC[:10]
    #print 'ROO ', ROO[:10, :10]
    #print 'avrROO ', avrROO[:10]

    print 'RVP ', RVP[:10, :10]
    #print 'avrTVL ', avrTVL[:10, :10]
    print 'avrRVP ', avrRVP[:10, :10]

    return RCO, avrRCO, ROC, avrROC, ROO, avrROO, TVL, avrTVL, RVP, avrRVP

def p2_opt():
    RCC, avrRCC = p1_strat()
    RCO, avrRCO, ROC, avrROC, ROO, avrROO, TVL, avrTVL, RVP, avrRVP = p2_strat()
    print 'Shapes'
    print 'RCC.shape', RCC.shape, avrRCC.shape
    print 'RCO.shape', RCO.shape, avrRCO.shape
    print 'ROC.shape', ROC.shape, avrROC.shape
    print 'ROO.shape', ROO.shape, avrROO.shape
    print 'TVL.shape', TVL.shape, avrTVL.shape
    print 'RVP.shape', RVP.shape, avrRVP.shape

    # 1001, 100
    B = np.zeros((12, RCC.shape[0] - 1, RCC.shape[1]))
    print "B shape ", B.shape
    # RCC(t-1)
    B[0] = (RCC[:-1] - avrRCC[:-1][:, None])/RCC.shape[1]
    # ROO(t)
    B[1] = (ROO[1:] - avrROO[1:][:, None])/ROO.shape[1]
    # ROC(t-1)
    B[2] = (ROC[1:-1] - avrROC[1:-1][:, None])/ROC.shape[1]
    # RCO(t)
    B[3] = (RCO[1:] - avrRCO[1:][:, None])/RCO.shape[1]
    # TVL(t-1)
    B[4] = (TVL[1:-1] / avrTVL[1:-1]) * (RCC[:-1] - avrRCC[:-1][:, None])/RCC.shape[1]
    B[5] = (TVL[1:-1] / avrTVL[1:-1]) * (ROO[1:] - avrROO[1:][:, None])/ROO.shape[1]
    B[6] = (TVL[1:-1] / avrTVL[1:-1]) * (ROC[1:-1] - avrROC[1:-1][:, None])/ROC.shape[1]
    B[7] = (TVL[1:-1] / avrTVL[1:-1]) * (RCO[1:] - avrRCO[1:][:, None])/RCO.shape[1]
    B[8] = (RVP[:-1] / avrRVP[:-1]) * (RCC[:-1] - avrRCC[:-1][:, None])/RCC.shape[1]
    B[9] = (RVP[:-1] / avrRVP[:-1]) * (ROO[1:] - avrROO[1:][:, None])/ROO.shape[1]
    B[10] = (RVP[:-1] / avrRVP[:-1]) * (ROC[1:-1] - avrROC[1:-1][:, None])/ROC.shape[1]
    B[11] = (RVP[:-1] / avrRVP[:-1]) * (RCO[1:] - avrRCO[1:][:, None])/RCO.shape[1]

    batch_size = 128
    # = (RCC - avrRCC[:, None])/RCC.shape[1]
    # = (ROO - avrROO[:, None])/ROO.shape[1]
    # = (ROC[:-1, :] - avrROC[:-1][:, None])/ROC.shape[1]
    # = (RCO[:, :] - avrRCO[:, None])/RCO.shape[1]
    # = (TVL[:-1, :] / avrTVL[:-1, :]) * (RCC - avrRCC[:, None])/RCC.shape[1]
    # = (TVL[:-1, :] / avrTVL[:-1, :]) * (ROO[:-1, :] - avrROO[:-1][:, None])/ROO.shape[1]
    # = (TVL[:-1, :] / avrTVL[:-1, :]) * (ROC[:-1, :] - avrROC[:-1][:, None])/ROC.shape[1]
    # = (TVL[:-1, :] / avrTVL[:-1, :]) * (RCO[:-1, :] - avrRCO[:-1][:, None])/RCO.shape[1]
    # = (RVP[:-1, :] / avrRVP[:-1, :]) * (RCC[:-1, :] - avrRCC[:-1][:, None])/RCC.shape[1]
    # = (RVP[:-1, :] / avrRVP[:-1, :]) * (ROO[:, :] - avrROO[:-1][:, None])/ROO.shape[1]
    #  = (RVP[:-1, :] / avrRVP[:-1, :]) * (ROC[:-1, :] - avrROC[:-1][:, None])/ROC.shape[1]
    #  = (RVP[:-1, :] / avrRVP[:-1, :]) * (RCO[:-1, :] - avrRCO[:-1][:, None])/RCO.shape[1]

    # Build computation graph
    batch_size = 200
    g = tf.Graph() 
    with g.as_default(): 
        #a_coef = tf.Variable(tf.random_normal([12], stddev=0.35, name="weights")) 
        a0 =  tf.Variable(0.5, name="a0")
        a1 =  tf.Variable(0.5, name="a1")
        a2 =  tf.Variable(0.5, name="a2")
        a3 =  tf.Variable(0.5, name="a3")
        a4 =  tf.Variable(0.5, name="a4")
        a5 =  tf.Variable(0.5, name="a5")
        a6 =  tf.Variable(0.5, name="a6")
        a7 =  tf.Variable(0.5, name="a7")
        a8 =  tf.Variable(0.5, name="a8")
        a9 =  tf.Variable(0.5, name="a9")
        a10 = tf.Variable(0.5, name="a10")
        a11 = tf.Variable(0.5, name="a11")
        
        plROC = tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plROC')

        plB0 =  tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plB0')
        plB1 =  tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plB1')
        plB2 =  tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plB2')
        plB3 =  tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plB3')
        plB4 =  tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plB4')
        plB5 =  tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plB5')
        plB6 =  tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plB6')
        plB7 =  tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plB7')
        plB8 =  tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plB8')
        plB9 =  tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plB9')
        plB10 = tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plB10')
        plB11 = tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plB11')
        
        plW2 = tf.placeholder(tf.float32, shape=[None, RCC.shape[1]], name='plW2')
        sharpe = tf.placeholder(tf.float32, shape=[None])

        #plW2 = tf.mul(tf.slice(a_coef, 0, 1), plB0)
        #plW2 = plW2 + tf.mul(tf.slice(a_coef, 1, 1), plB1)
        #plW2 = plW2 + tf.mul(tf.slice(a_coef, 2, 1), plB2)
        #plW2 = plW2 + tf.mul(tf.slice(a_coef, 3, 1), plB3)
        #plW2 = plW2 + tf.mul(tf.slice(a_coef, 4, 1), plB4)
        #plW2 = plW2 + tf.mul(tf.slice(a_coef, 5, 1), plB5)
        #plW2 = plW2 + tf.mul(tf.slice(a_coef, 6, 1), plB6)
        #plW2 = plW2 + tf.mul(tf.slice(a_coef, 7, 1), plB7)
        #plW2 = plW2 + tf.mul(tf.slice(a_coef, 8, 1), plB8)
        #plW2 = plW2 + tf.mul(tf.slice(a_coef, 9, 1), plB9)
        #plW2 = plW2 + tf.mul(tf.slice(a_coef, 10, 1), plB10)
        #plW2 = plW2 + tf.mul(tf.slice(a_coef, 11, 1), plB11)

        plW2 = tf.mul(a0, plB0)
        plW2 = plW2 + tf.mul(a1 , plB1)
        plW2 = plW2 + tf.mul(a2 , plB2)
        plW2 = plW2 + tf.mul(a3 , plB3)
        plW2 = plW2 + tf.mul(a4 , plB4)
        plW2 = plW2 + tf.mul(a5 , plB5)
        plW2 = plW2 + tf.mul(a6 , plB6)
        plW2 = plW2 + tf.mul(a7 , plB7)
        plW2 = plW2 + tf.mul(a8 , plB8)
        plW2 = plW2 + tf.mul(a9 , plB9)
        plW2 = plW2 + tf.mul(a10, plB10)
        plW2 = plW2 + tf.mul(a11, plB11)
        
        W2_ROC = tf.reduce_sum(plW2 * plROC, reduction_indices=1)
        norm_W2 = tf.reduce_sum(tf.abs(plW2), reduction_indices=1)
        powers = (tf.range(1, batch_size+1))
        RP2 = tf.truediv(tf.reduce_sum(plW2 * plROC, reduction_indices=1), tf.reduce_sum(tf.abs(plW2), reduction_indices=1))

        #nsharpe = (-1) * tf.truediv(tf.reduce_mean(RP2), tf.pow(tf.reduce_sum(tf.pow(RP2 - tf.reduce_mean(RP2), 2)) / batch_size, 0.5))
        nsharpe = (-1) * tf.reduce_sum(tf.truediv( tf.truediv(tf.accumulate_n(RP2), powers), tf.pow(tf.accumulate_n(tf.pow(RP2 - tf.truediv(tf.accumulate_n(RP2), powers), 2)), 0.5)))
        
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(nsharpe)
        #train_step = optimizer.minimize(nsharpe)
        #grads = optimizer.compute_gradients(nsharpe, [a_coef])

    num_steps=10001
    #batch_size = 200 #(int)(RCC.shape[0] - 2)/num_steps
    with tf.Session(graph=g) as session:
        tf.initialize_all_variables().run()
        print 'Initialized'
        #print 'train_step', train_step
        print 'nsharpe ', nsharpe
        for step in range(num_steps):
            offset = np.random.randint(200, 500) #(step * batch_size) % (RCC.shape[0] - 2 - batch_size)
            batchROC = ROC[2 + offset:(2 + offset + batch_size)]
            batchB0 = B[0][offset:(offset + batch_size)]
            batchB1 = B[1][offset:(offset + batch_size)]
            batchB2 = B[2][offset:(offset + batch_size)]
            batchB3 = B[3][offset:(offset + batch_size)]
            batchB4 = B[4][offset:(offset + batch_size)]
            batchB5 = B[5][offset:(offset + batch_size)]
            batchB6 = B[6][offset:(offset + batch_size)]
            batchB7 = B[7][offset:(offset + batch_size)]
            batchB8 = B[8][offset:(offset + batch_size)]
            batchB9 = B[9][offset:(offset + batch_size)]
            batchB10 = B[10][offset:(offset + batch_size)]
            batchB11 = B[11][offset:(offset + batch_size)]
            feed_dict = {plROC: batchROC, 
                         plB0: batchB0,
                         plB1: batchB1,
                         plB2: batchB2,
                         plB3: batchB3,
                         plB4: batchB4,
                         plB5: batchB5,
                         plB6: batchB6,
                         plB7: batchB7,
                         plB8: batchB8,
                         plB9: batchB9,
                         plB10: batchB10,
                         plB11: batchB11
                         }
            rvals = session.run([optimizer, nsharpe, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11], feed_dict=feed_dict)
            loss = rvals[1]
            a_vals = rvals[2:]
            #print rvals
            #raw_input('press enter')
            if step % 100 == 0:
                print 'Offset for batch is %d' % offset
                print('Sharpe ratio at step %d: %f'%(step, -loss))
                print('Intermediate vals at step %d:'%(step))
                print 'a_vals: ', a_vals
                #raw_input('press enter')

    return



    





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
    #save_p1()
    #p2_strat()
    p2_opt()
