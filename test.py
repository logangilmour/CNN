from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from matplotlib.pyplot import imshow, pause, show

import cupy as cp

import cgrad as cg


DT = cp.float32
x_train = cp.array(x_train,dtype=DT)
y_train = cp.array(y_train,dtype=DT)
x_test = cp.array(x_test,dtype=DT)
y_test = cp.array(y_test,dtype=DT)

test = cp.asnumpy(x_train[0,3:27,2:25])

# chain = [Conv(28,28,1,12),Relu(),Conv(28,28,12,12),Relu(),Pool(28,28,12),Conv(14,14,12,12),Relu(),Conv(14,14,12,6),Relu(),Pool(14,14,6),Flat(7,7,6),FC(7*7*6,100),Relu(),FC(100,10)]
#chain = [cg.Conv(28, 28, 1, 12),cg.BatchNorm(12,14,14), cg.Relu(), cg.Pool(28, 28, 12), cg.Conv(14, 14, 12, 12), cg.BatchNorm(12,7,7), cg.Relu(), cg.Pool(14, 14, 12), cg.Flat(7, 7, 12),cg.FC(7 * 7 * 12, 100), cg.BatchNorm(100,1,1),cg.Relu(), cg.FC(100, 10)]
chain = [cg.Conv(28, 28, 1, 12), cg.Relu(), cg.Pool(28, 28, 12), cg.Conv(14, 14, 12, 12), cg.Relu(), cg.Pool(14, 14, 12), cg.Flat(7, 7, 12),cg.FC(7 * 7 * 12, 100), cg.Relu(), cg.FC(100, 10)]

# chain = [Conv(28,28,1,2)]
# chain = [Conv(28,28,1,3),Pool(28,28,3),Flat(14,14,3),FC(14*14*3,128),Relu(),FC(128,10)]
# test = testConv(28,28,1,2)

batch = 50
batches = 60000 // batch

lr = 3e-2

B1 = 0.9
B2 = 0.999
step = 0.005
eps = 1e-8

moments = list()
for c in chain:
    mi = list()
    for gi in c.grad():
        mi.append((cp.zeros(gi.shape), cp.zeros(gi.shape)))
    moments.append(mi)

t = 0

for ep in range(0, 5):
    error = 0
    print(ep)
    for i in range(0, batches):
        t += 1

        trn = cp.reshape(x_train[i * batch:(i + 1) * batch, :, :] / 255 + 0.001, (batch, 1, 28, 28))
        vals = trn
        for c in chain:
            vals = c.go(vals)

        d = (vals.reshape(batch,10).transpose(1,0) - cg.hot(y_train[i * batch:(i + 1) * batch], 10)).transpose(1,0).reshape(batch,10,1,1)
        error += cp.sum((d ** 2).reshape(-1)) / (batch * batches)

        for i in reversed(range(len(chain))):
            c = chain[i]
            d = c.backprop(d)
            grad = c.grad()
            dp = list()

            for j in range(len(grad)):
                g = grad[j] / batch
                m, v = moments[i][j]
                m *= B1
                m += (1 - B1) * g
                mh = m / (1 - B1 ** t)

                v *= B2
                v += (1 - B2) * (g ** 2)
                vh = v / (1 - B2 ** t)

                dp.append(-step * mh / (cp.sqrt(vh + eps)))

            c.update(dp)
    print(error)

good = 0

batch=100
total = 10000

for i in range(total//batch):

    vals = cp.reshape(x_test[i * batch:(i + 1) * batch, :, :] / 255 + 0.001, (batch, 1, 28, 28))
    for cg in chain:
        vals = cg.go(vals)

    good += cp.sum(cp.argmax(vals.reshape(batch,10).transpose(1,0), axis=0) == y_test[i * batch:(i + 1) * batch])

print(good / total)