import cupy as cp
DT = cp.float32

class CG:
    def forward(self, X):
        pass

    def backprop(self, D):
        pass

    def update(self, step):
        pass

    def grad(self):
        return []

    def finalize(self,X,cur,total):
        return self.forward(X)

    def infer(self,X):
        return self.forward(X)

class Conv(CG):
    def __init__(self, w, h, d, n):
        # self.K = cp.array([1, 3, 1, 0, 0, 0, -1, -3, -1, -1, 0, 1, -3, 0, 3, -1, 0, 1]).reshape((2,1,3,3))
        self.K = cp.random.normal(0, 0.1, 3 * 3 * d * n).astype(DT).reshape((n, d, 3, 3))
        self.b = cp.zeros(n, dtype=DT)
        self.w = w
        self.h = h
        self.n = n
        self.d = d
        self.dK = cp.zeros((self.n, self.d, 3, 3), dtype=DT)
        self.db = cp.zeros(self.n, dtype=DT)

    def grad(self):
        return self.dK, self.db

    def update(self, delta):
        (dK, db) = delta
        self.K += dK
        self.b += db

    def forward(self, X):
        self.X = X

        batch = X.shape[0]

        B = cp.zeros((batch, self.n, self.w, self.h), dtype=DT)
        for x in range(-1, 2):
            for y in range(-1, 2):
                kslice = self.K[:, :, x + 1, y + 1].reshape(1, self.n, self.d, 1, 1)

                Xoff = X[:, :, max(0, x):min(0, x) + self.w, max(0, y):min(0, y) + self.h]
                Xoff = Xoff.reshape(batch, 1, self.d, self.w - abs(x), self.h - abs(y))

                weighted = cp.sum(kslice * Xoff, axis=2)

                B[:, :, max(0, -x):min(0, -x) + self.w, max(0, -y):min(0, -y) + self.h] += weighted

        return B

    def backprop(self, D):
        batch = D.shape[0]
        dK = cp.zeros((self.n, self.d, 3, 3), dtype=DT)
        db = cp.sum(D, axis=(0, 2, 3), dtype=DT)

        for x in range(-1, 2):
            for y in range(-1, 2):
                xc = self.X[:, :, max(0, -x):min(0, -x) + self.w, max(0, -y):min(0, -y) + self.h].reshape(batch, 1,
                                                                                                          self.d,
                                                                                                          self.w - abs(
                                                                                                              x),
                                                                                                          self.h - abs(
                                                                                                              y))

                d = D[:, :, max(0, x):min(0, x) + self.w, max(0, y):min(0, y) + self.h].reshape(batch, self.n, 1,
                                                                                                self.w - abs(x),
                                                                                                self.h - abs(y))
                dK[:, :, 1 - x, 1 - y] = cp.sum(xc * d, axis=(0, 3, 4))

        dX = cp.zeros((batch, self.d, self.w, self.h), dtype=DT)

        pk = cp.transpose(self.K, (1, 0, 2, 3))

        for x in range(-1, 2):
            for y in range(-1, 2):
                kslice = pk[:, :, x + 1, y + 1].reshape(1, self.d, self.n, 1, 1)
                Doff = D[:, :, max(0, x):min(0, x) + self.w, max(0, y):min(0, y) + self.h].reshape(batch, 1, self.n,
                                                                                                   self.w - abs(x),
                                                                                                   self.h - abs(y))
                weighted = cp.sum(kslice * Doff, axis=2)

                dX[:, :, max(0, -x):min(0, -x) + self.w, max(0, -y):min(0, -y) + self.h] += weighted

        self.dK = dK
        self.db = db

        return dX


class Pool(CG):
    def __init__(self, w, h, d):
        self.w = w
        self.h = h
        self.d = d

    def forward(self, X):
        self.X = X

        h = self.h
        w = self.w
        b = X.shape[0]
        d = self.d

        rows = X.reshape(b * d, h // 2, 2, w).transpose(0, 2, 1, 3).reshape(b * d, 2, w // 2 * h // 2, 2).transpose(0,
                                                                                                                    2,
                                                                                                                    1,
                                                                                                                    3).reshape(
            b * d * w // 2 * h // 2, 4)

        indices = rows.argmax(axis=1)
        self.indices = indices

        return rows[cp.arange(b * d * (h // 2) * (w // 2)), indices].reshape(b, d, w // 2, h // 2)

    def backprop(self, D):
        w = self.w // 2
        h = self.h // 2
        b = D.shape[0]
        d = self.d

        D2 = cp.zeros((d * b * h * w, 4), dtype=DT)
        D2[cp.arange(b * d * h * w), self.indices] = D.reshape(b * d * h * w)
        return D2.reshape(b * d, h * w, 2, 2).transpose(0, 2, 1, 3).reshape(b * d, 2, h, w * 2).transpose(0, 2, 1,
                                                                                                          3).reshape(b,
                                                                                                                     d,
                                                                                                                     h * 2,
                                                                                                                     w * 2)


class FC(CG):

    def __init__(self, w, s):
        self.s = s
        self.w = w
        self.W = cp.random.normal(0, 0.1, w * s).astype(DT).reshape(s, w)
        self.b = cp.full((s, 1), 0.5, dtype=DT)
        self.dW = cp.zeros((self.s, self.w), dtype=DT)
        self.db = cp.zeros((self.s, 1), dtype=DT)

    def grad(self):
        return (self.dW, self.db)

    def update(self, delta):
        (dW, db) = delta

        self.W += dW
        self.b += db

    def forward(self, X):
        self.X = X.reshape(-1,self.w).transpose(1,0)
        return (cp.dot(self.W, self.X) + self.b).transpose(1,0).reshape(-1,self.s,1,1)

    def backprop(self, D):
        D = D.reshape(-1,self.s).transpose(1,0)
        dW = cp.dot(D, self.X.T)
        dB = cp.sum(D, axis=1, keepdims=True)
        D = cp.dot(self.W.T, D).transpose(1,0).reshape(-1,self.w,1,1)

        self.dW = dW
        self.db = dB

        return D

class BatchNorm(CG):
    def __init__(self,d,w,h):
        self.d = d
        self.h = h
        self.w = w
        self.g = cp.ones((1,d,1,1),dtype=DT)
        self.B = cp.zeros((1,d,1,1),dtype=DT)
        self.gF = cp.zeros((1,d,1,1),dtype=DT)
        self.BF = cp.zeros((1,d,1,1),dtype=DT)
        self.dg = cp.zeros((1,d,1,1),dtype=DT)
        self.dB = cp.zeros((1,d,1,1),dtype=DT)

        self.eps = 1e-8

    def grad(self):
        return self.dg, self.dB

    def update(self, delta):
        (dg,dB) = delta
        self.g+=dg
        self.B+=dB

    def forward(self, X):
        batch = X.shape[0]

        self.m = (batch*self.w*self.h)
        u = cp.sum(X,axis=(0,2,3),keepdims=True)/self.m
        self.centered = X-u
        self.v_eps = cp.sum(self.centered**2,axis=(0,2,3),keepdims=True)/self.m+self.eps
        self.v_eps_inv_sqrt = self.v_eps**-0.5
        self.xh = self.centered*self.v_eps_inv_sqrt
        return self.xh*self.g+self.B

    def finalize(self,X,cur,batches):
        if cur==0:
            self.uP = cp.zeros((1, self.d, 1, 1), dtype=DT)
            self.vP = cp.zeros((1, self.d, 1, 1), dtype=DT)
        batch = X.shape[0]
        m = (batch*self.w*self.h)

        u = cp.sum(X, axis=(0, 2, 3), keepdims=True) / m
        v = cp.sum((X-u)**2,axis=(0,2,3),keepdims=True)/m
        self.uP += u * (1/batches)
        self.vP += v * (1/batches) * (m/(m-1))
        ret = self.forward(X)

        g_v_eps_inv_sqrt = self.g*(self.vP+self.eps)**-0.5

        if cur==batches-1:
            self.BF = self.B - self.uP*g_v_eps_inv_sqrt
            self.gF = g_v_eps_inv_sqrt

        return ret

    def infer(self,X):
        return X*self.gF+self.BF

    def backprop(self, D):

        dxh = D*self.g

        dv = cp.sum(dxh*self.centered*-0.5*self.v_eps_inv_sqrt**3,axis=(0,2,3),keepdims=True)
        du = cp.sum(dxh*-self.v_eps_inv_sqrt,axis=(0,2,3),keepdims=True)+dv*cp.sum(-2*self.centered,axis=(0,2,3),keepdims=True)/self.m
        dx = dxh*self.v_eps_inv_sqrt + dv*2*self.centered/self.m + du/self.m
        self.dg = cp.sum(D*self.xh,axis=(0,2,3),keepdims=True)
        self.dB = cp.sum(D,axis=(0,2,3),keepdims=True)

        return dx

class Relu(CG):
    def forward(self, X):
        self.route = X <= 0
        return cp.maximum(X, 0)

    def backprop(self, D):
        D[self.route] =0#*= 0.01
        return D


def hot(n, k):
    rng = cp.arange(0, k).reshape(10, 1)
    return (n == rng).astype(DT)


class Flat(CG):
    def __init__(self, w, h, d):
        self.w = w
        self.h = h
        self.d = d

    def forward(self, X):
        return X.reshape(-1, self.d*self.w * self.h,1,1)

    def backprop(self, D):
        ret = D.reshape(-1, self.d,self.w, self.h)
        return ret
