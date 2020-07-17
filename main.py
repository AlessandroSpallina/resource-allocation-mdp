import mdptoolbox.example

import numpy as np


class SliceCalculator:
    def __init__(self, arrival_rate, departure_rate, buffer_size):
        self._arrival_rate = arrival_rate
        self._departure_rate = departure_rate
        self._server_num = 1
        self._buffer_size = buffer_size

        self._rho = self._get_rho()
        self._p0 = self._get_p0()


    def _get_rho(self):
        self._rho = self._arrival_rate / (self._departure_rate * self._server_num)
        return self._rho

    def _get_p0(self):
        # ----
        pre_sum = 1
        power_term = 1
        factor_term = 1
        for i in range(1, self._server_num + 1):
            power_term *= self._arrival_rate / self._departure_rate
            factor_term /= float(i)
            pre_sum += power_term * factor_term
        self._p0 = 1 / pre_sum

        return self._p0
        # ----

    def _get_pk(self, k):
        if k == 0:
            pass


if __name__ == '__main__':

    # P, R = mdptoolbox.example.forest()
    # vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    # vi.run()
    # print(vi.policy)  # result is (0, 0, 0)

    P = np.array([
            [
                [0.41, 0.59, 0, 0, 0, 0],  # matrice transizione ad azione a0
                [0, 0.41, 0.59, 0, 0, 0],
                [0, 0.41, 0.59, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0.41, 0.59, 0, 0, 0, 0],
                [0.41, 0.59, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0.41, 0.59, 0],  # matrice transizione ad azione a1
                [0, 0, 0, 0, 0.41, 0.59],
                [0, 0, 0, 0, 0.41, 0.59],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0.41, 0.59, 0],
                [0, 0, 0, 0.41, 0.59, 0]
            ]
    ])

    R = np.array([
        [0, 0],
        [-1, -2],
        [-2, -3],
        [-1, -4],
        [-2, -5],
        [-3, -6]
    ])

    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.8)
    vi.setVerbose()
    vi.run()
    print(f"policy {vi.policy}, iter {vi.iter}")



    # -----------------------------------------

    """
    assumo M/M/s/b con: s = 1; b = 2; lambda = 2; miu = 3

    i-esimo stato del tipo (k job, n server)
    S0: (0,0)       S3: (0,1)
    S1: (1,0)       S4: (1,1)
    S2: (2,0)       S5: (2,1)
    
    azioni:
    A0: idle
    A1: +1 server
    A2: -1 server
    """
    p_0 = "qualcosa"
    p_1 = "qualcosaltro"
    p_2 = "qualcosaltroancora"
    p_alloc = 1.0
    p_dealloc = 1.0

    P = [
            [
                [p_0,   p_1,    p_2,    0,  0,  0],
                [p_0,   ],
                [],
                [],
                [],
                []
            ],
            [],
            []
    ]


