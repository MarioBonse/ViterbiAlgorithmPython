
# INPUt:
# ObservationSpace = {o_1..o_n} list StateSpace = {s_1...s_k} ->list of object
# Pini = {p_1..p_k} : p_i = p(x_1 = s_i) -> np.array-> shape = (k,1)
# ObservatedSequence = Y = {y_1..y_t} -> array of integers (indexes of the states)
# TransitionMatrix = A(K*K) : A_ij = p(s_j| s_i) -> np.matrix -> shape = (K,K)
# EmissionMatrix = B(K*N) : B_ij = p(o_j: s_i) -> np.matrix -> shape = (K,N)
#
#
# OUTPUT
# X = {x_1..x_n} most likely hidden sequence


def viterbi(ObservationSpace, StateSpace, Pini, ObservatedSequence, TransitionMatrix , EmissionMatrix):
    # just some controls
    N = size(ObservationSpace) 
    K = size(StateSpace) 
    T = ObservatedSequence.size
    if Pini.shape != (K,) or Pini.shape != (K,1):
        print("Error the init probability vector has a wrong shape")
        return False
    if TransitionMatrix.shape != (K,K):
        print("Error the TransitionMatrix has a wrong shape")
        return False
    if EmissionMatrix.shape != (K,N):
        print("Error the TransitionMatrix has a wrong shape")
        return False
    # now the algorithm begin
    T1 = np.zeros((K,T))
    T2 = np.zeros((K,T))
    # The first column of T1 are symply the probability that the firsta state is i multiplied 
    # for the probability that the observation is 0_1 given the state i
    T1[:,1] = EmissionMatrix[:,ObservatedSequence[1]]*Pini[:]
    for i, observation in enumerate(ObservationSpace[1:]):
        for j, state in enumerate(StateSpace):
                k, kProb = maximize(T1, i, j, TransitionMatrix, EmissionMatrix, ObservatedSequence)
                T1[i,j] = kProb
                T2[i,j] = k
    z = np.argmax(T1[:,T])
    x = []
    x.append(StateSpace[zT])
    for i in range(T-1,-1,-1):
        z = T2[z,i]  
        x.insert(0, z)
    return x


def maximize(T, i, j, A, B, Seq):
    kOptimum = 0
    val = 0
    for k in range(T.shape[0])
        step = T[k, i-1]*A[k,j]*B[j,Seq[i]]
        if step > val:
            val = step
            kOptimum = k
    return kOptimum, val
    



    
