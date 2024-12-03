import numpy as np
import scipy as sp
from scipy import linalg as la
import math
import cmath
import itertools


def ClipMatrix(matrix, threshold):
    #Removes small entries from a matrix:
    clipmatrix = matrix
    clipmatrix[abs(clipmatrix) < threshold] = 0
    return clipmatrix

def GetExponential(matrix):
    #First, write U as e^(iH) for some traceless Hermitian H
    #Since U is unitary, U = PDP* for some P unitary and D diagonal. 
    #Assume the diagonal elements are of the form exp(i theta_k). We form H as
    #P*EP, where E has theta_k on the diagonal
    eigenval, P = la.eig(matrix)
    P = np.matrix(P)
    diagval = -1.j * np.log(eigenval)
    diagmat = np.diag(diagval)
    Herm = np.matmul(np.matmul(P.getH(),diagmat),P)

    return Herm

def GroupCommutator(matrix1,matrix2):
    #Takes in two invertible matrices, returns the group commutator UVU*V*

    matrix1 = np.matrix(matrix1, dtype=complex)
    matrix2 = np.matrix(matrix2, dtype=complex)

    bracket = np.matmul(np.matmul(np.matmul(matrix1,matrix2),la.inv(matrix1)),la.inv(matrix2))
    return bracket

def Commutator(matrix1,matrix2):
    #Takes in two matrices, returns the commutator XY-YX

    matrix1 = np.matrix(matrix1, dtype=complex)
    matrix2 = np.matrix(matrix2, dtype=complex)

    commutator = (np.matmul(matrix1,matrix2)-np.matmul(matrix2,matrix1))
    return commutator

def NormBalance(matrix1,matrix2):
    #Takes in two matrices, returns them so their norms are scaled to be equal
    psd1 = np.matrix(matrix1).getH()@matrix1
    psd2 = matrix2.getH()@matrix2
    norm1 = la.norm(psd1,2)
    norm2 = la.norm(psd2,2)
    balnorm = math.sqrt(norm1 * norm2)
    return ((balnorm / norm1)*matrix1, (balnorm / norm2)*matrix2)

def ApproxDecompose(matrix):
    #Takes in a matrix, converts it to the exponential space, and finds balanced group commutators 
    dim = len(matrix)

    Herm = np.matrix(GetExponential(matrix))
    dvals = np.linspace(-1*(dim-1)/2, (dim-1)/2, dim, dtype=complex)
    GMat = np.diag(dvals)
    tempmatrix = (np.subtract.outer(dvals,dvals) + np.diag(np.ones(dim, dtype = complex))).transpose()
    FMat = 1.j * (Herm / tempmatrix)
    FMat = FMat - np.diag(np.diag(FMat))

    #BalancedG, BalancedF = NormBalance(GMat, FMat)

    return (FMat,GMat)

def SK(matrix, gates, dim, depth, estimations):
    #Returns both the code and the matrix
    if(depth == 0):
        return BasicFromList(matrix, estimations)
    
    MApproxCode, MApprox = SK(matrix, gates, dim,depth-1, estimations)
    (F,G) = ApproxDecompose(MApprox)
    (V, W) = (la.expm(1.j*F),la.expm(1.j*G))
    VApproxCode,VApprox = SK(V,gates, dim,depth-1, estimations)
    WApproxCode,WApprox = SK(W,gates, dim,depth-1, estimations)
    
    MNew = np.matmul(GroupCommutator(VApprox,WApprox),MApprox)
    ApproxCode = list(VApproxCode) + list(WApproxCode) + list(InverseWord(VApproxCode)) + list(InverseWord(WApproxCode)) + list(MApproxCode)
    return ApproxCode, MNew

def BasicFromList(gate, estimations):
    estimation_gates = estimations[1]
    estimation_codes = estimations[0]
    gate = np.asarray(gate)
    constant_list = [gate]*len(estimation_gates)
    translation = estimation_gates - constant_list
    norms = np.array([la.norm(M,2) for M in translation])
    index = np.argmin(norms)
    return estimation_codes[index], estimation_gates[index]

def ZGSUd(target, gates, depth, word = [1,2,-1,-2], b=0.25):
    # Performs the ZG algorithm on the Lie Algebra SU(d) with the given inputs. We default to b = 0.25 for group word given as the group commutator
    # Note we do not need to do LDG as SU(d) is compact
    #m = math.ceil((1-b)*depth)
    #if(m == depth or False):
    #    return BasicApproximation(target, gates, depth, word, b)
    #...
    #...
    #...
    #base = aaa
    #similarities = SolveSimilarities(base, target)
    #for u in similarities:
    #    stuf
    raise NotImplemented
  
def BasicApproximation(target, gates, depth, word, b):
    #This can effectively be done through epsilon nets
    raise NotImplemented

def REQB(gates, word, depth, cc_degree=2):
    raise NotImplemented

def REU(gates, word, depth, resolution, cc_degree=2,datapath = './precomp/sn_values.npy'):
    #Finds "small" circuits, ones that approximate the identity to error 2^(-depth). Note cc_degree is a property of your word.
    #Honestly we probably really want to precompute this for all sufficiently interesting n. 

    assert  (resolution > 0 and resolution < 1), "Resolution error"


    KnownValues = np.load(datapath)
    if (not (len(KnownValues) <= depth or KnownValues[depth] == 0)):
        return KnownValues[depth]
    else:
        for m in range(depth):
            if math.abs(depth/cc_degree - m) < resolution:
                KnownValues[m] = REU(gates, word, m, resolution, cc_degree, datapath)



    raise NotImplemented
    
def BrutalREUWordSearch(word, contestant,gates,length,error):
    #Brute force search for u formed from at most length many gates such that word(contestant, contestant^u) < error
    #Such matrices like u are stored as nparrays taking values (0,2,..,|gates|), where the entry determines which gate. 0 is reserved for identity
    code = np.zeros(length)
    while True:
        u = np.matrix(code_to_matrix(u))
        sim_contestant = u.inv()@contestant@u
        candidate = WordExecutor(word, contestant, sim_contestant)
        if (la.norm(candidate - np.identity(8),2)< error):
            return candidate
        #else if (s)


    raise NotImplemented

def code_to_matrix(gates,code,partial = np.eye(4,dtype=complex)):
    #Takes in a list of invertible gates, so that the 0th is the identity. Code is a list of signed indices corresponding to the gates in the product, with
    #the sign representing whether it is the matrix or the inverse. The partial parameter is primarily for internal use, but could be used if you want to 
    #left multiply the code by a given matrix. 
    if len(code) == 0:
        return partial
    else:
        if(code[0] >=0):
            partial = partial @ gates[code[0]]
        else:
            partial = partial @ la.inv(gates[-1*code[0]])
        return code_to_matrix(gates, code[1:],partial)
        
def DirectSum(terms):
    n = len(terms)
    if n == 0:
        return np.matrix([])
    
    sizes = [None]*n
    for i in range(n):
        sizes[i] = np.array(terms[i].shape)

    dsum = np.zeros(sum(sizes),dtype = complex)

    index = [0,0]
    for i in range(n):
        dsum[index[0]:(index[0]+sizes[i][0]),index[1]:(index[1]+sizes[i][1])]=terms[i]
        index += sizes[i]

    return dsum

def ApproximatelyEqual(a,b,epsilon):
    return (la.norm(a-b,2) < epsilon)

def GenerateGateSet(incomplete, dimension):
    num_terms = len(incomplete)
    sizes= [None] *num_terms
    CompletedSet = [np.eye(dimension,dtype=complex)]
    for i in range(num_terms):
        sizes[i] = len(incomplete[i])
    for subset in itertools.product(range(num_terms),repeat = math.ceil(dimension/2)):
        if sum(sizes[index] for index in subset) == dimension:
            DirectSummand = [incomplete[index] for index in subset]
            GeneratedMatrix = DirectSum(DirectSummand)
            if not any(ApproximatelyEqual(GeneratedMatrix,found,1e-10) for found in CompletedSet):
                CompletedSet.append(GeneratedMatrix)
    return CompletedSet

def WordExecutor(word, g, h):
    #Given a group word, recursively finds the evaluation of the word at the two given elements. Functionality would be a lot easier here. 
    iden = np.eye(len(g),dtype=complex)
    gates = [iden,g,h]
    return code_to_matrix(gates,word)
    
def GetElkasapyWords(depth):
    #Returns the nth Elkasapy word, a special group word that has good optimization bounds for above.
    
    word_list = [None]*(depth+1)
    word_list[0]=[]
    word_list[1]=[1]
    word_list[2]=[2]
    
    for i in range(3,depth+1):
        word_list[i] = CollapseWord(InverseWord(word_list[i-1])+word_list[i-2] + word_list[i-1] + InverseWord(word_list[i-2]))

    return word_list

def InverseWord(word):
    return np.flip(np.negative(word)).tolist()
    
def CollapseWord(word):
    #Shortens a word if possible. 
    done = False
    i = 0
    while not done:
        if i >= len(word)-1:
            return word
        if word[i] == -1*word[i+1]:
            del word[i:i+2]
            i=0
        else:
            i+=1
            
def SolveSimilarities(base, target,N):
    #Given a target matrix, we want to find a tuple of matrices (a_1,...,a_N) such that s^(a_1)s^(a_2)...s^(a_N) = g, where superscript here means conjugation.
    #N will be fixed depending on our space

    #Idea, first write s = exp(x). Then find y that maximizes ||[x,y]||, ... idk. I might just have to do this numerically.
    raise NotImplemented

def BuildApproximateList(gates,alpha = 0.2, length = 4):
    initial_approximations_codes = []
    initial_approximations_gates = []
    num_gates = len(gates)-1
    for candidate_length in range(length):
        for candidate_word in itertools.product(range(-1*num_gates,num_gates),repeat = candidate_length):
            word = code_to_matrix(gates,candidate_word)
            valid_candidate = True
            for i in range(len(initial_approximations_codes)):
                if la.norm(word-initial_approximations_gates[i],2) < alpha:
                    valid_candidate = False
                    break
            if(valid_candidate):
                initial_approximations_codes.append(candidate_word)
                initial_approximations_gates.append(word)
    return np.asarray(initial_approximations_gates), initial_approximations_codes



def DFTTest():
    dftmtx = (1/math.sqrt(3))*np.fft.fft(np.eye(3))
    
    [BalancedF, BalancedG] = ApproxDecompose(dftmtx)

    diff =   (1.j * np.matrix(GetExponential(dftmtx))) - np.matrix(Commutator(BalancedF,BalancedG),dtype=complex)
    ClippedDiff = ClipMatrix(diff, 1e-6)
    print(ClippedDiff)
def WordTest():
    #commutator = [1,2,-1,-2]
    #G,H = [[1,1],[0,1]],[[1,0],[1,1]]
    #Guess = WordExecutor(commutator, G,H)
    #print(Guess)
    #Exact = GroupCommutator(G,H)
    #print(Exact)
    #ClippedDiff = ClipMatrix(Guess - Exact, 1e-6)
    #print(ClippedDiff)

    word = [1,2,2,2,-2,-2,1,-1,2,2]
    print(CollapseWord(word))
def CodeTest():
    gate_Id = np.eye(2,dtype=complex)
    gate_H = np.matrix([[1,1],[1,-1]],dtype=complex)/math.sqrt(2)
    gate_S = np.matrix([[1,0],[0,1j]],dtype=complex)
    gate_X = np.matrix([[0,1],[1,0]],dtype=complex)
    gate_Z = np.matrix([[1,0],[0,-1]],dtype=complex)

    gates = [gate_Id,gate_H,gate_S,gate_X,gate_Z]
    code = [1,3,-1,-4]
    print(code_to_matrix(gates,code))
    print(WordExecutor([1,2,-1,-2],gate_H,gate_X))
def SumTest():
    gate_H = np.matrix([[1,1],[1,-1]])/math.sqrt(2)
    gate_S = np.matrix([[1,0],[0,1j]],dtype = complex)
    gate_CNOT = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    gate_T = np.matrix([[1,0],[0,cmath.exp(0.25j*math.pi)]])
    gate_I2 = np.eye(2)
    print(DirectSum([gate_H,gate_S,gate_I2]))
    print(DirectSum([gate_CNOT,gate_T]))
def GateTest():
    gate_1 = np.eye(2,dtype=complex)
    gate_2 = np.matrix([[0,1.j],[1.j,0]])
    gates = GenerateGateSet([gate_1,gate_2],4)
    for gate in gates:
        print(gate)
def ElkasapyTest():
    print(GetElkasapyWords(5))
    
if __name__ == '__main__':
    dftmtx = 0.5*sp.fft(sp.eye(4))
    gate_H = np.matrix([[1,1],[1,-1]])/math.sqrt(2)
    gate_S = np.matrix([[1,0],[0,1j]])
    gate_CNOT = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    gate_T = np.matrix([[1,0],[0,cmath.exp(0.25j*math.pi)]])
    gate_I2 = np.eye(2)
    incomplete = [gate_H,gate_S,gate_CNOT,gate_T,gate_I2]
    gates = GenerateGateSet(incomplete,4)
    
    #medium_list_gates, medium_list_codes = BuildApproximateList(gates)
    #np.save('medium_list_gates.npy',medium_list_gates)
    #np.save('medium_list_codes.npy',medium_list_codes)
    list_gates = np.load('medium_list_gates.npy')
    list_codes = np.load('medium_list_codes.npy',allow_pickle=True)

    estimated_code, estimated_gate = BasicFromList(dftmtx, [list_codes,list_gates])
    distance = la.norm(estimated_gate-dftmtx,2)
    print(estimated_gate)
    print(estimated_code)
    print(distance)
    #print(la.norm(dftmtx,2))
    #test_code,test_matrix = SK(dftmtx, gates, 4, 3, [list_codes,list_gates])
    #print(test_code)
    #print(test_matrix)
    #print(la.norm(dftmtx-test_matrix,2))









#Some Notes:
# I need to find some good group words for these algos. Elkasapy words seem the best bet
# It is unclear how much (pre)computation is "specfic" and/or parallelizable. It is clear we can find the s_n without knowing g, 
# but how much of the "golf game" can we precompute
# The golf game seems rather "dependent", it might not be as easily parallelizable
# It seems Step 1-5 are all pre-computable. 