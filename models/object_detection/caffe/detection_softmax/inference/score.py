import numpy as np

def face_recognize_risk(risk_difference=None, risk_vector1=None, risk_vector2=None):
    error = np.isclose(risk_vector1, risk_vector2, rtol=None, 
    atol=risk_difference)
    if np.prod(error.astype(np.int8)) == 1:
        return True
    else:
        return False

def process_outputs(output, significant, to_significant):
    result = np.zeros_like(output)
    for d in range(significant,to_significant+1,1):
        result += np.round(output,d)
    return 1 / (to_significant-significant+1) * result / output.flatten().shape[0]

# 1, 1000, 1, 1
def lognorm(weights, probability, alpha):
    return np.dot(weights, 
        alpha * np.exp(probability-np.mean(probability))+\
        np.square(probability-np.mean(probability)) + \
        np.min(probability) * np.log(np.square(probability-np.mean(probability)))
    ) / weights.sum()