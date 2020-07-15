import numpy as np

# transform the feature vector
def score_model(x, significant, to_significant):
    mul = 1/(to_significant-significant+1)
    total = 0.0
    for sig in range(significant,to_significant,1):
        total += np.round(x,sig)
    return total * mul

def recognize_risk(risk_difference=None, risk_vector1=None, risk_vector2=None):
    error = np.isclose(risk_vector1, risk_vector2, rtol=risk_difference, 
    atol=risk_difference)
    # image orientation risk vector differences
    # measures closeness of the original bounding image with augmented image
    diff = np.abs(risk_vector1 - risk_vector2) / risk_vector1
    if np.prod(error.astype(np.int8)) == 1:
        return True, 1 - np.mean(np.abs(diff))
    else:
        return False, 1 - np.mean(np.abs(diff))

def process_outputs(output, significant, to_significant):
    result = np.zeros_like(output)
    for d in range(significant,to_significant+1,1):
        result += np.round(output,d)
    return 1 / (to_significant-significant+1) * result
