def percentage_difference(num1,num2):
    """
    Calculate the percentage difference between two numbers.
    Parameters:
    num1 (float): The first number (used as the reference value).
    num2 (float): The second number (to compare against `num1`).
    Returns:
    float: The percentage difference between `num1` and `num2`.
    """
    perc = abs((num1-num2)/num1)*100
    return perc

def percentage_contribution(array):
    """
    Calculate the percentage contribution of each element in an array.
    Parameters:
    array (list or iterable): A list of numerical values (e.g., integers or floats).
    Returns:
    list: A list of percentages where each element represents the percentage contribution 
          of the corresponding element in the input array.
    """
    total_sum = sum(array)
    percentages = [element / total_sum for element in array]
    return percentages

def single_gate_counts(counts_dict):
    """
    Retrieve the counts for the '0' and '1' states from a dictionary.
    Parameters:
    counts_dict (dict): A dictionary where the keys are the states ('0', '1') 
                        and the values are the corresponding counts.
    Returns:
    list: A list of two elements: the count of state '0' and the count of state '1'.
          If either state is not present in `counts_dict`, it is treated as 0.
    """
    state0 = 0
    state1 = 0
    if '0' in counts_dict:
        state0 = counts_dict['0']
    if '1' in counts_dict:
        state1 = counts_dict['1']
    return [state0, state1]

def multi_gate_count(counts_dict):
    state00 = 0
    state01 = 0
    state10 = 0
    state11 = 0
    if '00' in counts_dict:
        state00 = counts_dict['00']
    if '01' in counts_dict:
        state01 = counts_dict['01']
    if '10' in counts_dict:
        state10 = counts_dict['10']
    if '11' in counts_dict:
        state11 = counts_dict['11']
    return [state00, state01, state10, state11]

def fidelity(ideal_probs, realistic_probs):
    """
    Calculate the fidelity between two probability distributions.
    Parameters:
    ideal_probs (list or array-like): A list or array of probabilities representing 
                                       the ideal distribution.
    realistic_probs (list or array-like): A list or array of probabilities representing 
                                           the realistic distribution.
    Returns:
    float: The fidelity value, which ranges from 0 (completely different) to 1 
           (identical distributions).
    """
    psi_ideal = np.sqrt(ideal_probs)
    psi_realistic = np.sqrt(realistic_probs)
    fidelity_value = abs(np.dot(psi_ideal, psi_realistic))**2
    return fidelity_value
