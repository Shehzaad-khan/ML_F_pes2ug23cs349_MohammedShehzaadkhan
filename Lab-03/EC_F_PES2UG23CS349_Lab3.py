import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).

    Args:
        data (np.ndarray): Dataset where the last column is the target variable

    Returns:
        float: Entropy value calculated using the formula:
        Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i

    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        entropy = get_entropy_of_dataset(data)
        # Should return entropy based on target column ['yes', 'no', 'yes']
    """
    if len(data) == 0:
        return 0.0

    # Get the target column (last column)
    target_column = data[:, -1]

    # Get unique classes and their counts
    unique_classes, counts = np.unique(target_column, return_counts=True)

    # Calculate probabilities
    total_samples = len(target_column)
    probabilities = counts / total_samples

    # Calculate entropy
    entropy = 0.0
    for prob in probabilities:
        if prob > 0:  # Handle the case when probability is 0 to avoid log2(0)
            entropy -= prob * np.log2(prob)

    return entropy

def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.

    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate average information for

    Returns:
        float: Average information calculated using the formula:
        Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v))
        where S_v is subset of data with attribute value v

    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        avg_info = get_avg_info_of_attribute(data, 0) # For attribute at index 0
        # Should return weighted average entropy for attribute splits
    """
    if len(data) == 0:
        return 0.0

    # Get unique values in the attribute column
    unique_values = np.unique(data[:, attribute])

    total_samples = len(data)
    avg_info = 0.0

    # For each unique value in the attribute column:
    for value in unique_values:
        # 1. Create a subset of data with that value
        subset_mask = data[:, attribute] == value
        subset_data = data[subset_mask]

        # 2. Calculate the entropy of that subset
        subset_entropy = get_entropy_of_dataset(subset_data)

        # 3. Weight it by the proportion of samples with that value
        weight = len(subset_data) / total_samples

        # 4. Sum all weighted entropies
        avg_info += weight * subset_entropy

    return avg_info

def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.

    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate information gain for

    Returns:
        float: Information gain calculated using the formula:
        Information_Gain = Entropy(S) - Avg_Info(attribute)
        Rounded to 4 decimal places

    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        gain = get_information_gain(data, 0) # For attribute at index 0
        # Should return the information gain for splitting on attribute 0
    """
    # Information Gain = Dataset Entropy - Average Information of Attribute
    dataset_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    information_gain = dataset_entropy - avg_info

    # Round the result to 4 decimal places
    return round(information_gain, 4)

def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.

    Args:
        data (np.ndarray): Dataset where the last column is the target variable

    Returns:
        tuple: A tuple containing:
        - dict: Dictionary mapping attribute indices to their information gains
        - int: Index of the attribute with the highest information gain

    Example:
        data = np.array([[1, 0, 2, 'yes'],
                        [1, 1, 1, 'no'],
                        [0, 0, 2, 'yes']])
        result = get_selected_attribute(data)
        # Should return something like: ({0: 0.123, 1: 0.456, 2: 0.789}, 2)
        # where 2 is the index of the attribute with highest gain
    """
    if len(data) == 0:
        return {}, -1

    # Calculate information gain for all attributes (except target variable)
    num_attributes = data.shape[1] - 1  # Exclude target column
    gain_dictionary = {}

    # Store gains in a dictionary with attribute index as key
    for attr_idx in range(num_attributes):
        gain = get_information_gain(data, attr_idx)
        gain_dictionary[attr_idx] = gain

    # Find the attribute with maximum gain using max() with key parameter
    if gain_dictionary:
        selected_attribute_index = max(gain_dictionary, key=gain_dictionary.get)
    else:
        selected_attribute_index = -1

    # Return tuple (gain_dictionary, selected_attribute_index)
    return gain_dictionary, selected_attribute_index
