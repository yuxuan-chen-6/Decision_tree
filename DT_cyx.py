import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

data_clean = np.loadtxt("CW1-60012/wifi_db/clean_dataset.txt")
x_clean = data_clean[:, :-1]  #the characteristic
y_clean = data_clean[:, -1]   #the label
data_noisy = np.loadtxt("CW1-60012/wifi_db/noisy_dataset.txt")
x_noisy = data_noisy[:, :-1]
y_noisy = data_noisy[:, -1]


class TreeNode:
    def __init__(self, split_attr=None, split_value=None, left=None, right=None, label=None):
        self.split_attr = split_attr    #best attribute to split
        self.split_value = split_value  #best point
        self.left = left    #left tree
        self.right = right  #right tree
        self.label = label
    
    def is_leaf(self):
        return self.label is not None


def calculate_entropy(labels):
    #calculate entropy
    _, label_counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    probability = label_counts / total_samples
    return -np.sum(probability * np.log2(probability))


def calculate_information_gain(dataset, left_subset, right_subset):
    #data is the data set, attribute is the attribute column index to be divided, split_point is the dividing point
    total_entropy = calculate_entropy(dataset)

    #calculate the weight of the left and right subsets
    left_weight = len(left_subset) / len(dataset)
    right_weight = len(right_subset) / len(dataset)

    left_entropy = calculate_entropy(left_subset)
    right_entropy = calculate_entropy(right_subset)

    information_gain = total_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    return information_gain


def find_split(dataset):
    num_attributes = dataset.shape[1] - 1 
    best_split_value = None
    best_split_feature = None
    max_information_gain = 0

    for feature in range(num_attributes):
        unique_values = dataset[:, feature]

        for i in range(len(unique_values)): 
            split_value = unique_values[i]
            
            left_mask = dataset[:, feature] <= split_value
            right_mask = ~left_mask

            #Calculate information gain based on attributes and split points
            information_gain = calculate_information_gain(dataset[:, -1], dataset[left_mask, -1], dataset[right_mask, -1])

            #If the information gain is higher, update the best attributes and split points
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                best_split_value = split_value
                best_split_feature = feature
                
    return best_split_feature, best_split_value


def decision_tree(data, depth=0, max_depth=4):
    #Check if the maximum depth is reached or if there are no attributes left
    if depth >= max_depth or len(data) == 0:
        #Create a leaf node with the most common label
        most_common_label = Counter(data[:, -1]).most_common(1)[0][0]
        return TreeNode(label=most_common_label), depth
    
    last_col = data[:, -1]
    
    if len(np.unique(last_col)) == 1:
        return TreeNode(label=data[0, -1]), depth
    else:
        split_feature, split_value = find_split(data)
        left_mask = data[:, split_feature] <= split_value
        right_mask = ~left_mask
        
        #Recursively build the left and right branches, incrementing the depth
        left_branch, left_depth = decision_tree(data[left_mask], depth + 1, max_depth)
        right_branch, right_depth = decision_tree(data[right_mask], depth + 1, max_depth)
        
        return TreeNode(split_feature, split_value, left_branch, right_branch), max(left_depth, right_depth)


def predict(tree, sample):
        if tree.is_leaf():
            return tree.label
        
        if sample[tree.split_attr] <= tree.split_value:
            return predict(tree.left, sample)
        else:
            return predict(tree.right, sample)

def confusion_matrix(p_labels, t_labels):
    matrix = np.zeros((4, 4), dtype=int)  # Initialize the 4*4 matrix
    
    for true, predict in zip(t_labels, p_labels):
        true_index = int(true) - 1 # Assuming labels are 1-indexed. Subtract one to convert to 0-index
        pred_index = int(predict) - 1
        
        matrix[true_index, pred_index] += 1 # increase the value in the cell
    
    return matrix

def cross_validation(data,k=10):
    predicted_labels_set = []
    true_labels_set = []
    
    accuracy = 0
    for i in range(k):
        #Select the i-th subset as the validation set, and combine the rest as the training set
        #Train the decision tree on the training data
        root,_ = decision_tree(np.vstack((data[:i*200], data[i*200+200:])))

        correct_predictions = 0
        for sample in data[i*200:i*200+200]:
            predicted_label = predict(root, sample[:-1])
            true_label = sample[-1]
            
            predicted_labels_set.append(predicted_label)
            true_labels_set.append(true_label)
            
            if predict(root, sample) == sample[-1]:
                correct_predictions += 1

        accuracy += correct_predictions / data[i*200:i*200+200].shape[0]
        
    conf_matrix = confusion_matrix(predicted_labels_set, true_labels_set)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    accuracy /= k
    print(f"Accuracy: {accuracy * 100:.2f}%")


def shuffle_data(data):
    shuffled_indices = np.arange(data.shape[0])
    np.random.shuffle(shuffled_indices)
    
    return data[shuffled_indices]

user_input = input("please enter the dataset file path: ")
input_data = np.loadtxt(user_input)

np.random.shuffle(input_data)
cross_validation(input_data)

# GPT generated, not allowed to use graphviz in report
# from graphviz import Digraph

# def visualize_tree(tree):
#     dot = Digraph(comment="Decision Tree", format='png')
#     visualize_tree_recursive(tree, dot)
#     dot.view()

# def visualize_tree_recursive(node, graph, parent_name=None):
#     if node is None:
#         return

#     # Create a unique name for the node
#     current_name = str(id(node))

#     # Display the tree node details
#     if node.is_leaf():
#         graph.node(current_name, label=str(node.label))
#     else:
#         graph.node(current_name, label="X[{}] <= {}".format(node.split_attr, node.split_value))

#     # Connect the current node to its parent
#     if parent_name:
#         graph.edge(parent_name, current_name)

#     # Visualize left and right children
#     if node.left:
#         visualize_tree_recursive(node.left, graph, current_name)
#     if node.right:
#         visualize_tree_recursive(node.right, graph, current_name)

# # Create the tree and visualize
# visualize_tree(root)
