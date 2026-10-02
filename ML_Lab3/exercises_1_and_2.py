"""
Lab 4: Decision Trees, Exercises 1 and 2
Machine Learning (H9MLAI), National College of Ireland

Brian Ezeanya   (26134667)
Joseph Gimba    (25243888)
Gloria Ezeanya  (26128756)

Computes entropy and information gain for the class distributions
given in the lab brief.
"""

from collections import Counter
from math import log2


# -----------------------------------
# ENTROPY
# -----------------------------------
def entropy(labels):
    """
    Entropy tells us how mixed the classes are.

    Entropy = 0  -> all values belong to one class
    Higher value -> classes are more mixed
    """

    total = len(labels)

    if total == 0:
        return 0

    class_counts = Counter(labels)
    entropy_value = 0

    for count in class_counts.values():

        probability = count / total

        entropy_value += -probability * log2(probability)

    return entropy_value


# -----------------------------------
# INFORMATION GAIN
# -----------------------------------
def information_gain(parent, subsets):
    """
    Information gain tells us how much better
    the data becomes after splitting it.
    """

    entropy_before = entropy(parent)

    entropy_after = 0

    for subset in subsets:

        weight = len(subset) / len(parent)

        entropy_after += weight * entropy(subset)

    gain = entropy_before - entropy_after

    return gain

# ===================================
# EXERCISE 1
# ===================================

dataset = [
    "A", "A", "A", "A", "A",
    "B", "B", "B",
    "C", "C"
]

print("\n--- EXERCISE 1 ---")

print("Dataset:", dataset)
print("Class counts:", Counter(dataset))

dataset_entropy = entropy(dataset)

print(f"Entropy = {dataset_entropy:.4f}")

print(
    "Meaning: The entropy is above 0 because "
    "the dataset contains different classes."
)


# ===================================
# EXERCISE 2A
# ===================================

subset_1 = ["A", "A", "A", "B", "B", "C"]
subset_2 = ["A", "A", "B", "C"]


print("\n--- EXERCISE 2A ---")

print("Subset 1:", subset_1)
print(f"Entropy = {entropy(subset_1):.4f}")

print()

print("Subset 2:", subset_2)
print(f"Entropy = {entropy(subset_2):.4f}")


# ===================================
# EXERCISE 2B
# ===================================

print("\n--- EXERCISE 2B ---")

gain = information_gain(
    dataset,
    [subset_1, subset_2]
)

print(f"Entropy before split = {entropy(dataset):.4f}")

print(f"Information Gain = {gain:.4f}")

print(
    "Meaning: Information gain shows how much "
    "the split reduced the disorder in the dataset."
)


# ===================================
# EXERCISE 2C: TESTS
# ===================================

print("\n--- EXERCISE 2C: TESTS ---")

# A pure dataset has no uncertainty, so entropy is 0.
assert entropy(["A", "A", "A", "A"]) == 0

# A 50/50 split of two classes has the maximum entropy of 1 bit.
assert entropy(["A", "A", "B", "B"]) == 1.0

print("All tests passed.")
