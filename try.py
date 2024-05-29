def geometric_progression(a, l, T):
    if T == 1:
        # If there is only one term, both a and l must be the same
        return [a]
    
    # Calculate the common ratio
    r = (l / a) ** (1 / (T - 1))
    
    # Generate the terms in the geometric progression
    gp = [a * (r ** i) for i in range(T)]
    return gp

# Example usage:
a = 2   # Initial value
l = 32  # End value
T = 5   # Total number of terms

sequence = geometric_progression(a, l, T)
print("Geometric Progression Sequence:", sequence)
