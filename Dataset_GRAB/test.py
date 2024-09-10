import numpy as np

# Load the NPY file
grab_data = np.load('grab_test.npy', allow_pickle=True)

# Print basic information
print("Shape of GRAB data:", grab_data.shape)
print("Data type:", grab_data.dtype)

# If it's a structured array, print the field names
if grab_data.dtype.names is not None:
    print("Field names:", grab_data.dtype.names)

# Print the shape of the first item (assuming it's a sequence of poses)
print("Shape of first item:", grab_data[0].shape)

# Print the first few entries of the first item
print("First few entries of first item:")
print(grab_data[0][:5, :5])  # Adjust slicing as needed

# If there are multiple sequences, print information about a few of them
if len(grab_data) > 1:
    for i in range(min(3, len(grab_data))):
        print(f"Sequence {i} shape:", grab_data[i].shape)
