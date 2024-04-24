# %%
import array_api_strict as np
def is_row_echelon_form(matrix):
	"""
    Checks if a matrix is in row echelon form.
    - Each new leading entry (non-zero) must be to the right of the leading entry in the previous row.
    - Rows entirely of zeros must be at the bottom.
    """ 
	if matrix.size == 0: 
		return False # empty matrix technically doesn't qualify under standard definitions
	rows = matrix.shape[0] 
	cols = matrix.shape[1] 
	# track the column index of the last found leading entry (pivot)
	prev_leading_col = -1

	for row in range(rows): 
		leading_col_found = False
		for col in range(cols): 
			# non-zero curr element
			if matrix[row, col] != 0: 
				# If curr non-zero element's column index is not to the right of the last pivot's column
				if col <= prev_leading_col: 
					return False
				prev_leading_col = col 
				leading_col_found = True
				break
		# If no leading column was found but there are still non-zero elements
		if not leading_col_found and np.any(matrix[row, :] != 0): 
			return False
	return True

def find_nonzero_row(matrix, pivot_row, col):
	"""
    Finds the first row with a non-zero entry in a specified column, starting from a given row.
    """ 
	nrows = matrix.shape[0] 
	for row in range(pivot_row, nrows): 
		if matrix[row, col] != 0: 
			return row 
	return None

# Swapping rows so that we can have our non zero row on the top of the matrix 
def swap_rows(matrix, row1, row2):
	"""
    Swaps two rows in the matrix.
    """
	temp = matrix[row1, :]
	matrix[row1, :] = matrix[row2, :]
	matrix[row2, :] = temp

def make_pivot_one(matrix, pivot_row, col): 
	"""
    Scales a row to make the pivot element equal to one.
    """
	pivot_element = matrix[pivot_row, col] 
	matrix[pivot_row, :] /= pivot_element 

def eliminate_below(matrix, pivot_row, col): 
	"""
    Eliminates entries below the pivot by making them zero, adjusting the matrix towards REF.
    """
	nrows = matrix.shape[0] 
	for row in range(pivot_row + 1, nrows): 
		# Factor by which the pivot row must be multiplied to eliminate the column element below the pivot
		factor = matrix[row, col]
		elem = factor * matrix[pivot_row, :] 
		# Subtract this from the current row to zero out the column below the pivot
		matrix[row, :] -= elem 

# Implementing above functions 
def row_echelon_form(matrix): 
	"""
    Converts a matrix to row echelon form using a series of row operations.
    """
	ncols = matrix.shape[1] 
	pivot_row = 0
# this will run for number of column times. If matrix has 3 columns this loop will run for 3 times 
	for col in range(ncols): 
		# Find the first row with a non-zero entry starting within col loop
		nonzero_row = find_nonzero_row(matrix, pivot_row, col)
		# Check if a non-zero row was found 
		if nonzero_row is not None: 
			# Swap the current pivot row with the found non-zero row
			swap_rows(matrix, pivot_row, nonzero_row) 
			# Normalize pivot (to one)
			make_pivot_one(matrix, pivot_row, col) 
			# Kill all entries below pivot
			eliminate_below(matrix, pivot_row, col) 
			pivot_row += 1
	return matrix 


# %%
# List of matrices with varying sizes and values
matrices = [
    np.asarray([[1, 2, -1], [2, 0, 3], [4, -3, 8]], dtype=np.float32),
    np.asarray([[2,-2,4,-2],[2,1,10,7],[-4,4,-8,4],[4,-1,14,6]], dtype=np.float32),
    np.asarray([[0, 3, -6, 6, 4, -5], [3, -7, 8, -5, 8, 9], [3, -9, 12, -9, 6, 15]], dtype=np.float32),
    np.asarray([[2,3],[9,4]], dtype=np.float32)
]

for matrix in matrices:
    print("Matrix Before Converting:")
    print(matrix)
    print()
    result = row_echelon_form(matrix)
    print("After Converting to Row Echelon Form:")
    print(result)
    if is_row_echelon_form(result):
        print("In REF")
    else:
        print("Not in REF--------------->")
    print()  # Adding a newline for better separation between matrices

# %%
# Verbose Version
def find_nonzero_row(matrix, pivot_row, col):
    """
    Finds the first row with a non-zero entry in a specified column, starting from a given row.
    """
    nrows = matrix.shape[0]
    for row in range(pivot_row, nrows):
        if matrix[row, col] != 0:
            return row
    return None

def swap_rows(matrix, row1, row2):
    """
    Swaps two rows in the matrix.
    """
    if row1 != row2:  # Only swap if different rows to avoid unnecessary operations
        print(f"Swapping row {row1} with row {row2}")
        temp = matrix[row1, :]
        matrix[row1, :] = matrix[row2, :]
        matrix[row2, :] = temp
def make_pivot_one(matrix, pivot_row, col):
    """
    Scales a row to make the pivot element equal to one.
    """
    pivot_element = matrix[pivot_row, col]
    if pivot_element != 1:  # Only scale if pivot is not already 1 to avoid unnecessary operations
        print(f"Scaling row {pivot_row} to make pivot 1")
        matrix[pivot_row, :] /= pivot_element
        print(matrix)

def eliminate_below(matrix, pivot_row, col):
    """
    Eliminates entries below the pivot by making them zero, adjusting the matrix towards REF.
    """
    nrows = matrix.shape[0]
    for row in range(pivot_row + 1, nrows):
        factor = matrix[row, col]
        if factor != 0:  # Only eliminate if there is something to eliminate
            print(f"Eliminating column {col} in row {row} using row {pivot_row}")
            matrix[row, :] -= factor * matrix[pivot_row, :]
            print(matrix)

def row_echelon_form_verbose(matrix):
    """
    Converts a matrix to row echelon form using a series of row operations.
    """
    nrows = matrix.shape[0]
    ncols = matrix.shape[1]
    pivot_row = 0
    for col in range(ncols):
        nonzero_row = find_nonzero_row(matrix, pivot_row, col)
        if nonzero_row is not None:
            swap_rows(matrix, pivot_row, nonzero_row)
            make_pivot_one(matrix, pivot_row, col)
            eliminate_below(matrix, pivot_row, col)
            pivot_row += 1
        if pivot_row >= nrows:
            break
    print("\nMatrix After Converting to Row Echelon Form:")
    return matrix

# Example of using the verbose REF function
A = np.asarray([[1, 2, -1], [2, 0, 3], [4, -3, 8]], dtype=np.float32)
print("Initial Matrix:")
print(A)
print()
result = row_echelon_form_verbose(A)
print(result)


