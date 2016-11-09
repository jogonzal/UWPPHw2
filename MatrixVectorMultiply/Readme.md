The approach taken hereis very similar to the one taken in the example "vector scalar multiply".

Instead of splitting the vector by rows, we now split the matrix by rows. Every GPU work item gets the full matrix and the full vector, and only multiplies the rows that are assigned to it.

Additionally, the matrix is now represented by a unidimensional array instead of a bidimensional array, so we can transfer it to the GPU more easily.