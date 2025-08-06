"""
Representing data in n dimensional spaces


Scalar -> 1d vector

Vector -> vector of 1d vectors

Matrices -> vector of vectors

But this is not true 

R^1 -= [1]

R^2 = [1,2,2,3,3]

R^3 = [1,2,3]

"""

class Tensor {
public: 




private: 

std::vector<size_t> compute_strides(std::vector<size_t> shape){
    std::vector<size_t> strides(shape.size()); 
    strides[strides.size() - 1] = 1; 
    for (int i = strides.size() - 1; i >= 0; i--){
        strides[i - 1] = strides[i] * shape[i]; 
    }
    return strides; 
}


}