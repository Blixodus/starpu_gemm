#pragma once

#include <numeric>
#include <vector>

template <typename IntType>
class DimIter {
  std::vector<std::vector<IntType>> iterates;
  
public:
  DimIter(IntType* size, size_t len, size_t start_dim = 0) {
    std::vector<IntType> curr(len, 0);
    size_t nelem = static_cast<size_t>(std::accumulate(size, size+len, 1, std::multiplies<IntType>()));
    for(size_t i = 0; i < nelem; i++) {
      iterates.push_back(curr);
      // Move to next element
      for(size_t j = start_dim; j < len; j++) {
        curr[j] = (curr[j] < size[j] - 1) ? (curr[j] + 1) : 0;
        if(curr[j]) { break; }
      }
    }
  }
  
  DimIter(std::vector<IntType>& size, size_t start_dim = 0) : DimIter(&size[0], size.size(), start_dim) { }

  auto begin() {
    return iterates.begin();
  }

  auto end() {
    return iterates.end();
  }
};
