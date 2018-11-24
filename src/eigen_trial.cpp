//template
//compile & link - g++ trial.cpp
//execute       ./a.out
#include <iostream>
#include <vector>
#include <algorithm>

#include <Eigen>
using namespace Eigen;
using namespace std;

void print_vec(auto &a){
    for (auto i = a.begin(); i != a.end(); ++i)
            std::cout << *i << ' ';
}

int main()
{
    std::vector<int>  a {1, 2, 3, 4};
    print_vec(a);
}
