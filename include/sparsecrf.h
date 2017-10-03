#include "densecrf_utils.h"
#include <vector>
class SparseCRF{

private:
    int N_, M_, L_; //N_ -> number of variables (= H_ X W_), M_ -> number of meta-labels, L_ -> number of labels
    int H_, W_; //H_ -> height of grid, W_ -> width of grid
    
    // Store the unary term
    Eigen::MatrixXf unary_;
    Eigen::VectorXf pairwise_weight_;    
//    // Store all pairwise potentials

public:
    // Create a dense CRF model of size N with M labels
//    SparseCRF( int N, int M, int H, int W );
    SparseCRF(int W, int H, int M);
    SparseCRF(int W, int H, int M, int L);

    void readUnary(std::string file, int rows, int cols);
    void setUnary(Eigen::MatrixXf unary);
    void setTreeUnary(Eigen::MatrixXf unary);
    
    void setPottsWeight(float weight);
    void setTreeWeight(Eigen::VectorXf pairwise);

    void getNeighbors(int var, int grid_size, int *neighbor);
    
    float gridEnergyChange(int var, std::vector<int> S, int grid_size,  int label);
    
    void greedyAlgorithm(Eigen::MatrixXf &out, Eigen::MatrixXf &grad, int grid_size);
    
    void getConditionalGradient(Eigen::MatrixXf &Qs, Eigen::MatrixXf & Q, int grid_size);

    void getConditionalGradient_rhst(Eigen::MatrixXf &Qs, Eigen::MatrixXf & Q, int grid_size, const std::vector<node> &G);
    
    Eigen::MatrixXf getUnary();

    void submodularFrankWolfe(Eigen::MatrixXf & init, int grid_size, std::string log_filename);

};