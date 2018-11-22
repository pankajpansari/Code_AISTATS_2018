#include <chrono>
#include <vector>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <ctime>
#include <math.h> 
#include "densecrf_utils.h"
#include "permutohedral.h"
#include "pairwise.h"
#include "file_storage.hpp"
#include "sparsecrf.h"

struct Neighbor{
    int n1;
    int n2;
    int n3;
    int n4;
};

/////////////////////////////
/////  Alloc / Dealloc  /////
/////////////////////////////

SparseCRF::SparseCRF(int W, int H, int M): W_(W), H_(H), M_(M), N_(H*W), L_(M){
    unary_ = Eigen::MatrixXf::Zero(M_, N_);
    pairwise_weight_ = Eigen::VectorXf::Zero(M_, N_); 
}

SparseCRF::SparseCRF(int W, int H, int M, int L): W_(W), H_(H), M_(M), N_(H*W), L_(L){
    unary_ = Eigen::MatrixXf::Zero(M_, N_);
    pairwise_weight_ = Eigen::VectorXf::Zero(M_, N_); 
}
//////////////////////////////
/////  Unary Potentials  /////
//////////////////////////////


void SparseCRF::readUnary(std::string file, int rows, int cols) {

  std::ifstream in(file);
  
  std::string line;

  int row = 0;
  int col = 0;

  if (in.is_open()) {

    while (std::getline(in, line)) {

      char *ptr = (char *) line.c_str();
      int len = line.length();

      col = 0;

      char *start = ptr;
      for (int i = 0; i < len; i++) {

        if (ptr[i] == ',') {
          unary_(row, col++) = atof(start);
          start = ptr + i + 1;
        }
      }
      unary_(row, col) = atof(start);

      row++;
    }

    in.close();
  }
}

Eigen::MatrixXf SparseCRF::getUnary() {
    return unary_;
}

void SparseCRF::setUnary(Eigen::MatrixXf unary) {
    assert(unary_.rows() == M_ && unary_cols() == N_);
    unary_ = unary;
}

void SparseCRF::setTreeUnary(Eigen::MatrixXf unary) {
    assert(unary_.rows() == L_ && unary_cols() == N_);
    unary_.block(0, 0, L_, N_) = unary;
}
/////////////////////////////////
/////  Pairwise Potentials  /////
/////////////////////////////////

void SparseCRF::setPottsWeight(float weight) {
    for(int i = 0; i < M_; i++){
        pairwise_weight_(i) = weight;
    }
}

void SparseCRF::setTreeWeight(Eigen::VectorXf pairwise_weight) {
    pairwise_weight_ = pairwise_weight;
}

/////////////////////////////////
/////  Helper functions ////
/////////////////////////////////


void SparseCRF::getNeighbors(int var, int grid_size, int *neighbor){ //there are grid_size*grid_size variables

    int var_x = var/grid_size; //row number for elem in the N X N grid
    int var_y = var % grid_size; //column number for elem

    std::fill_n(neighbor, 4, -1);

    if(var_x > 0)
        neighbor[0] = (var_x - 1)*grid_size + (var_y);
    if(var_x < grid_size - 1)
        neighbor[1] = (var_x + 1)*grid_size + (var_y);
    if(var_y > 0)
        neighbor[2] = (var_x)*grid_size + (var_y - 1);
    if(var_y < grid_size - 1)
        neighbor[3] = (var_x)*grid_size + (var_y + 1);

}
/////////////////////////////////
/////  Conditional Gradient for Optimal Potts Extension  /////
/////////////////////////////////

float SparseCRF::gridEnergyChange(int var, std::vector<int> S, int grid_size, int label){ //elem is between 0 - (NL - 1), N = grid_size ^ 2
    
   
    float unary_val = unary_(label, var);
    
    //get pairwise change
    float pairwise_val = 0;
    int neighbor[4];
    getNeighbors(var, grid_size, neighbor);
  
    for(int i = 0; i < 4; i++){
        if(neighbor[i] != -1){
            if(find(S.begin(), S.end(), neighbor[i]) != S.end()){
               pairwise_val = pairwise_val - pairwise_weight_(label); //0.5 not required here because tree assumes that
            }
            else{
               pairwise_val =  pairwise_val + pairwise_weight_(label); 
            }
        }
    }
    return unary_val + pairwise_val;
}

void SparseCRF::greedyAlgorithm(MatrixXf &out, MatrixXf &grad, int grid_size){

    //negative gradient at current point is input
    //LP solution is the output
    
//    cout << "Sorted grad" << endl;
    for(int j = 0; j < grad.rows(); j++){
        VectorXf grad_j = grad.row(j);

        std::vector<int> y(grad_j.size());
        auto comparator = [&grad_j](int a, int b){ return grad_j[a] > grad_j[b]; };
        sort(y.begin(), y.end(), comparator);

        std::sort(grad_j.data(), grad_j.data() + grad_j.size(), std::greater<float>()); 
//        cout << grad_j << endl << endl;
        std::vector<int> S = {};
        for(int i = 0; i < y.size(); i++){
            out(j, y[i]) = gridEnergyChange(y[i], S, grid_size, j);
            S.push_back(y[i]);
        }
    }
//    cout << endl << "conditional gradient = " << out << endl;
}

void SparseCRF::getConditionalGradient(MatrixXf &Qs, MatrixXf & Q, int grid_size){
    //current solution is the input matrix (in)
    //conditional gradient is output

	MatrixXf negGrad( M_, N_ );
	getNegGradient(negGrad, Q); //negative gradient

        Qs.fill(0);
	greedyAlgorithm(Qs, negGrad, grid_size);	
}
/////////////////////////////////
/////  Conditional Gradient for Alternate Potts Extension  /////
////////////////////////////////

float SparseCRF::gridEnergyChangeBadExtension(int var, std::vector<int> S, int grid_size, int label){ //elem is between 0 - (NL - 1), N = grid_size ^ 2
    
   
    //get pairwise change
    float pairwise_val = 0;
    int neighbor[4];
    getNeighbors(var, grid_size, neighbor);
  
    for(int i = 0; i < 4; i++){
        if(neighbor[i] != -1){
            if(find(S.begin(), S.end(), neighbor[i]) != S.end()){
               pairwise_val = pairwise_val - pairwise_weight_(label); //0.5 not required here because tree assumes that
            }
            else{
               pairwise_val =  pairwise_val + pairwise_weight_(label); 
            }
        }
    }
    return pairwise_val;
}


float SparseCRF::unaryChangeBadExtension(int var, int label, std::vector<int> S){

    std::vector<int> st_vec(M_); // 0 -> in s, 1 -> in t
    std::vector<int> temp(M_);  //which unary costs to add
    float unary_cost1 = 0, unary_cost2 = 0;

    //without label assigned, only S assigned
    for(int i = 0; i <M_; i++){
        st_vec[i] = 0;
        temp[i] = 0;
    }

   for(int i = 0; i < S.size(); i++){
       st_vec[S[i]] = 1;
   }

   for(int i = 0; i < st_vec.size(); i++){
       if(i == st_vec.size() - 1){
        if(st_vec[i] != st_vec[0]){
           temp[i] = 1;
        }
       }
       else if(st_vec[i] != st_vec[i + 1])
           temp[i] = 1;
      }

   for(int i = 0; i < st_vec.size(); i++){
       if(i == st_vec.size() - 1){
        if(st_vec[i] != st_vec[0]){
           temp[i] = 1;
        }
       }
       else if(st_vec[i] != st_vec[i + 1])
           temp[i] = 1;
      }

    for(int i = 0; i < temp.size(); i++){
        if(temp[i] == 1)
        unary_cost1 += unary_(i, var);
    }

    //with label and S assigned
    for(int i = 0; i <M_; i++){
        st_vec[i] = 0;
        temp[i] = 0;
    }

   for(int i = 0; i < S.size(); i++){
       st_vec[S[i]] = 1;
   }
       st_vec[label] = 1;

   for(int i = 0; i < st_vec.size(); i++){
       if(i == st_vec.size() - 1){
        if(st_vec[i] != st_vec[0]){
           temp[i] = 1;
        }
       }
       else if(st_vec[i] != st_vec[i + 1])
           temp[i] = 1;
      }

   for(int i = 0; i < st_vec.size(); i++){
       if(i == st_vec.size() - 1){
        if(st_vec[i] != st_vec[0]){
           temp[i] = 1;
        }
       }
       else if(st_vec[i] != st_vec[i + 1])
           temp[i] = 1;
      }

    for(int i = 0; i < temp.size(); i++){
        if(temp[i] == 1)
        unary_cost2 += unary_(i, var);
}
   return unary_cost2 - unary_cost1;
}
void SparseCRF::greedyAlgorithmBadExtension(MatrixXf &out, MatrixXf &grad, int grid_size){

    //negative gradient at current point is input
    //LP solution is the output
    
    MatrixXf unary_out = MatrixXf::Zero(M_, N_); 
    MatrixXf pairwise_out = MatrixXf::Zero(M_, N_);

    //for unaries
    for(int j = 0; j < grad.cols(); j++){
        VectorXf grad_j = grad.col(j);

        std::vector<int> y(grad_j.size());
        auto comparator = [&grad_j](int a, int b){ return grad_j[a] > grad_j[b]; };
        sort(y.begin(), y.end(), comparator);

        std::sort(grad_j.data(), grad_j.data() + grad_j.size(), std::greater<float>()); 
        std::vector<int> S = {};

        for(int i = 0; i < y.size(); i++){
            unary_out(y[i], j) = unaryChangeBadExtension(j, y[i], S);
            S.push_back(y[i]);
        }
    }


    //for pairwise
    for(int j = 0; j < grad.rows(); j++){
        VectorXf grad_j = grad.row(j);

        std::vector<int> y(grad_j.size());
        auto comparator = [&grad_j](int a, int b){ return grad_j[a] > grad_j[b]; };
        sort(y.begin(), y.end(), comparator);

        std::sort(grad_j.data(), grad_j.data() + grad_j.size(), std::greater<float>()); 
        std::vector<int> S = {};

        for(int i = 0; i < y.size(); i++){
            pairwise_out(j, y[i]) = gridEnergyChangePairwise(y[i], S, grid_size, j);
            S.push_back(y[i]);
        }
    }
   
    out = unary_out + pairwise_out;
//    cout << endl << "conditional gradient = " << out << endl;
}

float SparseCRF::gridEnergyChangePairwise(int var, std::vector<int> S, int grid_size, int label){ //elem is between 0 - (NL - 1), N = grid_size ^ 2
    
   
    //get pairwise change
    float pairwise_val = 0;
    int neighbor[4];
    getNeighbors(var, grid_size, neighbor);
  
    for(int i = 0; i < 4; i++){
        if(neighbor[i] != -1){
            if(find(S.begin(), S.end(), neighbor[i]) != S.end()){
               pairwise_val = pairwise_val - pairwise_weight_(label); //0.5 not required here because tree assumes that
            }
            else{
               pairwise_val =  pairwise_val + pairwise_weight_(label); 
            }
        }
    }
    return pairwise_val;
}

void SparseCRF::getConditionalGradientBad(MatrixXf &Qs, MatrixXf & Q, int grid_size){
    //current solution is the input matrix (in)
    //conditional gradient is output

	MatrixXf negGrad( M_, N_ );
	getNegGradient(negGrad, Q); //negative gradient

        Qs.fill(0);
	greedyAlgorithmBadExtension(Qs, negGrad, grid_size);	
}



/////////////////////////////////
/////  Conditional Gradient for Optimal Tree Extension  /////
/////////////////////////////////

void SparseCRF::getConditionalGradient_tree(MatrixXf &Qs, MatrixXf & Q, int grid_size, const std::vector<node> &G){
    //current solution is the input matrix (in)
    //conditional gradient is output

	MatrixXf negGrad( M_, N_ );
	getNegGradient_rhst(negGrad, Q, G); //negative gradient

        Qs.fill(0);
	greedyAlgorithm(Qs, negGrad, grid_size);	
}

/////////////////////////////////
/////  Conditional Gradient for Alternate Tree Extension  /////
/////////////////////////////////
void SparseCRF::getConditionalGradientBad_tree(MatrixXf &Qs, MatrixXf & Q, int grid_size, const std::vector<node> &G){
    //current solution is the input matrix (in)
    //conditional gradient is output

	MatrixXf negGrad( M_, N_ );
	getNegGradient_rhst(negGrad, Q, G); //negative gradient

        Qs.fill(0);
	greedyAlgorithmBadExtension(Qs, negGrad, grid_size);	
}



/////////////////////////////////
/////  Submodular Inference for Potts potentials /////
/////////////////////////////////

void SparseCRF::submodularFrankWolfe_Potts(MatrixXf & init, int grid_size, std::string log_filename, int good){

    //clock
    typedef std::chrono::high_resolution_clock::time_point htime;
    htime start, end;
    double timing;
    start = std::chrono::high_resolution_clock::now();

    MatrixXf Q( M_, N_ ), Qs( M_, N_); //Q is the current point, Qs is the conditional gradient

    MatrixXf Qs_bf = MatrixXf::Zero(M_, N_);
    MatrixXf negGrad = MatrixXf::Zero( M_, N_ );

    Q = init;	//initialize to unaries
    
    //log file
    std::ofstream logFile;
    logFile.open(log_filename);

    float step = 0;
    
    float objVal = 0;

    objVal = getObj(Q);

    logFile << "0 " << objVal << " " << step << std::endl;

    for(int k = 1; k <= 100; k++){

      getNegGradient(negGrad, Q); //negative gradient

      if(good == 1)
          getConditionalGradient(Qs, Q, grid_size);
    else if (good == 0)
          getConditionalGradientBad(Qs, Q, grid_size);

      float fenchelGap = (Qs - Q).cwiseProduct(negGrad).sum();
    
      step = 2.0/(k + 2);
//      step = doLineSearch(Qs, Q);

      Q = Q + step*(Qs - Q); 

      objVal = getObj(Q);

    end = std::chrono::high_resolution_clock::now();
    timing = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
    
      logFile << timing << '\t' << objVal << '\t' << step << std::endl;
 //     std::cout << "Iter: " << k << " Obj = " << objVal << " Step size = " << step << " Gap = " << fenchelGap << std::endl;
      std::cout << "Iter: " << k << " Obj = " << objVal << " Step size = " << step << " Time = " << timing << " Gap = " << fenchelGap << std::endl;
   }
    std::cout << "Upper bound = " << objVal << std::endl;
}


/////////////////////////////////
/////  Submodular Inference for metric potentials /////
/////////////////////////////////


void SparseCRF::submodularFrankWolfe_tree(MatrixXf & init, int grid_size, std::string log_filename, const std::vector<node> &G, int good){

    //clock
    typedef std::chrono::high_resolution_clock::time_point htime;
    htime start, end;
    double timing;
    start = std::chrono::high_resolution_clock::now();

    MatrixXf Q = MatrixXf::Zero(M_, N_); //current point 
    MatrixXf Qs = MatrixXf::Zero(M_, N_);//conditional gradient
    MatrixXf negGrad = MatrixXf::Zero( M_, N_ );

    Q.block(0, 0, L_, N_) = init;

    //log file
    std::ofstream logFile;
    logFile.open(log_filename);

    float step = 0;
    
    float objVal = 0;

    objVal = getObj_rhst(Q, G);

    logFile << "0 " << objVal << " " << step << std::endl;

    for(int k = 1; k <= 100; k++){

      getNegGradient_rhst(negGrad, Q, G); //negative gradient

      if(good == 1)
          getConditionalGradient_tree(Qs, Q, grid_size, G);
        else if (good == 0)
          getConditionalGradientBad_tree(Qs, Q, grid_size, G);


      assert(checkNan(negGrad) && "negGrad has nan");
      assert(checkNan(Qs) && "Qs has nan");
      assert(checkNan(Q) && "Q has nan");

      float fenchelGap = (Qs - Q).cwiseProduct(negGrad).sum();
    
      step = 2.0/(k + 2);
//      step = doLineSearch_rhst(Qs, Q, 1, G);

      Q = Q + step*(Qs - Q); 

      objVal = getObj_rhst(Q, G);

    end = std::chrono::high_resolution_clock::now();
    timing = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
    
      logFile << timing << '\t' << objVal << '\t' << step << std::endl;
 //     std::cout << "Iter: " << k << " Obj = " << objVal << " Step size = " << step << " Gap = " << fenchelGap << std::endl;
      std::cout << "Iter: " << k << " Obj = " << objVal << " Step size = " << step << " Time = " << timing << " Gap = " << fenchelGap << std::endl;
//        if(fenchelGap < 1)
//            break;
    }
    std::cout << "Upper bound = " << objVal << std::endl;
}
