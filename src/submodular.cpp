#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <ctime>
#include "densecrf.h"
#include <math.h> 
#include "densecrf_utils.h"
#include "permutohedral.h"
#include "pairwise.h"
#include "file_storage.hpp"

void DenseCRF::load_cliques(const std::string & fileName) {

    std::fstream myfile(fileName.c_str(), std::ios_base::in);
    
    float dummy;
    
    myfile >> nvar >> nlabel;
    
    myfile.get();
    
    for(int variable_count = 0; variable_count < nvar; variable_count++){
        std::string line;
        getline(myfile, line);
        std::istringstream iss(line);
        float unary;
        variable_clique_id.push_back(std::vector<int>());
        while(iss >> unary){
            dummy = unary;
        }
    }
    
    myfile >> dummy;

    myfile >> nclique;

    std::vector<float> temp(nlabel, 0);
    for(int clique = 0; clique < nclique; clique++){
        int size;
        myfile >> size;
        clique_sizes.push_back(size);
        std::vector<int> current_clique_members;
        for(int variable_count = 0; variable_count < size; variable_count++){
            int variable_id;
            myfile >> variable_id;
            variable_clique_id[variable_id - 1].push_back(clique);
            current_clique_members.push_back(variable_id - 1);
        }
        clique_members.push_back(current_clique_members);
        double weight;
        myfile >> weight;
        clique_weight.push_back(weight);
        clique_state.push_back(temp);
        last_clique_val.push_back(0);
    }

    myfile.close();
}


void DenseCRF::compareWithBf(MatrixXf &pairwise_filter, MatrixXf & grad){

    MatrixXf pairwise_bf = MatrixXf::Zero(M_, N_);

    std::cout << "Doing bf computation" << std::endl;
    applyBruteForce(pairwise_bf, grad);

   //check the angle
    MatrixP dot_tmp(M_, N_);
    double costh = dotProduct(pairwise_filter, pairwise_bf, dot_tmp)/
       (sqrt(dotProduct(pairwise_filter, pairwise_filter, dot_tmp))*sqrt(dotProduct(pairwise_bf, pairwise_bf, dot_tmp)));
    std::cout << "bf-filter #cos-theta: " << costh << std::endl;

    //check norms
    std::cout << "Filter o/p norm = " << pairwise_filter.norm() << std::endl;
    std::cout << "Bf o/p norm = " << pairwise_bf.norm() << std::endl;

}

float DenseCRF::update_clique_state(int v, int c){
        int label = v % M_;         
        clique_state[c][label] += 1.0/clique_sizes[c];
        float fn_val = 0;
        for(int j = 0; j < M_; j++){
            if(clique_state[c][j] > 0 && clique_state[c][j] < 1)
                fn_val += 0.5;
        }
        if(fn_val >= 1)
            return 1;
        else
            return fn_val;
}

float DenseCRF::delta_submodular_higher_Potts(int v){
        int var = v/M_;
        int label = v % M_;
        std::vector<int> cliques = variable_clique_id[var];  
        double fn_difference = 0;

        for(int c = 0; c < cliques.size(); c++){
            int clique_id = cliques[c];
            float last_val = last_clique_val[clique_id];
            float new_val = update_clique_state(v, clique_id);
            fn_difference += new_val - last_val;
            last_clique_val[clique_id] = new_val;
        }
        return fn_difference;
}

MatrixXf DenseCRF::get_clique_term(MatrixXf &grad){

    Map<VectorXf> grad_vec(grad.data(), grad.rows()*grad.cols());     
    VectorXf out_vec =  VectorXf::Zero(grad.rows()* grad.cols());

    std::vector<int> y(grad_vec.size());
    for(int i = 0; i < y.size(); i++)
        y[i] = i;
    auto comparator = [&grad_vec](int p, int q){ return grad_vec[p] > grad_vec[q]; };
    sort(y.begin(), y.end(), comparator);

    std::vector<float> temp(M_, 0);
    for(int i = 0; i < nclique; i++){
        clique_state[i] = temp;
    }
    for(int i = 0; i < y.size(); i++){
        out_vec(y[i]) = delta_submodular_higher_Potts(y[i]);
    }
    
    Map<MatrixXf> out(out_vec.data(), grad.rows(), grad.cols());    
    return out;
}


void DenseCRF::getConditionalGradient(MatrixXf &Qs, MatrixXf & Q){
    //current solution is the input matrix (in)
    //conditional gradient is output

        M_ = Q.rows();
        N_ = Q.cols();

	MatrixXf negGrad( M_, N_ );
	getNegGradient(negGrad, Q); //negative gradient

        Qs.fill(0);
	greedyAlgorithm(Qs, negGrad);	
}

void DenseCRF::greedyAlgorithm(MatrixXf &out, MatrixXf &grad){

    //negative gradient at current point is input
    //LP solution is the output

    out.fill(0);
    //get unaries
    MatrixXf unary = unary_->get();   
    
    //get pairwise
    MatrixXf pairwise = MatrixXf::Zero(M_, N_);
    
    clock_t start = std::clock();
    double duration = 0;
    applyFilter(pairwise, grad);
    pairwise = pairwise.array() * 0.5;

    MatrixXf clique = MatrixXf::Zero(M_, N_);
    clique = get_clique_term(grad);
 
    out = unary - pairwise + clique_potts*clique; //-ve because original code makes use of negative Potts potential (in labelcompatibility.cpp), but we want to use positive weights

}


MatrixXf DenseCRF::submodularFrankWolfe_Potts( MatrixXf & init, int width, int height, std::string output_path, std::string dataset_name){

    MatrixXf Q = MatrixXf::Zero(M_, N_); //current point 
    MatrixXf Qs = MatrixXf::Zero(M_, N_);//conditional gradient
    MatrixXf negGrad = MatrixXf::Zero( M_, N_ );

    MatrixP dot_tmp(M_, N_);
    MatrixXf temp(M_, N_); //current point 

    Q = init;	//initialize to unaries

    float step = 0;
    img_size size;
    size.width = width;
    size.height = height;

    //log file
    std::string log_output = output_path;
    log_output.replace(log_output.end()-4, log_output.end(),"_log.txt");
    std::ofstream logFile;
    logFile.open(log_output);

    //image file
    std::string image_output = output_path;

    clock_t start;
    float duration = 0;
    float dualGap = 0;
    float objVal = 0;

    start = clock();

    objVal = getObj(Q);
    logFile << "0 " << objVal << " " <<  duration << " " << step << std::endl;

    for(int k = 1; k <= 10; k++){

      getConditionalGradient(Qs, Q);

      step = doLineSearch(Qs, Q, 1);

      Q = Q + step*(Qs - Q); 

      duration = (clock() - start ) / (double) CLOCKS_PER_SEC;

      objVal = getObj(Q);

      //write to log file
      logFile << k << " " << objVal << " " <<  duration << " " << step << std::endl;
      std::cout << "Iter: " << k << " Obj = " << objVal << " Time = " <<  duration << " Step size = " << step << std::endl;

      
      if(k % 10 == 0){
            //name the segmented image and Q files
                std::string img_file_extn = "_" + std::to_string(k) + ".png";
                image_output = output_path;
                image_output.replace(image_output.end()-4, image_output.end(), img_file_extn);
                
             //save segmentation
               expAndNormalize(temp, -Q);
               save_map(temp, size, image_output, dataset_name);

      }
   }

   logFile.close();

   //convert Q to marginal probabilities
   MatrixXf marginal(M_, N_);
   expAndNormalize(marginal, -Q); 

   return marginal;
}
