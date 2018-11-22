#pragma once
#include <Eigen/Core>
#include <vector>

struct node{
    int id;
    node* parent;
    std::vector<node*> children;
    float weight; //weight of edge from node to its parent
};

std::vector<node> readTree(const std::string filename);

int getNumMetaLabels(const std::vector<node> &G);

int getNumLabels(const std::vector<node> &G, int M = 0);

Eigen::VectorXf getWeight(const std::vector<node> &G);

node getRoot(std::vector<node> G);

std::vector<node> getLeafNodes(const std::vector<node> &G);

float sumPath(node t);
std::vector<node> getPath(node t);

Eigen::MatrixXf getPairwiseTable(const std::vector<node> &G);
