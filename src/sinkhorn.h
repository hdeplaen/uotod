//
// Created by hdeplaen on 3/4/22.
//

#ifndef SINKHORN_H
#define SINKHORN_H

#include <torch/extension.h>
#include <iostream>
#include <cstdio>
#include <limits>

#define EPS 0.00000000001

torch::Device dev = torch::kCPU;

torch::Tensor base(
        const torch::Tensor &h_s,
        const torch::Tensor &h_t,
        const torch::Tensor &C,
        double reg,
        int numIter);

torch::Tensor base_single(
        const torch::Tensor &h_s,
        const torch::Tensor &h_t,
        const torch::Tensor &C,
        double reg,
        int numIter);


torch::Tensor unbalanced(
        const torch::Tensor &h_s,
        const torch::Tensor &h_t,
        const torch::Tensor &C,
        double reg,
        int numIter,
        double tau1,
        double tau2);

torch::Tensor stable(
        const torch::Tensor &h_s,
        const torch::Tensor &h_t,
        const torch::Tensor &C,
        double reg,
        int numIter);

torch::Tensor unbalanced_stable(
        const torch::Tensor &h_s,
        const torch::Tensor &h_t,
        const torch::Tensor &C,
        double reg,
        int numIter,
        double tau1,
        double tau2);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("unbalanced", &unbalanced, "Sinkhorn Unbalanced");
m.def("stable", &stable, "Stabilized Sinkhorn");
m.def("unbalanced_stable", &unbalanced_stable, "Stabilized Unbalanced Sinkhorn");
m.def("base", &base, "Sinkhorn");
m.def("base_single", &base_single, "Sinkhorn with single cost matrix");
}

#endif //SINKHORN_H
