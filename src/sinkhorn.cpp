#include "sinkhorn.h"

torch::Tensor base(const torch::Tensor &h_s,
                   const torch::Tensor &h_t,
                   const torch::Tensor &C,
                   double reg,
                   int numIter) {

    int N, num_s;
    N = C.size(0);
    num_s = C.size(1);
    ::dev = C.device();

    torch::Tensor K = (C / -reg).exp();
    torch::Tensor u = torch::ones({N, num_s}, at::device(::dev).dtype(C.scalar_type()).requires_grad(false));

    for (int iter = 0; iter < numIter; ++iter) {
        u = h_s / ((K * (h_t / ((K * u.unsqueeze(2)).sum(1))).unsqueeze(1)).sum(2));
    }

    return torch::einsum("ni,nij,nj->nij", {u, K, h_t / ((K * u.unsqueeze(2)).sum(1))});
}

torch::Tensor base_single(const torch::Tensor &h_s,
                           const torch::Tensor &h_t,
                           const torch::Tensor &C,
                           double reg,
                           int numIter) {

    int N, num_s, num_t;
    N = h_t.size(0);
    num_s = C.size(0);
    num_t = C.size(1);
    ::dev = C.device();

    torch::Tensor K = (C / -reg).exp();

//    std::cout << "test1" << std::endl ;
    torch::Tensor Kp = torch::einsum("i,ij->ji",{1. / h_s, K});
//    std::cout << "test2 Kp: " << Kp.size(0) << " " << Kp.size(1) << std::endl ;

    torch::Tensor u = torch::ones({N, num_s}, at::device(::dev).dtype(C.scalar_type()).requires_grad(false));
    torch::Tensor v = torch::ones({N, num_t}, at::device(::dev).dtype(C.scalar_type()).requires_grad(false));

    for (int iter = 0; iter < numIter; ++iter) {
//        std::cout << "test3 u:" << u.size(0) << u.size(1) << std::endl ;
        v = h_t / (torch::matmul(u, K));
//        std::cout << "test4 " << std::endl ;
        u = 1. / (torch::matmul(v, Kp));
    }

//    std::cout << "test5" << std::endl ;
    return torch::einsum("ni,ij,nj->nij", {u, K, v});
}

torch::Tensor unbalanced(const torch::Tensor &h_s,
                         const torch::Tensor &h_t,
                         const torch::Tensor &C,
                         double reg,
                         int numIter,
                         double tau1,
                         double tau2) {
    double factor1 = tau1 / (tau1 + reg);
    double factor2 = tau2 / (tau2 + reg);

    int N, num_s;
    N = C.size(0);
    num_s = C.size(1);
    ::dev = C.device();

    torch::Tensor K = (-C / reg).exp();
    torch::Tensor u = torch::ones({N, num_s}, at::device(::dev).dtype(C.scalar_type()).requires_grad(false));

    for (int iter = 0; iter < numIter; ++iter) {
        u = (h_s / ((K *
                     (h_t / ((K * u.unsqueeze(2)).sum(1))).pow(factor2)
                             .unsqueeze(1)).sum(2))).pow(factor1);
    }

    return torch::einsum("ni,nij,nj->nij", {u, K, (h_t / ((K * u.unsqueeze(2)).sum(1))).pow(factor2)});
}

/////////////////////////////////////////////////////////////////////////////////////
// STABLE VERSIONS //////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

void _update_K(const torch::Tensor &alpha,
               const torch::Tensor &beta,
               const torch::Tensor &C,
               double reg,
               torch::Tensor &K) {
    K = torch::exp(-(C - alpha.unsqueeze(2).expand_as(C)
                     - beta.unsqueeze(1).expand_as(C)) / reg);
}

torch::Tensor _get_P(const torch::Tensor &alpha,
                     const torch::Tensor &beta,
                     const torch::Tensor &u,
                     const torch::Tensor &v,
                     const torch::Tensor &C,
                     double reg) {
    return torch::exp(-((C - alpha.unsqueeze(2).expand_as(C)
                         - beta.unsqueeze(1).expand_as(C)) / reg)
                      + u.log().unsqueeze(2).expand_as(C)
                      + v.log().unsqueeze(1).expand_as(C));
}

torch::Tensor stable(const torch::Tensor &h_s,
                     const torch::Tensor &h_t,
                     const torch::Tensor &C,
                     double reg,
                     int numIter) {
    int N, num_s, num_t;
    N = C.size(0);
    num_s = C.size(1);
    num_t = C.size(2);
    ::dev = C.device();

    torch::Tensor K;
    torch::Tensor u = torch::ones({N, num_s}, at::device(::dev).dtype(C.scalar_type()).requires_grad(false)) / num_s;
    torch::Tensor v = torch::ones({N, num_t}, at::device(::dev).dtype(C.scalar_type()).requires_grad(false)) / num_t;
    torch::Tensor alpha = torch::zeros({N, num_s}, at::device(::dev).dtype(C.scalar_type()).requires_grad(false));
    torch::Tensor beta = torch::zeros({N, num_t}, at::device(::dev).dtype(C.scalar_type()).requires_grad(false));
    _update_K(alpha, beta, C, reg, K);

    for (int iter = 0; iter < numIter; ++iter) {
        u = h_s / ((K * v.unsqueeze(1)).sum(2));
        v = h_t / ((K * u.unsqueeze(2)).sum(1));

        alpha += reg * std::get<0>(u.log().max(0, true));
        beta += reg * std::get<0>(v.log().max(0, true));

        u.fill_(1. / num_s);
        v.fill_(1. / num_t);

        _update_K(alpha, beta, C, reg, K);
    }
    u = h_s / ((K * v.unsqueeze(1)).sum(2));
    v = h_t / ((K * u.unsqueeze(2)).sum(1));
    return _get_P(alpha, beta, u, v, C, reg);
}

torch::Tensor unbalanced_stable(const torch::Tensor &h_s,
                                const torch::Tensor &h_t,
                                const torch::Tensor &C,
                                double reg,
                                int numIter,
                                double tau1,
                                double tau2) {
    double factor1 = tau1 / (tau1 + reg);
    double factor2 = tau2 / (tau2 + reg);

    int N, num_s, num_t;
    N = C.size(0);
    num_s = C.size(1);
    num_t = C.size(2);
    ::dev = C.device();

    torch::Tensor K;
    torch::Tensor u = torch::ones({N, num_s}, at::device(::dev).dtype(C.scalar_type()).requires_grad(false));
    torch::Tensor v = torch::ones({N, num_t}, at::device(::dev).dtype(C.scalar_type()).requires_grad(false));
    torch::Tensor alpha = torch::zeros({N, num_s}, at::device(::dev).dtype(C.scalar_type()).requires_grad(false));
    torch::Tensor beta = torch::zeros({N, num_t}, at::device(::dev).dtype(C.scalar_type()).requires_grad(false));
    _update_K(alpha, beta, C, reg, K);

    for (int iter = 0; iter < numIter; ++iter) {
        u = (h_s / ((K * v.unsqueeze(1)).sum(2))).pow(factor1);
        v = (h_t / ((K * u.unsqueeze(2)).sum(1))).pow(factor2);

        alpha += reg * std::get<0>(u.log().max(1, true));
        beta += reg * std::get<0>(v.log().max(1, true));

        u.fill_(1.);
        v.fill_(1.);

        _update_K(alpha, beta, C, reg, K);
    }

    return _get_P(alpha, beta, u, v, C, reg);
}


