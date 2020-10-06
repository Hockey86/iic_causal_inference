data {
    int W;
    int N;
    int T;
    int AR_p;  // p, AR init time step (order)
    int MA_q;  // q, MA init time step (order)
    int ND;
    matrix[N,ND] D[T];
    matrix[N,AR_p] A_start;
    int NClust;
    int cluster[N];
    
    int N_sample;
    vector[N] alpha0[N_sample];
    matrix[N, AR_p] alpha[N_sample];
    vector[MA_q] theta[N_sample];
    matrix[N,ND] b[N_sample];
    vector[NClust] sigma_err[N_sample];
    vector[NClust] nu[N_sample];
}

parameters {
}

model {
}

generated quantities {
    matrix[N,T] P_output[N_sample];
    
    matrix[N,T] err;
    vector[ND] ones_b;
    vector[N] ones_N;
    vector[N] tmp1;
    real tmp2;
    real eta;
    
    ones_b = rep_vector(1, ND);
    ones_N = rep_vector(1, N);
    
    for (n in 1:N_sample) {
        for (i in 1:N)
            err[i] = to_row_vector(student_t_rng(nu[n][cluster[i]], 0, rep_vector(sigma_err[n][cluster[i]], T)));
        
        for (t in 1:T) {
            if (t<=AR_p)
                P_output[n][:,t] = A_start[:,t];
            else {
                // AR(p)
                tmp1 = rep_vector(0, N);
                for (p in 1:AR_p) {
                    tmp1 += alpha[n][:,p] .* P_output[n][:,t-p];
                }
                P_output[n][:,t] = alpha0[n] + tmp1;
            
                // MA(q)
                tmp1 = rep_vector(0, N);
                for (q in 1:MA_q) {
                    if (t>q)
                        tmp1 += theta[n][q] * err[:,t-q];
                }
                P_output[n][:,t] += tmp1;
                
                // drug
                P_output[n][:,t] -= (D[t] .* b[n])*ones_b;
                
                // sample P_output
                for (i in 1:N) {
                    //Binomial throws erros when given probability of 0
                    eta = Phi(P_output[n][i,t]);
                    if(eta<= 0.001)
                        eta = 0.001;
                    else if(eta >= 0.999)
                        eta = 0.999;
                    
                    tmp2 = binomial_rng(W, eta);
                    
                    if (tmp2<1)
                        tmp2 = 1;
                    else if (tmp2>W-1)
                        tmp2 = W-1;
                    P_output[n][i,t] = inv_Phi(tmp2/W);
                }
            }
        }
        P_output[n] = Phi(P_output[n]);
    }
}
