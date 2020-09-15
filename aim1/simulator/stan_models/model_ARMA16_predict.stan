data {
    int W;
    int N;
    int T;
    int AR_T0;  // p, AR init time step (order)
    int MA_T0;  // q, MA init time step (order)
    int ND;
    matrix[N,ND] D[T];
    matrix[N,AR_T0] A_start;
    int NClust;
    int cluster[N];
    
    int N_sample;
    vector[N] a0[N_sample];
    vector[N] a1[N_sample];
    matrix[N, MA_T0] theta[N_sample];
    matrix[N,ND] b[N_sample];
    vector[N] sigma_err[N_sample];
}

parameters {
}

model {
}

generated quantities {
    matrix[N,T] P_output[N_sample];
    
    real tmp1;
    vector[N] tmp2;
    vector[ND] ones_b;
    ones_b = rep_vector(1, ND);
    
    for (i in 1:N_sample) {
        for (t in 1:T) {
            // AR(1) only, TODO code for general AR(p), but don't know how to code stationery constraints in general
            if (t<=AR_T0)
                P_output[i][:,t] = A_start[:,t];
            else {
                P_output[i][:,t] = a0[i] + a1[i] .* P_output[i][:,t-1] - (D[t] .* b[i])*ones_b;
            }
            
            // MA(q)
            tmp2 = rep_vector(0, N);
            for (q in 1:MA_T0) {
                if (t-q>0)
                    tmp2 += theta[i][:,q] .* to_vector(normal_rng(0, sigma_err[i]));
            }
            P_output[i][:,t] += tmp2;
            
            for (n in 1:N) {
                if (P_output[i][n,t]<-12)
                    P_output[i][n,t] = -12;
                else if (P_output[i][n,t]>12)
                    P_output[i][n,t] = 12;
                P_output[i][n,t] = inv_logit(P_output[i][n,t]);
                
                tmp1 = binomial_rng(W, P_output[i][n,t]);
                
                if (tmp1<1)
                    tmp1 = 1;
                else if (tmp1>W-1)
                    tmp1 = W-1;
                P_output[i][n,t] = logit(tmp1/W);
            }
        }
        P_output[i] = inv_logit(P_output[i]);
    }
}
