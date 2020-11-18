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
    vector[N] t0[N_sample];
    vector[N] sigma0[N_sample];
    vector[N] alpha0[N_sample];
    matrix[N, AR_p] alpha[N_sample];
    vector[MA_q] theta[N_sample];
    matrix[N,ND] b[N_sample];
    vector[NClust] sigma_err[N_sample]; // 
}

parameters {
}

model {
}

generated quantities {
    matrix[N,T] P_output[N_sample];
    
    matrix[N,T] err;
    vector[ND] ones_b;
    vector[N] tmp1;
    vector[N] tmp2;
    
    ones_b = rep_vector(1, ND);
    
    for (n in 1:N_sample) {
        for (i in 1:N)
            err[i] = to_row_vector(cauchy_rng(0, rep_vector(sigma_err[n][cluster[i]], T)));
        
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
                if (t>1)
                    P_output[n][:,t] -= (D[t-1] .* b[n])*ones_b;
            }
        }
                
        // sample P_output
        for (t in 1:T) {
            for (i in 1:N) {
                tmp1[i] = P_output[n][i,t];
                if (tmp1[i]>12)
                    tmp1[i] = 12;
                else if (tmp1[i]<-12)
                    tmp1[i] = -12;
                tmp1[i] = inv_logit(tmp1[i]);
                tmp2[i] = exp( -(log(t-1-t0[n][i]))^2 / (2*sigma0[n][i]^2) );
            }
            P_output[n][:,t] = to_vector(binomial_rng(W, tmp1 .* tmp2))/W;
        }
    }
}
