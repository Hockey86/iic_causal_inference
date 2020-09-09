data {
    int W;
    int N;
    int T;
    int T0;  // initialization time step
    int ND;
    matrix[N,ND] D[T];
    matrix[N,T0] A_start;
    int NClust;
    int cluster[N];
    
    int N_sample;
    vector[N] a0[N_sample];
    vector[N] a1[N_sample];
    vector[N] a2[N_sample];
    matrix[N,ND] b[N_sample];
}

parameters {
}

model {
}

generated quantities {
    matrix[N,T] P_output[N_sample];
    
    real tmp2;
    vector[ND] ones_b;
    ones_b = rep_vector(1, ND);
    
    for (i in 1:N_sample) {
        for (t in 1:T0) { P_output[i][:,t] = A_start[:,t]; }
        for (t in T0+1:T) {
            P_output[i][:,t] = a0[i] + a1[i] .* P_output[i][:,t-1] + a2[i] .* P_output[i][:,t-2] - (D[t] .* b[i])*ones_b;
            
            for (n in 1:N) {
                if (P_output[i][n,t]<-12)
                    P_output[i][n,t] = -12;
                else if (P_output[i][n,t]>12)
                    P_output[i][n,t] = 12;
                P_output[i][n,t] = inv_logit(P_output[i][n,t]);
                
                tmp2 = binomial_rng(W, P_output[i][n,t]);
                
                if (tmp2<1)
                    tmp2 = 1;
                else if (tmp2>W-1)
                    tmp2 = W-1;
                P_output[i][n,t] = logit(tmp2/W);
            }
        }
        P_output[i] = inv_logit(P_output[i]);
    }
}
