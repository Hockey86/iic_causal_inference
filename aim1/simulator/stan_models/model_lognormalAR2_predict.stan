data {
    int W;
    int N;
    int T;
    int ND;
    matrix[N,ND] D[T];
    
    int N_sample;
    vector[N] a0[N_sample];
    vector[N] a1[N_sample];
    vector[N] a2[N_sample];
    matrix[N,ND] b[N_sample];
    
    vector[N] mu[N_sample];
    vector[N] sigma[N_sample];
}

parameters {
}

model {
}

generated quantities {
    matrix[N,T] AR[N_sample];
    matrix[N,T] P_output[N_sample];
    
    vector[N] tmp;
    real tmp2;
    vector[ND] ones_b;
    ones_b = rep_vector(1, ND);
    
    for (i in 1:N_sample) {
        for (t in 1:T) {
            if (t==1)
                AR[i][:,t] = a0[i];
            else if (t==2)
                AR[i][:,t] = a0[i] + a1[i] .* AR[i][:,t-1];
            else
                AR[i][:,t] = a0[i] + a1[i] .* AR[i][:,t-1] + a2[i] .* AR[i][:,t-2];
            tmp = log(t)-mu[i];
            P_output[i][:,t] = AR[i][:,t] .* exp(-(tmp .* tmp) ./ (2* (sigma[i] .* sigma[i]))) - (D[t] .* b[i])*ones_b;
        }
        P_output[i] = inv_logit(P_output[i]);
        
        for (t in 1:T) {
            for (n in 1:N) {
                tmp2 = binomial_rng(W, P_output[i][n,t]);
                P_output[i][n,t] = tmp2/W;
            }
        }
    }
}
