data {
    int W;
    int N;
    int T;
    int ND;
    matrix[N,ND] D[T];
    
    //real mu_mu;
    //real<lower=0> sigma_mu;
    //real<lower=0> sigma_sigma;
    //real<lower=0> sigma_alpha;
    //vector<lower=0>[ND] sigma_b;
    
    int N_sample;
    vector[N] t0[N_sample];
    vector[N] mu[N_sample];
    vector[N] sigma[N_sample];
    vector[N] alpha[N_sample];
    matrix[N,ND] b[N_sample];
}

parameters {
}

model {
}

generated quantities {
    matrix[N,T] P_output[N_sample];
    
    vector[N] tmp;
    real tmp2;
    vector[ND] ones_b;
    ones_b = rep_vector(1, ND);
    
    for (i in 1:N_sample) {
        for (t in 1:T) {
            tmp = log(t+t0[i])-mu[i];
            P_output[i][:,t] = alpha[i] .* exp(-(tmp .* tmp) ./ (2* (sigma[i] .* sigma[i]))) - (D[t] .* b[i])*ones_b;
        }
        
        P_output = inv_logit(P_output);
        
        for (t in 1:T) {
            for (n in 1:N) {
                tmp2 = binomial_rng(W, P_output[i][n,t]);
                P_output[i][n,t] = tmp2/W;
            }
        }
    }
}
