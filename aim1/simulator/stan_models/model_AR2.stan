data {
    int W;
    int N;
    int T;
    int T0;  // initialization time step
    int ND;
    int not_empty_num;
    int not_empty_ids[not_empty_num];
    int Eobs_flatten_nonan[not_empty_num];
    real sample_weights[not_empty_num];
    matrix[N,ND] D[T];
    matrix[N,T0] A_start;
    int NClust;
    int cluster[N];
}

parameters {
    real mu_a0[NClust];
    real mu_a1[NClust];
    real<lower=-0.999,upper=0.999> mu_a2[NClust];
    vector<lower=0>[ND] mu_b[NClust];
    
    real<lower=0.001> sigma_a0[NClust];
    real<lower=0.001> sigma_a1[NClust];
    real<lower=0.001> sigma_a2[NClust];
    vector<lower=0.001>[ND] sigma_b[NClust];
    
    vector[N] a0;
    vector<lower=-0.999,upper=0.999>[N] a2;
    vector[N] a1_raw;
    matrix<lower=0>[N,ND] b;
}

transformed parameters {
    vector[N] a1;
    for (i in 1:N) {
        a1[i] = inv_logit(a1_raw[i])*(2-2*a2[i]) + a2[i]-1;
    }
}

model {
    matrix[N,T] A;
    matrix[N,T] p;
    vector[N*(T-T0)] p_flatten;
    vector[not_empty_num] p_flatten_nonan;
    vector[ND] ones_b;
    
    for (i in 1:N){
        a0[i] ~ normal(mu_a0[cluster[i]], sigma_a0[cluster[i]]);
        a1_raw[i] ~ normal(mu_a1[cluster[i]], sigma_a1[cluster[i]]);
        a2[i] ~ normal(mu_a2[cluster[i]], sigma_a2[cluster[i]]);
    }
    for (j in 1:ND) {
        for (i in 1:N)
            b[i,j] ~ normal(mu_b[cluster[i]][j], sigma_b[cluster[i]][j]);
    }
    
    for (t in 1:T0) { A[:,t] = A_start[:,t]; }
    
    ones_b = rep_vector(1, ND);
    for (t in T0+1:T)
        A[:,t] = a0 + a1 .* A[:,t-1] + a2 .* A[:,t-2] - (D[t] .* b)*ones_b; // + theta1 .* err[:,t-1];
    
    // clip At
    for (t in T0+1:T) {
        for (i in 1:N) {
            if (A[i,t]<-12) {
                A[i,t] = -12;
            }
            else if (A[i,t]>12) {
                A[i,t] = 12;
            }
        }
    }
    
    p = inv_logit(A);
    
    p_flatten = to_vector(p[:,T0+1:]'); // to_vector is flattening p in column-major order, but we need row-major order, so transpose it
    p_flatten_nonan = p_flatten[not_empty_ids];
    
    //Eobs_flatten_nonan ~ binomial(W, p_flatten_nonan);
    for (i in 1:not_empty_num) {
        target += binomial_lpmf(Eobs_flatten_nonan[i] | W, p_flatten_nonan[i]) * sample_weights[i]; // not_empty_num;
    }
}

