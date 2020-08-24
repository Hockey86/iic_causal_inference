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
}

parameters {
    real mu_a0;
    real mu_a1;
    real mu_a2;
    vector[ND] mu_b;
    
    real<lower=0> sigma_a0;
    real<lower=0> sigma_a1;
    real<lower=0> sigma_a2;
    vector<lower=0>[ND] sigma_b;
    
    vector[N] a0;
    vector<lower=-1,upper=1>[N] a2;
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
    
    vector[N] mu_a0_vec;
    vector[N] mu_a1_vec;
    vector[N] mu_a2_vec;
    vector[N] mu_b_vec;
    vector[N] sigma_a0_vec;
    vector[N] sigma_a1_vec;
    vector[N] sigma_a2_vec;
    vector[N] sigma_b_vec;
    
    vector[ND] ones_b;
    
    mu_a0_vec = rep_vector(mu_a0, N);
    sigma_a0_vec = rep_vector(sigma_a0, N);
    a0 ~ normal(mu_a0_vec, sigma_a0_vec);
    mu_a1_vec = rep_vector(mu_a1, N);
    sigma_a1_vec = rep_vector(sigma_a1, N);
    a1_raw ~ normal(mu_a1_vec, sigma_a1_vec);
    mu_a2_vec = rep_vector(mu_a2, N);
    sigma_a2_vec = rep_vector(sigma_a2, N);
    a2 ~ normal(mu_a2_vec, sigma_a2_vec);
    for (i in 1:ND) {
        mu_b_vec = rep_vector(mu_b[i], N);
        sigma_b_vec = rep_vector(sigma_b[i], N);
        b[:,i] ~ normal(mu_b_vec, sigma_b_vec);
    }
    
    for (t in 1:T0) { A[:,t] = A_start[:,t]; }
    
    ones_b = rep_vector(1, ND);
    for (t in T0+1:T) {
        A[:,t] = a0 + a1 .* A[:,t-1] + a2 .* A[:,t-2] - (D[t] .* b)*ones_b;//
    }
    
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
        target += binomial_lpmf(Eobs_flatten_nonan[i] | W, p_flatten_nonan[i]) * sample_weights[i]/not_empty_num;
    }
}

