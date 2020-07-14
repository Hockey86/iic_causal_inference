data {
    int W;
    int N;
    int T;
    int T0;  // initialization time step
    int ND;
    int NC;
    int not_empty_num;
    int not_empty_ids[not_empty_num];
    int Eobs_flatten_nonan[not_empty_num];
    real sample_weights[not_empty_num];
    matrix[N,ND] D[T];
    matrix[N,NC] C;
    matrix[N,T0] A_start;
}

parameters {
    real a0;
    real a1;
    real a2;
    vector<lower=0>[ND] b;
    vector<lower=0>[NC] beta;
}

transformed parameters {
    matrix[N,T] A;
    matrix[N,T] p;
    
    vector[N*(T-T0)] p_flatten;
    vector[not_empty_num] p_flatten_nonan;
    
    for (t in 1:T0) { A[:,t] = A_start[:,t]; }
    
    for (t in T0+1:T) {
        A[:,t] = a0 + a1*A[:,t-1] + a2*A[:,t-2] - D[t]*b + C*beta;
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
}

model {
    a0 ~ normal(0,1);
    a1 ~ normal(0,1);
    a2 ~ normal(0,1);
    beta ~ multi_normal(rep_vector(0,NC), diag_matrix(rep_vector(1, NC)));
    b ~ multi_normal(rep_vector(0,ND), diag_matrix(rep_vector(1, ND)));
    
    //Eobs_flatten_nonan ~ binomial(W, p_flatten_nonan);
    for (i in 1:not_empty_num) {
        target += binomial_lpmf(Eobs_flatten_nonan[i] | W, p_flatten_nonan[i]) * sample_weights[i];
    }
}
    
generated quantities {
    vector[not_empty_num] log_lik;
    for (i in 1:not_empty_num) {
        log_lik[i] = binomial_lpmf(Eobs_flatten_nonan[i] | W, p_flatten_nonan[i]);
    }
}
