data {
    int W;
    int N;
    int T;
    int AR_T0;  // p, AR init time step (order)
    int MA_T0;  // q, MA init time step (order)
    int ND;
    int not_empty_num;
    int not_empty_ids[not_empty_num];
    int Eobs_flatten_nonan[not_empty_num];
    matrix[N,T] sample_weights;
    real sample_weights2[not_empty_num];
    matrix[N,T] Pobs;//
    matrix[N,ND] D[T];
    int NClust;
    int cluster[N];
}

parameters {    
    real<lower=0.01> sigma_a0[NClust];
    real<lower=0.01> sigma_a1[NClust];
    real<lower=0.01> sigma_theta[NClust];//
    real<lower=0.01> sigma_sigma_err[NClust];//
    vector<lower=0.01>[ND] sigma_b[NClust];
    vector<lower=0.01>[N] sigma_err;
    
    vector[N] a0;
    vector<lower=-0.01,upper=0.01>[N] a1;
    matrix<lower=-0.01,upper=0.01>[N, MA_T0] theta;//
    matrix<lower=0>[N,ND] b;
}

model {
    matrix[N,T] A;
    matrix[N,T] err;//
    vector[N*(T-AR_T0)] A_flatten;
    vector[not_empty_num] A_flatten_nonan;
    vector[ND] ones_b;
    vector[N] tmp2;
    
    for (i in 1:N){
        a0[i] ~ normal(0, sigma_a0[cluster[i]]);
        a1[i] ~ normal(0, sigma_a1[cluster[i]]);
        theta[i] ~ normal(0, sigma_theta[cluster[i]]);//
        sigma_err[i] ~ cauchy(0, sigma_sigma_err[cluster[i]]);
    }
    for (j in 1:ND) {
        for (i in 1:N)
            b[i,j] ~ normal(0, sigma_b[cluster[i]][j]);
    }
    
    ones_b = rep_vector(1, ND);
    for (t in 1:T) {
        // AR(1) only, TODO code for general AR(p), but don't know how to code stationery constraints in general
        if (t<=1)
            A[:,t] = logit(Pobs[:,t]);
        else
            A[:,t] = a0 + a1 .* A[:,t-1];
            
        // MA(q)
        tmp2 = rep_vector(0, N);
        for (q in 1:MA_T0) {
            if (t-q>0)
                tmp2 += theta[:,q] .* err[:,t-q];
        }
        A[:,t] = A[:,t] + tmp2 - (D[t] .* b)*ones_b;
        
        for (i in 1:N) {
            if (Pobs[i,t]>0) {  // not missing at time t
                err[i,t] = logit(Pobs[i,t]) - A[i,t];
                //err[i,t] ~ normal(0, sigma_err[i]);
                target += normal_lpdf(err[i,t] | 0, sigma_err[i]) * sample_weights[i,t];
            }
            else              // missing at time t, carry error forward
                err[i,t] = err[i,t-1];
        }
    }
    
    A_flatten = to_vector(A[:,AR_T0+1:]'); // to_vector is flattening p in column-major order, but we need row-major order, so transpose it
    A_flatten_nonan = A_flatten[not_empty_ids];
    
    for (i in 1:not_empty_num) {
        target += binomial_logit_lpmf(Eobs_flatten_nonan[i] | W, A_flatten_nonan[i]) * sample_weights2[i]; // not_empty_num;
    }
}
