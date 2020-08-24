data {
    int W;
    int N;
    int T;
    int ND;
    int not_empty_num;
    int not_empty_ids[not_empty_num];
    int Eobs_flatten_nonan[not_empty_num];
    real sample_weights[not_empty_num];
    matrix[N,ND] D[T];
}

parameters {
    real mu_mu;
    real<lower=0> sigma_mu;
    real<lower=0> sigma_sigma;
    real<lower=0> sigma_alpha;
    vector<lower=0>[ND] sigma_b;
    
    //vector<lower=0, upper=336>[N] t0;  // max 7 days (7*24*2) (0.5h/step)
    vector<upper=5.8>[N] mu;  // max 7 days ln(7*24*2) (0.5h/step)
    vector<lower=0>[N] sigma;
    vector<lower=0>[N] alpha;
    matrix<lower=0>[N,ND] b;
    
    //real<lower=0> sigma_epsilon;
}

model {
    matrix[N,T] A;
    matrix[N,T] p;
    
    vector[N*T] p_flatten;
    vector[not_empty_num] p_flatten_nonan;
    
    vector[N] mu_mu_vec;
    vector[N] sigma_mu_vec;
    vector[N] sigma_sigma_vec;
    vector[N] sigma_alpha_vec;
    vector[N] sigma_b_vec;
    
    vector[ND] ones_b;
    vector[N] zeros;
    vector[N] tmp;
    ones_b = rep_vector(1, ND);
    zeros = rep_vector(0, N);
    
    mu_mu_vec = rep_vector(mu_mu, N);
    sigma_mu_vec = rep_vector(sigma_mu, N);
    mu ~ normal(mu_mu_vec, sigma_mu_vec);
    sigma_sigma_vec = rep_vector(sigma_sigma, N);
    sigma ~ normal(0, sigma_sigma_vec);
    sigma_alpha_vec = rep_vector(sigma_alpha, N);
    alpha ~ normal(0, sigma_alpha_vec);
    for (i in 1:ND) {
        sigma_b_vec = rep_vector(sigma_b[i], N);
        b[:,i] ~ normal(0, sigma_b_vec);
    }
    //sigma_epsilon ~ normal(0, 0.01);
    
    for (t in 1:T) {
        tmp = log(t)-mu;
        A[:,t] = alpha .* exp(-(tmp .* tmp) ./ (2* (sigma .* sigma))) - (D[t] .* b)*ones_b;
    }
    
    // clip At
    for (t in 1:T) {
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
    
    p_flatten = to_vector(p'); // to_vector is flattening p in column-major order, but we need row-major order, so transpose it
    p_flatten_nonan = p_flatten[not_empty_ids];
    
    //Eobs_flatten_nonan ~ binomial(W, p_flatten_nonan);
    for (i in 1:not_empty_num) {
        target += binomial_lpmf(Eobs_flatten_nonan[i] | W, p_flatten_nonan[i]) * sample_weights[i];// not_empty_num;
    }
}
