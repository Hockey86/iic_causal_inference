data {
    int W;  
    int N;   //number of patients
    int T;
    int T0;  // initialization time step
    int ND;
    int not_empty_num;
    int not_empty_ids[not_empty_num];
    int Eobs_flatten_nonan[not_empty_num];
    real sample_weights[not_empty_num];
    matrix[N,ND] D[T];
    matrix[N,T0] A_start;
    //int cluster[N];
    //int NClust;
}

parameters {
    real mu_a0;
    real mu_a1;
    //vector[ND] mu_b;
    real mu_b;
    
    real<lower=0> sigma_a0;
    real<lower=0> sigma_a1;
    //vector<lower=0>[ND] sigma_b;
    real<lower=0> sigma_b;
    
    vector[N] a0;
    vector<lower=-0.95,upper=0.95>[N] a1;
    vector[ND] b;
    //matrix<lower=0>[N,ND] b;
}


model {
    matrix[N,T] A;
    matrix[N,T] p;
    vector[N*(T-T0)] p_flatten;
    vector[not_empty_num] p_flatten_nonan;
    
    vector[ND] ones_b;
    

    a0 ~ normal(mu_a0, sigma_a0);
    a1 ~ normal(mu_a1, sigma_a1);
    b  ~ normal(mu_b, sigma_b);
    
    //for (i in 1:ND) {
    //    mu_b_vec = rep_vector(mu_b[i], N);
    //    sigma_b_vec = rep_vector(sigma_b[i], N);
    //    b[i] ~ normal(mu_b_vec, sigma_b_vec);
    //}
    
    
    
    for (t in 1:T0) { A[:,t] = A_start[:,t]; }


    
    for (t in T0+1:T) {
        
       // A[:,t] = a0 + a1 .* A[:,t-1] - (D[t] .* b)*ones_b;;//
        A[:,t] = a0 + a1 .* A[:,t-1] - (D[t] * b);
       // A[:,t] = a0 + a1 .* A[:,t-1];
    }
    
    //for (i in 1:N){
    //    for (j in (T0+1):T){
    //        current_D = D[j];
    //        cd = current_D[i,:];
     //       b .* cd[j];
    //        A[i,j] = a0[i] + a1[i] *A[i,j-1]  ;
    //    }
    //}
    

    
    p = inv_logit(A);
    p_flatten = to_vector(p[:,T0+1:]'); // to_vector is flattening p in column-major order, but we need row-major order, so transpose it
    p_flatten_nonan = p_flatten[not_empty_ids];
    
    
    Eobs_flatten_nonan ~ binomial(W, p_flatten_nonan);
    //for (i in 1:not_empty_num) {
    //    target += binomial_lpmf(Eobs_flatten_nonan[i] | W, p_flatten_nonan[i]);
    //}
}

