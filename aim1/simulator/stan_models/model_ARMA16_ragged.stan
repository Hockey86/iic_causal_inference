data {
    int W;
    int N;
    int AR_T0;  // p, AR init time step (order)
    int MA_T0;  // q, MA init time step (order)
    int ND;
    
    int not_empty_num;
    int Eobs_flatten_nonan[not_empty_num];
    vector[not_empty_num] Aobs_flatten_nonan;
    real sample_weights[not_empty_num];
    int not_empty_ids[not_empty_num]; 
    int patient_nonnan_len[N];
    //int cum_patient_nonnan_len[N];
    vector[ND] D[not_empty_num];
       
    int NClust;
    int cluster[N];
}

parameters {
    real<lower=0.001> sigma_a0;
    real<lower=0.001> sigma_a1;
    
    vector[N] a0;
    vector<lower=-0.01,upper=0.01>[N] a1;
    vector<lower=0>[ND] b[N];
    
    real<lower=0.001> sigma_theta;
    matrix<lower=-0.01,upper=0.01>[N, MA_T0] theta;
}

model {
    vector[not_empty_num] A;
    vector[not_empty_num] err;
    real tmp1;
    real tmp2;
    int pos;
    vector[ND] ones_b;
    ones_b = rep_vector(1, ND);
    
    for (i in 1:N){
        a0[i] ~ normal(0,sigma_a0);
        a1[i] ~ normal(0,sigma_a1);
        theta[i] ~ normal(0, sigma_theta);
        //b[i] ~ normal();
    }
    
    pos = 0;
    for (i in 1:N){
        for (j in 1:patient_nonnan_len[i]){
            // AR(1) only, TODO code for general AR(p), but don't know how to code stationery constraints in general
            if (j-1==0)
                tmp1 = a0[i];
            else
                tmp1 = Aobs_flatten_nonan[pos+j-1];

            // MA(q)
            tmp2 = 0;
            for (t in 1:MA_T0) {
                if (j-t>0)
                    tmp2 += theta[i,t]*err[pos+j-t];
            }

            A[pos+j] =  a0[i] + a1[i] * tmp1 + tmp2 - (D[pos+j] .* b[i])' * ones_b;
            err[pos+j] = Aobs_flatten_nonan[pos+j]  - A[pos+j];
            //print("A[", i, ",", j, "] = ",A[pos+j]);
            //print("err[", i, ",", j, "] = ",err[pos+j]);

            target += binomial_logit_lpmf(Eobs_flatten_nonan[pos+j] |W, A[pos+j] ) * sample_weights[pos+j];
        }
        pos += patient_nonnan_len[i];
    }
}

