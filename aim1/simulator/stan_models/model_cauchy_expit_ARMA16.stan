data {
    int W;
    int N;
    int AR_p;  // AR init time step (order)
    int MA_q;  // MA init time step (order)
    int ND;
    
    int total_len;
    int patient_lens[N];
    int Eobs[total_len];
    real f_Eobs[total_len];
    real sample_weights[total_len];
    vector[ND] D[total_len];
       
    int NClust;
    int cluster[N];
    
    real loss_weight;
}

parameters {
    real<lower=0.001> sigma_alpha0[NClust];
    real<lower=0.001> sigma_alpha[NClust];
    vector<lower=0.01>[ND] sigma_b[NClust];
    
    vector[N] alpha0;
    matrix<lower=-0.999,upper=0.999>[N, AR_p] alpha;// TODO general stationery constraint for AR(p)?
    vector<lower=0>[ND] b[N];
    vector<lower=-0.999,upper=0.999>[MA_q] theta;//[NClust]
    real<lower=0.001> sigma_err[NClust];
}

model {
    vector[total_len] A;
    vector[total_len] err;
    vector[ND] ones_b;
    real tmp1;
    int pos;
    ones_b = rep_vector(1, ND);
    
    for (i in 1:N){
        alpha0[i] ~ normal(0, sigma_alpha0[cluster[i]]);
        alpha[i] ~ normal(0, sigma_alpha[cluster[i]]);
    }
    for (j in 1:ND) {
        for (i in 1:N)
            b[i,j] ~ normal(0, sigma_b[cluster[i]][j]);
    }
    
    pos = 0;
    for (i in 1:N){
        for (j in 1:patient_lens[i]){
            if (j<=AR_p) {
                A[pos+j] = f_Eobs[pos+j];
                err[pos+j] = 0;
                continue;
            }
            
            // AR(p)
            tmp1 = 0;
            for (t in 1:AR_p) {
                if (Eobs[pos+j-t]<0)  // miss data
                    tmp1 += alpha[i,t]*A[pos+j-t];
                else                  // not miss data
                    tmp1 += alpha[i,t]*f_Eobs[pos+j-t];
            }
            A[pos+j] = alpha0[i] + tmp1;
            //print("alpha0[", i, "] = ",alpha0[i]);
            //print("AR = ",tmp1);
            //print("after AR, A[", i, ",", j, "] = ",A[pos+j]);

            // MA(q)
            tmp1 = 0;
            for (t in 1:MA_q) {
                if (j>t)
                    tmp1 += theta[t]*err[pos+j-t];
            }
            A[pos+j] += tmp1;
            //print("MA = ",tmp1);
            //print("after MA, A[", i, ",", j, "] = ",A[pos+j]);
            
            // drug
            A[pos+j] -= (D[pos+j] .* b[i])' * ones_b;
            //print("after drug, A[", i, ",", j, "] = ",A[pos+j]);
            
            if (Eobs[pos+j]<0)  // miss data
                err[pos+j] = 0;
            else {              // not miss data
                err[pos+j] = f_Eobs[pos+j] - A[pos+j];
                target += cauchy_lpdf(f_Eobs[pos+j] | A[pos+j], sigma_err[cluster[i]]) * sample_weights[pos+j] * loss_weight;
            }
            
            //print("A[", i, ",", j, "] = ",A[pos+j]);
            //print("err[", i, ",", j, "] = ",err[pos+j]);

            if (Eobs[pos+j]>=0) { // not miss data
                target += binomial_logit_lpmf(Eobs[pos+j] | W, A[pos+j] ) * sample_weights[pos+j];
            }
        }
        pos += patient_lens[i];
    }
}

