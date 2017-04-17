
/*
 * Author: Ao Zhang
 *
 * Resolve following linear constrainted optimization problem:
 *
 *      min ||a - a0||^2 + ||q - q0||^2 
 *      s.t.    a >= 0,  q >= 0, sigma_a = sigma_q, q <= sigma_q / K.
 *
 * Input:  a0, q0, K
 * Output:  a, q
 */

#include <math.h>
#include "mex.h"
#include <string.h>

double preTrainingC(double* a0, double* q0, mwIndex m, mwIndex n)
{
    int nga=0, nla=0, nea=0, ngq=0, nlq=0, neq=0, nua=m, nuq=n, i, j, k=0, l=0;
    double *vga=(double*)mxMalloc(sizeof(double)*m);
    double *vla=(double*)mxMalloc(sizeof(double)*m);
    double *vgq=(double*)mxMalloc(sizeof(double)*n);
    double *vlq=(double*)mxMalloc(sizeof(double)*n);
    double *vua=a0, *vuq=q0;
    double sa=0.0, sq=0.0, sq0=0.0, rho, val, dsa, dsq, df, C;
    
    for(i=0;i<n;i++){// compute the sum of q0
        q0[i]=-q0[i];    
        sq0+=q0[i]; 
    }
    while(nua+nuq>0){
        if (nua>0) rho=vua[0]; else rho=vuq[0];    //select the threshold
        dsa=0.0; dsq=0.0;
        nga=0; nla=0; nea=0;
        ngq=0; nlq=0; neq=0;
        for(i=0;i<nua;i++){   // split the vector a
            val=vua[i];
            if (val>rho)        {vga[nga]=val; nga++; dsa+=val;} 
            else if (val<rho)   {vla[nla]=val; nla++; }
            else                {dsa+=val; nea++;} 
        }
        for(i=0;i<nuq;i++){   // split the vector q
            val=vuq[i];
            if (val>rho)        {vgq[ngq]=val; ngq++; dsq+=val;} 
            else if (val<rho)   {vlq[nlq]=val; nlq++; }
        }
        df = sa+dsa+(sq0-sq-dsq)-(k+nga+nea)*rho-(n-l-ngq)*rho;        
        if (df<0){
            vua=vla;  nua=nla;  sa+=dsa; k+=(nga+nea);
            vuq=vlq;  nuq=nlq;  sq+=dsq; l+=ngq;
        }else{
            vua=vga;  nua=nga;  
            vuq=vgq;  nuq=ngq; 
        }   
    }
    rho = (sa+sq0-sq)/(k+n-l);
    
    C = 0;
    for(i = 0; i < m; i++) {
        val = a0[i] - rho;
        C += val > 0 ? val : 0;
    }
    
    for(i=0;i<n;i++) 
    {
        q0[i]=-q0[i];
    }
    
    mxFree(vga);mxFree(vla);mxFree(vgq);mxFree(vlq);
    return C;
}

double find_simple_root(double* a, mwIndex m, double* psa, int* px, double C)
{
    int debug = 0;
    
    //these parameters will be tuning and return
    double sa = *psa;
    int x = *px;
    
    double *vga=(double*)mxMalloc(sizeof(double)*m);
    double *vla=(double*)mxMalloc(sizeof(double)*m);   
    double *vua = a;
    
    int nga=0, nla=0, nea=0, nua=m, i;
    double rho, val, dsa, df;
    int iter = 0;
    
    while(nua > 0)
    {
        rho=vua[0];    //select the threshold
        dsa=0.0; 
        nga=0; nla=0; nea=0;
        
        for(i=0; i<nua; i++)
        {   // split the vector a
            val = vua[i];
            if (val > rho)        {vga[nga]=val; nga++; dsa+=val;} 
            else if (val < rho)   {vla[nla]=val; nla++; }
            else                {dsa+=val; nea++;} 
        }
        
        df = (dsa + sa) - (nea + nga + x)*rho - C;
        if(df > 0)
        {
            vua = vga;  nua = nga;
        } else {
            vua = vla; nua = nla;
            sa += dsa; 
            x += (nea + nga);  
        }
        
        iter ++;
        if(debug)
            printf("rho: %lf, df: %lf, nga: %d, nla: %d, nea: %d, nua: %d\n", rho, df, nga, nla, nea, nua);
    }
    
    if(debug)
        printf("iter: %d\n", iter);
    
    mxFree(vga);mxFree(vla); 
    
    //return parameters 
    *psa = sa;
    *px = x;
    
    return (sa - C)/x;
} 

double find_root(double* a, double* q, mwIndex m, mwIndex n, double* psa, double* psq, int* px, int* py, int k, double C)
{
    int debug = 0;
    
    //get value
    double sa = *psa;
    double sq = *psq;
    int x = *px;
    int y = *py;
    
    double *vga=(double*)mxMalloc(sizeof(double)*m);
    double *vla=(double*)mxMalloc(sizeof(double)*m);   
    double *vgq=(double*)mxMalloc(sizeof(double)*n);
    double *vlq=(double*)mxMalloc(sizeof(double)*n);
    
    double *vua = a, *vuq = q;
    int nga=0, nla=0, nea=0, ngq=0, nlq=0, neq=0, nua=m, nuq=n, i;
    double rho, val, dsa, dsq, df;
    int iter = 0;

    while(nua + nuq > 0)
    {
        if(nuq >0) rho=vuq[0]; else rho=vua[0] - C/k;    //select the threshold
        dsa=0.0; dsq=0.0;
        nga=0; nla=0; nea=0;
        ngq=0; nlq=0; neq=0;
        
        for(i=0; i<nua; i++)
        {   // split the vector a
            val = vua[i];
            if (val - C/k > rho)        {vga[nga]=val; nga++; dsa+=val;} 
            else if (val - C/k < rho)   {vla[nla]=val; nla++; }
            else {dsa+=val; nea++;} 
        }
        for(i=0; i<nuq; i++)
        {   // split the vector q
            val=vuq[i];
            if (val>rho)        {vgq[ngq]=val; ngq++; dsq+=val;} 
            else if (val<rho)   {vlq[nlq]=val; nlq++; }
            else                {dsq+=val; neq++;} 
        }
        
        df = (dsq + sq) - (dsa + sa) + (nea + nga + x - neq - ngq - y)*rho + (nea + nga + x)*C/k - C;
        if(df > 0)
        {
            vua = vga;  nua = nga;
            vuq = vgq; nuq = ngq; 
        } else {
            vua = vla; nua = nla;
            vuq = vlq; nuq = nlq;
            sa += dsa; sq += dsq;
            x += (nea + nga);
            y+= (neq + ngq);    
        }
        
        iter ++;
        if(debug)
        {
            printf("sa: %lf, sq: %lf, x: %d, y %d, C: %lf, k: %d, iter: %d\n", sa, sq, x, y, C, k, iter);
            printf("rho: %lf, df: %lf, nga: %d, nla: %d, nea: %d, ngq: %d, nlq: %d, neq: %d, nua: %d, nuq: %d\n", rho, df, nga, nla, nea, ngq, nlq, neq, nua, nuq);
        }
    }
    
    if(debug)
        printf("iter: %d\n", iter);
    
    mxFree(vga);mxFree(vla);mxFree(vgq);mxFree(vlq);
    
    //get value
    *psa = sa;
    *psq = sq;
    *px = x;
    *py = y;
    
    if (x == y)
    {
        return rho;
    } else{
        
        return (sq - sa + x*C/k - C)/(y - x);
    }   
}

static double computeProjection(double* a0, double* q0, double* a, double *q, mwIndex m, mwIndex n, int K)
{
    double step_sz = 2;
    double tol = 1e-5;
    double lb_C = 0;
    double ub_C = 1;
    int debug = 0;
    
    double lambda, mu, t, fC, mid, sum, val, C;
    //partial sum of three group of dual variables.
    double sa = 0, sq = 0, st = 0;
    int nsa = 0, nsq = 0, nst = 0;
    //list of dual variables we need to consider currently.
    double *ua = a0, *uq = q0, *ut = q0;
    mwIndex nua = m, nuq = n, nut = n;
    //temp cache of above variables. 
    double tsa, tsq, tst;
    int tnsa, tnsq, tnst;
    double *temp_ua=(double*)mxMalloc(sizeof(double)*m);   
    double *temp_uq=(double*)mxMalloc(sizeof(double)*n);
    double *temp_ut=(double*)mxMalloc(sizeof(double)*n);
    int t_nua, t_nuq, t_nut;
   
    int iter = 0, i;
    
    // find upper bound for C ( satisfies f(C) <=0 )
    ub_C = preTrainingC(a0, q0, m, n);
    while(1)
    {
        tsa = sa; tsq = sq; tst =st;
        tnsa = nsa; tnsq = nsq; tnst =nst;
        lambda = find_simple_root(ua, nua, &tsa, &tnsa, ub_C);
        mu = find_root(ut, uq, nut, nuq, &tst, &tsq, &tnst, &tnsq, K, ub_C);
        t = mu + ub_C/K;

        //calculate function value
        fC = lambda + mu + (tst - tnst*t)/K;
        
        //update lb and ub of C;
        //also update ua, uq and ut which we still need to explore later.
        //also update sa, sq, st, nsa, nsq, nst if we need to restore current partial sum. 
        if(fC > tol)
        {   
            //update lb and ub of C
            lb_C = ub_C;
            ub_C = ub_C * step_sz;
            
            //update ua, nua, sa, nsa.
            t_nua = 0;
            for(i = 0; i < nua; i++)
            {
                val = ua[i];
                if(val < lambda)  
                {
                    temp_ua[t_nua] = val;
                    t_nua ++;
                }
            }
            ua = temp_ua;
            nua = t_nua;
            sa = tsa;
            nsa = tnsa;
            
            //update uq, nuq, sq, nsq.
            t_nuq = 0;
            for(i = 0; i < nuq; i++)
            {
                val = uq[i];
                if(val < mu)  
                {
                    temp_uq[t_nuq] = val;
                    t_nuq ++;
                }
            }
            uq = temp_uq;
            nuq = t_nuq;
            sq = tsq;
            nsq = tnsq;
            
            //update ut, nut.
            t_nut = 0;
            for(i = 0; i < nut; i++)
            {
                val = ut[i];
                if(val > t)  
                {
                    temp_ut[t_nut] = val;
                    t_nut ++;
                }
            }
            ut = temp_ut;
            nut = t_nut;
        } else if (fC < -tol){
            //update ua, nua.
            t_nua = 0;
            for(i = 0; i < nua; i++)
            {
                val = ua[i];
                if(val > lambda)  
                {
                    temp_ua[t_nua] = val;
                    t_nua ++;
                }
            }
            ua = temp_ua;
            nua = t_nua;
            
            //update uq, nuq.
            t_nuq = 0;
            for(i = 0; i < nuq; i++)
            {
                val = uq[i];
                if(val > mu)  
                {
                    temp_uq[t_nuq] = val;
                    t_nuq ++;
                }
            }
            uq = temp_uq;
            nuq = t_nuq;
            
            //update ut, nut, st, nst.
            t_nut = 0;
            for(i = 0; i < nut; i++)
            {
                val = ut[i];
                if(val < t)  
                {
                    temp_ut[t_nut] = val;
                    t_nut ++;
                }
            }
            ut = temp_ut;
            nut = t_nut; 
            st = tst;
            nst = tnst;
            break;
        } else {
            lb_C = ub_C;
            break;
        }
        
        iter ++;
        if(debug)
        {
            printf("ub_C: %lf, lambda: %lf, mu: %lf, fC: %lf .\n", ub_C, lambda, mu, fC);
        }
    }

    // bisection method to find a root of equation fC = 0
    while(fabs(ub_C - lb_C) > tol)
    {
        if(debug) iter ++;
        
        mid = (lb_C + ub_C)/2;
        tsa = sa; tsq = sq; tst =st;
        tnsa = nsa; tnsq = nsq; tnst =nst;
        lambda = find_simple_root(ua, nua, &tsa, &tnsa, mid);
        mu = find_root(ut, uq, nut, nuq, &tst, &tsq, &tnst, &tnsq, K, mid);
        t = mu + mid/K;

        //calculate function value
        fC = lambda + mu + (tst - tnst*t)/K;
        
        //update lb and ub of C;
        //also update ua, uq and ut which we still need to explore later.
        //also update sa, sq, st, nsa, nsq, nst if we need to restore current partial sum. 
        if(fC > tol)
        {
            //update lb of C
            lb_C = mid;
            
            //update ua, nua, sa, nsa.
            t_nua = 0;
            for(i = 0; i < nua; i++)
            {
                val = ua[i];
                if(val < lambda)  
                {
                    temp_ua[t_nua] = val;
                    t_nua ++;
                }
            }
            ua = temp_ua;
            nua = t_nua;
            sa = tsa;
            nsa = tnsa;
            
            //update uq, nuq, sq, nsq.
            t_nuq = 0;
            for(i = 0; i < nuq; i++)
            {
                val = uq[i];
                if(val < mu)  
                {
                    temp_uq[t_nuq] = val;
                    t_nuq ++;
                }
            }
            uq = temp_uq;
            nuq = t_nuq;
            sq = tsq;
            nsq = tnsq;
            
            //update ut, nut.
            t_nut = 0;
            for(i = 0; i < nut; i++)
            {
                val = ut[i];
                if(val > t)  
                {
                    temp_ut[t_nut] = val;
                    t_nut ++;
                }
            }
            ut = temp_ut;
            nut = t_nut;
        } else if (fC < -tol){
            //update ub of C
            ub_C = mid;
            
            //update ua, nua.
            t_nua = 0;
            for(i = 0; i < nua; i++)
            {
                val = ua[i];
                if(val > lambda)  
                {
                    temp_ua[t_nua] = val;
                    t_nua ++;
                }
            }
            ua = temp_ua;
            nua = t_nua;
            
            //update uq, nuq.
            t_nuq = 0;
            for(i = 0; i < nuq; i++)
            {
                val = uq[i];
                if(val > mu)  
                {
                    temp_uq[t_nuq] = val;
                    t_nuq ++;
                }
            }
            uq = temp_uq;
            nuq = t_nuq;
            
            //update ut, nut, st, nst.
            t_nut = 0;
            for(i = 0; i < nut; i++)
            {
                val = ut[i];
                if(val < t)  
                {
                    temp_ut[t_nut] = val;
                    t_nut ++;
                }
            }
            ut = temp_ut;
            nut = t_nut; 
            st = tst;
            nst = tnst;
        } else {
            //update lb and ub of C
            lb_C = mid;
            ub_C = mid;
        }
    }

    C = (lb_C + ub_C)/2;
    // calculate a
    for(i = 0; i < m; i++) 
    {
        val = a0[i] - lambda;
        a[i] = val > 0? val: 0.0;
    }
    //calculate q
    for(i = 0; i<n; i++) 
    {
        val = q0[i] - mu;
        val = val > 0 ? val : 0.0;
        q[i] = val < C/K ? val : C/K;
    }
    
    mxFree(temp_ua); mxFree(temp_uq); mxFree(temp_ut);
    return C;
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
{
    double *p_a0, *p_q0, *p_a, *p_q;
    double C;
    int K;
    mwIndex m, n;
    
    m = mxGetM(prhs[0]);
    plhs[0] = mxCreateDoubleMatrix(m, (mwIndex)1, mxREAL);
    if (plhs[0]==NULL) {
        fprintf(stderr, "epne.c: Out of Memory!");
        return;
    }
    
    n = mxGetM(prhs[1]);
    plhs[1] = mxCreateDoubleMatrix(n, (mwIndex)1, mxREAL);
    if (plhs[1]==NULL) {
        fprintf(stderr, "epne.c: Out of Memory!");
        return;
    }
    
    p_a0 = mxGetPr(prhs[0]);
    p_q0 = mxGetPr(prhs[1]);
    p_a  = mxGetPr(plhs[0]);
    p_q  = mxGetPr(plhs[1]);
    K = mxGetScalar(prhs[2]); 
    
    C = computeProjection(p_a0, p_q0, p_a, p_q, m, n, K);
    plhs[2] = mxCreateDoubleScalar(C);
}
