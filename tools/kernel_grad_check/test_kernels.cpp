// Exhaustive finite-difference check of every hand-coded kernel gradient in
// batch_sdt_core.hpp (config-independent; no Stan/MCMC). Compares analytic
// gradients to central finite differences of the kernel's own value/lp.
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include "batch_sdt_core.hpp"
using namespace batch_sdt;

static int g_bad = 0, g_tot = 0;
static void chk(const char* fn, const char* nm, double ana, double fd) {
  double e = std::fabs(ana - fd); g_tot++;
  bool bad = e > 1e-4 && e > 1e-4 * (std::fabs(fd) + 1.0);
  if (bad) { g_bad++; printf("  %-26s %-14s ana=% .6g fd=% .6g err=%.2e  <<< BAD\n", fn, nm, ana, fd, e); }
}
// central finite diff of a scalar->scalar value function
static double cfd(std::function<double(double)> f, double x, double h=1e-6) {
  return (f(x+h) - f(x-h)) / (2*h);
}

int main() {
  // ---------- cell_log_prob (probit) : lp wrt z_lo, z_hi ----------
  { double zl=-0.4, zh=0.7; auto r=cell_log_prob(zl,zh);
    chk("cell_log_prob","d_z_lo",r.d_z_lo, cfd([&](double x){return cell_log_prob(x,zh).lp;},zl));
    chk("cell_log_prob","d_z_hi",r.d_z_hi, cfd([&](double x){return cell_log_prob(zl,x).lp;},zh)); }
  // ---------- cell_log_prob_logit : lp wrt z_lo, z_hi ----------
  { double zl=-0.4, zh=0.7; auto r=cell_log_prob_logit(zl,zh);
    chk("cell_log_prob_logit","d_z_lo",r.d_z_lo, cfd([&](double x){return cell_log_prob_logit(x,zh).lp;},zl));
    chk("cell_log_prob_logit","d_z_hi",r.d_z_hi, cfd([&](double x){return cell_log_prob_logit(zl,x).lp;},zh)); }
  // ---------- cell_prob (raw p) : p wrt z_lo, z_hi ----------
  { double zl=-0.4, zh=0.7; auto r=cell_prob(zl,zh);
    chk("cell_prob","d_z_lo",r.d_z_lo, cfd([&](double x){return cell_prob(x,zh).p;},zl));
    chk("cell_prob","d_z_hi",r.d_z_hi, cfd([&](double x){return cell_prob(zl,x).p;},zh)); }
  // ---------- binormal_cdf_grad : val wrt z1, z2, rho  [FOUNDATIONAL] ----------
  for (double rho : {-0.6,-0.2,0.3,0.7}) for (double z1 : {-1.0,0.5}) for (double z2 : {-0.3,0.8}) {
    auto r=binormal_cdf_grad(z1,z2,rho);
    chk("binormal_cdf_grad","d_z1",r.d_z1, cfd([&](double x){return binormal_cdf_grad(x,z2,rho).val;},z1));
    chk("binormal_cdf_grad","d_z2",r.d_z2, cfd([&](double x){return binormal_cdf_grad(z1,x,rho).val;},z2));
    chk("binormal_cdf_grad","d_rho",r.d_rho, cfd([&](double x){return binormal_cdf_grad(z1,z2,x).val;},rho)); }
  // ---------- uni_cell : p wrt mu, sigma, thresh ----------
  { double t[5]={-1.0,-0.3,0.2,0.7,1.4}; int K=6,n=5;
    for (int y : {1,3,6}) { double mu=0.4, sg=1.2; auto r=uni_cell(y,K,mu,sg,t,n);
      chk("uni_cell","d_mu",r.d_mu, cfd([&](double x){return uni_cell(y,K,x,sg,t,n).p;},mu));
      chk("uni_cell","d_sigma",r.d_sigma, cfd([&](double x){return uni_cell(y,K,mu,x,t,n).p;},sg));
      int klo=(y>1)?y-2:-1, khi=(y<K)?y-1:-1;
      if(klo>=0){double sv=t[klo]; auto f=[&](double x){double tt[5];for(int i=0;i<5;i++)tt[i]=t[i];tt[klo]=x;return uni_cell(y,K,mu,sg,tt,n).p;}; chk("uni_cell","d_thr_lo",r.d_thresh_lo,cfd(f,sv));}
      if(khi>=0){double sv=t[khi]; auto f=[&](double x){double tt[5];for(int i=0;i<5;i++)tt[i]=t[i];tt[khi]=x;return uni_cell(y,K,mu,sg,tt,n).p;}; chk("uni_cell","d_thr_hi",r.d_thresh_hi,cfd(f,sv));} } }
  // ---------- bivariate_cell : lp wrt z1_lo,z1_hi,z2_lo,z2_hi,rho ----------
  for (double rho : {-0.4,0.3,0.6}) { double a=-0.6,b=0.5,c=-0.3,d=0.7; auto r=bivariate_cell(a,b,false,false,c,d,false,false,rho);
    chk("bivariate_cell","d_z1_lo",r.d_z1_lo, cfd([&](double x){return bivariate_cell(x,b,false,false,c,d,false,false,rho).lp;},a));
    chk("bivariate_cell","d_z1_hi",r.d_z1_hi, cfd([&](double x){return bivariate_cell(a,x,false,false,c,d,false,false,rho).lp;},b));
    chk("bivariate_cell","d_z2_lo",r.d_z2_lo, cfd([&](double x){return bivariate_cell(a,b,false,false,x,d,false,false,rho).lp;},c));
    chk("bivariate_cell","d_z2_hi",r.d_z2_hi, cfd([&](double x){return bivariate_cell(a,b,false,false,c,x,false,false,rho).lp;},d));
    chk("bivariate_cell","d_rho",r.d_rho, cfd([&](double x){return bivariate_cell(a,b,false,false,c,d,false,false,x).lp;},rho)); }
  // ---------- bivariate_sdt_cell : lp wrt dprime,discrim,sigma1,sigma2,rho ----------
  { double t1[5]={-1.0,-0.3,0.2,0.7,1.4}, t2[5]={-0.9,-0.2,0.3,0.6,1.3}; int K1=6,K2=6,n1=5,n2=5;
    for (int y1 : {2,5}) for (int y2 : {3,6}) { double mu1=0.8,mu2=0.5,s1=1.1,s2=0.9,rho=0.3; auto r=bivariate_sdt_cell(y1,y2,K1,K2,mu1,mu2,s1,s2,rho,t1,n1,t2,n2);
      chk("bivariate_sdt_cell","d_dprime",r.d_dprime, cfd([&](double x){return bivariate_sdt_cell(y1,y2,K1,K2,x,mu2,s1,s2,rho,t1,n1,t2,n2).lp;},mu1));
      chk("bivariate_sdt_cell","d_discrim",r.d_discrim, cfd([&](double x){return bivariate_sdt_cell(y1,y2,K1,K2,mu1,x,s1,s2,rho,t1,n1,t2,n2).lp;},mu2));
      chk("bivariate_sdt_cell","d_sigma1",r.d_sigma1, cfd([&](double x){return bivariate_sdt_cell(y1,y2,K1,K2,mu1,mu2,x,s2,rho,t1,n1,t2,n2).lp;},s1));
      chk("bivariate_sdt_cell","d_sigma2",r.d_sigma2, cfd([&](double x){return bivariate_sdt_cell(y1,y2,K1,K2,mu1,mu2,s1,x,rho,t1,n1,t2,n2).lp;},s2));
      chk("bivariate_sdt_cell","d_rho",r.d_rho, cfd([&](double x){return bivariate_sdt_cell(y1,y2,K1,K2,mu1,mu2,s1,s2,x,t1,n1,t2,n2).lp;},rho)); } }
  // ---------- bounded_marginal_source : p wrt mu2,sigma2,rho (re-verify d_rho fix) ----------
  { double t2[5]={-0.9,-0.2,0.3,0.6,1.3}; int K2=6,n2=5;
    for (int y2 : {1,4,6}) for (double rho : {0.3,0.6}) { double mu1=1.0,mu2=0.5,s1=1.1,s2=0.9; auto r=bounded_marginal_source(y2,K2,mu1,mu2,s1,s2,rho,t2,n2);
      chk("bounded_marg_source","d_mu2",r.d_mu2, cfd([&](double x){return bounded_marginal_source(y2,K2,mu1,x,s1,s2,rho,t2,n2).p;},mu2));
      chk("bounded_marg_source","d_sigma2",r.d_sigma2, cfd([&](double x){return bounded_marginal_source(y2,K2,mu1,mu2,s1,x,rho,t2,n2).p;},s2));
      chk("bounded_marg_source","d_rho",r.d_rho, cfd([&](double x){return bounded_marginal_source(y2,K2,mu1,mu2,s1,s2,x,t2,n2).p;},rho)); } }
  // ---------- dpsdt_top_cell : lp wrt z, lambda ----------
  { double z=0.6, lam=0.35; auto r=dpsdt_top_cell(z,lam);
    chk("dpsdt_top_cell","d_z",r.d_z, cfd([&](double x){return dpsdt_top_cell(x,lam).lp;},z));
    chk("dpsdt_top_cell","d_lambda",r.d_lambda, cfd([&](double x){return dpsdt_top_cell(z,x).lp;},lam)); }
  // ---------- dpsdt_non_top_cell : lp wrt z_lo, z_hi, lambda ----------
  { double zl=-0.3, zh=0.5, lam=0.35; int K=6; for (int y : {1,3}) { auto r=dpsdt_non_top_cell(zl,zh,lam,y,K);
    if(y>1) chk("dpsdt_non_top","d_z_lo",r.d_z_lo, cfd([&](double x){return dpsdt_non_top_cell(x,zh,lam,y,K).lp;},zl));
    chk("dpsdt_non_top","d_z_hi",r.d_z_hi, cfd([&](double x){return dpsdt_non_top_cell(zl,x,lam,y,K).lp;},zh));
    chk("dpsdt_non_top","d_lambda",r.d_lambda, cfd([&](double x){return dpsdt_non_top_cell(zl,zh,x,y,K).lp;},lam)); } }
  // ---------- mixture_cell : lp wrt z1_lo,z1_hi,z2_lo,z2_hi,lambda ----------
  { double a=-0.3,b=0.5,c=-0.5,d=0.4,lam=0.4; int K=6,y=3; auto r=mixture_cell(a,b,c,d,lam,y,K);
    chk("mixture_cell","d_z1_lo",r.d_z1_lo, cfd([&](double x){return mixture_cell(x,b,c,d,lam,y,K).lp;},a));
    chk("mixture_cell","d_z1_hi",r.d_z1_hi, cfd([&](double x){return mixture_cell(a,x,c,d,lam,y,K).lp;},b));
    chk("mixture_cell","d_z2_lo",r.d_z2_lo, cfd([&](double x){return mixture_cell(a,b,x,d,lam,y,K).lp;},c));
    chk("mixture_cell","d_z2_hi",r.d_z2_hi, cfd([&](double x){return mixture_cell(a,b,c,x,lam,y,K).lp;},d));
    chk("mixture_cell","d_lambda",r.d_lambda, cfd([&](double x){return mixture_cell(a,b,c,d,x,y,K).lp;},lam)); }
  // ---------- vrdp2d_cell : lp wrt dprime_F,dprime_R,source_d,R,sigma_item,sigma_S ----------
  { double t1[5]={-1.0,-0.3,0.2,0.7,1.4}, t2[5]={-0.9,-0.2,0.3,0.6,1.3}; int K1=6,K2=6,n1=5,n2=5;
    for (int it : {1,2,3}) for (int y1 : {3,6}) { int y2=4; double dF=0.7,dR=0.5,sd=0.6,R=0.4,si=1.1,sS=0.9; auto r=vrdp2d_cell(y1,y2,it,K1,K2,dF,dR,sd,R,si,sS,t1,n1,t2,n2);
      chk("vrdp2d_cell","d_dprime_F",r.d_dprime_F, cfd([&](double x){return vrdp2d_cell(y1,y2,it,K1,K2,x,dR,sd,R,si,sS,t1,n1,t2,n2).lp;},dF));
      chk("vrdp2d_cell","d_dprime_R",r.d_dprime_R, cfd([&](double x){return vrdp2d_cell(y1,y2,it,K1,K2,dF,x,sd,R,si,sS,t1,n1,t2,n2).lp;},dR));
      chk("vrdp2d_cell","d_source_d",r.d_source_d, cfd([&](double x){return vrdp2d_cell(y1,y2,it,K1,K2,dF,dR,x,R,si,sS,t1,n1,t2,n2).lp;},sd));
      chk("vrdp2d_cell","d_lambda",r.d_lambda, cfd([&](double x){return vrdp2d_cell(y1,y2,it,K1,K2,dF,dR,sd,x,si,sS,t1,n1,t2,n2).lp;},R));
      chk("vrdp2d_cell","d_sigma_item",r.d_sigma_item, cfd([&](double x){return vrdp2d_cell(y1,y2,it,K1,K2,dF,dR,sd,R,x,sS,t1,n1,t2,n2).lp;},si));
      chk("vrdp2d_cell","d_sigma_S",r.d_sigma_S, cfd([&](double x){return vrdp2d_cell(y1,y2,it,K1,K2,dF,dR,sd,R,si,x,t1,n1,t2,n2).lp;},sS)); } }
  // ---------- cdp_strip_upper : p wrt z_cR, z_lo, z_hi, rho ----------
  for (double rho : {0.3,0.6}) { double zc=0.4,zl=-0.5,zh=0.8; auto r=cdp_strip_upper(zc,zl,zh,rho);
    chk("cdp_strip_upper","d_z_cR",r.d_z_cR, cfd([&](double x){return cdp_strip_upper(x,zl,zh,rho).p;},zc));
    chk("cdp_strip_upper","d_z_lo",r.d_z_lo, cfd([&](double x){return cdp_strip_upper(zc,x,zh,rho).p;},zl));
    chk("cdp_strip_upper","d_z_hi",r.d_z_hi, cfd([&](double x){return cdp_strip_upper(zc,zl,x,rho).p;},zh));
    chk("cdp_strip_upper","d_rho",r.d_rho, cfd([&](double x){return cdp_strip_upper(zc,zl,zh,x).p;},rho)); }

  // ---------- bivariate_dp_cell : full mixture (fast/slow/corner, item 2/3/new) ----------
  { double t1[5]={-1.0,-0.3,0.2,0.7,1.4}, t2[5]={-0.9,-0.2,0.3,0.6,1.3}; int K1=6,K2=6,n1=5,n2=5;
    struct C{int y1,y2,it;}; C cs[]={{3,4,2},{6,4,2},{6,1,2},{6,6,3},{6,3,3},{2,5,1}};
    for(auto&c:cs){ double mu1=1.1,mu2=0.6,s1=1.2,s2=0.9,rho=0.4,RI=0.35,RS=0.5;
      auto bd=[&](double a,double b,double e,double f,double g,double h,double i,double j,double k){return bivariate_dp_cell(c.y1,c.y2,c.it,K1,K2,a,b,e,f,g,h,i,j,k,t1,n1,t2,n2).lp;};
      auto r=bivariate_dp_cell(c.y1,c.y2,c.it,K1,K2,mu1,mu2,s1,s2,rho,RI,RS,RI,RS,t1,n1,t2,n2);
      chk("bivariate_dp_cell","d_dprime",r.d_dprime,cfd([&](double x){return bd(x,mu2,s1,s2,rho,RI,RS,RI,RS);},mu1));
      chk("bivariate_dp_cell","d_discrim",r.d_discrim,cfd([&](double x){return bd(mu1,x,s1,s2,rho,RI,RS,RI,RS);},mu2));
      chk("bivariate_dp_cell","d_sigma2",r.d_sigma2,cfd([&](double x){return bd(mu1,mu2,s1,x,rho,RI,RS,RI,RS);},s2));
      chk("bivariate_dp_cell","d_rho",r.d_rho,cfd([&](double x){return bd(mu1,mu2,s1,s2,x,RI,RS,RI,RS);},rho));
      if(c.it==2){chk("bivariate_dp_cell","d_lambda",r.d_lambda,cfd([&](double x){return bd(mu1,mu2,s1,s2,rho,x,RS,RI,RS);},RI));
                  chk("bivariate_dp_cell","d_lambda2",r.d_lambda2,cfd([&](double x){return bd(mu1,mu2,s1,s2,rho,RI,x,RI,RS);},RS));}
      else if(c.it==3){chk("bivariate_dp_cell","d_lambda_B",r.d_lambda_B,cfd([&](double x){return bd(mu1,mu2,s1,s2,rho,RI,RS,x,RS);},RI));
                  chk("bivariate_dp_cell","d_lambda2_B",r.d_lambda2_B,cfd([&](double x){return bd(mu1,mu2,s1,s2,rho,RI,RS,RI,x);},RS));} } }
  // ---------- bounded_bivariate_sdt_cell ----------
  { double t1[5]={-1.2,-0.5,0.1,0.7,1.4}, t2[5]={-1.0,-0.3,0.2,0.6,1.3}; int K1=6,K2=6,n1=5,n2=5;
    for(int y1:{3,6})for(int y2:{2,4}){ double mu1=1.1,mu2=0.6,s1=1.2,s2=0.9,rho=0.4;
      auto bb=[&](double a,double b,double f,double g){return bounded_bivariate_sdt_cell(y1,y2,K1,K2,a,b,s1,f,g,t1,n1,t2,n2).lp;};
      auto r=bounded_bivariate_sdt_cell(y1,y2,K1,K2,mu1,mu2,s1,s2,rho,t1,n1,t2,n2);
      chk("bounded_biv_sdt","d_dprime",r.d_dprime,cfd([&](double x){return bb(x,mu2,s2,rho);},mu1));
      chk("bounded_biv_sdt","d_discrim",r.d_discrim,cfd([&](double x){return bb(mu1,x,s2,rho);},mu2));
      chk("bounded_biv_sdt","d_sigma2",r.d_sigma2,cfd([&](double x){return bb(mu1,mu2,x,rho);},s2));
      chk("bounded_biv_sdt","d_rho",r.d_rho,cfd([&](double x){return bb(mu1,mu2,s2,x);},rho)); } }
  // ---------- bounded_bivariate_dp_cell ----------
  { double t1[5]={-1.2,-0.5,0.1,0.7,1.4}, t2[5]={-1.0,-0.3,0.2,0.6,1.3}; int K1=6,K2=6,n1=5,n2=5;
    struct C{int y1,y2,it;}; C cs[]={{6,4,2},{6,1,2},{6,6,3},{3,4,2}};
    for(auto&c:cs){ double mu1=1.1,mu2=0.6,s1=1.2,s2=0.9,rho=0.4,RI=0.35,RS=0.5;
      auto bd=[&](double a,double b,double e,double f,double g){return bounded_bivariate_dp_cell(c.y1,c.y2,c.it,K1,K2,a,b,e,f,g,RI,RS,RI,RS,t1,n1,t2,n2).lp;};
      auto r=bounded_bivariate_dp_cell(c.y1,c.y2,c.it,K1,K2,mu1,mu2,s1,s2,rho,RI,RS,RI,RS,t1,n1,t2,n2);
      chk("bounded_biv_dp","d_dprime",r.d_dprime,cfd([&](double x){return bd(x,mu2,s1,s2,rho);},mu1));
      chk("bounded_biv_dp","d_discrim",r.d_discrim,cfd([&](double x){return bd(mu1,x,s1,s2,rho);},mu2));
      chk("bounded_biv_dp","d_sigma2",r.d_sigma2,cfd([&](double x){return bd(mu1,mu2,s1,x,rho);},s2));
      chk("bounded_biv_dp","d_rho",r.d_rho,cfd([&](double x){return bd(mu1,mu2,s1,s2,x);},rho)); } }

  printf("\nKERNEL FINITE-DIFF: %d checks, %d BAD\n", g_tot, g_bad);
  return g_bad ? 1 : 0;
}
