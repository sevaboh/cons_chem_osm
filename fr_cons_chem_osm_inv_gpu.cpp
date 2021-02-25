#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <cfloat>
#include <sstream>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <sys/times.h>
#define OCL
#define USE_OPENCL
#define NEED_OPENCL
#include "../include/sarpok3d.h"

//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
unsigned int GetTickCount()
{
   struct tms t;
   long long time=times(&t);
   int clk_tck=sysconf(_SC_CLK_TCK);
   return (unsigned int)(((long long)(time*(1000.0/clk_tck)))%0xFFFFFFFF);    
}

// Note that the functions Gamma and LogGamma are mutually dependent.
double Gamma
(
    double x    // We require x > 0
);

double LogGamma
(
    double x    // x must be positive
)
{
	if (x <= 0.0)
	{
		std::stringstream os;
        os << "Invalid input argument " << x <<  ". Argument must be positive.";
        throw std::invalid_argument( os.str() ); 
	}

    if (x < 12.0)
    {
        return log(fabs(Gamma(x)));
    }

	// Abramowitz and Stegun 6.1.41
    // Asymptotic series should be good to at least 11 or 12 figures
    // For error analysis, see Whittiker and Watson
    // A Course in Modern Analysis (1927), page 252

    static const double c[8] =
    {
		 1.0/12.0,
		-1.0/360.0,
		1.0/1260.0,
		-1.0/1680.0,
		1.0/1188.0,
		-691.0/360360.0,
		1.0/156.0,
		-3617.0/122400.0
    };
    double z = 1.0/(x*x);
    double sum = c[7];
    for (int i=6; i >= 0; i--)
    {
        sum *= z;
        sum += c[i];
    }
    double series = sum/x;

    static const double halfLogTwoPi = 0.91893853320467274178032973640562;
    double logGamma = (x - 0.5)*log(x) - x + halfLogTwoPi + series;    
	return logGamma;
}
double Gamma
(
    double x    // We require x > 0
)
{
	if (x <= 0.0)
	{
		std::stringstream os;
        os << "Invalid input argument " << x <<  ". Argument must be positive.";
        throw std::invalid_argument( os.str() ); 
	}

    // Split the function domain into three intervals:
    // (0, 0.001), [0.001, 12), and (12, infinity)

    ///////////////////////////////////////////////////////////////////////////
    // First interval: (0, 0.001)
	//
	// For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
	// So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of x^3.
	// The relative error over this interval is less than 6e-7.

	const double gamma = 0.577215664901532860606512090; // Euler's gamma constant

    if (x < 0.001)
        return 1.0/(x*(1.0 + gamma*x));

    ///////////////////////////////////////////////////////////////////////////
    // Second interval: [0.001, 12)
    
	if (x < 12.0)
    {
        // The algorithm directly approximates gamma over (1,2) and uses
        // reduction identities to reduce other arguments to this interval.
		
		double y = x;
        int n = 0;
        bool arg_was_less_than_one = (y < 1.0);

        // Add or subtract integers as necessary to bring y into (1,2)
        // Will correct for this below
        if (arg_was_less_than_one)
        {
            y += 1.0;
        }
        else
        {
            n = static_cast<int> (floor(y)) - 1;  // will use n later
            y -= n;
        }

        // numerator coefficients for approximation over the interval (1,2)
        static const double p[] =
        {
            -1.71618513886549492533811E+0,
             2.47656508055759199108314E+1,
            -3.79804256470945635097577E+2,
             6.29331155312818442661052E+2,
             8.66966202790413211295064E+2,
            -3.14512729688483675254357E+4,
            -3.61444134186911729807069E+4,
             6.64561438202405440627855E+4
        };

        // denominator coefficients for approximation over the interval (1,2)
        static const double q[] =
        {
            -3.08402300119738975254353E+1,
             3.15350626979604161529144E+2,
            -1.01515636749021914166146E+3,
            -3.10777167157231109440444E+3,
             2.25381184209801510330112E+4,
             4.75584627752788110767815E+3,
            -1.34659959864969306392456E+5,
            -1.15132259675553483497211E+5
        };

        double num = 0.0;
        double den = 1.0;
        int i;

        double z = y - 1;
        for (i = 0; i < 8; i++)
        {
            num = (num + p[i])*z;
            den = den*z + q[i];
        }
        double result = num/den + 1.0;

        // Apply correction if argument was not initially in (1,2)
        if (arg_was_less_than_one)
        {
            // Use identity gamma(z) = gamma(z+1)/z
            // The variable "result" now holds gamma of the original y + 1
            // Thus we use y-1 to get back the orginal y.
            result /= (y-1.0);
        }
        else
        {
            // Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
            for (i = 0; i < n; i++)
                result *= y++;
        }

		return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Third interval: [12, infinity)

    if (x > 171.624)
    {
		// Correct answer too large to display. Force +infinity.
		double temp = DBL_MAX;
		return temp*2.0;
    }

    return exp(LogGamma(x));
}
//////////////////////////////////////
/////////direct solver////////////////
//////////////////////////////////////
class direct_solver {
public:
// variables
double beta;
double alpha;
double tau;
// constants
double cv=0.34;
double l=25.0;
double mu=0.00095;
double sigma=0.38;
double d=0.02;
double k=0.01;
double nu=2.8e-5;
double C0=200;
double H0=10;
double sa(double x) {return (1/(alpha*alpha))*pow(x,2.0*(1.0-alpha)); }
double ra(double x) {return ((1.0-alpha)/(alpha*alpha))*pow(x,1.0-2.0*alpha); }
double h=l/(M_+1.0);
// arrays
double H[M_+2],C[M_+2];
double A[M_+2],B[M_+2],S[M_+2],F[M_+2];
std::vector<double *> Hs,Cs;
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
double v(int i) { return k*H[i]-nu*C[i]; }
double w(int j,int v) { return (pow(tau,1.0-beta)/Gamma(2.0-beta))*(pow((double)(j-v+1.0),(double)(1.0-beta))-pow((double)(j-v),(double)(1.0-beta))); }
double AC(int i) { return (0.5/h)*(d*((sa(i*h)/h)-(ra(i*h)/2))-(sa(i*h)/(4.0*h))*(v(i+1)-v(i-1))); }
double BC(int i) { return (0.5/h)*(d*((sa(i*h)/h)+(ra(i*h)/2))+(sa(i*h)/(4.0*h))*(v(i+1)-v(i-1))); }
double SC(int i) { return AC(i)+BC(i)+sigma/(pow(tau,beta)*Gamma(2.0-beta));}
double FC(int i) 
{
    double sum=0;
    if (Cs.size()>=2)
    for (int j=0;j<Cs.size()-1;j++)
	sum+=w(Cs.size(),j)*(Cs[j+1][i]-Cs[j][i])/tau;
    return sigma*(sum-(C[i]/(pow(tau,beta)*Gamma(2.0-beta))))-
	   0.5*(d/h)*((sa(i*h)/h)*(C[i-1]-2.0*C[i]+C[i+1])+(ra(i*h)/2)*(C[i+1]-C[i-1]))-
	   0.5*(sa(i*h)/(4.0*h*h))*(v(i+1)-v(i-1)*(C[i+1]-C[i-1]));
}
double AH(int i) { return (0.5*cv/h)*((sa(i*h)/h)-(ra(i*h)/2)); }
double BH(int i) { return (0.5*cv/h)*((sa(i*h)/h)+(ra(i*h)/2)); }
double SH(int i) { return AH(i)+BH(i)+1.0/(pow(tau,beta)*Gamma(2.0-beta));}
double FH(int i) 
{
    double sum=0;
    if (Hs.size()>=2)
    for (int j=0;j<Hs.size()-1;j++)
	sum+=w(Hs.size(),j)*(Hs[j+1][i]-Hs[j][i])/tau;
    return (sum-(H[i]/(pow(tau,beta)*Gamma(2.0-beta))))-
	   0.5*(cv/h)*((sa(i*h)/h)*(H[i-1]-2.0*H[i]+H[i+1])+(ra(i*h)/2)*(H[i+1]-H[i-1]))+
	   0.5*(mu/h)*((sa(i*h)/h)*(C[i-1]-2.0*C[i]+C[i+1]+Cs[Cs.size()-1][i-1]-2.0*Cs[Cs.size()-1][i]+Cs[Cs.size()-1][i+1])+
		       (ra(i*h)/2)*(C[i+1]-C[i-1]+Cs[Cs.size()-1][i+1]-Cs[Cs.size()-1][i-1]));
}
void solveC()
{
	A[1]=0;
	B[1]=C0;
	for (int i=1;i<=M_;i++)
	    A[i+1]=BC(i)/(SC(i)-AC(i)*A[i]);
	for (int i=1;i<=M_;i++)
	    B[i+1]=(A[i+1]/BC(i))*(AC(i)*B[i]-FC(i));
	C[M_+1]=0;
	for (int i=M_;i>=0;i--)
	    C[i]=A[i+1]*C[i+1]+B[i+1];
}
void solveH()
{
	A[1]=0;
	B[1]=0;
	for (int i=1;i<=M_;i++)
	    A[i+1]=BH(i)/(SH(i)-AH(i)*A[i]);
	for (int i=1;i<=M_;i++)
	    B[i+1]=(A[i+1]/BH(i))*(AH(i)*B[i]-FH(i));
	H[M_+1]=0;
	for (int i=M_;i>=0;i--)
	    H[i]=A[i+1]*H[i+1]+B[i+1];
}
void calc_step()
{
	solveC();
	solveH();
	double *sC=new double[M_+2];
	memcpy(sC,C,(M_+2)*sizeof(double));
	Cs.push_back(sC);
	double *sH=new double[M_+2];
	memcpy(sH,H,(M_+2)*sizeof(double));
	Hs.push_back(sH);
}
void initialize()
{
	for (int i=0;i<M_+2;i++)
	{
	    C[i]=0.0;
	    H[i]=H0;
	}
	for (int i=1;i<Cs.size();i++)
	    delete [] Cs[i];
	for (int i=1;i<Hs.size();i++)
	    delete [] Hs[i];
	Cs.clear();
	Hs.clear();
	double *sC=new double[M_+2];
	memcpy(sC,C,(M_+2)*sizeof(double));
	Cs.push_back(sC);
	double *sH=new double[M_+2];
	memcpy(sH,H,(M_+2)*sizeof(double));
	Hs.push_back(sH);
}
direct_solver(double b,double a,double t) {alpha=a;beta=b;tau=t;initialize();}
void solve(double a,double b,double t,int n,int ns)
{
	alpha=a;
	beta=b;
	tau=t;
	initialize();
	for (int i=0;i<n;i++)
	{
		calc_step();
		if ((i%ns)==0)
		for (int j=0;j<=M_+1;j++)
			printf("a %g b %g tau %g t %g x %g C %g H %g\n",a,b,t,(i+1)*t,j*h,C[j],H[j]);
	}
}
};
/////////////////////////////////////////////////////////////
//////////////// opencl /////////////////////////////////////
/////////////////////////////////////////////////////////////
char *input_opencl_text = "\n\
#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n\
#define M_ %d\n\
#define cv %g\n\
#define l %g\n\
#define mu %g\n\
#define sigma %g\n\
#define d %g\n\
#define k %g\n\
#define nu %g\n\
#define C0 %g\n\
#define H0 %g\n\
#define h (l/(M_+1.0))\n\
double sa(double x,double alpha) {return (1/(alpha*alpha))*pow(x,2.0*(1.0-alpha)); }\n\
double ra(double x,double alpha) {return ((1.0-alpha)/(alpha*alpha))*pow(x,1.0-2.0*alpha); }\n\
double v(int i,__global double *C,__global double *H) { return k*H[i]-nu*C[i]; }\n\
double w(int j,int v,double tau,double beta,double gb) { return (pow(tau,1.0-beta)/gb)*(pow((double)(j-v+1.0),(double)(1.0-beta))-pow((double)(j-v),(double)(1.0-beta))); }\n\
double AC(int i,double alpha,__global double *C,__global double *H) { return (0.5/h)*(d*((sa(i*h,alpha)/h)-(ra(i*h,alpha)/2))-(sa(i*h,alpha)/(4.0*h))*(v(i+1,C,H)-v(i-1,C,H))); }\n\
double BC(int i,double alpha,__global double *C,__global double *H) { return (0.5/h)*(d*((sa(i*h,alpha)/h)+(ra(i*h,alpha)/2))+(sa(i*h,alpha)/(4.0*h))*(v(i+1,C,H)-v(i-1,C,H))); }\n\
double SC(int i,double alpha,double beta,double gb,__global double *C,__global double *H,double tau) { return AC(i,alpha,C,H)+BC(i,alpha,C,H)+sigma/(pow(tau,beta)*gb);}\n\
double FC(int i,int step,double tau,double alpha,double beta,double gb,__global double *C,__global double *H) \n\
{\n\
    double sum=0;\n\
    if (step>=2)\n\
	    for (int j=0;j<step-1;j++)\n\
		sum+=w(step,j,tau,beta,gb)*(C[(M_+2)*(j+1)+i]-C[(M_+2)*j+i])/tau;\n\
    step=step-1;\n\
    return sigma*(sum-(C[(M_+2)*step+i]/(pow(tau,beta)*gb)))-\n\
	   0.5*(d/h)*((sa(i*h,alpha)/h)*(C[(M_+2)*step+i-1]-2.0*C[(M_+2)*step+i]+C[(M_+2)*step+i+1])+(ra(i*h,alpha)/2)*(C[(M_+2)*step+i+1]-C[(M_+2)*step+i-1]))-\n\
	   0.5*(sa(i*h,alpha)/(4.0*h*h))*(v(i+1,C+((M_+2)*step),H+((M_+2)*step))-v(i-1,C+((M_+2)*step),H+((M_+2)*step))*(C[(M_+2)*step+i+1]-C[(M_+2)*step+i-1]));\n\
}\n\
double AH(int i,double alpha) { return (0.5*cv/h)*((sa(i*h,alpha)/h)-(ra(i*h,alpha)/2)); }\n\
double BH(int i,double alpha) { return (0.5*cv/h)*((sa(i*h,alpha)/h)+(ra(i*h,alpha)/2)); }\n\
double SH(int i,double alpha,double beta, double gb,double tau) { return AH(i,alpha)+BH(i,alpha)+1.0/(pow(tau,beta)*gb);}\n\
double FH(int i,int step,double tau,double alpha,double beta,double gb,__global double *C,__global double *H) \n\
{\n\
    double sum=0;\n\
    if (step>=2)\n\
	for (int j=0;j<step-1;j++)\n\
		sum+=w(step,j,tau,beta,gb)*(H[(M_+2)*(j+1)+i]-H[(M_+2)*j+i])/tau;\n\
    step=step-1;\n\
    return (sum-(H[(M_+2)*step+i]/(pow(tau,beta)*gb)))-\n\
	   0.5*(cv/h)*((sa(i*h,alpha)/h)*(H[(M_+2)*step+i-1]-2.0*H[(M_+2)*step+i]+H[(M_+2)*step+i+1])+(ra(i*h,alpha)/2)*(H[(M_+2)*step+i+1]-H[(M_+2)*step+i-1]))+\n\
	   0.5*(mu/h)*((sa(i*h,alpha)/h)*(C[(M_+2)*(step+1)+i-1]-2.0*C[(M_+2)*(step+1)+i]+C[(M_+2)*(step+1)+i+1]+C[(M_+2)*step+i-1]-2.0*C[(M_+2)*step+i]+C[(M_+2)*step+i+1])+\n\
		       (ra(i*h,alpha)/2)*(C[(M_+2)*(step+1)+i+1]-C[(M_+2)*(step+1)+i-1]+C[(M_+2)*step+i+1]-C[(M_+2)*step+i-1]));\n\
}\n\
void solveC(int step,double tau,double alpha,double beta,double gb,__global double *C,__global double *H,__global double *A,__global double *B,__global double *F)\n\
{\n\
	A[1]=0;\n\
	B[1]=C0;\n\
	for (int i=1;i<=M_;i++)\n\
	    A[i+1]=BC(i,alpha,C+(M_+2)*(step-1),H+(M_+2)*(step-1))/(SC(i,alpha,beta,gb,C+(M_+2)*(step-1),H+(M_+2)*(step-1),tau)-AC(i,alpha,C+(M_+2)*(step-1),H+(M_+2)*(step-1))*A[i]);\n\
	for (int i=1;i<=M_;i++)\n\
	    B[i+1]=(A[i+1]/BC(i,alpha,C+(M_+2)*(step-1),H+(M_+2)*(step-1)))*(AC(i,alpha,C+(M_+2)*(step-1),H+(M_+2)*(step-1))*B[i]-F[i]);\n\
	C[(M_+2)*step+M_+1]=0;\n\
	for (int i=M_;i>=0;i--)\n\
	    C[(M_+2)*step+i]=A[i+1]*C[(M_+2)*step+i+1]+B[i+1];\n\
}\n\
void solveH(int step,double tau,double alpha,double beta,double gb,__global double *C,__global double *H,__global double *A,__global double *B,__global double *F)\n\
{\n\
	A[1]=0;\n\
	B[1]=0;\n\
	for (int i=1;i<=M_;i++)\n\
	    A[i+1]=BH(i,alpha)/(SH(i,alpha,beta,gb,tau)-AH(i,alpha)*A[i]);\n\
	for (int i=1;i<=M_;i++)\n\
	    B[i+1]=(A[i+1]/BH(i,alpha))*(AH(i,alpha)*B[i]-F[i]);\n\
	H[(M_+2)*step+M_+1]=0;\n\
	for (int i=M_;i>=0;i--)\n\
	    H[(M_+2)*step+i]=A[i+1]*H[(M_+2)*step+i+1]+B[i+1];\n\
}\n\
__kernel void Solve(__global double *C,__global double *H,__global double *betas,__global double *alphas,__global double *gamma_betas,__global double *A,__global double *B,__global double *F,int nparticles,int nsteps,double tau)\n\
{\n\
	int id=get_global_id(0);\n\
	int p=id/(M_+2);\n\
	int cell=id%(M_+2);\n\
	__global double *Cp=&C[(M_+2)*(nsteps+1)*p];\n\
	__global double *Hp=&H[(M_+2)*(nsteps+1)*p];\n\
	Cp[cell]=0.0;\n\
	Hp[cell]=H0;\n\
	barrier(CLK_GLOBAL_MEM_FENCE);\n\
	for (int i=1;i<=nsteps;i++)\n\
	{\n\
		F[id]=FC(cell,i,tau,alphas[p],betas[p],gamma_betas[p],Cp,Hp);\n\
		barrier(CLK_GLOBAL_MEM_FENCE);\n\
		if (cell==0)\n\
			solveC(i,tau,alphas[p],betas[p],gamma_betas[p],Cp,Hp,A+(M_+2)*p,B+(M_+2)*p,F+(M_+2)*p);\n\
		barrier(CLK_GLOBAL_MEM_FENCE);\n\
		F[id]=FH(cell,i,tau,alphas[p],betas[p],gamma_betas[p],Cp,Hp);\n\
		barrier(CLK_GLOBAL_MEM_FENCE);\n\
		if (cell==0)\n\
			solveH(i,tau,alphas[p],betas[p],gamma_betas[p],Cp,Hp,A+(M_+2)*p,B+(M_+2)*p,F+(M_+2)*p);\n\
		barrier(CLK_GLOBAL_MEM_FENCE);\n\
	}\n\
}\n\
";
int silent=1;
OpenCL_program *prg=NULL;
OpenCL_commandqueue *queue;
OpenCL_prg *prog;
OpenCL_kernel *kSolve;
OpenCL_buffer *bC,*bH,*bBetas,*bAlphas,*bGB,*bA,*bB,*bF;
// initialize OpenCL
void init_opencl(int nsteps,int nparticles,direct_solver *bs)
{
	int iv;
	if (prg) return;
	if (silent==0)
		printf("OCL: init\n");
	prg = new OpenCL_program(1);
	queue = prg->create_queue(0, 0);
	if (silent==0)
		printf("OCL: compilation ");
	{
		char *text = new char[strlen(input_opencl_text)* 2];
		sprintf(text, input_opencl_text, M_, bs->cv, bs->l,bs->mu,bs->sigma,bs->d, bs->k,bs->nu,bs->C0,bs->H0);
		prog = prg->create_program(text);
		delete[] text;
	}
	if (silent==0)
		printf("done\n");
	kSolve = prg->create_kernel(prog, "Solve");
	// to save solutions on all steps for all particles
	bC = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*(M_+2)*nparticles*(nsteps+1) , NULL);
	bH = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*(M_+2)*nparticles*(nsteps+1) , NULL);
	// a and b for particles
	bAlphas = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*nparticles, NULL);
	bBetas = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*nparticles, NULL);
	bGB = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*nparticles, NULL);
	bA = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*(M_+2)*nparticles , NULL);
	bB = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*(M_+2)*nparticles , NULL);
	bF = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*(M_+2)*nparticles , NULL);
	if (silent==0)
		printf("OCL: init done\n");
}
// get from GPU a solution for a given particle on a given step
void fr_ocl_get(double *C,double *H,int step,int particle,int nsteps)
{
	queue->EnqueueBuffer(bC, C,sizeof(double)*(M_+2)*((nsteps+1)*particle+step),sizeof(double)*(M_+2));
	queue->EnqueueBuffer(bH, H,sizeof(double)*(M_+2)*((nsteps+1)*particle+step),sizeof(double)*(M_+2));
	queue->Finish();
}
void fr_solve(double *betas,double *alphas, double tau,double t,int nparticles)
{
	int err=0;
	if (silent==0)
		printf("OCL: solve ");
	int nsteps=ceil(t/tau);
	double *gb=new double[nparticles];
	for (int i=0;i<nparticles;i++)
		gb[i]=Gamma(2.0-betas[i]);
	queue->EnqueueWriteBuffer(bAlphas, alphas);
	queue->EnqueueWriteBuffer(bBetas, betas);
	queue->EnqueueWriteBuffer(bGB, gb);
	err |= kSolve->SetBufferArg(bC, 0);
	err |= kSolve->SetBufferArg(bH, 1);
	err |= kSolve->SetBufferArg(bBetas, 2);
	err |= kSolve->SetBufferArg(bAlphas, 3);
	err |= kSolve->SetBufferArg(bGB, 4);
	err |= kSolve->SetBufferArg(bA, 5);
	err |= kSolve->SetBufferArg(bB, 6);
	err |= kSolve->SetBufferArg(bF, 7);
	err |= kSolve->SetArg(8, sizeof(int), &nparticles);
	err |= kSolve->SetArg(9, sizeof(int), &nsteps);
	err |= kSolve->SetArg(10, sizeof(double), &tau);
	if (err) {printf("Error: Failed to set kernels args\n");exit(0);}
	if (silent==0)
		printf("(args set) ");
	size_t nth=nparticles*(M_+2), lsize=M_+2;
	queue->ExecuteKernel(kSolve, 1, &nth, &lsize);
	queue->Finish();
	if (silent==0)
		printf(" done\n");
	delete [] gb;
}
//////////////////////////////////////
//////inverse solver//////////////////
//////////////////////////////////////
int opencl=1;
int opencl_test=0;
int adaptive_pso=1; // adaptive pso: o - omega change parameter p , fi_p - fi_p,fi_g changing parameter C, fi_g - vmax calculating parameter N
double restart_prox=0.2; // probability of particles "restart" - random initialization
int init_print=1;
char *filename=NULL;
double rnd()
{
	return ((rand() % 10000) / (10000.0 - 1.0));
}
void init_particle(double *particle)
{
	particle[1]=0.5+0.5*rnd();
	particle[2]=0.5+0.5*rnd();
}
direct_solver *init_solver(double *p,double tau)
{
	if (p[1]<0.5) p[1]=0.5;
	if (p[2]<0.5) p[2]=0.5;
	if (p[1]>1.0) p[1]=1.0;
	if (p[2]>1.0) p[2]=1.0;
	while ((!finite(p[1]))||(!finite(p[2]))) init_particle(p);
	if (silent==0)
	if (init_print==1)
		printf("b %g a %g tau %g\n",p[1],p[2],tau);
	init_print=0;
	return new direct_solver(p[1],p[2],tau);
}
void solve_all_and_test(double **p,int nparticles,double *values_T,double *values_Z,double *values_F,int nvalues,double t,double tau)
{
	double *vs=new double [2*nvalues*nparticles];
	direct_solver **ss,**ss2;
	double *as,*bs;
	ss=new direct_solver *[nparticles];
	if (opencl_test)
	    ss2=new direct_solver *[nparticles];
	for (int i=0;i<nparticles;i++)
	{
		ss[i]=init_solver(p[i],tau);
		p[i][0]=0.0; // error
		if (opencl_test)
			ss2[i]=init_solver(p[i],tau);
	}
	if (opencl)
	{
		as=new double[nparticles];
		bs=new double[nparticles];
		for (int i=0;i<nparticles;i++)
		{
			bs[i]=p[i][1];
			as[i]=p[i][2];
		}
		fr_solve(bs,as, tau,t,nparticles);
	}
	int step=0;
	for (double tt = 0;tt < t;tt += ss[0]->tau,step++)
	{
		// save old values
		double got=0;
		for (int i=0;i<nvalues;i++)
			if ((values_T[i]>=tt)&&(values_T[i]<(tt+ss[0]->tau)))
			{
				for (int ii=0;ii<M_+1;ii++)
					if (((ss[0]->h*ii)<=values_Z[i])&&((ss[0]->h*(ii+1))>values_Z[i]))
					{
						if ((opencl)&&(got==0))
						{
							for (int i1=0;i1<nparticles;i1++)
							    if (opencl_test==0)
								fr_ocl_get(ss[i1]->C,ss[i1]->H,step,i1,ceil(t/tau));
							    else
								fr_ocl_get(ss2[i1]->C,ss2[i1]->H,step,i1,ceil(t/tau));
							got=1;
						}
						double k2x=(values_Z[i]-(ss[0]->h*ii))/ss[0]->h;
						for (int i1=0;i1<nparticles;i1++)
						{
							vs[i1*2*nvalues+2*i+0]=(1-k2x)*ss[i1]->H[ii]+k2x*ss[i1]->H[ii+1];
							vs[i1*2*nvalues+2*i+1]=(1-k2x)*ss[i1]->C[ii]+k2x*ss[i1]->C[ii+1];
						}
						break;
					}
			}
		// solve
		if ((opencl==0)||(opencl_test))
#pragma omp parallel for
		    for (int i=0;i<nparticles;i++)
			ss[i]->calc_step();
		// add to err
		got=0;
		for (int i=0;i<nvalues;i++)
			if ((values_T[i]>=tt)&&(values_T[i]<(tt+ss[0]->tau)))
			{
				double k1=(values_T[i]-tt)/ss[0]->tau;
				for (int ii=0;ii<M_+1;ii++)
					if (((ss[0]->h*ii)<=values_Z[i])&&((ss[0]->h*(ii+1))>values_Z[i]))
					{
						if ((opencl)&&(got==0))
						{
							for (int i1=0;i1<nparticles;i1++)
							    if (opencl_test==0)
								fr_ocl_get(ss[i1]->C,ss[i1]->H,step+1,i1,ceil(t/tau));
							    else
								fr_ocl_get(ss2[i1]->C,ss2[i1]->H,step+1,i1,ceil(t/tau));
							got=1;
						}
						double k2x=(values_Z[i]-(ss[0]->h*ii))/ss[0]->h;
						double v1,v2;
						for (int i1=0;i1<nparticles;i1++)
						{
							v1=(1-k2x)*ss[i1]->H[ii]+k2x*ss[i1]->H[ii+1];
							v2=(1-k2x)*ss[i1]->C[ii]+k2x*ss[i1]->C[ii+1];
							v1=(1-k1)*vs[i1*2*nvalues+2*i+0]+k1*v1;
							v2=(1-k1)*vs[i1*2*nvalues+2*i+1]+k1*v2;
							p[i1][0]+=(values_F[2*i+0]-v1)*(values_F[2*i+0]-v1);
							p[i1][0]+=(values_F[2*i+1]-v2)*(values_F[2*i+1]-v2);
						}
						break;
					}					
			}
		if ((opencl_test)&&(got==0))
			for (int i1=0;i1<nparticles;i1++)
				fr_ocl_get(ss2[i1]->C,ss2[i1]->H,step+1,i1,ceil(t/tau));
		if (opencl_test)
		for (int i=0;i<nparticles;i++)
		{
		    double ocl_err_c=0.0,ocl_err_h=0.0;
		    for (int ii=0;ii<M_+2;ii++)
		    {
			ocl_err_c+=(ss2[i]->C[ii]-ss[i]->C[ii])*(ss2[i]->C[ii]-ss[i]->C[ii]);
			ocl_err_h+=(ss2[i]->H[ii]-ss[i]->H[ii])*(ss2[i]->H[ii]-ss[i]->H[ii]);
		    }
			printf("OCL test %g: %d(%g,%g) cerr %g herr %g\n",tt,i,as[i],bs[i],ocl_err_c,ocl_err_h);
		}
	}
	if (opencl_test)
		exit(0);
	for (int i=0;i<nparticles;i++)
		if (!finite(p[i][0]))
		{
			p[i][0]=1e300;
			init_particle(p[i]);
		}
	delete [] vs;
	if (opencl)
	{
		delete [] as;
		delete [] bs;
	}
	delete [] ss;
	if (opencl_test)
		delete [] ss2;
}
void fit_and_solve(double t, double tau,double *values_T,double *values_Z,double *values_F,int nvalues,
	int pso_n,double pso_o,double pso_fi_p,double pso_fi_g,double pso_eps,int pso_max_iter)
{
	double ad_pso_p,ad_pso_C; // adaptive PSO parameters
	double ad_v0,*ad_vmax;
	double *ad_pso_fi_p,*ad_pso_fi_g; // per-variable fi_p, fi_g for adaptive PSO
	int size=2;
	double **particles;
	int best;
	double *best_p;
	int iter=0;
	unsigned int t1=GetTickCount();
	best_p=new double[2*size+1];
	particles=new double *[pso_n];
	for (int i=0;i<pso_n;i++)
		particles[i]=new double[3*size+2]; // particles[0] contains f value, particles[1+size].. contain velocities, particles[1+2*size]... contain particle's best known position, particles[1+3*size] contains best known f value
		// initialize
	if (opencl)
	{
		direct_solver *bs=new direct_solver(1,1,tau);
		init_opencl(ceil(t/tau),pso_n,bs);
		delete bs;
	}
	for (int i=0;i<pso_n;i++)
		init_particle(particles[i]);
	solve_all_and_test(particles,pso_n,values_T,values_Z,values_F,nvalues,t,tau);
	for (int i=0;i<pso_n;i++)
		particles[i][1+3*size]=particles[i][0];
	best=0;
        if (silent==0)
	    printf("initial 0 - %g\n", particles[0][0]);
	for (int i=1;i<pso_n;i++)
	{
		if (particles[i][0]<particles[best][0])
			best=i;
	        if (silent==0)
			printf("initial %d - %g\n",i, particles[i][0]);
	}
	// save best known position
	for (int j=0;j<=size;j++)
		best_p[j]=particles[best][j];
        if (silent==0)
	{
		printf("initial best: ");
		for (int j = 0;j <= size;j++)
			printf("%2.2g ", best_p[j]);
		printf("\n");
		fflush(stdout);
	}
	// adaptive PSO - calc initial and max velocity
	if (adaptive_pso)
	{
		double *xm=new double [2*size]; // min-max x in initial population
		double *ad_v0v=new double[size];
		ad_v0=0;
		for (int i=0;i<size;i++) ad_v0v[i]=0;
		ad_vmax=new double[size];
		ad_pso_fi_p=new double[size];
		ad_pso_fi_g=new double[size];
		for (int i=0;i<pso_n;i++)
			for (int j=0;j<size;j++)
			{
				ad_v0+=particles[i][j+1]*particles[i][j+1];
				ad_v0v[j]+=particles[i][j+1]*particles[i][j+1];
				if (i==0)
					xm[2*j+0]=xm[2*j+1]=particles[i][j+1];
				else
				{
					if (particles[i][j+1]<xm[2*j+0]) xm[2*j+0]=particles[i][j+1];
					if (particles[i][j+1]>xm[2*j+1]) xm[2*j+1]=particles[i][j+1];
				}
			}
		ad_v0=sqrt(ad_v0);
		ad_v0/=pso_n;
		for (int i=0;i<size;i++)
		{
			ad_vmax[i]=(xm[2*i+1]-xm[2*i+0])/pso_fi_g;
			if (ad_vmax[i]==0.0) ad_vmax[i]=1.0/pso_fi_g;
			ad_v0v[i]=sqrt(ad_v0v[i]);
			ad_v0v[i]/=pso_n;
		}
		delete [] xm;
		ad_pso_p=pso_o; ad_pso_C=pso_fi_p;
		pso_o=1.0;
		for (int i=0;i<size;i++)
		{
			ad_pso_fi_p[i]=ad_pso_C*ad_v0v[i]/ad_vmax[i];
			ad_pso_fi_g[i]=ad_pso_C*(1-ad_v0v[i]/ad_vmax[i]);
			if (ad_pso_fi_p[i]>2.0) ad_pso_fi_p[i]=2.0;
			if (ad_pso_fi_g[i]>2.0) ad_pso_fi_g[i]=2.0;
				if (ad_pso_fi_p[i]<0.0) ad_pso_fi_p[i]=0.0;
			if (ad_pso_fi_g[i]<0.0) ad_pso_fi_g[i]=0.0;
		}
		delete [] ad_v0v;
	        if (silent==0)
		{
			printf("adaptive PSO: ||v0||=%g, initial PSO params values %g",ad_v0,pso_o);
			for (int i=0;i<size;i++)
				printf(",(%g->%g,%g)",ad_vmax[i],ad_pso_fi_p[i],ad_pso_fi_g[i]);
			printf("\n");
		}
	}
	// process
	if (pso_max_iter>=1)
	do
	{
	// adaptive PSO - change PSO parameters
	if (adaptive_pso)
	{
		double ve=ad_v0*exp(-(2*(iter+1)/(float)(pso_max_iter-iter))*(2*(iter+1)/(float)(pso_max_iter-iter)));
		double vavg=0.0;
		double *vavgc=new double[size];
		for (int j=0;j<size;j++)
			vavgc[j]=0;
		for (int i=0;i<pso_n;i++)
			for (int j=0;j<size;j++)
			if (finite(particles[i][j+1+size]))
			{
				vavg+=particles[i][j+1+size]*particles[i][j+1+size];
				vavgc[j]+=particles[i][j+1+size]*particles[i][j+1+size];
			}
		vavg=sqrt(vavg);
		vavg/=pso_n;
		for (int j=0;j<size;j++)
		{
			vavgc[j]=sqrt(vavgc[j]);
			vavgc[j]/=pso_n;
		}
		// change omega
		if (vavg>ve)
			pso_o/=ad_pso_p;
		if (vavg<ve)
			pso_o*=ad_pso_p;
		if (pso_o>2) pso_o=2;
		if (pso_o<0) pso_o=0;
		// change fi_p,fi_g
		for (int i=0;i<size;i++)
		{
			ad_pso_fi_p[i]=ad_pso_C*vavgc[i]/ad_vmax[i];
			ad_pso_fi_g[i]=ad_pso_C*(1-vavgc[i]/ad_vmax[i]);
			if (ad_pso_fi_p[i]>2.0) ad_pso_fi_p[i]=2.0;
			if (ad_pso_fi_g[i]>2.0) ad_pso_fi_g[i]=2.0;
				if (ad_pso_fi_p[i]<0.0) ad_pso_fi_p[i]=0.0;
			if (ad_pso_fi_g[i]<0.0) ad_pso_fi_g[i]=0.0;
		}
	        if (silent==0)
		{
			printf("adaptive PSO - vavg %g ve %g params values %g",vavg,ve,pso_o);
			for (int i=0;i<size;i++)
				printf(",(%g->%g,%g)",vavgc[i],ad_pso_fi_p[i],ad_pso_fi_g[i]);
			printf("\n");
		}
		delete [] vavgc;
	}
	for (int i=0;i<pso_n;i++)
	{
		// update velocity
		for (int j=0;j<size;j++)
		{
			double rp=(rand()%10000)/10000.0;
			double rg=(rand()%10000)/10000.0;
			if (adaptive_pso) // percoordinate fi_p,fi_g
			{
				pso_fi_p=ad_pso_fi_p[j];
				pso_fi_g=ad_pso_fi_g[j];
			}
			particles[i][j+1+size]=pso_o*particles[i][j+1+size]+pso_fi_p*rp*(particles[i][j+1+2*size]-particles[i][j+1])+pso_fi_g*rg*(best_p[j+1]-particles[i][j+1]);				    
		}
		// update position
		for (int j=0;j<size;j++)
			particles[i][1+j]+=particles[i][j+1+size];
		// restart particle
		double rp=(rand()%10000)/10000.0;			
		if (rp<((pso_o<1)?(1-pso_o):0.0)*restart_prox)
		{
			init_particle(particles[i]);
		        if (silent==0)
				printf("r");
		}
	}
	// calc f values
	solve_all_and_test(particles,pso_n,values_T,values_Z,values_F,nvalues,t,tau);
	for (int i=0;i<pso_n;i++)
	{
		// update bests
		if (particles[i][0]<particles[i][1+3*size])
		{
			for (int j=0;j<size;j++)
				particles[i][j+1+2*size]=particles[i][j+1];
			particles[i][1+3*size]=particles[i][0];
		}
		if (particles[i][0]<best_p[0])
		{
			for (int j=0;j<size;j++)
				best_p[j+1]=particles[i][j+1];
			best_p[0]=particles[i][0];
		}
	}
	// check best-worst difference
	double max = 0.0;
	double avg=0.0;
	for (int i = 0;i < pso_n;i++)
	{
		if (max < particles[i][0])
		max = particles[i][0];
		avg+= particles[i][0];
	}
	avg/=pso_n;
        if (silent==0)
	{
		printf("%d avg %g best: ", iter,avg);
		for (int j = 0;j <= size;j++)
			printf("%g ", best_p[j]);
		printf("\n");
	}
	if ((max - best_p[0]) < pso_eps)
		break;
	iter++;
	}
	while ((iter<pso_max_iter)&&(best_p[0]>pso_eps));
	// solve with best parameters values
	direct_solver *ss;
	init_print=1;
    ss=init_solver(best_p,tau);
	double err=0.0;	
	double *vs=new double [2*nvalues];
	for (double tt = 0;tt <= t;tt += ss->tau)
	{
		// save old values
		for (int i=0;i<nvalues;i++)
			if ((values_T[i]>=tt)&&(values_T[i]<(tt+ss->tau)))
			{
				for (int ii=0;ii<M_+1;ii++)
					if (((ss->h*ii)<=values_Z[i])&&((ss->h*(ii+1))>values_Z[i]))
					{
						double k2x=(values_Z[i]-(ss->h*ii))/ss->h;
						vs[2*i+0]=(1-k2x)*ss->H[ii]+k2x*ss->H[ii+1];
						vs[2*i+1]=(1-k2x)*ss->C[ii]+k2x*ss->C[ii+1];
						break;
					}
			}
		// solve
		ss->calc_step();
		// add to err
		for (int i=0;i<nvalues;i++)
			if ((values_T[i]>=tt)&&(values_T[i]<(tt+ss->tau)))
			{
				double k1=(values_T[i]-tt)/ss->tau;
				for (int ii=0;ii<M_+1;ii++)
					if (((ss->h*ii)<=values_Z[i])&&((ss->h*(ii+1))>values_Z[i]))
					{
						double k2x=(values_Z[i]-(ss->h*ii))/ss->h;
						double v1,v2;
						v1=(1-k2x)*ss->H[ii]+k2x*ss->H[ii+1];
						v2=(1-k2x)*ss->C[ii]+k2x*ss->C[ii+1];
						v1=(1-k1)*vs[2*i+0]+k1*v1;
						v2=(1-k1)*vs[2*i+1]+k1*v2;
						err+=(values_F[2*i+0]-v1)*(values_F[2*i+0]-v1);
						err+=(values_F[2*i+1]-v2)*(values_F[2*i+1]-v2);
					        if (silent==0)
							printf("t %g x %g H %g Hmodel %g C %g Cmodel %g\n",values_T[i],values_Z[i],values_F[2*i+0],v1,values_F[2*i+1],v2);
						break;
					}					
			}
		if (!finite(ss->H[M_/2]))
		{
		     err=1e300;
		     break;
		}
	}
	printf("total err %g values %g %g time %d ms\n",err,best_p[1],best_p[2],GetTickCount()-t1);
}
void test_inverse(double b,double a,double tau,double t,double noise,double sel_prob,int pso_n,double pso_o,double pso_fi_p,double pso_fi_g,double pso_eps,int pso_max_iter)
{
	double values_T[M_+2],values_Z[M_+2],values_F[2*(M_+2)];
	int nvalues=0;
	direct_solver *ss=new direct_solver(b,a,tau);
	for (double tt=0;tt<=t;tt+=tau)
		ss->calc_step();
	for (int i=0;i<M_+2;i++)
	if (rnd()<sel_prob)
	{
		values_T[nvalues]=t;
		values_Z[nvalues]=ss->h*i;
		values_F[2*nvalues+0]=ss->H[i]+noise*(rnd()-0.5);
		values_F[2*nvalues+1]=ss->C[i]+noise*(rnd()-0.5);
		if (silent==0)
			printf("value %d t %g z %g H %g noisedH %g C %g noisedC %g\n",nvalues,t,ss->h*i,ss->H[i],values_F[2*nvalues+0],ss->C[i],values_F[2*nvalues+1]);
		nvalues++;
	}
	printf("b %g a %g tau %g t %g noise %g sel %g pso (%d %g %g %g %g %d)\n",b,a,tau,t,noise,sel_prob,pso_n,pso_o,pso_fi_p,pso_fi_g,pso_eps,pso_max_iter);
	for (int i=0;i<2;i++)
		fit_and_solve(t, tau, (double *)values_T,(double *)values_Z,(double *)values_F,nvalues,pso_n,pso_o,pso_fi_p,pso_fi_g,pso_eps,pso_max_iter);
}
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
int main(int argc, char**argv)
{
    if (argc==6)
	{
		direct_solver *ss=new direct_solver(atof(argv[1]),atof(argv[2]),atof(argv[3]));
        ss->solve(atof(argv[1]),atof(argv[2]),atof(argv[3]),atoi(argv[4]),atoi(argv[5]));
	}
	if (argc==14)
	{
		int i=atoi(argv[13]);
		opencl=(i&1);
		opencl_test=((i>>1)&1);
		silent=((i>>2)&1);
		adaptive_pso=((i>>3)&1);
		printf("ocl %d (test %d) silent %d apso %d\n",opencl,opencl_test,silent,adaptive_pso);
		test_inverse(atof(argv[1]),atof(argv[2]),atof(argv[3]),atof(argv[4]),atof(argv[5]),atof(argv[6]),atoi(argv[7]),atof(argv[8]),atof(argv[9]),atof(argv[10]),atof(argv[11]),atoi(argv[12]));
	}
    return 0;
}