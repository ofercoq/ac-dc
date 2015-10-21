/*
 * cdn_prox_common.h
 *
 *  Created on: Oct 17, 2015
 *      Author: fercoq
 *  SDNA for smooth hinge
 */

#ifndef CDN_PROX_COMMON_H_
#define CDN_PROX_COMMON_H_

#include "../helpers/c_libs_headers.h"
#include "../utils/randomNumbersUtil.h"
#include <gsl/gsl_linalg.h>

#include <gsl/gsl_cblas.h>
#include "../utils/my_cblas_wrapper.h"
#include "../helpers/utils.h"

#include "../utils/distributed_instances_loader.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

template<typename L, typename D>
  D computeDualityGapSparse(L m, L n, ProblemData<int, double>& part,
			    std::vector<D> &b, std::vector<D> &x, std::vector<D> &w, D& lambda,
			    D& primal, D&dual) {

  primal = 0;
  dual = 0;

  for (int i = 0; i < n; i++) {
    D tmp = 0;
    for (int j = part.A_csr_row_ptr[i]; j < part.A_csr_row_ptr[i + 1];
	 j++) {
      tmp += part.A_csr_values[j] * w[part.A_csr_col_idx[j]];
    }
    tmp *= b[i];
    if (tmp < 1. - part.mu)
      primal += 1. - 0.5 * part.mu - tmp;
    else if (tmp < 1.)
      primal += 0.5 / part.mu * (1. - tmp) * (1. - tmp);
    else
      primal += 0.0;

    if (x[i] < 0)
      dual += 1e300;
    else if (x[i] <= 1)
      dual += part.mu / 2 * x[i] * x[i] - x[i];
    else dual += 1e300;
  }
  D norm = cblas_l2_norm(m, &w[0], 1);

  primal = 1 / (0.0 + n) * primal + lambda * 0.5 * norm * norm;
  dual = 1 / (0.0 + n) * dual + lambda * 0.5 * norm * norm;
  return primal + dual;

}


// function prototype for L-BFGS-B routine
// NOTES: All arguments must be passed as pointers. Must add an underscore
//  to the function name. 'extern "C"' keeps c++ from mangling the function
//  name. Be aware that fortran indexes 1d arrays [1,n] not [0,n-1], and
//  expects 2d arrays to be in col-major order not row-major order.
extern "C" void setulb_(int* n, int* m, double x[], double l[], double u[],
	    int nbd[], double* f, double g[], double* factr, 
	    double* pgtol, double wa[], int iwa[], char task[],
	    int* iprint, char csave[], bool lsave[], int isave[],
	    double dsave[]);


template<typename L, typename D>
  void runCDNExperimentSparse(L m, L n, ProblemData<int, double>& part,
			      std::vector<D>& b, D & lambda, int tau, 
			      ofstream& logFile,
			      D maxTime, std::vector<D>& Hessian, std::vector<D> & Li) {
  bool hessianPrecomputed = (Hessian.size() > 0);
  bool blockOfHessianComputed = (tau < 20) && !(tau == 1);
  std::vector < D > x(n, 0);
  for (int i = 0; i < x.size(); i++) {
    x[i] = 0;
  }
  std::vector < D > w(m, 0);
  for (int i = 0; i < w.size(); i++) {
    w[i] = 0;
  }

  D primal;
  D dual;
  D gap;
  gap = computeDualityGapSparse(m, n, part, b, x, w, lambda, primal, dual);
  cout << "0   Duality Gap: " << gap << "   " << primal << "   " << dual << " tau = " << tau << endl;

  logFile << setprecision(16) << tau << "," << m << "," << n
	  << "," << lambda << "," << part.mu << "," << primal << "," << dual << "," << 0
	  << endl;

  std::vector<D> Qdata;
  if (blockOfHessianComputed)
    Qdata.resize(tau * tau, 0.);
  else
    Qdata.resize(tau, 0.);
  std::vector<D> bS(tau);
  std::vector<int> S(tau);
  gsl_vector *T = gsl_vector_alloc(tau);
  gsl_permutation * p = gsl_permutation_alloc(tau);
  //  std::vector < D > AS(m * tau);

  double scaling = 1 / (lambda * n);

  //c     Declare the variables needed by the L-BFGS-B code.
  //c       A description of all these variables is given at the end of 
  //c       this file.

  //c        rank is the maximum number of limited memory corrections.
  int rank = min(20, tau);

  const int SIXTY=60;

  char task[SIXTY], csave[SIXTY];
  bool lsave[4];
  int iprint, isave[44];
  std::vector<int> nbd(tau);
  std::vector<int> iwa(3*tau); 
  double f_S, factr, pgtol, dsave[29];
  std::vector<double> x_S(tau);
  std::vector<double> l_S(tau);
  std::vector<double> u_S(tau);
  std::vector<double> g0_S(tau);
  std::vector<double> g_S(tau);
  std::vector<double> wa(2*rank*tau+4*tau+12*rank*rank+12*rank);
  std::vector<double> dw_S;
  if (!blockOfHessianComputed)
    dw_S.resize(m);

  D tol = 1e-13;
  double elapsedTime = 0;
  double start;

  long long it = 0;
  int nbaff = 0;
  for (;;) {    
    start = gettime_();
    it = it + tau;

    // Calculate a tau-nice sampling
    if (tau < n) {
      for (int i = 0; i < tau; i++) {
	bool done = true;
	do {
	  done = true;
	  S[i] = gsl_rng_uniform_int(gsl_rng_r, n);
	  for (int j = 0; j < i; j++) {

	    if (S[i] == S[j]) {
	      done = false;
	      break;
	    }
	  }

	} while (!done);
      }
    }
    else {
      for (int i = 0; i < tau; i++)
	S[i] = i;
    }

    // Build the according block of the Hessian matrix
    if (hessianPrecomputed) {

      for (int row = 0; row < tau; row++) {
	for (int col = row; col < tau; col++) {

	  D tmp = Hessian[S[row] * n + S[col]] * scaling;

	  Qdata[row * tau + col] = tmp;
	  Qdata[col * tau + row] = tmp;
	}
      }
    } 
    else if (blockOfHessianComputed){

      std::vector<double>& vals = part.A_csr_values;
      std::vector<int> &rowPtr = part.A_csr_row_ptr;
      std::vector<int> &colIdx = part.A_csr_col_idx;

      for (int row = 0; row < tau; row++) {
	for (int col = row; col < tau; col++) {

	  double tmp = 0;

	  int id1 = rowPtr[S[row]];
	  int id2 = rowPtr[S[col]];

	  while (id1 < rowPtr[S[row] + 1] && id2 < rowPtr[S[col] + 1]) {

	    if (colIdx[id1] == colIdx[id2]) {
	      tmp += vals[id1] * vals[id2];
	      id1++;
	      id2++;
	    } 
	    else if (colIdx[id1] < colIdx[id2]) {
	      id1++;
	    } 
	    else {
	      id2++;
	    }

	  }

	  Qdata[row * tau + col] = b[row] * tmp * scaling * b[col];
	  Qdata[col * tau + row] = b[col] * tmp * scaling * b[row];

	}
	Qdata[row * tau + row] += part.mu / n;
      }
    }
    else {
      // If tau == 1: the diagonal is enough
      // If tau > 20: We will estimate Qdata by BFGS
      for (int row = 0; row < tau; row++) {
	Qdata[row] = Li[S[row]] * scaling + part.mu / n;
      }
    }



    // We first try two runs of proximal coordinate descent.
    // This is much cheaper than L-BFGS-B and may be enough
    int do_lbfgsb = 0;
    int nb_tries = ((tau>1) ? 2:1);
    for (int ii=0; ii < nb_tries*tau; ii++) {
      // Compute partial derivative
      int i = ii % tau;
      g0_S[i] = 0;
      for (int j = part.A_csr_row_ptr[S[i]];
	   j < part.A_csr_row_ptr[S[i] + 1]; j++) {
	g0_S[i] += part.A_csr_values[j] * w[part.A_csr_col_idx[j]];
      }
      g0_S[i] *= b[S[i]] / n;
      g0_S[i] += - 1. / (n + 0.0) + part.mu / n * x[S[i]];

      // Compute prox
      x_S[i] = max(0., min(1., x[S[i]] - 1. / Qdata[i,i] * g0_S[i] ));

      // Update and test whether one run of cd was enough
      for (int j = part.A_csr_row_ptr[S[i]];
	   j < part.A_csr_row_ptr[S[i] + 1]; j++) {
	w[part.A_csr_col_idx[j]] += scaling * part.A_csr_values[j]
	  * bS[i] * (x_S[i] - x[S[i]]);
      }
      if (ii >= tau && abs(x[S[i]] - x_S[i]) > tol) {
	do_lbfgsb = 1;
      }
      x[S[i]] = x_S[i];
    }

    if (do_lbfgsb) {
      // Compute the partial derivatives that are needed 
      // g0_S = -1/n + 1/n bS (A w)_S + mu/n x_S
      for (int i = 0; i < tau; i++) {
	g0_S[i] = 0;
	for (int j = part.A_csr_row_ptr[S[i]];
	     j < part.A_csr_row_ptr[S[i] + 1]; j++) {
	  g0_S[i] += part.A_csr_values[j] * w[part.A_csr_col_idx[j]];
	}
      }
      for (int i = 0; i < tau; i++) {
	g0_S[i] *= b[S[i]] / n;
	g0_S[i] += - 1. / (n + 0.0) + part.mu / n * x[S[i]];
      }

      //--------------------------------------------------------------------//
      // Run L-BFGS-B to solve the sub-problem (in necessary)//
      //--------------------------------------------------------------------//
 
      //c     We do not wish to have output at every iteration.
      iprint = -1;

      //c     We specify the tolerances in the stopping criteria.
      factr=1.0e4;  // 1.0e7;
      pgtol=1.0e-8; // 1.0e-5;

      //c     We now provide nbd which defines the bounds on the variables:
      //c                    l   specifies the lower bounds,
      //c                    u   specifies the upper bounds. 
 
      //c     Set bounds on the variables.
      for (int i=0; i<tau; i+=1){
	nbd[i]=2;
	l_S[i]=0.0;
	u_S[i]=1.0;
      }

      //c     We now define the starting point.
      for (int i=0; i<tau; i++)
	x_S[i] = x[S[i]];

      //c     We start the iteration by initializing task.
      // (**MUST clear remaining chars in task with spaces (else crash)!**)
      strcpy(task,"START");
      for (int i=5; i<SIXTY; i++)
	task[i]=' ';

      //c     This is the call to the L-BFGS-B code.
      // (* call the L-BFGS-B routine with task='START' once before loop *)
      setulb_(&tau, &rank, &x_S[0], &l_S[0], &u_S[0], &nbd[0], &f_S, &g_S[0],
	      &factr, &pgtol, &wa[0], &iwa[0], task, &iprint,
	      csave,lsave,isave,dsave);

      // (* while routine returns "FG" or "NEW_X" in task, keep calling it *)
      while (strncmp(task,"FG",2)==0 || strncmp(task,"NEW_X",5)==0) {

	if (strncmp(task,"FG",2)==0) {
	  //c   the minimization routine has returned to request the
	  //c   function f_S and gradient g_S values at the current x

	  //c        Compute function value f_S for the sample problem.
	  // f_S(x) - f_S(x0) = -dual(x) + dual(x0) 
	  //   = < g0, x - x0 > + 0.5 < x - x0, Q (x - x0) >
	  f_S = 0;
	  for (int i=1; i<tau; i++)
	    f_S += g0_S[i] * (x_S[i] - x[S[i]]);
	  if (blockOfHessianComputed)
	    for (int row=1; row<tau; row++)
	      for (int col=1; col<tau; col++)
		f_S += 0.5 * (x_S[row] - x[S[row]]) * Qdata[row * tau + col]
		            * (x_S[col] - x[S[col]]);
	  else {
	    for (int j = 0; j < m; j++)
	      dw_S[j] = 0.;
	    for (int i = 0; i < tau; i++) {
	      for (int j = part.A_csr_row_ptr[S[i]];
		   j < part.A_csr_row_ptr[S[i] + 1]; j++) {
		dw_S[part.A_csr_col_idx[j]] += scaling * part.A_csr_values[j]
		  * bS[i] * (x_S[i] - x[S[i]]);
	      }
	    }
	    for (int j = 0; j < m; j++)
	      f_S += 0.5 * lambda * dw_S[j] * dw_S[j];
	    for (int i = 0; i < tau; i++)
	      f_S += 0.5 * part.mu / n * (x_S[i] - x[S[i]]) * (x_S[i] - x[S[i]]);
	  }
	  // end compute function value

	  //c        Compute gradient g_S for the sample problem.
	  for (int i=1; i<tau; i++)
	    g_S[i] = g0_S[i];
	  if (blockOfHessianComputed) {
	    for (int row=1; row<tau; row++)
	      for (int col=1; col<tau; col++)
		g_S[row] += Qdata[row * tau + col] * (x_S[col] - x[S[col]]);
	  }
	  else {
	    for (int i = 0; i < tau; i++) {
	      for (int j = part.A_csr_row_ptr[S[i]];
		   j < part.A_csr_row_ptr[S[i] + 1]; j++) {
		g_S[i] += b[S[i]] / n * part.A_csr_values[j] * dw_S[part.A_csr_col_idx[j]];
	      }
	    }
	    for (int i = 0; i < tau; i++) {
	      g_S[i] += part.mu / n * (x[S[i]] - x[S[i]]);
	    }
	  }
	  // end compute gradient

	}
	else {
	  // the minimization routine has returned with a new iterate,
	  // and we have opted to continue the iteration (do nothing here)
	}

	//c          go back to the minimization routine.
	setulb_(&tau, &rank, &x_S[0], &l_S[0], &u_S[0], &nbd[0], &f_S, &g_S[0],
		&factr, &pgtol, &wa[0], &iwa[0], task, &iprint,
		csave,lsave,isave,dsave);
      } // end lbfgsb loop

      // Apply the update
      for (int i = 0; i < tau; i++) {
	for (int j = part.A_csr_row_ptr[S[i]];
	     j < part.A_csr_row_ptr[S[i] + 1]; j++) {
	  w[part.A_csr_col_idx[j]] += scaling * part.A_csr_values[j]
	    * bS[i] * (x_S[i] - x[S[i]]);
	}
      }
      for (int i = 0; i < tau; i++) {
	x[S[i]] = x_S[i];
      }

      // end do_lbfgsb
    }

    elapsedTime += (gettime_() - start);
    // if ((it + tau) % n < tau) {
    if (nbaff < elapsedTime || elapsedTime > maxTime || gap < tol) {
      gap = computeDualityGapSparse(m, n, part, b, x, w, lambda, primal,
				    dual);
      cout << it << "   Duality Gap: " << gap << "   " << primal
	   << "   " << dual << "  " << elapsedTime << endl;
      logFile << setprecision(16) << tau << "," << m << ","
	      << n << "," << lambda << "," << part.mu << "," << primal << "," << dual << ","
	      << elapsedTime << endl;
      nbaff++;
    }

    if (elapsedTime > maxTime || gap < tol) {
      break;
    }
  }

  gsl_permutation_free(p);
  gsl_vector_free(T);

}

#include "../helpers/matrix_conversions.h"

#endif /* CDN_PROX_COMMON_H_ */


/*******************************************************************
c     --------------------------------------------------------------
c             DESCRIPTION OF THE VARIABLES IN L-BFGS-B
c     --------------------------------------------------------------
c
c     n is an INTEGER variable that must be set by the user to the
c       number of variables.  It is not altered by the routine.
c
c     m is an INTEGER variable that must be set by the user to the
c       number of corrections used in the limited memory matrix.
c       It is not altered by the routine.  Values of m < 3  are
c       not recommended, and large values of m can result in excessive
c       computing time. The range  3 <= m <= 20 is recommended. 
c
c     x is a DOUBLE PRECISION array of length n.  On initial entry
c       it must be set by the user to the values of the initial
c       estimate of the solution vector.  Upon successful exit, it
c       contains the values of the variables at the best point
c       found (usually an approximate solution).
c
c     l is a DOUBLE PRECISION array of length n that must be set by
c       the user to the values of the lower bounds on the variables. If
c       the i-th variable has no lower bound, l(i) need not be defined.
c
c     u is a DOUBLE PRECISION array of length n that must be set by
c       the user to the values of the upper bounds on the variables. If
c       the i-th variable has no upper bound, u(i) need not be defined.
c
c     nbd is an INTEGER array of dimension n that must be set by the
c       user to the type of bounds imposed on the variables:
c       nbd(i)=0 if x(i) is unbounded,
c              1 if x(i) has only a lower bound,
c              2 if x(i) has both lower and upper bounds, 
c              3 if x(i) has only an upper bound.
c
c     f is a DOUBLE PRECISION variable.  If the routine setulb returns
c       with task(1:2)= 'FG', then f must be set by the user to
c       contain the value of the function at the point x.
c
c     g is a DOUBLE PRECISION array of length n.  If the routine setulb
c       returns with taskb(1:2)= 'FG', then g must be set by the user to
c       contain the components of the gradient at the point x.
c
c     factr is a DOUBLE PRECISION variable that must be set by the user.
c       It is a tolerance in the termination test for the algorithm.
c       The iteration will stop when
c
c        (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr*epsmch
c
c       where epsmch is the machine precision which is automatically
c       generated by the code. Typical values for factr on a computer
c       with 15 digits of accuracy in double precision are:
c       factr=1.d+12 for low accuracy;
c             1.d+7  for moderate accuracy; 
c             1.d+1  for extremely high accuracy.
c       The user can suppress this termination test by setting factr=0.
c
c     pgtol is a double precision variable.
c       On entry pgtol >= 0 is specified by the user.  The iteration
c         will stop when
c
c                 max{|proj g_i | i = 1, ..., n} <= pgtol
c
c         where pg_i is the ith component of the projected gradient.
c       The user can suppress this termination test by setting pgtol=0.
c
c     wa is a DOUBLE PRECISION  array of length 
c       (2mmax + 4)nmax + 12mmax^2 + 12mmax used as workspace.
c       This array must not be altered by the user.
c
c     iwa is an INTEGER  array of length 3nmax used as
c       workspace. This array must not be altered by the user.
c
c     task is a CHARACTER string of length 60.
c       On first entry, it must be set to 'START'.
c       On a return with task(1:2)='FG', the user must evaluate the
c         function f and gradient g at the returned value of x.
c       On a return with task(1:5)='NEW_X', an iteration of the
c         algorithm has concluded, and f and g contain f(x) and g(x)
c         respectively.  The user can decide whether to continue or stop
c         the iteration. 
c       When
c         task(1:4)='CONV', the termination test in L-BFGS-B has been 
c           satisfied;
c         task(1:4)='ABNO', the routine has terminated abnormally
c           without being able to satisfy the termination conditions,
c           x contains the best approximation found,
c           f and g contain f(x) and g(x) respectively;
c         task(1:5)='ERROR', the routine has detected an error in the
c           input parameters;
c       On exit with task = 'CONV', 'ABNO' or 'ERROR', the variable task
c         contains additional information that the user can print.
c       This array should not be altered unless the user wants to
c          stop the run for some reason.  See driver2 or driver3
c          for a detailed explanation on how to stop the run 
c          by assigning task(1:4)='STOP' in the driver.
c
c     iprint is an INTEGER variable that must be set by the user.
c       It controls the frequency and type of output generated:
c        iprint<0    no output is generated;
c        iprint=0    print only one line at the last iteration;
c        0<iprint<99 print also f and |proj g| every iprint iterations;
c        iprint=99   print details of every iteration except n-vectors;
c        iprint=100  print also the changes of active set and final x;
c        iprint>100  print details of every iteration including x and g;
c       When iprint > 0, the file iterate.dat will be created to
c                        summarize the iteration.
c
c     csave  is a CHARACTER working array of length 60.
c
c     lsave is a LOGICAL working array of dimension 4.
c       On exit with task = 'NEW_X', the following information is
c         available:
c       lsave(1) = .true.  the initial x did not satisfy the bounds;
c       lsave(2) = .true.  the problem contains bounds;
c       lsave(3) = .true.  each variable has upper and lower bounds.
c
c     isave is an INTEGER working array of dimension 44.
c       On exit with task = 'NEW_X', it contains information that
c       the user may want to access:
c         isave(30) = the current iteration number;
c         isave(34) = the total number of function and gradient
c                         evaluations;
c         isave(36) = the number of function value or gradient
c                                  evaluations in the current iteration;
c         isave(38) = the number of free variables in the current
c                         iteration;
c         isave(39) = the number of active constraints at the current
c                         iteration;
c
c         see the subroutine setulb.f for a description of other 
c         information contained in isave
c
c     dsave is a DOUBLE PRECISION working array of dimension 29.
c       On exit with task = 'NEW_X', it contains information that
c         the user may want to access:
c         dsave(2) = the value of f at the previous iteration;
c         dsave(5) = the machine precision epsmch generated by the code;
c         dsave(13) = the infinity norm of the projected gradient;
c
c         see the subroutine setulb.f for a description of other 
c         information contained in dsave
c
c     --------------------------------------------------------------
c           END OF THE DESCRIPTION OF THE VARIABLES IN L-BFGS-B
c     --------------------------------------------------------------
c
c     << An example of subroutine 'timer' for AIX Version 3.2 >>
c
c     subroutine timer(ttime)
c     double precision ttime
c     integer itemp, integer mclock
c     itemp = mclock()
c     ttime = dble(itemp)*1.0d-2
c     return
c     end

********************************************************************/
/*******************************************************************
c                       this code uses DRIVER 1
c     --------------------------------------------------------------
c                SIMPLE DRIVER FOR L-BFGS-B (version 2.1)
c     --------------------------------------------------------------
c        *** MODIFIED to drive in C++ (M. Coahran, 11/21/03) ***
c     --------------------------------------------------------------
c
c        L-BFGS-B is a code for solving large nonlinear optimization
c             problems with simple bounds on the variables.
c
c        The code can also be used for unconstrained problems and is
c        as efficient for these problems as the earlier limited memory
c                          code L-BFGS.
c
c        This is the simplest driver in the package. It uses all the
c                    default settings of the code.
c
c
c     References:
c
c        [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
c        memory algorithm for bound constrained optimization'',
c        SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.
c
c        [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
c        Subroutines for Large Scale Bound Constrained Optimization''
c        Tech. Report, NAM-11, EECS Department, Northwestern University,
c        1994.
c
c
c          (Postscript files of these papers are available via anonymous
c           ftp to eecs.nwu.edu in the directory pub/lbfgs/lbfgs_bcm.)
c
c                              *  *  *
c
c        NEOS, November 1994. (Latest revision June 1996.)
c        Optimization Technology Center.
c        Argonne National Laboratory and Northwestern University.
c        Written by
c                           Ciyou Zhu
c        in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
c
c     NOTE: The user should adapt the subroutine 'timer' if 'etime' is
c           not available on the system.  An example for system 
c           AIX Version 3.2 is available at the end of this driver.
c
*******************************************************************/
