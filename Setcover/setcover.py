__license__ = "MIT"
__author__ = "Guangtun Ben Zhu (BGT) @ Johns Hopkins University"
__startdate__ = "2015.12.10"
__name__ = "setcover"
__module__ = "Setcover"
__python_version__ = "3.50"

__lastdate__ = "2015.12.31"
__version__ = "0.90"

import warnings

import numpy as np
from scipy import sparse

# To_do: 
#    1. Remove redundant columns (Calculate the sum first then remove column one by one)
#    2. Clean up wrapper CFT: add fix_col option (to cut time for large instances)
#    3. Better converging criteria needed (to cut time for small instances)
#    4. Documentation!

# Some magic numbes
_stepsize = 0.1
_pi = 0.1
_largenumber = 1E5
_smallnumber = 1E-5

class SetCover:
    """
    Set Cover Problem:
    Instantiation:
       g = SetCover(a_matrix, cost)
    Run the optimization finder:
       g.CFT()
    Once it's done:
       g.s - the (near-optimal) minimal set
       g.total_cost - the (near-optimal) solution 
    """

    def __init__(self, amatrix, cost):
        self.a = np.copy(amatrix)
        self.a_csr = sparse.csr_matrix(amatrix, copy=True) # Compressed sparse row
        self.a_csc = sparse.csr_matrix(amatrix.T, copy=True) # Compressed sparse column (transposed for convenience)
        self.c = np.copy(cost)
        self.mrows = amatrix.shape[0]
        self.ncols = amatrix.shape[1]

        # Magic Numbers 

        ## subgradient method
        self.subg_nadaptive = 20
        self.subg_nsteps = self.subg_nadaptive*15
        self.subg_maxsteps = 100 # How many times we want to perturb the best u and then recalculate 
        self.subg_maxfracchange = 0.00001 # convergence criteria, fractional change
        self.subg_maxabschange = 0.01 # convergence criteria, absolute change
        self.max_adapt = 0.05 # threshold to half the stepsize
        self.min_adapt = 0.005 # threshold to increase the stepsize by 1.5
        self.u_perturb = 0.08 # perturbations

        ## column fixing iteration
        self.colfix_maxfracchange = 0.02 # convergence criteria, fractional change
        self.maxiters = 10
        self.maxfracchange = 0.001 # convergence criteria, fractional change
        self.alpha = 1.1
        self.beta = 1.0

        # setting up
        self.f_uniq = self._fix_uniq_col() # fix unique columns
        self.f = np.copy(self.f_uniq)
        self.f_covered = np.any(self.a[:,self.f], axis=1)
        self.s = np.copy(self.f_uniq) # (current) solution, selected column
        self.u = self._u_naught() # (current best) Lagrangian multiplier
        self.stepsize = _stepsize
        self.pi = _pi

    @property
    def total_cost(self):
        """
        """
        return np.einsum('i->', self.c[self.s])

    def reset_all(self):
        self.stepsize = _stepsize
        self.pi = _pi
        self.reset_f()
        self.reset_s()
        self.reset_u()

    def reset_s(self):
        self.s = np.copy(self.f_uniq) # (current) solution, selected column

    def reset_f(self):
        self.f = np.copy(self.f_uniq)
        self.f_covered = np.any(self.a[:,self.f], axis=1)

    def reset_u(self):
        self.u = self._u_naught()

    #def _u_naught(self):
    def _u_naught_simple(self):
        """
        initial guess of the Lagrangian multiplier
        """
        return np.random.rand(self.mrows) # Random is better to give different multipliers in the subgradient phase

    def _fix_uniq_col(self):
        """
        """
        n_covered_col = self.a_csr.dot(np.ones(self.ncols)) # subgradient; for two boolean arrays, multiplication seems to be the best way (equivalent to logical_and)
        ifix = np.zeros(self.ncols, dtype=bool)
        if (np.count_nonzero(n_covered_col) != self.mrows):
           raise ValueError("There are uncovered rows! Please check your input!")
        if (np.any(n_covered_col==1)):
           inonzero = self.a_csr[n_covered==1,:].nonzero()
           ifix[inonzero[1]] = True

        return ifix

    #def _u_naught_complicated(self):
    def _u_naught(self):
        """
        initial guess of the Lagrangian multiplier
        """
        adjusted_cost = self.c/self.a_csc.dot(np.ones(self.mrows))
        cost_matrix = adjusted_cost*self.a + np.amax(adjusted_cost)*(~self.a)
        return adjusted_cost[np.argmin(cost_matrix, axis=1)]

    def greedy(self, u=None, niters_max=1000):
        """
        heuristic greedy method to select a set of columns to cover all the rows
        """

        niters = 1
        if (u is None):
            u = self.u

        utmp = np.copy(u)
        iuncovered = ~np.any(self.a[:,self.s], axis=1)
        
        score = np.zeros(self.ncols)
        while (np.count_nonzero(iuncovered) > 0) and (niters <= niters_max):
            mu = (self.a_csc.dot((iuncovered).astype(int))).astype(float) # It's 5 times faster without indexing, the advantage is made possible by csc_matrix.dot
            mu[mu<=_smallnumber] = _smallnumber

            utmp[~iuncovered] = 0
            gamma = (self.c - self.a_csc.dot(utmp))
            select_gamma = (gamma>=0)

            if (np.count_nonzero(select_gamma)>0):
                score[select_gamma] = gamma[select_gamma]/mu[select_gamma]
            if (np.count_nonzero(~select_gamma)>0):
                score[~select_gamma] = gamma[~select_gamma]*mu[~select_gamma]

            inewcolumn = (np.nonzero(~self.s)[0])[np.argmin(score[~self.s])]
            self.s[inewcolumn] = True
            iuncovered = ~np.logical_or(~iuncovered, self.a[:,inewcolumn])
            niters = niters+1
        if (niters == niters_max): 
           warnings.warn("Iteration reaches maximum = {0}".format(niters_max))
        #else:
        #   pass
        #   print("Success with {0} iterations!".format(niters))
        return self.total_cost

    def update_core(self):
        """
        Removing fixed columns
        """
        if (~np.any(self.f)):
           a_csr = sparse.csr_matrix(self.a, copy=True) # Compressed sparse row
           a_csc = sparse.csr_matrix(self.a.T, copy=True) # Compressed sparse column (transposed for convenience)
        else:
           a_csr = sparse.csr_matrix(self.a[:,~self.f][~self.f_covered,:], copy=True) # Compressed sparse row
           a_csc = sparse.csr_matrix(self.a[:,~self.f][~self.f_covered,:].T, copy=True) # Compressed sparse column (transposed for convenience)
        return (a_csr, a_csc)

    def subgradient(self):
        """
        subgradient step for the core problem N\S. 
        """
        
        UB_full = self.total_cost
        ufull = np.copy(self.u)

        # Update core: possible bottleneck
        (a_csr, a_csc) = self.update_core()
        mrows = a_csr.shape[0]
        ncols = a_csr.shape[1]
        u_this = self.u[~self.f_covered]
        UB_fixed = np.einsum('i->', self.c[self.f])
        UB = UB_full - UB_fixed
        cost = self.c[~self.f]

        u_sequence = np.zeros((mrows, self.subg_nsteps)) # save nsteps calculations (Lagrangian multipliers and lower bounds)
        Lu_sequence = np.zeros(self.subg_nsteps)
        # update u
        x = np.zeros(ncols, dtype=bool) # has to be integer to use scipy sparse matrix
        niters_max = self.subg_maxsteps
        maxfracchange = self.subg_maxfracchange
        maxabschange = self.subg_maxabschange

        # initialization
        f_change = _largenumber
        a_change = _largenumber
        niters = 0
        Lu_max0 = 0
        while ((f_change>maxfracchange) or (a_change>maxabschange)) and (niters<niters_max):
            u_this = (1.0+(np.random.rand(mrows)*2.-1)*self.u_perturb)*u_this
            u_sequence[:,0] = u_this
            cost_u = cost - a_csc.dot(u_sequence[:,0]) # Lagrangian cost
            Lu_sequence[0] = np.einsum('i->', cost_u[cost_u<0])+np.einsum('i->', u_sequence[:,0]) # next lower bound of the Lagrangian subproblem

            for i in np.arange(self.subg_nsteps-1):
                # current solution to the Lagrangian subproblem
                x[0:] = False
                x[cost_u<0] = True

                # core problem
                # x[self.f] = True # ignore those fixed

                s_u = 1. - a_csr.dot(x.astype(int)) # subgradient; for two boolean arrays, multiplication seems to be the best way (equivalent to logical_and)
                s_u_norm = np.einsum('i,i',s_u,s_u) # subgradient's norm squared

                # Update
                u_temp = u_sequence[:,i]+self.stepsize*(UB - Lu_sequence[i])/s_u_norm*s_u # next Lagrangian multiplier
                u_temp[u_temp<0] = 0

                u_sequence[:,i+1] = u_temp
                cost_u = cost - a_csc.dot(u_sequence[:,i+1]) # Lagrangian cost
                Lu_sequence[i+1] = np.einsum('i->', cost_u[cost_u<0])+np.einsum('i->', u_sequence[:,i+1]) # next lower bound of the Lagrangian subproblem
            
                # Check the last nadaptive steps and see if the step size needs to be adapted
                if (np.mod(i+1,self.subg_nadaptive)==0):
                    Lu_max_adapt = np.amax(Lu_sequence[i+1-self.subg_nadaptive:i+1])
                    Lu_min_adapt = np.amin(Lu_sequence[i+1-self.subg_nadaptive:i+1])
                    if (Lu_max_adapt <= 0.):
                        Lu_max_adapt = _smallnumber
                    f_change_adapt = (Lu_max_adapt-Lu_min_adapt)/np.fabs(Lu_max_adapt)
                    if f_change_adapt > self.max_adapt:
                        self.stepsize = self.stepsize*0.5
                    elif f_change_adapt < self.min_adapt:
                        self.stepsize = self.stepsize*1.5
                    # swap the last multiplier with the optimal one
                    i_optimal = np.argmax(Lu_sequence[i+1-self.subg_nadaptive:i+1])
                    if (i_optimal != (self.subg_nadaptive-1)):
                       u_temp = u_sequence[:,i]
                       u_sequence[:,i] = u_sequence[:,i+1-self.subg_nadaptive+i_optimal]
                       u_sequence[:,i+1-self.subg_nadaptive+i_optimal] = u_temp
                       Lu_sequence[i+1-self.subg_nadaptive+i_optimal] = Lu_sequence[i]
                       Lu_sequence[i] = Lu_max_adapt

            Lu_max = np.amax(Lu_sequence)
            i_optimal = np.argmax(Lu_sequence)
            u_this = u_sequence[:,i_optimal]
            niters = niters + 1
            a_change = Lu_max - Lu_max0
            f_change = a_change/np.fabs(Lu_max)
            Lu_max0 = Lu_max # Just a copy. Not the reference (It's a number object)
            if (niters == niters_max): 
                warnings.warn("Iteration reaches maximum = {0}".format(niters))
            #if ((f_change<=maxfracchange) and (a_change<=maxabschange)):
            #    print("Found a near-optimal solution!!!: n_iter={0}, Lu_max={1} for {2} uncovered rows ".format(niters, Lu_max, u_this.size))

        # update multipliers
        self.u[~self.f_covered] = u_this

        # return the last nsteps multipliers
        u_sequence_full = np.zeros((self.mrows, self.subg_nsteps)) # save nsteps calculations (Lagrangian multipliers and lower bounds)
        Lu_sequence_full = np.zeros(self.subg_nsteps)
        u_sequence_full[self.f_covered,:] = self.u[self.f_covered][:, np.newaxis]
        u_sequence_full[~self.f_covered,:] = u_sequence

        cost_u_full = self.c - self.a_csc.dot(u_sequence_full[:,0]) # Lagrangian cost
        Lu_sequence_full[0] = np.einsum('i->', cost_u_full[cost_u_full<0])+np.einsum('i->', u_sequence_full[:,0]) # next lower bound of the Lagrangian subproblem
        Lu_sequence_full = Lu_sequence + (Lu_sequence_full[0] - Lu_sequence[0])

        return (u_sequence_full, Lu_sequence_full)

    def subgradient_solution(self, u=None):
        """
        """
        if (u is None):
            u = np.copy(self.u)
        cost_u = self.c - self.a_csc.dot(u) # Lagrangian cost
        x = np.zeros(self.ncols, dtype=bool) # has to be integer to use scipy sparse matrix
        x[cost_u<0] = True # current solution to the Lagrangian subproblem
        return x

    def fix_col(self, u=None):
        """
        Note this needs to be done after greedy()
        """
        # calculate delta (gap between the lower and upper bounds)
        if (u is None): 
            u = self.u

        utmp = np.copy(u)
        cost_u = self.c - self.a_csc.dot(utmp) # Lagrangian cost

        #x = np.zeros(self.ncols, dtype=bool) # has to be integer to use scipy sparse matrix
        x = np.copy(self.s)
        #x[cost_u<0] = True # current solution to the Lagrangian subproblem

        UB = self.total_cost
        LB = np.einsum('i->', cost_u[cost_u<0])+np.einsum('i->', utmp) # next lower bound of the Lagrangian subproblem

        cost_u[cost_u<0] = 0.
        s_u = (self.a_csr.dot(x.astype(int))).astype(float) # the number of columns that cover each row
        if (np.count_nonzero(s_u)<1.):
           raise ValueError("The solution you give is not a solution!")
        #gap = utmp*(s_u-1.)/s_u
        gap = utmp*(s_u-1.)/s_u
        delta = cost_u + self.a_csc.dot(gap)
        #delta = cost_u + self.a_csc.dot(utmp) - (np.einsum('i->', utmp)*self.a_csc.dot(np.ones(self.mrows)))/(self.a_csc[x,:].nonzero()[0].size)
        isort = np.argsort(delta[x])

        print("Gap = {0}, UB-LB = {1}.".format(np.sum(delta[x]), UB-LB))

        # total (cumulative) number of covered rows by the first j columns
        n_covered_row = self.a_csc.dot(np.ones(self.mrows))
        ntotal_covered_row = np.cumsum(n_covered_row[x][isort])
        iwhere = (np.where(ntotal_covered_row>=(self.mrows*self.pi)))[0]
        self.reset_f()
        if (iwhere.size>0):
           itemp = iwhere[0]
           self.f[x.nonzero()[0][isort[0:itemp]]] = True # This doesn't work because the left hand side returns a copy, not a view
        self.f_covered = np.any(self.a[:,self.f], axis=1)

    def CFT(self):
        """
        """

        # Some predicates
        Lu_min = 0.
        niters_max = self.maxiters
        maxfracchange = self.maxfracchange

        # initialization, resetting ...
        self.reset_all() # including _u_naught(), first application
        scp_min = self.greedy()
        # self.fix_col() # Let's not fix any column the first time

        # column fixing iteration
        niters = 0
        f_change = _largenumber
        while (f_change>maxfracchange) and (niters<niters_max):
            u_tmp, Lu_tmp = self.subgradient() # find a near-optimal solution 
            u, Lu = self.subgradient() # rerun subgradient to get a set of Lagrangian multipliers

            scp_all = np.zeros(self.subg_nsteps)
            for i in np.arange(self.subg_nsteps):
                self.reset_s()
                scp_all[i] = self.greedy(u=u[:,i])

            # check if the solution is gettting better
            imin_tmp = (np.where(scp_all==np.amin(scp_all)))[0]
            imin = imin_tmp[np.argmax(Lu[imin_tmp])]
            imax = np.argmax(Lu)
            print("Best solution: UB={0}, LB={1}, UB1={2}, LB1={3}".format(scp_all[imin], Lu[imin], scp_all[imax], Lu[imax]))
            if (niters==0) or ((scp_all[imin]<=scp_min) and ((Lu[imin]-Lu_min)>-(np.fabs(Lu_min)*self.colfix_maxfracchange))):
                scp_min = scp_all[imin]
                print(scp_min)
                u_min = np.copy(u[:,imin])
                Lu_min = Lu[imin]
                #self.stepsize = _stepsize

            # final step, needs to get u_min back
            # self.fix_col(u=u[:,imin])
            # self.s = np.copy(self.f)
            self.reset_s()
            UB1 = self.greedy(u=u[:,imin])

            self.reset_s()
            self.u = np.copy(u_min)
            UB = self.greedy()
            LB = Lu_min

            #self.pi = self.pi*self.alpha
            #if (self.pi>0.8):
            #   self.pi = 0.8
            #self.fix_col()

            GAP = (UB-LB)/np.fabs(UB)
            f_change = GAP
            print("UB={0}, UB1={1}, LB={2}, change={3}%".format(UB,UB1,LB,f_change*100.))
            niters = niters + 1
            if (niters == niters_max): 
                warnings.warn("Iteration reaches maximum = {0}".format(niters))

        # Need to remove redundant columns

        print("Best solution: {0}".format(UB))
        #return scp_all

