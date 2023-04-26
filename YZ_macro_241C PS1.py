import numpy as np
import matplotlib.pyplot as plt
import os  # lets you set your working directory easily
from numpy import linalg 
import matplotlib.ticker as ticker
import pandas as pd
# Set your working directory to save files into
#os.chdir('Users/zhouy/Desktop/UCSB/Academic years/2022Core/ECON 204C-macro/204C ps/ps1')




class FindValPlicyFunc(object):
    def __init__(self,alpha,beta,A,N,stepfunc, part):
        self.alpha=alpha
        self.beta=beta
        self.A=A
        self.N=N
        self.stepfunc=stepfunc
        self.part=part

    def iterValFunc(self,figname,**kwargs):
        #initiate capital grid and return matrix 
        ##steady-state capital stock
        k_ss = (self.A*self.alpha*self.beta)**(1/(1-self.alpha))
        #print(alpha,beta,A,N,k_ss)
        #create the 5 element capital grid
        k_grid = np.zeros(self.N)
        #populate the grid with 5 entries
        for i in range(self.N):
            k_grid[i] = k_ss*(self.stepfunc(i))
            #k_grid[i] = k_ss*(1 + 1/(2*N) * (i-(np.floor(N/2)))) #np.floor: the nearest smaller integer for every element, prevent negative consumption for this length of step
        ##check the grid
        #print(k_grid)

        # create blank v_0 array
        v_0 = np.zeros(self.N) #array([0., 0., 0., 0., 0.])
        #create grid of return function:
        """
        This is essentially the period return (in utils) given by the implied consumption resulting from choosing some 
        next-period capital level k' when we start with capital k.

        In matrix form, let row i be current capital stock and column j be our choice of next-period's capital stock. 
        We can then form a matrix of utility returns from the consumption level implied by this k and k' pair that is invariant over time.
        We now don't need to solve it every iteration of the bellman operator.
        """
        return_matrix = np.zeros([self.N,self.N],)
        for i in range(self.N):
            for j in range(self.N):
                #solve each return
                if kwargs is None:
                #log utility
                    return_matrix[i,j] = np.log(self.A*k_grid[i]**self.alpha - k_grid[j]) #ln(f(k)-k')=ln(c) fix k, iterate k', same first component for all value func
                else:
                #CRRA,σ=0.5
                    return_matrix[i,j]=((self.A*k_grid[i]**self.alpha - k_grid[j])**(1-kwargs["sigma"])-1)/(1-kwargs["sigma"])
        if self.part=="d":     
            ##V1
            value_matrix = return_matrix + self.beta * np.tile(v_0,(self.N,1)) # N*N mx of 0s
            ##Solve for the maximum in each row
            v_1 = np.amax(value_matrix, axis = 1) #Maxima in each row, expression of v(1) is determined after get the maxima
            #v1 is decribed by the (maximized) value at k iterating k'. when the grid is dense, we kinda get the shape of v1, which means we get the func of v1
            #for each k, find the maxima of V(1) corresponding to a certain k'

            ##V2
            value_matrix = return_matrix + self.beta * np.tile(v_1,(self.N,1)) #Since v1 is fixed, copy v1 for each k of v2
            # use the same return mx since the first component is the same for all v
            v_2 = np.amax(value_matrix, axis = 1)

            ##V3 
            value_matrix = return_matrix + self.beta * np.tile(v_2,(self.N,1)) 
            v_3 = np.amax(value_matrix, axis = 1)


            plt.figure()
            plt.plot(k_grid,v_1, alpha=0.8, linewidth=1, label='V_1(k)')
            plt.plot(k_grid,v_2, alpha=0.8, linewidth=1, label='V_2(k)')
            plt.plot(k_grid,v_3, alpha=0.8, linewidth=1, label='V_3(k)')
            plt.grid(True, linestyle='dotted')
            plt.xlabel('Capital Stock k')
            plt.ylabel('Value v(k)')
            plt.title('Value Function')
            plt.legend()
            plt.savefig(figname, dpi=300)
            plt.show()
        
        if self.part=="e":
            v=[]
            k_map_k_prime=[]
            v.append(np.zeros(self.N)) 
            value_matrix = return_matrix + self.beta * np.tile(v[0],(self.N,1)) 
            v.append(np.amax(value_matrix, axis = 1))
            k_map_k_prime.append(value_matrix.argmax(axis=1))
            n=1
            while np.max(abs(v[n]-v[n-1]))>10**(-4):
                value_matrix = return_matrix + self.beta * np.tile(v[n],(self.N,1)) 
                v.append(np.amax(value_matrix, axis = 1))
                k_map_k_prime.append(value_matrix.argmax(axis=1))
                n+=1
            final_kprime=[k_grid[i] for i in k_map_k_prime[-1]]
            fig=plt.figure(dpi=300, figsize=(13,7))
            ax1=fig.add_subplot(1, 2, 1)
            ax2=fig.add_subplot(1, 2, 2)
            ax1.plot(k_grid,v[-1], alpha=0.8, linewidth=1, label='V_n(k)')
            ax1.set_title("Value Function")
            ax1.set_xlabel('Capital Stock k')
            ax1.set_ylabel('Value v(k)')
            ax1.grid(True, linestyle='dotted')
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax1.legend()

            ax2.plot(k_grid,final_kprime, alpha=0.8, linewidth=1, label='k_prime')
            ax2.set_title("Policy Function")
            ax2.set_xlabel('Capital Stock k')
            ax2.set_ylabel('Policy Function g(k)')
            ax2.grid(True, linestyle='dotted')
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax2.legend()
            if kwargs is None:
                plt.suptitle(f"alpha={self.alpha}, beta={self.beta}, A={self.A}, N={self.N}") 
                plt.savefig(f"{figname}.png", dpi=300)
            else:
                plt.suptitle(f"alpha={self.alpha}, beta={self.beta}, A={self.A}, N={self.N}, sigma={kwargs['sigma']}") #cannot use kwargs["sigma"]
                plt.savefig(f"{figname}_{kwargs['sigma']}.png", dpi=300)
            plt.show()
            policyFunc=pd.DataFrame(k_map_k_prime[-1],index=k_grid, columns=["k_prime"])
            valueFunc=pd.DataFrame(v[-1],index=k_grid, columns=["v(k)"])
            return k_ss,k_grid,valueFunc, policyFunc, v[-1], k_map_k_prime[-1]

    def FindOptimalCK(self,T,figname,**kwargs):
        k_ss, k_grid,valueFunc, policyFunc, valFun,policy=self.iterValFunc(figname,**kwargs)
        k0=0.9*k_ss
        ctTable=pd.DataFrame(index=list(range(0,T+2)),columns=["ct", "k_t"])
        ctTable.iloc[0,1]=k0
        for t in range(ctTable.index[-1]):
            if kwargs is None:
                ctTable.iloc[t+1,1]=self.alpha*self.beta*self.A*(ctTable.iloc[t,1]**self.alpha)
                ctTable.iloc[t,0]=self.A*(ctTable.iloc[t,1]**self.alpha)-ctTable.iloc[t+1,1]
            else:
                ctTable.iloc[t+1,1]=ctTable.iloc[t,1]*(self.beta*self.A)**(1/kwargs['sigma'])
                ctTable.iloc[t,0]=(1-(self.beta**(1/kwargs['sigma']))*(self.A**((1-kwargs['sigma'])/kwargs['sigma'])))*self.A*ctTable.iloc[t,1]
        
        return ctTable
        # for t in ctTable.index:
        #     i=t+1
        #     ctTable.iloc[i,1]=policyFunc.loc[ctTable.iloc[i-1,1]]
        #     ctTable.iloc[i-1,0]=ctTable.iloc[i,1]-ctTable.iloc[i-1,1]
        # return ctTable







if __name__ == "__main__":
    #part de
    stepD=lambda x: 1 + 0.1 * (x-3)
    Q1e=FindValPlicyFunc(0.3,0.6,20,5,stepD, "e")
    Q1e.iterValFunc("part_e")
    #part f
    stepF=lambda x: 1 + 0.2 * (x-101)/100
    Q1f=FindValPlicyFunc(0.3,0.6,20,201,stepF, "e")
    ctTable=Q1f.FindOptimalCK(100,"part_f")
    ctTable.to_csv('g_ctTable.csv')
    #part h
    Q1h=FindValPlicyFunc(0.3,0.85,20,201,stepF, "e")
    ctTable2=Q1h.FindOptimalCK(100,"part_h")
    ctTable2.to_csv('h_ctTable.csv')
    #part i
    Q1i=FindValPlicyFunc(0.3,0.85,20,201,stepF, "e")
    ctTable2=Q1i.FindOptimalCK(100,"part_i",sigma=0.5)
    ctTable2.to_csv('i_0.5_ctTable.csv')

    Q1i=FindValPlicyFunc(0.3,0.85,20,201,stepF, "e")
    ctTable2=Q1i.FindOptimalCK(100,"part_i",sigma=3)
    ctTable2.to_csv('i_3_ctTable.csv')
