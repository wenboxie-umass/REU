#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>
#include <algorithm>
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>
using namespace std;
// type declarations

//#define MIN(a, b) ((a < b) ? a : b)
//#define RATE(a,b) sqrt(MIN(a,b))

//Print real and integer 1D array
template <typename T>
void print_v(T* Array, int size)
{
    if(size >= 0)
    {
        for(int i = 0; i < size; i++)
            cout<< Array[i] << "  ";
        cout<<endl;
    }
    else
    {
        cout<<"Error! negative size"<<endl;
    }
}

double RATE(double a, double b)
{
    if(a > b)
        return sqrt(b);
    else
        return sqrt(a);
}

//find the minimum in an 1D array. Return the minimum value and update min_node to the index of the minimum value
template <typename T>
double findmin(T* Array, const int N, int& min_node)
{
    if(N >= 0)
    {
        T tmp = Array[0];
        int tmpi = 0;
        for(int i = 1; i < N; i++)
        {
            if(tmp > Array[i])
            {
                tmp = Array[i];
                tmpi = i;
            }
        }
        min_node = tmpi;
        return tmp;
    }
    else
    {
        cout<<"Error! negative size"<<endl;
        return 0;
    }
}


void update_process(const int min_node, const int N, const double Tl, const double Tr, const double current_time, double* energy_array, double* time_array,trng::yarn2 &r, trng::uniform01_dist<> &u)
{
    if(min_node < 0)
    {
        cout<<"Error! Negative index!"<<endl;
    }
    else
    {
        double p = u(r);
        if( min_node == 0)
        {
            double old_rate = RATE(energy_array[1],energy_array[2]);
            energy_array[1] = p*( Tl*(-log(u(r))) + energy_array[1]);
            time_array[1] = (old_rate/RATE(energy_array[1],energy_array[2]))*(time_array[1] - current_time) + current_time;
//            time_array[1] = -log(u(r))/RATE(energy_array[1],energy_array[2]) + current_time;
            time_array[0] = current_time  + (-log(1-u(r)))/RATE(energy_array[0],energy_array[1]);
        }
        else if(min_node == N)
        {
            double old_rate = RATE(energy_array[N-1],energy_array[N]);
            energy_array[N] = p*( Tr*(-log(u(r))) + energy_array[N]);
//            time_array[N-1] = current_time  + (-log(u(r)))/RATE(energy_array[N - 1],energy_array[N]);
            time_array[N-1] = (old_rate/RATE(energy_array[N-1],energy_array[N]))*(time_array[N-1] - current_time) + current_time;
            time_array[N] = current_time  + (-log(1-u(r)))/RATE(energy_array[N],energy_array[N+1]);
        }
        else
        {
            double total_tmp = energy_array[min_node] + energy_array[min_node + 1];
        
            double old_rate = RATE(energy_array[min_node - 1], energy_array[min_node]);
            energy_array[min_node] = p*total_tmp;
//            time_array[min_node - 1] = current_time + -log(u(r))/RATE(energy_array[min_node - 1], energy_array[min_node]);
            time_array[min_node - 1] = (old_rate/RATE(energy_array[min_node - 1], energy_array[min_node]))*(time_array[min_node - 1] - current_time) + current_time;
        
            old_rate = RATE(energy_array[min_node + 1], energy_array[min_node + 2]);
            energy_array[min_node + 1] = (1 - p)*total_tmp;
            time_array[min_node + 1] = (old_rate/RATE(energy_array[min_node + 1], energy_array[min_node + 2]))*(time_array[min_node + 1] - current_time) + current_time;
        
//            time_array[min_node] = current_time + (-log(u(r)))/RATE(energy_array[min_node + 1], energy_array[min_node + 2]);
            
            time_array[min_node] = current_time +(-log(1-u(r)))/RATE(energy_array[min_node + 1], energy_array[min_node]);
        
        }
    }

}

bool checkreturn(double* energy_array, const int N){
    double a = 0.1;
    double b = 100;
    for(int y = 1; y < N+1; y++)
    {
        if (energy_array[y] < a || energy_array[y] > b){
            return false;
        }
    }
    return true;
}

int main()
{
    struct timeval t1, t2;
    gettimeofday(&t1,NULL);
    ofstream myfile;
    myfile.open("KMP_ss_return.txt");
    double Tl = 2.0;
    double Tr = 1.0;
    int N = 3;
    int N_thread = 8;
    int Step = 10;
    const int N_trials = 1000000;//number of total number of simulations
    double dt = 10;
    long long int Max_time = 1000;//
    long long int** tail_stat= new long long int* [N_thread];
    for(int i = 0 ; i < N_thread; i++)
        tail_stat[i] = new long long int[Max_time + 1];//tail_stat[i][j] = prob[ return time > j*dt] from thread i;
    for(int i = 0; i < N_thread; i++)
    {
        for(int j = 0; j <= Max_time; j++)
            tail_stat[i][j] = 0;
    }

    for(long long int round = 0; round < 1; round ++)
    {
        const long long int N_trials = 1000000;//number of total number of simulations
 
        #pragma omp parallel num_threads(N_thread)
        {
            long long int rank = omp_get_thread_num();
            long long int size = omp_get_num_threads();
            trng::yarn2 r;
            trng::uniform01_dist<> u;
            r.seed(time(NULL));
            r.split(N_thread, rank);

            double *energy_array = new double[N+2];
            double *time_array = new double[N+1];
            int local_size = (rank + 1)*N_trials/size - rank*N_trials/size;
            double* return_time = new double[local_size];//record return times locally
            
            for(long long int trial = rank*N_trials/size; trial < (rank + 1)*N_trials/size; trial++)
            {
                energy_array[0] = Tl;
                energy_array[N+1] = Tr;
                double T_mid = (Tl + Tr)/2;
                for(int n = 1; n < N+1; n++)
                    energy_array[n] = -log(1-u(r))*T_mid;
                //initial conditions
                for(int n = 0; n < N+1; n++)
                {
                    time_array[n] = (-log(1-u(r)))/RATE(energy_array[n],energy_array[n+1]);
                }
                double current_time = -1;
                while ( (!(checkreturn(energy_array,N)) && current_time < Max_time*dt )|| current_time <= 200)
                {
                    int min_node = 0;
                    current_time = findmin<double>(time_array,N+1,min_node);
                    update_process(min_node, N, Tl, Tr, current_time, energy_array, time_array,r,u);
                }
                return_time[trial - rank*N_trials/size] = current_time - 200;
 
            }

            sort(return_time, return_time + local_size);
            int iter = 0;//location in Time array
            int flag = 0;
            for(int i = 0; i < Max_time ; i++)
            {
                flag = 0;
                while(iter < local_size && flag == 0)
                {
                    if( return_time[iter] > i*dt && return_time[iter] < (i+1)*dt)
                    {
                        tail_stat[rank][i] += local_size - iter;
                        flag = 1;
                        iter ++;
                    }
                    else if(return_time[iter] > (i+1)*dt)
                    {
                        tail_stat[rank][i] += local_size - iter;
                        flag = 1;
                    }
                    else
                    {
                        iter++;
                    }
                }
                
            }
            
            delete[] energy_array;
            delete[] time_array;
            delete[] return_time;

        }//end of parallelization
        
//        print_v<double>(return_time, N_trials);


        cout<<"round "<<round<<endl;
    }
    
    
    long long int tmp;
    for(int i = 0; i < Max_time; i++)
    {
        tmp = 0;
        for(int j = 0; j < N_thread; j++)
        {
            tmp += tail_stat[j][i];
        }
        myfile<< tmp << "  ";

    }
    myfile.close();
    for(int i = 0; i < N_thread; i++ )
    {
        delete[] tail_stat[i];
    }
    delete[] tail_stat;

    gettimeofday(&t2, NULL);
    double delta = ((t2.tv_sec  - t1.tv_sec) * 1000000u +
                    t2.tv_usec - t1.tv_usec) / 1.e6;
    
    cout << "total CPU time = " << delta <<endl;
 
    
}

