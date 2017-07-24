#include <iostream>
#include <fstream>
#include <random>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <list>
#include <vector>
#include <map>
using namespace std;

typedef pair<int, int> Pair;
typedef vector<Pair> P_list;
typedef vector<int> List;
typedef vector<double> Double_list;
typedef List::iterator iter;
typedef map<Pair, bool> Dictionary;

const double TL = 1.0;
const double TR = 2.0;

/*
horizontal:
        |-------------|
        |             |
        |             |
  |------------c------------|
  |     |             |     |
  |     c             c     |
  |     |             |     |
  |-----------c_i-----------|
  |     |             |     |
  |     c             c     |
  |     |             |     |
  |------------c------------|
        |             |
        |             |
        |-------------|

vertical:
        |------|------|
        |      |      |
  |--------c-------c--------|
  |     |      |      |     |
  |     c     c_i     c     |
  |     |      |      |     |
  |--------c-------c--------|
        |      |      |
        |------|------|
*/

enum Direction {
    horizontal,
    vertical
 };

struct interaction
{
    double time;
    int location; // Store the index in 1D time_array
    Pair coordinate;
    List adjacent_clocks;
    Direction direction; //indicates position of clock. it can be either on vertical line or on horizontal line
    interaction* left;
    interaction* right;
};

struct site {
    double previous_time;
    double previous_energy;
    Pair coordinate;
    Double_list record;
    double summation;
};

P_list find_coordinate_of_site_in_energy_array(interaction info, int rows, int cols);

int find_index(int i, int j, int N, int M, Direction direction)
{
	if(direction == vertical)
	{
		return (int)((M + 1) * (int)(i / 2) + M * (int)(i / 2) + (j));
	}

	else
	{
		return (int)((M + 1) * (int)(i / 2 + 1) + M * (int)(i / 2) + (j));
	}
}

double rate_function(double x, double y)
{
  return sqrt(x+y);
}

//find_adjacent_clocks will return all possible clocks which are adjacent to clock at (i,j).
List find_adjacent_clocks(int i, int j, Direction direction, int N, int M, int current_index) {
    List lst;
    switch (direction) {
      case vertical:
          if(i != 0) {
              if(j != 0) {
                  //we have a clock at (i - 1, j - 1);
                  lst.push_back(find_index(i - 1, j - 1, N, M, horizontal));
              }

              if(j != M) {
                  //we have a clock at (i - 1, j)
                  lst.push_back(find_index(i - 1, j, N, M, horizontal));
              }
          }

          if(i != 2 * N - 2) {
              if(j != 0) {
                  //we have a clock at (i + 1, j - 1);
                  lst.push_back(find_index(i + 1, j - 1, N, M, horizontal));
              }

              if(j != M) {
                  //we have a clock at (i + 1, j)
                  //cout<<i + 1<<","<<j<<", index: "<<find_index(i + 1, 0, N, M, horizontal)<<endl;
                  lst.push_back(find_index(i + 1, j, N, M, horizontal));
              }
          }

          if(j != 0) {
              lst.push_back(find_index(i, j - 1, N, M, vertical));
          }

          if(j != M) {
              lst.push_back(find_index(i, j + 1, N, M, vertical));
          }
          break;
      case horizontal:
          if(i != 1) {
              lst.push_back(find_index(i - 2, j, N, M, horizontal));
          }

          if(i != 2 * N - 3) {
              lst.push_back(find_index(i + 2, j, N, M, horizontal));
          }

          lst.push_back(find_index(i - 1, j, N, M, vertical));
          lst.push_back(find_index(i - 1, j + 1, N, M, vertical));
          lst.push_back(find_index(i + 1, j, N, M, vertical));
          lst.push_back(find_index(i + 1, j + 1, N, M, vertical));
          break;
    }
    //lst.push_back(10);
    return lst;
}

double energy_summation(double **energy_array, int x, int y, Direction direction) {
    double ret = 10.0;
    switch (direction) {
      case vertical:
        ret = rate_function(energy_array[x / 2][y],energy_array[x / 2][y + 1]);
        break;
      case horizontal:
        ret = rate_function(energy_array[int(x / 2)][y + 1], energy_array[int(x / 2) + 1][y + 1]);
        break;
    }
    return ret;
}

void print_v(double* Array, int size)
{
    for(int i = 0; i < size; i++)
        cout<< Array[i] << "  ";
    cout<<endl;
}

int find_min(interaction* Link)
{
        double tmp = Link->time;
        interaction* pt = Link;
        interaction* tmp_pt = Link;
        while(tmp_pt != NULL)
        {
            if(tmp_pt->time < tmp)
            {
                tmp = tmp_pt->time;
                pt = tmp_pt;
            }
            tmp_pt = tmp_pt->right;
        }

        return pt->location;

}

void remove(interaction** Link, interaction* pt)
//remove pt from link but keep pt for future use
{
    if( *Link == NULL || pt == NULL)
        return;
    else if( pt->left == NULL && pt->right == NULL)
    {
        *Link = NULL;
        return;
    }
    else
    {
        if((*Link) == pt )
            *Link = pt->right;
        if( pt->right != NULL)
            pt->right->left = pt->left;
        if( pt->left != NULL)
            pt->left->right = pt->right;
    }
    pt->left = NULL;
    pt->right = NULL;
}

void push_front(interaction** Link, interaction* pt)
//push the interaction pointed by pt into the front of the list
{
    pt->left = NULL;
    if(*Link == NULL)
    {
        pt->right = NULL;
        *Link = pt;
    }
    else
    {
        //cout<<"ok";
        pt->right = *Link;
        //if((*Link)->left == NULL) cout<<"NULL"<<endl;
        (*Link)->left = pt;
         *Link = pt;
    }
}

void print_list(interaction* Link)
{
    interaction* tmp = Link;
    while(tmp!= NULL)
    {
        //cout<<" location: "<< tmp->location <<" time: " << tmp->time << "  " ;
        tmp = tmp->right;
    }
    cout<<endl;
}


void big_step_distribute(interaction** &clock_time_in_step, interaction* time_array, const int N, const double small_tau, const int ratio, const int Step)
//distribute clock times of a big step into vectors that represent small steps. If the clock time is bigger than a big tau, then it is arranged in the right location
{
    for(int i = 0; i < N; i++)
    {
        int tmp;
        if(time_array[i].time > (Step + 1)*ratio*small_tau)
        {
            tmp = ratio;
        }
        else
        {
            tmp = int((time_array[i].time - ratio*small_tau*Step)/small_tau);
        }

      //  if(tmp < 0 || tmp > ratio) {
      //      tmp = ratio;
      //  }

      //cout<<"Time: "<<time_array[i].time<<" Tmp: "<<tmp<<endl;

        push_front(&clock_time_in_step[tmp], &time_array[i] );
        //cout<<tmp<<endl;


    }
}



void move_interaction(interaction** &clock_time_in_step, interaction* pt, const double small_tau, const int ratio, const int Step, const double new_time)
//move the interaction pointed by *pt from old bucket to new bucket
//n_move1: move without relinking pointers
//n_move2: move with relinking pointers
{
    double old_time = pt->time;
    int old_level, new_level;
    pt->time = new_time;
    if (old_time > (Step + 1)*ratio*small_tau )
    {
        old_level = ratio;
    }
    else
    {
        old_level = int( (old_time - Step*ratio*small_tau)/small_tau );
    }

    if ( new_time > (Step + 1)*ratio*small_tau )
    {
        new_level = ratio;
    }
    else
    {
        new_level = int( (new_time - Step*ratio*small_tau)/small_tau );
    }
    //    cout<<"start to move "<< pt->location << " from " << old_level << " to " << new_level<<endl;
    if(old_level == new_level )
    {
        pt->time = new_time;
    }
    else
    {
        remove(&(clock_time_in_step[old_level]), pt);
        push_front(&(clock_time_in_step[new_level]), pt);
    }
}



void update(interaction** &clock_time_in_step, const int level, const int N, const int M, const double small_tau, const int ratio, const int Step, interaction* time_array, double **energy_array, uniform_real_distribution<double> &u, mt19937 &mt, int *count, site** E_avg_profile)
//update clock_time_in_step[level]
{
    //cout<<"Start!"<<endl;
    double next_time = (Step*ratio + level + 1)*small_tau;
    int min_loc = find_min(clock_time_in_step[level]); //find_min should return index of clock in time_array
    //cout<<"OK, Min_Loc: "<<min_loc<<endl;
    interaction* pt = &time_array[min_loc];
    double current_time = pt->time;
//    cout<<"at level " << level << endl;
    Direction direction_min_loc = pt -> direction;
    Pair coordinate_min_loc = pt -> coordinate; // Find the coordinate of min_loc in 2D energy_array;
    int x = coordinate_min_loc.first / 2; //Compute the x coordinate of cell that the min_loc is corresponding to.
    int y = coordinate_min_loc.second;//(direction_min_loc == vertical) ? coordinate_min_loc.second : coordinate_min_loc.second + 1; //Compute the y coordinate of cell......
    List* adjacent_clocks_min_loc = &(pt -> adjacent_clocks);
    Dictionary old_e_value_classifier; // To determine old_e_value I should use to compute the tmp_double
    Pair left = (direction_min_loc == vertical) ? make_pair(x, y) : make_pair(x, y + 1);
    Pair right = (direction_min_loc == vertical) ? make_pair(x, y + 1) : make_pair(x + 1, y + 1);
    old_e_value_classifier[left] = true;
    old_e_value_classifier[right] = true;

    Pair tmp_loc;
    Pair tmp_left, tmp_right;
    int tmp_x = -1;
    int tmp_y = -1;
    Direction dir = vertical;
    //int a = 0;
    //cout<<"I'm In the Function"<<endl;
    double total_energy, tmp_double, old_e_left, old_e_right, tmp_rnd_uni, tmp_rate;

    P_list tmp_list;
    double pre_time = 0.0;
    //double curr_time = 0.0;
    double curr_energy = 0.0;

    while(current_time < next_time)
    {
        //cout<<"OK"<<a<<endl;
//        old_e_value_classifier[left] = true;
//        old_e_value_classifier[right] = true;
        (*count)++;
        // a++;

        tmp_list = find_coordinate_of_site_in_energy_array(time_array[min_loc], N, M);
        for(P_list::iterator i = tmp_list.begin() ; i != tmp_list.end() ; i++) {
            //E_avg_profile[(*i).first][(*i).second].previous_time = time_array[min_loc].time;
            E_avg_profile[(*i).first][(*i).second].previous_energy = energy_array[(*i).first][(*i).second];
            //cout<<"("<<(*i).first<<","<<(*i).second<<"),";
        }
        //Step 1: update min interaction and energy
        total_energy =  (direction_min_loc == vertical) ? energy_array[x][y]+ energy_array[x][y + 1] : energy_array[x][y + 1] + energy_array[x + 1][y + 1];
        tmp_rate = (direction_min_loc == vertical) ? rate_function(energy_array[x][y],energy_array[x][y + 1]) : rate_function(energy_array[x][y + 1] , energy_array[x + 1][y + 1]); //If the direction is horizontal, then we are going to sum the energy of cell at index (x, y) and (x, y + 1). Otherwise, we have to compute the total energy of cell at index (x , y + 1) and (x + 1, y + 1);
        tmp_double = -log(1 - u(mt))/tmp_rate;
//        cout<<"added time = " << tmp_double << endl;
        old_e_left = (direction_min_loc == vertical) ? energy_array[x][y] : energy_array[x][y + 1];
        old_e_right = (direction_min_loc == vertical) ? energy_array[x][y + 1] : energy_array[x + 1][y + 1];
        Pair old_e_left_coordinate, old_e_right_coordinate;
        old_e_left_coordinate = (direction_min_loc == vertical) ? make_pair(x, y) : make_pair(x, y + 1);
        old_e_right_coordinate = (direction_min_loc == vertical) ? make_pair(x, y + 1) : make_pair(x + 1, y + 1);
        tmp_rnd_uni = u(mt);
        if(y == 0 && direction_min_loc == vertical)
        {
            total_energy = old_e_right  -log(1 - u(mt))*TL;
        }
        if(y == M && direction_min_loc == vertical)
        {
            total_energy = old_e_left  -log(1 - u(mt))*TR;
        }

        if(direction_min_loc == vertical) {
            if(y != 0) {
                energy_array[x][y] = tmp_rnd_uni*total_energy;
            }

            if(y != M) {
                energy_array[x][y + 1] = (1 - tmp_rnd_uni)*total_energy;//update energy
            }
        }
        else {
            energy_array[x][y + 1] = tmp_rnd_uni*total_energy;
            energy_array[x + 1][y + 1] = (1 - tmp_rnd_uni)*total_energy;//update energy
        }
        move_interaction(clock_time_in_step, pt,small_tau,ratio, Step,  current_time + tmp_double);

        for(P_list::iterator i = tmp_list.begin() ; i != tmp_list.end() ; i++) {
//                    if((*i).first == 0 &&  (*i).second == 0) {
//                        //cout<<"OK";
//                        cout<<"Info: "<<time_array[min_loc].coordinate.first<<","<<time_array[min_loc].coordinate.second<<endl;
//                    }
            //cout<<(*i).first<<","<<(*i).second<<endl;
            pre_time = E_avg_profile[(*i).first][(*i).second].previous_time;
            //curr_time = time_array[min_loc].time;
            curr_energy = E_avg_profile[(*i).first][(*i).second].previous_energy;//energy_array[(*i).first][(*i).second];
            //cout<<curr_energy * (curr_time - pre_time)<<endl;
            //E_avg_profile[(*i).first][(*i).second].record.push_back(curr_energy * (curr_time - pre_time));
            E_avg_profile[(*i).first][(*i).second].summation += (curr_energy * (current_time - pre_time));
            E_avg_profile[(*i).first][(*i).second].previous_time = current_time;
        }

        //Step 2: update other interactions
        for(iter i = adjacent_clocks_min_loc->begin() ; i != adjacent_clocks_min_loc->end() ; i++) {
            pt = &time_array[(*i)];
            dir = pt -> direction;
            tmp_loc = pt -> coordinate;
            tmp_x = tmp_loc.first / 2;
            tmp_y = tmp_loc.second;
            if(dir == vertical) {
                //cout<<"Vertical"<<endl;
                tmp_left = make_pair(tmp_x, tmp_y);
                tmp_right = make_pair(tmp_x, tmp_y + 1);
                if(old_e_value_classifier[tmp_left] && direction_min_loc == vertical) {
		  tmp_double = (pt->time - current_time)*rate_function(energy_array[tmp_right.first][tmp_right.second],old_e_right)/energy_summation(energy_array, tmp_loc.first, tmp_loc.second, dir) + current_time;
                    move_interaction(clock_time_in_step, pt,small_tau,ratio, Step, tmp_double);
                }

                if(old_e_value_classifier[tmp_right] && direction_min_loc == vertical) {
		  tmp_double = (pt->time - current_time)*rate_function(energy_array[tmp_left.first][tmp_left.second], old_e_left)/energy_summation(energy_array, tmp_loc.first, tmp_loc.second, dir) + current_time;
                    move_interaction(clock_time_in_step, pt,small_tau,ratio, Step, tmp_double);
                }

                if(old_e_value_classifier[tmp_left] && direction_min_loc == horizontal) {
                    if(tmp_left == left) {
		      tmp_double = (pt->time - current_time)*rate_function(energy_array[tmp_right.first][tmp_right.second], old_e_left)/energy_summation(energy_array, tmp_loc.first, tmp_loc.second, dir) + current_time;
                        move_interaction(clock_time_in_step, pt,small_tau,ratio, Step, tmp_double);
                    }
                    else if(tmp_left == right){
		      tmp_double = (pt->time - current_time)*rate_function(energy_array[tmp_right.first][tmp_right.second], old_e_right)/energy_summation(energy_array, tmp_loc.first, tmp_loc.second, dir) + current_time;
                        move_interaction(clock_time_in_step, pt,small_tau,ratio, Step, tmp_double);
                    }
                }

                if(old_e_value_classifier[tmp_right] && direction_min_loc == horizontal) {
                    if(tmp_right == left) {
                      tmp_double = (pt->time - current_time)*rate_function(energy_array[tmp_left.first][tmp_left.second], old_e_left)/energy_summation(energy_array, tmp_loc.first, tmp_loc.second, dir) + current_time;
                      move_interaction(clock_time_in_step, pt,small_tau,ratio, Step, tmp_double);
                    }
                    else if(tmp_right == right){
                      tmp_double = (pt->time - current_time)*rate_function(energy_array[tmp_left.first][tmp_left.second] , old_e_right)/energy_summation(energy_array, tmp_loc.first, tmp_loc.second, dir) + current_time;
                      move_interaction(clock_time_in_step, pt,small_tau,ratio, Step, tmp_double);
                    }
                }
            }
            else {
                tmp_left = make_pair(tmp_x, tmp_y + 1);
                tmp_right = make_pair(tmp_x + 1, tmp_y + 1);
                if(old_e_value_classifier[tmp_right] && direction_min_loc == vertical) {
                    if(tmp_right == left) {
		      tmp_double = (pt->time - current_time)*rate_function(energy_array[tmp_left.first][tmp_left.second] , old_e_left)/energy_summation(energy_array, tmp_loc.first, tmp_loc.second, dir) + current_time;
                        move_interaction(clock_time_in_step, pt,small_tau,ratio, Step, tmp_double);
                    }
                    else if(tmp_right == right) {
		      tmp_double = (pt->time - current_time)*rate_function(energy_array[tmp_left.first][tmp_left.second] , old_e_right)/energy_summation(energy_array, tmp_loc.first, tmp_loc.second, dir) + current_time;
                        move_interaction(clock_time_in_step, pt,small_tau,ratio, Step, tmp_double);
                    }
                }

                if(old_e_value_classifier[tmp_left] && direction_min_loc == vertical) {
                    if(tmp_left == left) {
                      tmp_double = (pt->time - current_time)*rate_function(energy_array[tmp_right.first][tmp_right.second] , old_e_left)/energy_summation(energy_array, tmp_loc.first, tmp_loc.second, dir) + current_time;
                      move_interaction(clock_time_in_step, pt,small_tau,ratio, Step, tmp_double);
                    }
                    else if(tmp_left == right) {
                      tmp_double = (pt->time - current_time)*rate_function(energy_array[tmp_right.first][tmp_right.second] , old_e_right)/energy_summation(energy_array, tmp_loc.first, tmp_loc.second, dir) + current_time;
                      move_interaction(clock_time_in_step, pt,small_tau,ratio, Step, tmp_double);
                    }
                }

                if(old_e_value_classifier[tmp_right] && direction_min_loc == horizontal) {
		  tmp_double = (pt->time - current_time)*rate_function(energy_array[tmp_left.first][tmp_left.second] , old_e_left)/energy_summation(energy_array, tmp_loc.first, tmp_loc.second, dir) + current_time;
                    move_interaction(clock_time_in_step, pt,small_tau,ratio, Step, tmp_double);
                }

                if(old_e_value_classifier[tmp_left] && direction_min_loc == horizontal) {
		  tmp_double = (pt->time - current_time)*rate_function(energy_array[tmp_right.first][tmp_right.second] , old_e_right)/energy_summation(energy_array, tmp_loc.first, tmp_loc.second, dir) + current_time;
                    move_interaction(clock_time_in_step, pt,small_tau,ratio, Step, tmp_double);
                }
            }
        }

        //Step 3: update current time
        if(clock_time_in_step[level] != NULL)
        {
            min_loc = find_min(clock_time_in_step[level]);
            pt = &time_array[min_loc];

            current_time = pt->time;

            direction_min_loc = pt -> direction;
            coordinate_min_loc = pt -> coordinate; // Find the coordinate of min_loc in 2D energy_array;
            x = coordinate_min_loc.first / 2; //Compute the x coordinate of cell that the min_loc is corresponding to.
            y = coordinate_min_loc.second;//(direction_min_loc == vertical) ? coordinate_min_loc.second : coordinate_min_loc.second + 1; //Compute the y coordinate of cell......
            adjacent_clocks_min_loc = &(pt -> adjacent_clocks);

            old_e_value_classifier.clear();

            left = (direction_min_loc == vertical) ? make_pair(x, y) : make_pair(x, y + 1);
            right = (direction_min_loc == vertical) ? make_pair(x, y + 1) : make_pair(x + 1, y + 1);
            old_e_value_classifier[left] = true;
            old_e_value_classifier[right] = true;
        }
        else
        {
            current_time = next_time + 1;
        }


    }

}

P_list find_coordinate_of_site_in_energy_array(interaction info, int rows, int cols) {
    P_list ret;
    Pair tmp_pair = info.coordinate;
    Direction dir = info.direction;
    int tmp_x = tmp_pair.first / 2;
    int tmp_y = tmp_pair.second;
    if(dir == vertical) {
        if(tmp_y != 0) {
            ret.push_back(make_pair(tmp_x, tmp_y));
        }

        if(tmp_y != cols) {
            ret.push_back(make_pair(tmp_x, tmp_y + 1));
        }
    }
    else {
        ret.push_back(make_pair(tmp_x, tmp_y + 1));
        ret.push_back(make_pair(tmp_x + 1, tmp_y + 1));
    }
    return ret;
}

int main(int argc, char** argv)
{
    //cout<<sizeof(interaction);
    struct timeval t1, t2;
    ofstream myfile;
    ofstream profile;
    profile.open("Profile_final2.txt", ios_base::trunc);
    int N = 20, M = 20;
    if(argc > 1)
    {
        N = strtol(argv[1], NULL, 0);
        M = strtol(argv[2], NULL, 0);
    }
    int size_of_time_array = N * (M + 1) + (N - 1) * M;
    int N_times_M = N * M; // N is row, M is column
    double big_tau = 0.2;//big time step of tau leaping
    const int ratio = int(size_of_time_array/5);//ratio of big step and small step
    double small_tau = big_tau/double(ratio);//small time step

    //int size_of_energy_array = N * (M + 2); //Make the energy array in 2D
    double** energy_array = new double*[N];
    for(int i = 0 ; i < N ; i ++) {
        energy_array[i] = new double[M+2];
    }
    double *E_avg = new double[N_times_M];
    double* last_update = new double[N_times_M];
    for(int i = 0; i <N_times_M; i++)
    {
        E_avg[i] = 0;
        last_update[i] = 0;
    }
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> u(0,1);

    //Initialize energy_array
    for(int i = 0 ; i < N ; i++) {
        for(int j = 0 ; j < M + 2 ; j++) {
            if(j == 0) {
                energy_array[i][j] = TL;
            }
            else if(j == M + 1) {
                energy_array[i][j] = TR;
            }
            else {
                energy_array[i][j] = 1;
            }
        }
    }


    interaction* time_array = new interaction[size_of_time_array];
    int index = 0;
    int total_number_of_row = 2 * N - 2;
    // int test_index = 0;
    // cout<<total_number_of_row<<" "<<
    for(int i = 0 ; i <= total_number_of_row ; i++) {
        if(i % 2 == 0) {
            for(int j = 0 ; j <= M ; j++) {
                time_array[index].time = -log(1 - u(mt))/energy_summation(energy_array, i, j, vertical); //sqrt(energy_array[index] + energy_array[index+1]);
                time_array[index].location = index;
                time_array[index].coordinate = make_pair(i, j);
                time_array[index].direction = vertical;
                time_array[index].adjacent_clocks = find_adjacent_clocks(i, j, vertical, N, M, index);
                time_array[index].left = NULL;
                time_array[index].right = NULL;
                index++;
            }
        }
        else {
            for(int j = 0 ; j < M ; j++) {
                time_array[index].time = -log(1 - u(mt))/energy_summation(energy_array, i, j, horizontal);
                time_array[index].location = index;
                time_array[index].coordinate = make_pair(i, j);
                time_array[index].direction = horizontal;
                time_array[index].adjacent_clocks = find_adjacent_clocks(i, j, horizontal, N, M, index);
                time_array[index].left = NULL;
                time_array[index].right = NULL;
                index++;
            }
        }
        //cout<<"i: "<<i<<" j: "<<j<<" index: "<<index<<endl;
    }

     /*for(int i = 0; i < size_of_time_array; i++) {
         //cout<<"Current Index: "<<time_array[i].location<<endl;
          cout<<"Current coord: "<<time_array[i].coordinate.first<<","<<time_array[i].coordinate.second<<endl;
        cout<<"Adjacent List: ";
        for(iter j = time_array[i].adjacent_clocks.begin() ; j != time_array[i].adjacent_clocks.end() ; j++) {
            cout<<(*j)<<" ";
        }
         cout<<endl;
         //<<time_array[i].adjacent_clocks
     }*/
    //cout<<N_times_M<<endl;
    //cout<<index;


    /*for(int n = 0; n < size_time_array; n++)
    {
        time_array[n].time = -log(1 - u(mt))/sqrt(energy_array[n] + energy_array[n+1]);
        time_array[n].location = n;
        time_array[n].left = NULL;
        time_array[n].right = NULL;
    }*/

    //E_avg profile Begin

    site** E_avg_profile = new site*[N];

    for(int i = 0 ; i < N ; i++) {
        E_avg_profile[i] = new site[M + 2];
    }

    //Initialization
    for(int i = 0 ; i < N ; i++) {
        for(int j = 0 ; j < M + 2 ; j++) {
            E_avg_profile[i][j].previous_time = 0;
            E_avg_profile[i][j].coordinate = make_pair(i, j);
            E_avg_profile[i][j].summation = 0;
        }
    }

    //End

    int count = 0;

    interaction** clock_time_in_step = new interaction*[ratio + 1];//each element in the array is the head of a list
    for(int i = 0; i < ratio + 1; i++)
    {
        clock_time_in_step[i] = NULL;
    }

    gettimeofday(&t1,NULL);



    int Step = 50000;
    //double fixed_number = 0.0;
    //int min_loc = 0;
    // //int tmp_x = 0, tmp_y = 0;
    // P_list tmp_list;
    // double pre_time = 0.0;
    // double curr_time = 0.0;
    // double curr_energy = 0.0;

    // for(int i = 0 ; i < size_of_time_array ; i++) {
    //     tmp_list = find_coordinate_of_site_in_energy_array(time_array[i], N, M);
    //     // cout<<"Index: "<<i<<endl;
    //     // for(P_list::iterator j = tmp_list.begin() ; j != tmp_list.end(); j++) {
    //     //     cout<<"("<<(*j).first<<","<<(*j).second<<"),";
    //     // }
    //     // cout<<endl;
    // }

    // for(int i = 0 ; i < size_of_time_array ; i++) {
    //     tmp_list = find_coordinate_of_site_in_energy_array(time_array[i], N, M);
    //     cout<<"Current Index: "<<time_array[i].location<<endl;
    //     for(P_list::iterator j = tmp_list.begin() ; j != tmp_list.end() ; j++) {
    //         cout<<"("<<j -> first<<","<<j->second<<"), ";
    //     }
    //     cout<<endl;
    // }

    for(int out_n = 0; out_n < Step; out_n++)
    {
        //cout<<out_n<<endl;
        //cin.ignore();
        big_step_distribute(clock_time_in_step,time_array, size_of_time_array ,small_tau,ratio,out_n);
        //cout<<"-----------------"<<endl;
        // for(int i = 0 ; i < size_of_time_array ; i++) {
        //     //cout<<-log(1 - u(mt))<<endl;
        //     time_array[i].time = -log(1 - u(mt));
        // }
        //cout<<"Number Of Buckets: "<<ratio<<endl;
        for(int in_n = 0; in_n < ratio; in_n++)
        {
            //cout<<"OK"<<out_n<<endl;
            if(clock_time_in_step[in_n]!= NULL)
            {
                /*min_loc = find_min(clock_time_in_step[in_n]);
                tmp_list = find_coordinate_of_site_in_energy_array(time_array[min_loc], N, M);
                //cout<<"min_loc: "<<min_loc<<endl;
                for(P_list::iterator i = tmp_list.begin() ; i != tmp_list.end() ; i++) {
                    E_avg_profile[(*i).first][(*i).second].previous_time = time_array[min_loc].time;
                    E_avg_profile[(*i).first][(*i).second].previous_energy = energy_array[(*i).first][(*i).second];
                    //cout<<"("<<(*i).first<<","<<(*i).second<<"),";
                }*/
                //cout<<endl;
                //cout<<"OKOK: "<<in_n<<endl;
                //cout<<min_loc<<endl;
                update(clock_time_in_step, in_n, N, M, small_tau, ratio, out_n, time_array, energy_array, u, mt, &count, E_avg_profile);
                //cout<<"OKOKOK1"<<endl;
                /*for(P_list::iterator i = tmp_list.begin() ; i != tmp_list.end() ; i++) {
//                    if((*i).first == 0 &&  (*i).second == 0) {
//                        //cout<<"OK";
//                        cout<<"Info: "<<time_array[min_loc].coordinate.first<<","<<time_array[min_loc].coordinate.second<<endl;
//                    }
                    //cout<<(*i).first<<","<<(*i).second<<endl;
                    pre_time = E_avg_profile[(*i).first][(*i).second].previous_time;
                    curr_time = time_array[min_loc].time;
                    curr_energy = E_avg_profile[(*i).first][(*i).second].previous_energy;//energy_array[(*i).first][(*i).second];
                    //cout<<curr_energy * (curr_time - pre_time)<<endl;
                    E_avg_profile[(*i).first][(*i).second].record.push_back(curr_energy * (curr_time - pre_time));
                    E_avg_profile[(*i).first][(*i).second].summation += curr_energy * (curr_time - pre_time);
                }*/
                //cout<<count<<endl;
            }
        }

        clock_time_in_step[ratio] = NULL;





    }

    // for(int i = 0 ; i < N ; i++) {
    //   for(int j = 0 ; j < M ; j++) {
    //     profile<<"Previous Time: "<<E_avg_profile[i][j].previous_time<<endl;
    //     profile<<"Coordinate: "<<E_avg_profile[i][j].coordinate.first<<","<<E_avg_profile[i][j].coordinate.second<<endl;
    //     profile<<"Records: "<<endl;
    //     for(Double_list::iterator k = E_avg_profile[i][j].record.begin() ; k != E_avg_profile[i][j].record.end() ; k++) {
    //         //cout<<(*k)<<endl;
    //         profile<<(*k)<<" ";
    //     }
    //     //E_avg_profile[i][j].coordinate = make_pair(i, j);
    //     profile<<endl;
    //     profile<<"==============================================="<<endl;
    //   }
    // }
    /*
    profile<<endl<<"Profile Array: "<<"(Size: "<<N<<"*"<<M<<")"<<endl;
    profile<<endl<<"Without /(Step*big_tau)"<<endl;
    profile<<"==============================================="<<endl;
    for(int i = 0 ; i < N ; i++) {
      for(int j = 0 ; j < M + 2 ; j++) {
          profile<<E_avg_profile[i][j].summation<<" ";
      }
      profile<<endl;
    }
    profile<<endl<<"With /(Step*big_tau)"<<endl;
    profile<<"==============================================="<<endl;
    */
    for(int i = 0 ; i < N ; i++) {
      for(int j = 0 ; j < M + 2 ; j++) {
          profile<<E_avg_profile[i][j].summation/(Step*big_tau)<<" ";
      }
      profile<<endl;
    }

    /*for(int i = 0 ; i < N ; i++) {
      for(int j = 0 ; j < M + 2 ; j++) {
          profile<<energy_array[i][j]/(Step*big_tau)<<" ";
      }
      profile<<endl;
    }*/

    gettimeofday(&t2, NULL);
    delete[] energy_array;
    delete[] E_avg;
    delete[] time_array;
    delete[] clock_time_in_step;

    double delta = ((t2.tv_sec  - t1.tv_sec) * 1000000u +
                    t2.tv_usec - t1.tv_usec) / 1.e6;

    cout<<" N = "<<N <<endl;
    cout<<" M = "<<M <<endl;

    double CPU_time = 1000000*delta/double(count);

    cout<<" Speed is "<<CPU_time<<" second per million operations"<<endl;

    myfile.close();
    profile.close();
}
