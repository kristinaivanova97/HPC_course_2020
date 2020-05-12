#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
//#include <time.h>
#include <string.h>

#define N 10

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Status status;
    /*
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
     */
    
    //int n = strlen("I'm rank 1");
    int n = atoi(argv[1]);
    //char message = "i";
    int iters = atoi(argv[2]);
    char *receive_message;
    char format[20], *name;
    //char *full_send_message;
    //full_send_message = (char *) malloc(world_size*n*sizeof(char)+1);
    name = (char *)malloc((n+ 1) * sizeof(char));
    sprintf(format, "%s%d%s", "%.", n - 2, "f");
    sprintf(name, format, 1.0/(world_rank + 1));

    receive_message =(char *)malloc( (n+1)*sizeof(char)+1);
    
    int ping_pong_count;
    int circle;
    ping_pong_count=-1;
    double time_work = MPI_Wtime();
    for (circle = 0; circle < iters; circle++)
    {
    
        if (world_rank != 0) {
            MPI_Recv(&ping_pong_count, 1, MPI_INT, world_rank - 1, 0,
                     MPI_COMM_WORLD, &status);
            MPI_Recv(receive_message, n+1, MPI_CHAR, world_rank - 1, 1, MPI_COMM_WORLD, &status);
            //printf("Process %d received pingpong %d from process %d\n",
             //      world_rank, ping_pong_count, world_rank - 1);
            //strcat(receive_message, send_message);
            //printf("%s\n", receive_message);
            //strcpy(full_send_message, receive_message);
            
            //printf("My name is %s %d\n", processor_name, world_rank);
            ping_pong_count++;
        
        } else
            {
                //strcat(full_send_message, send_message);
                ping_pong_count++;
                //printf("%s", full_send_message);
            }
        
        MPI_Ssend(&ping_pong_count, 1, MPI_INT, (world_rank + 1) % world_size,
        0, MPI_COMM_WORLD);
        MPI_Ssend(name, n+1, MPI_CHAR, (world_rank + 1) % world_size,
                 1, MPI_COMM_WORLD);

        // Now process 0 can receive from the last process.
        if (world_rank == 0) {
            MPI_Recv(&ping_pong_count, 1, MPI_INT, world_size - 1, 0,
                     MPI_COMM_WORLD, &status);
            MPI_Recv(receive_message, n+1, MPI_CHAR, world_rank - 1, 1, MPI_COMM_WORLD, &status);
            
            //printf("Process %d received pingpong %d from process %d\n",
             //      world_rank, ping_pong_count, world_size - 1);
            //strcat(receive_message, send_message);
            //printf("New circle\n");
            //strcpy(full_send_message, receive_message);
        }
       // printf("My name is %s %d\n", processor_name, world_rank);
        //free(full_send_message);
    }
    time_work = MPI_Wtime() - time_work;
    printf("sending during %d loops finished in %f seconds\n", iters, time_work);
    free(receive_message);
    free(name);
    MPI_Finalize();
}


/* Here I will write down the table
 
 size        # iterations        total time(secs)        time per message(msecs)        bandwidth(MB/secs)

 1           1000000            27.549651                2.7549651                          0.363
 3           1000000            30.536396                3.0536396                          0.989
 5           1000000            31.784399                3.1784399                          1.573
 50          1000000            32.566806                3.2566806                          15.356
 1240        1000000            39.565726                3.565726                           256.73
 327680      1000000            491.600934               49.1600934                         667.37
 20971520    1000               52.512696                5.2512696                          3930
 */
