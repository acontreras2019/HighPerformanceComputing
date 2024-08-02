/* File:  
 *    openmp-entrega.c
 *
 * Purpose:
 *    Apply openmp in order to calculate heat map
 *
 * Input:
 *    none
 * Output:
 *    process resume
 *
 * Compile:  mpicc openmp-entrega.c -o openmp-entrega -fopenmp -Wall -std=c99 -g
 * Usage:    mpirun -np <numProcess> ./openmp-entrega
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#define VACIO -1.00

void Gen_matrix(float* matriz, int radius2, int center_x, int center_y);
void PrintMatrix(float *matriz, int fila, int col);
void showFinishMessage(double time, int nProcs);
void Hello(int rank); 
void *mainLoop(void *args);


int getIndex(const int i, const int j, const int width) {
    return i * width + j;
}

float* Un;
float* Unp1;
float* tmp;


const float a = 0.1;
const float dx = 0.01;
const float dy = 0.01;
const float dx2 = dx * dx;
const float dy2 = dy * dy;
const float dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));
const int numSteps = 5000;
const int outputEvery = 1000;
const int nx = 200;
const int ny = nx;
int numElements = nx * ny;
double startTime, endTime;
const int numHilos = 2; 


void PrintMatrix(float *matriz, int fila, int col)
{
//    printf("Matriz resultante:\n");
   for (int i = 0; i < fila; i++)
   {
      for (int j = 0; j < col; j++)
      {
         int index = getIndex(i, j, col);
         printf("%.2f ", matriz[index]);
      }
      printf("\n");
   }
}

//Imprimir mensaje de tiempos
void showFinishMessage(double time, int nProcs){
	printf("SIMULACIÓN FINALIZADA\n");
	printf("----------------------\n");
	printf("Tamaño de matriz %d x %d\n", nx, ny);
	printf("Numero de iteraciones: %d\n", numSteps);
	printf("Numero de Procesos: %d\n", nProcs);
    printf("Numero de Hilos: %d\n", numHilos);
	printf("\x1b[34mTiempo total de ejecucion: %f\n", time);
	printf("\x1b[0m-------------------------------------\n");
}


void Gen_matrix(float *matriz, int radius2, int center_x, int center_y) {
    for (int i = 0; i < nx; i++) {
        for (int j = 1; j < ny - 1; j++) {
            int index = getIndex(i, j, ny);
            float ds2 = (i - center_x) * (i - center_x) + (j - center_y) * (j - center_y);
            if (ds2 < radius2) {
                matriz[index] = VACIO;
            } else {
                matriz[index] = 50.0;
            }
        }
        {
            int index = getIndex(i, 0, ny);
            matriz[index] = 0.0;
        }
        {
            int index = getIndex(i, nx - 1, ny);
            matriz[index] = 100.0;
        }
    }
}

int main(int argc, char** argv) {
   
    Un = (float *)calloc(numElements, sizeof(float));
    Unp1 = (float *)calloc(numElements, sizeof(float));
    

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    startTime = MPI_Wtime();
    // Verificar que el número de procesos sea igual a 4
    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Este programa requiere mas de 1 procesos.\n");
        }
        MPI_Finalize();
        exit(1);
    }

    // Inicializar la matriz en el proceso 0

    if (rank == 0) {
        float radius2 = (nx / 6.0) * (nx / 6.0);
        Gen_matrix(Un,radius2,nx / 2, 5 * ny / 6 );
        // printf("Matriz original en el proceso 0:\n");
        // PrintMatrix(Un, nx, ny);
       
    }

    // Distribuir la matriz entre los procesos()
    int elemByProc = (numElements/(float)(size));
    int filasByProc= (nx/(float)(size));
    float* matrizProceso = (float*)malloc(elemByProc * sizeof(float));
 
    

     MPI_Scatter(Un, elemByProc, MPI_FLOAT, matrizProceso, elemByProc , MPI_FLOAT, 0, MPI_COMM_WORLD);


     float* filaInferiorRev = (float*)malloc(ny* sizeof(float));
     float* filaSuperiorRev = (float*)malloc(ny* sizeof(float));
     float* matrizProcesoNext = (float*)malloc(elemByProc * sizeof(float));
     memcpy(matrizProcesoNext, matrizProceso, elemByProc * sizeof(float));

    // Cada proceso realiza cálculos en su parte de la matriz
    #pragma omp parallel num_threads(numHilos)
    {
        // Hello(rank);

            for (int step = 0; step < numSteps; step++)
            {
                 #pragma omp master
                {
                                                // Enviar valores superior
                    if (rank > 0) {
                        MPI_Send(matrizProceso, ny, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                        //  printf("Matriz Superior Enviada: %d \n" , filasByProc);
                        //  PrintMatrix(matrizProceso, 10, ny);
                    }
                    
                    //Enviar Valores Inferiores
                    if (rank < size - 1) {

                        MPI_Send(matrizProceso + elemByProc-ny, ny, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                        // printf("Matriz Inferior Enviada: %d \n" , filasByProc);
                        // PrintMatrix(matrizProceso + elemByProc-ny, 1, ny);

                    }   

                // Recibir valores inferior
                    if (rank < size - 1) {
                        MPI_Recv(filaInferiorRev, ny, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        // printf("Matriz Inferior Recibida: %d \n" , filasByProc);
                        // PrintMatrix(filaInferiorRev, 1, ny);
                    }

                    // Recibir valores superior
                    if (rank > 0) {
                        MPI_Recv(filaSuperiorRev , ny, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        // printf("Matriz superior Recibida: %d \n" , filasByProc);
                        // PrintMatrix(filaSuperiorRev, 1, ny);
                    }

                }
               
               
                #pragma omp for
                
                
                for (int i = 0; i < filasByProc; i++) {

                    for (int j = 1; j < ny -1; j++) {
                            
                    const int index = getIndex(i, j, ny);
                        float uij = matrizProceso[index];
                        float uim1j;
                        //superior
                        if (i>0) // No est� en la primera fila y entonces existe la fila superior
                            uim1j = matrizProceso[getIndex(i-1, j, ny)];
                        if (rank>0 && i==0) // No esta en primer rango, primera fila... entonces tomamos los datos de la fila superior que recibimos
                        uim1j = filaSuperiorRev[j];
                        if(rank ==0 && i==0)
                        uim1j = uij; // Si estaba en la primera fila no hay nada arriba y ponemos su mismo valor para que no haya transferencia de calor
                    
                        if (uim1j==VACIO) // Si el hueco estaba vacio ponemos su mismo valor para que no haya transferencia de calor
                        uim1j=uij;

                        float uijm1 = matrizProceso[getIndex(i, j-1, ny)];  // Nunca estamos en la primera columna porque el bucle empieza en j=1
                        if (uijm1==VACIO)
                        uijm1=uij ; // Si el hueco estaba vacio ponemos su mismo valor para que no haya transferencia de calor

                        //inferior
                        float uip1j;
                        if (i< (filasByProc - 2) ) // No est� en la �ltima fila y entonces existe la fila inferior
                            uip1j = matrizProceso[getIndex(i+1, j, ny)];
                        if (rank ==(size -1) && i== (filasByProc - 1) )
                            uip1j = uij;
                        if (rank <(size -1) && i== (filasByProc - 1))
                                uip1j = filaInferiorRev[j]; // Si estaba en la �ltima fila no hay nada abajo y ponemos su mismo valor para que no haya transferencia de calor
                            

                        if (uip1j==VACIO) // Si el hueco estaba vacio ponemos su mismo valor para que no haya transferencia de calor
                        uip1j=uij ;
                        
                        float uijp1 = matrizProceso[getIndex(i, j+1, ny)]; // Nunca estamos en la �ltima columna porque el bucle termina en ny-1
                        if (uijp1==VACIO)
                        uijp1=uij ; // Si el hueco estaba vacio ponemos su mismo valor para que no haya transferencia de calor
                        
                        // Explicit scheme
                        if (uij==VACIO)  // Si era un hueco lo dejamos hueco
                        matrizProcesoNext[index]=uij;
                        else   // Si no aplicamos la formula de transferencia de calor
                        matrizProcesoNext[index] = uij + a * dt * ( (uim1j - 2.0*uij + uip1j)/dx2 + (uijm1 - 2.0*uij + uijp1)/dy2 );
                    }
                                
                }

                #pragma omp critical
                {

                tmp = matrizProceso;
                matrizProceso = matrizProcesoNext;
                matrizProcesoNext = tmp;
                }
            
        }   
    
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    // Reunir los resultados en el proceso 0
    MPI_Gather(matrizProceso, elemByProc, MPI_FLOAT, Unp1, elemByProc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Imprimir los resultados en el proceso 0
    if (rank == 0) {
        printf("\nMatriz resultante en el proceso 0:\n");
        PrintMatrix(Unp1, nx, ny);

        endTime = MPI_Wtime();
        showFinishMessage(endTime-startTime, size);
        // Liberar la memoria utilizada para la matriz
        free(Unp1);
        free(Un);
    }

	
    // Liberar la memoria utilizada para la submatriz local
    free(matrizProceso);
    free(matrizProcesoNext);
    free(filaInferiorRev);
    free(filaSuperiorRev);

    MPI_Finalize();
    return 0;
}

void Hello(int rank) {
   int my_rank = rank;
   int thread_count = omp_get_thread_num();

   printf("Hello desde proceso %d hilo %d\n", my_rank, thread_count);

}

