/* File:
 *    pth_heat.c
 *
 * Purpose:
 *    Apply threads in order to calculate heat map
 *
 * Input:
 *    none
 * Output:
 *    message from each thread
 *
 * Compile:  gcc -g -Wall -o pth_hello pth_hello.c -lpthread
 * Usage:    ./pth_hello <thread_count>
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#define VACIO -1.00
#define nx 20
#define ny 20

typedef struct
{
   int start;
   int end;
   //  float *Un;
   //  float *Unp1;
   int threadId;
} ThreadArgs;

void PrintMatrix(float *matriz);
int getIndex(const int i, const int j, const int width)
{
   return i * width + j;
}

float *Un;
float *Unp1;
float *tmp;
const float a = 0.1;   // Diffusion constant
const float dx = 0.01; // Horizontal grid spacing
const float dy = 0.01; // Vertical grid spacing
const float dx2 = dx * dx;
const float dy2 = dy * dy;
const float dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2)); // Largest stable time step
const int numSteps = 5000;                            // Number of time steps
const int outputEvery = 1000;

pthread_barrier_t barrier;

void Gen_matrix(float *matriz, int radius2, int center_x, int center_y)
{
   // Initializing the data with a pattern of disk of radius of 1/6 of the width
   for (int i = 0; i < nx; i++)
   {
      for (int j = 1; j < ny - 1; j++)
      {
         int index = getIndex(i, j, ny);
         // Distance of point i, j from the origin
         float ds2 = (i - center_x) * (i - center_x) + (j - center_y) * (j - center_y);
         if (ds2 < radius2)
         {
            matriz[index] = VACIO;
         }
         else
         {
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

void *calculateTemperature(void *arguments)
{
   ThreadArgs *args = (ThreadArgs *)arguments;

   for (int n = 0; n <= numSteps; n++)
   {
      for (int i = args->start; i <= args->end; i++)
      {
         for (int j = 1; j < ny - 1; j++)
         {
            const int index = getIndex(i, j, ny);
            float uij = Un[index];
            float uim1j;
            if (i > 0) // No est� en la primera fila y entonces existe la fila superior
               uim1j = Un[getIndex(i - 1, j, ny)];
            else
               uim1j = uij;     // Si estaba en la primera fila no hay nada arriba y ponemos su mismo valor para que no haya transferencia de calor
            if (uim1j == VACIO) // Si el hueco estaba vacio ponemos su mismo valor para que no haya transferencia de calor
               uim1j = uij;

            float uijm1 = Un[getIndex(i, j - 1, ny)]; // Nunca estamos en la primera columna porque el bucle empieza en j=1
            if (uijm1 == VACIO)
               uijm1 = uij; // Si el hueco estaba vacio ponemos su mismo valor para que no haya transferencia de calor

            float uip1j;
            if (i < nx - 1) // No est� en la �ltima fila y entonces existe la fila inferior
               uip1j = Un[getIndex(i + 1, j, ny)];
            else
               uip1j = uij; // Si estaba en la �ltima fila no hay nada abajo y ponemos su mismo valor para que no haya transferencia de calor

            if (uip1j == VACIO) // Si el hueco estaba vacio ponemos su mismo valor para que no haya transferencia de calor
               uip1j = uij;

            float uijp1 = Un[getIndex(i, j + 1, ny)]; // Nunca estamos en la �ltima columna porque el bucle termina en ny-1
            if (uijp1 == VACIO)
               uijp1 = uij; // Si el hueco estaba vacio ponemos su mismo valor para que no haya transferencia de calor

            // Explicit scheme
            if (uij == VACIO) // Si era un hueco lo dejamos hueco
               Unp1[index] = uij;
            else // Si no aplicamos la formula de transferencia de calor
               Unp1[index] = uij + a * dt * ((uim1j - 2.0 * uij + uip1j) / dx2 + (uijm1 - 2.0 * uij + uijp1) / dy2);
         }
      }
      pthread_barrier_wait(&barrier);
      if (args->threadId == 0)
      {
         tmp = Un;
         Un = Unp1;
         Unp1 = tmp;
      }
      pthread_barrier_wait(&barrier);
   }
   pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
   if (argc != 2)
   {
      printf("Usage: %s <num_threads>\n", argv[0]);
      return 1;
   }

   int numThreads = atoi(argv[1]);

   Un = (float *)calloc(nx * ny, sizeof(float));
   Unp1 = (float *)calloc(nx * ny, sizeof(float));

   float radius2 = (nx / 6.0) * (nx / 6.0);
   Gen_matrix(Un, radius2, nx / 2, 5 * ny / 6);
   memcpy(Unp1, Un, nx * ny * sizeof(float));

   // Timing
   struct timespec start, finish;
   double elapsed;
   clock_gettime(CLOCK_MONOTONIC, &start);

   // Thread-related variables
   pthread_t *threads = (pthread_t *)malloc(numThreads * sizeof(pthread_t));
   ThreadArgs *threadArgs = (ThreadArgs *)malloc(numThreads * sizeof(ThreadArgs));
   float processByThreads = (float)(nx / (float)numThreads);

   pthread_barrier_init(&barrier, NULL, numThreads);
   // Set up thread arguments and create threads
   for (int i = 0; i < numThreads; i++)
   {
      threadArgs[i].start = (int)(i * processByThreads);
      threadArgs[i].end = ((i + 1) * processByThreads)-1;
      threadArgs[i].threadId = i;
      pthread_create(&threads[i], NULL, calculateTemperature, (void *)&threadArgs[i]);
      printf("Usage: %d", threadArgs[i].start);
      printf("+ %0.2f ", processByThreads);
      printf(" : %d \n,", threadArgs[i].end);
   }
   for (int i = 0; i < numThreads; i++)
   {
      pthread_join(threads[i], NULL);
   }
   pthread_barrier_destroy(&barrier);
   // Timing
   clock_gettime(CLOCK_MONOTONIC, &finish);
   elapsed = (finish.tv_sec - start.tv_sec);
   elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
   printf("It took %f seconds\n", (double)elapsed);
   PrintMatrix(Unp1);

   // Memory cleanup
   free(threads);
   free(threadArgs);
   free(Un);
   free(Unp1);
   printf("Memory freed\n");

   return 0;
}

void PrintMatrix(float *matriz)
{
   printf("Matriz resultante:\n");
   for (int i = 0; i < nx; i++)
   {
      for (int j = 0; j < ny; j++)
      {
         int index = getIndex(i, j, ny);
         printf("%.2f ", matriz[index]);
      }
      printf("\n");
   }
}