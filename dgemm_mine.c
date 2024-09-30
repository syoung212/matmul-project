#include <emmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <omp.h>

const char *dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int)32)
#endif

#ifndef SUPER_BLOCK_SIZE
#define SUPER_BLOCK_SIZE ((int)800)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void cpy_to_bufferA(const int lda, const int rows, const int cols, const double * restrict src, double * restrict buffer)
{
    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            buffer[i * cols + j] = src[i + j * lda];
        }
    }
}

void col_to_col(const int lda, const int rows, const int cols, const double * restrict src, double * restrict buffer)
{
    for (int j = 0; j < cols; ++j)
    {
        // printf("starting col = %d ", j);
        for (int i = 0; i < rows; ++i)
        {

            buffer[j * rows + i] = src[i + j * lda];
            // printf("%d, ",  buffer[j * rows + i]);
        }
        // printf("\n");
    }
}

// void simd_dgemm(const int lda, const int M, const int N, const int K,
//                 const double *A, const double *B, double *C)
// {
//     int i, j, k;
//     for (i = 0; i < M; ++i)
//     {
//         for (j = 0; j < N; ++j)
//         {
//             __m128d c_reg = _mm_setzero_pd();
//             for (k = 0; k < K - 1; k += 2)
//             {

//                 __m128d a_reg = _mm_load_pd(&A[i * K + k]);
//                 __m128d b_reg = _mm_load_pd(&B[j * K + k]);

//                 c_reg = _mm_add_pd(c_reg, _mm_mul_pd(a_reg, b_reg));
//             }
//             if (K % 2 != 0)
//             {
//                 __m128d a_reg = _mm_load_sd(&A[i * K + K - 1]); // Load single double
//                 __m128d b_reg = _mm_load_sd(&B[j * K + K - 1]);
//                 c_reg = _mm_add_pd(c_reg, _mm_mul_pd(a_reg, b_reg));
//             }

//             _mm_store_pd(&C[j * lda + i], c_reg);
//         }
//     }
// }

// VERSION 2
void simd_dgemm(const int lda, const int M, const int N, const int K,
                const double * restrict A, const double * restrict B, double * restrict C) {
    int i, j, k;

    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            __m256d c_reg1 = _mm256_setzero_pd();  // Initialize SSE register to zero
            // __m128d c_reg2 = _mm_setzero_pd();
            //  double cij = C[j * lda + i];

            // __m128d c_reg = _mm_loadu_pd(&C[j * lda + i])

            // SIMD loop, processing 2 elements at a time
            for (k = 0; k <= K - 4; k += 4) {
                // cij += A[i * K + k] * B[j * K + k];
                // cij += A[i * K + k + 1] * B[j * K + k + 1];
                __m256d a_reg1 = _mm256_loadu_pd(&A[i * K + k]);  // Load 2 doubles from A
                // __m128d a_reg2 = _mm_loadu_pd(&A[i * K + k + 2]);  // Load 2 doubles from A
                __m256d b_reg1 = _mm256_loadu_pd(&B[j * K + k]);  // Load 2 doubles from B
                // __m128d b_reg2 = _mm_loadu_pd(&B[j * K + k + 2]);  // Load 2 doubles from B
                // c_reg += a_reg * b_reg
                c_reg1 = _mm256_fmadd_pd(a_reg1, b_reg1, c_reg1);
                // c_reg2 = _mm_add_pd(c_reg2, _mm_mul_pd(a_reg2, b_reg2));
            }

            // c_reg = _mm_loadu_pd(&C[j * lda + i])

            // Store the 2 values from the SIMD register into a regular array
            double c_val1[4];
            // double c_val2[2];
            _mm256_storeu_pd(c_val1, c_reg1);
            // _mm_storeu_pd(c_val2, c_reg2);

            // Manually sum the two values
            double cij = c_val1[0] + c_val1[1] + c_val1[2] + c_val1[3] + C[j * lda + i];

            // Handle remaining elements if K is not a multiple of 2
            for (; k < K; ++k) {
                cij += A[i * K + k] * B[j * K + k];
            }

            // Store the result back to C
            C[j * lda + i] = cij;
        }
    }
}

void simd_dgemm2(const int lda, const int M, const int N, const int K,
                const double *A, const double *B, double *C)
{
    int i, j, k;
    for (int j = 0; j < N; ++j) // Loop through columns of C
    {
        for (int k = 0; k < K; ++k) // Loop through rows of B and columns of A
        {
            // __m256d b_reg = _mm256_set1_pd(B[j * K + k]);
            __m256d b_reg = _mm256_broadcast_sd(&B[j * K + k]);

            for (int i = 0; i < M - 4; i += 4) // Loop through rows of C
            {
                
                __m256d a_reg1 = _mm256_loadu_pd(&A[k * M + i]);
                
                __m256d c_reg1 = _mm256_loadu_pd(&C[j * M + i]);
                c_reg1 = _mm256_fmadd_pd(a_reg1, b_reg, c_reg1);

                _mm256_storeu_pd(&C[j * M + i], c_reg1);
                // printf("after[%d] = %f\n", i, C[j * lda + i]);
    
            }
            for (int i = M - (M % 4); i < M; ++i)
            {
                C[j * M + i] += A[k * M + i] * B[j * K + k];
                //printf("= %d\n", C[j * lda + i]);
            }
        }
    }
}


void loop_unroll_dgemm(const int lda, const int M, const int N, const int K,
                       const double * restrict A, const double * restrict B, double * restrict C)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            double cij = C[j * lda + i];
            for (k = 0; k < K - 4; k += 4)
            {
                cij += A[i * K + k] * B[j * K + k];
                cij += A[i * K + k + 1] * B[j * K + k + 1];
                cij += A[i * K + k + 2] * B[j * K + k + 2];
                cij += A[i * K + k + 3] * B[j * K + k + 3];
            }
            for (k = k; k < K; ++k)
            {
                cij += A[i * K + k] * B[j * K + k];
            }
            C[j * lda + i] = cij;
        }
    }
}

void dgemm_ref(const int lda, const int M, const int N, const int K,
               const double *A, const double *B, double *C)
{
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            for (int i = 0; i < M; ++i) {
                C[j * lda + i] += A[k * M + i] * B[j * K + k]; // Matrix multiplication in column-major B
            }
        }
    }
}

void do_block(const int lda,
              const double * restrict A, const double * restrict B, double *restrict C,
              const int i, const int j, const int k, double *A_buffer, double*B_buffer, double*C_buffer)
{
    const int M = (i + BLOCK_SIZE > lda ? lda - i : BLOCK_SIZE);
    const int N = (j + BLOCK_SIZE > lda ? lda - j : BLOCK_SIZE);
    const int K = (k + BLOCK_SIZE > lda ? lda - k : BLOCK_SIZE);

    // printf("copying A\n");
    // col_to_col(lda, M, K, A + i + k * lda, A_buffer);
    cpy_to_bufferA(lda, M, K, A + i + k * lda, A_buffer);
    // printf("copying B\n");
    col_to_col(lda, K, N, B + k + j * lda, B_buffer);
    // col_to_col(lda, M, N, C + i + j * lda, C_buffer);



    // printf("A_buffer[%d] = %f\n", i, A_buffer[0]);
    // printf("A[%d] = %f\n", i, A[i + k * lda + 0]);
    

    // printf("B_buffer[%d] = %f\n", i, B_buffer[0]);

    simd_dgemm2(lda, M, N, K,
               A_buffer, B_buffer, C + i + j * lda);

    // dgemm_ref(lda, M, N, K,
    //            A_buffer, B_buffer, C + i + j * lda);

    

    // loop_unroll_dgemm(lda, M, N, K,
    //                   A_buffer, B_buffer, C + i + j * lda);
    // col_to_col(lda, M, N, C_buffer, C + i + j * lda);
    //  buffer[j * rows + i] = src[i + j * lda];
    // for (int cj = 0; cj < N; ++cj)
    // {
    //     for (int ci = 0; ci < M; ++ci)
    //     {
    //         C[i + j * lda + cj * M + ci] = C_buffer[ci + cj * M];
    //     }
    // }
}

void do_block2(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k,
              const int i_block_size, const int j_block_size, const int k_block_size,
              double *A_buffer, double *B_buffer, double *C_buffer)
{
    cpy_to_bufferA(lda, i_block_size, k_block_size, A + i + k * lda, A_buffer);
    // col_to_col(lda, i_block_size, k_block_size, A + i + k * lda, A_buffer);
    col_to_col(lda, k_block_size, j_block_size, B + k + j * lda, B_buffer);
    //col_to_col(lda, i_block_size, j_block_size, C + i + j * lda, C_buffer);
    simd_dgemm(lda, i_block_size, j_block_size, k_block_size,
               A_buffer, B_buffer, C + i + j * lda);
    //col_to_col(lda, i_block_size, j_block_size, C_buffer, C + i + j * lda);
}

void square_dgemm(const int M, const double * restrict A, const double * restrict B, double * restrict C)
{
    const int n_super_blocks = M / SUPER_BLOCK_SIZE + (M % SUPER_BLOCK_SIZE ? 1 : 0);
    const int n_blocks = M / BLOCK_SIZE + (M % BLOCK_SIZE ? 1 : 0);
    int sbi, sbj, sbk;
    int bi, bj, bk;

    double *A_buffer = (double *)aligned_alloc(64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    double *B_buffer = (double *)aligned_alloc(64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    double *C_buffer = (double *)aligned_alloc(64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    // int bi, bj, bk;
    
    // #pragma omp parallel for collapse(3)
    // for (bi = 0; bi < n_blocks; ++bi)
    // {
    //     const int i = bi * BLOCK_SIZE;
    //     for (bj = 0; bj < n_blocks; ++bj)
    //     {
    //         const int j = bj * BLOCK_SIZE;
    //         for (bk = 0; bk < n_blocks; ++bk)
    //         {
    //             const int k = bk * BLOCK_SIZE;
    //             do_block(M, A, B, C, i, j, k, A_buffer, B_buffer, C_buffer);
    //         }
    //     }
    // }

        for (int sbi = 0; sbi < n_super_blocks; ++sbi)
        {
            const int i_super = sbi * SUPER_BLOCK_SIZE;
            const int i_super_size = (i_super + SUPER_BLOCK_SIZE > M) ? (M - i_super) : SUPER_BLOCK_SIZE;

            for (int sbj = 0; sbj < n_super_blocks; ++sbj)
            {
                const int j_super = sbj * SUPER_BLOCK_SIZE;
                const int j_super_size = (j_super + SUPER_BLOCK_SIZE > M) ? (M - j_super) : SUPER_BLOCK_SIZE;

                for (int sbk = 0; sbk < n_super_blocks; ++sbk)
                {
                    const int k_super = sbk * SUPER_BLOCK_SIZE;
                    const int k_super_size = (k_super + SUPER_BLOCK_SIZE > M) ? (M - k_super) : SUPER_BLOCK_SIZE;

                    for (int bi = 0; bi < i_super_size / BLOCK_SIZE + (i_super_size % BLOCK_SIZE ? 1 : 0); ++bi)
                    {
                        const int i = i_super + bi * BLOCK_SIZE;
                        const int i_block_size = (i + BLOCK_SIZE > M) ? (M - i) : BLOCK_SIZE;

                        for (int bj = 0; bj < j_super_size / BLOCK_SIZE + (j_super_size % BLOCK_SIZE ? 1 : 0); ++bj)
                        {
                            const int j = j_super + bj * BLOCK_SIZE;
                            const int j_block_size = (j + BLOCK_SIZE > M) ? (M - j) : BLOCK_SIZE;

                            for (int bk = 0; bk < k_super_size / BLOCK_SIZE + (k_super_size % BLOCK_SIZE ? 1 : 0); ++bk)
                            {
                                const int k = k_super + bk * BLOCK_SIZE;
                                const int k_block_size = (k + BLOCK_SIZE > M) ? (M - k) : BLOCK_SIZE;

                                do_block2(M, A, B, C, i, j, k,
                                         i_block_size, j_block_size, k_block_size,
                                         A_buffer, B_buffer, C_buffer);
                            }
                        }
                    }
                }
            }
        }



    free(A_buffer);
    free(B_buffer);
    free(C_buffer);
}