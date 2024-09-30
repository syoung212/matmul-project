#include <emmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

const char *dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int)32)
#endif

#ifndef SUPER_BLOCK_SIZE
#define SUPER_BLOCK_SIZE ((int)128)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void col_to_row(const int lda, const int rows, const int cols, const double * restrict src, double * restrict buffer)
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
        for (int i = 0; i < rows; ++i)
        {
            buffer[j * rows + i] = src[i + j * lda];
        }
    }
}

void row_to_col(const int lda, const int rows, const int cols, const double *src, double *buffer)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            buffer[j * rows + i] = src[i + j * lda];
        }
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
            __m256d c_reg1 = _mm256_loadu_pd(&C[j * lda + i]);  // Initialize SSE register to zero
            //  double cij = C[j * lda + i];

            // __m128d c_reg = _mm_loadu_pd(&C[j * lda + i])

            // SIMD loop, processing 2 elements at a time
            for (k = 0; k <= K - 4; k += 4) {
                // cij += A[i * K + k] * B[j * K + k];
                // cij += A[i * K + k + 1] * B[j * K + k + 1];
                __m256d a_reg1 = _mm256_loadu_pd(&A[i * K + k]);  // Load 2 doubles from A
                __m256d b_reg1 = _mm256_loadu_pd(&B[j * K + k]);  // Load 2 doubles from B
                // c_reg += a_reg * b_reg
                c_reg1 = _mm256_add_pd(c_reg1, _mm256_mul_pd(a_reg1, b_reg1));
            }

            // c_reg = _mm_loadu_pd(&C[j * lda + i])

            // Store the 4 values from the SIMD register into a regular array
            double c_val1[4];
            _mm256_storeu_pd(c_val1, c_reg1);

            // Manually sum the two values
            double cij = c_val1[0] + c_val1[1] + c_val1[2] + c_val1[3];

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
    for (; j < N; ++j) // Loop through columns of C
    {
        for (; k < K; ++k) // Loop through rows of B and columns of A
        {
            __m512d b_reg = _mm512_loadu_pd(&B[k * N + j]);

            for (int i = 0; i <= M - 8; i += 8) // Loop through rows of C
            {
                __m512d a_reg1 = _mm512_loadu_pd(&A[k * M + i]);

                __m512d c_reg1 = _mm512_loadu_pd(&C[j * lda + i]);
                c_reg1 = _mm512_add_pd(c_reg1, _mm512_mul_pd(b_reg, a_reg1));

                _mm512_storeu_pd(&C[j * lda + i], c_reg1);
            }
            for (; i < M; ++i)
            {
                C[j * lda + i] += A[i * K + k] * B[k * N + j]; // Accumulate remaining elements
            }
        }
    }
}




void do_block(const int lda,
              const double * restrict A, const double * restrict B, double *restrict C,
              const int i, const int j, const int k, double * restrict A_buffer, double * restrict B_buffer, double * restrict C_buffer)
{
    const int M = (i + BLOCK_SIZE > lda ? lda - i : BLOCK_SIZE);
    const int N = (j + BLOCK_SIZE > lda ? lda - j : BLOCK_SIZE);
    const int K = (k + BLOCK_SIZE > lda ? lda - k : BLOCK_SIZE);


    col_to_row(lda, M, K, A + i + k * lda, A_buffer);
    col_to_col(lda, K, N, B + k + j * lda, B_buffer);

    col_to_col(lda, M, N, C + i + j * lda, C_buffer);

    // int array_size = 4;

    // printf("A_buffer[%d] = %f\n", i, A_buffer[0]);
    

    // printf("B_buffer[%d] = %f\n", i, B_buffer[0]);
    

    simd_dgemm(lda, M, N, K,
               A_buffer, B_buffer, C + i + j * lda);
    // loop_unroll_dgemm(lda, M, N, K,
    //                   A_buffer, B_buffer, C + i + j * lda);

    col_to_col(lda, M, N, C_buffer, C + i + j * lda);
    // printf("C_buffer[%d] = %f\n", i, C_buffer[0]);

}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_super_blocks = M / SUPER_BLOCK_SIZE + (M % SUPER_BLOCK_SIZE ? 1 : 0);
    const int n_blocks = M / BLOCK_SIZE + (M % BLOCK_SIZE ? 1 : 0);
    int sbi, sbj, sbk;
    int bi, bj, bk;

    double *A_buffer = (double *)aligned_alloc(64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    double *B_buffer = (double *)aligned_alloc(64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    double *C_buffer = (double *)aligned_alloc(64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    // for (bi = 0; bi < n_blocks; ++bi)
    // {
    //     const int i = bi * BLOCK_SIZE;
    //     for (bj = 0; bj < n_blocks; ++bj)
    //     {
    //         const int j = bj * BLOCK_SIZE;
    //         for (bk = 0; bk < n_blocks; ++bk)
    //         {
    //             const int k = bk * BLOCK_SIZE;
    //             do_block(M, A, B, C, i, j, k, A_buffer, B_buffer);
    //         }
    //     }
    // }

    for (bi = 0; bi < n_blocks; ++bi)
    {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj)
        {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk)
            {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k, A_buffer, B_buffer, C_buffer);

               
            }
        }
    }


    // if < 64, handle the 64x64 in a block and then the rest without. else expand the matrix to become 128 block size in the last part to append 


    // for (int sbi = 0; sbi < n_super_blocks; ++sbi) { // Super-block row
    //     const int i_super = sbi * SUPER_BLOCK_SIZE;
    //     for (int sbj = 0; sbj < n_super_blocks; ++sbj) { // Super-block column
    //         const int j_super = sbj * SUPER_BLOCK_SIZE;
    //         for (int sbk = 0; sbk < n_super_blocks; ++sbk) { // Super-block inner
    //             const int k_super = sbk * SUPER_BLOCK_SIZE;

    //             for (int bi = 0; bi < n_blocks; ++bi) { // Inner-block row
    //                 const int i = i_super + bi * BLOCK_SIZE;
    //                 for (int bj = 0; bj < n_blocks; ++bj) { // Inner-block column
    //                     const int j = j_super + bj * BLOCK_SIZE;
    //                     for (int bk = 0; bk < n_blocks; ++bk) { // Inner-block k
    //                         const int k = k_super + bk * BLOCK_SIZE;
    //                         do_block(M, A, B, C, i, j, k, A_buffer, B_buffer, C_buffer);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    free(A_buffer);
    free(B_buffer);
    free(C_buffer);
}