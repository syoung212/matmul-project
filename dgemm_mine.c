#include <emmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <omp.h>

const char *dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int)64)
#endif

#ifndef SUPER_BLOCK_SIZE
#define SUPER_BLOCK_SIZE ((int)1024)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

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

void col_to_col_add(const int lda, const int rows, const int cols, const double * restrict buffer, double * restrict src)
{
    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {

            src[i + j * lda] += buffer[j * rows + i];
            //src[j * rows + i] += buffer[i + j * lda];
        }
    }
}

void dgemm_ref2(const int lda, const int M, const int N, const int K,
               const double *A, const double *B, double *C)
{
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            double b = B[j * K + k];
            // __m512d b_reg = _mm512_broadcast_sd(&B[j * K + k]);
            __m512d b_reg = _mm512_set1_pd((double)b);
            int i = 0;
            for (; i < M - 8; i+=8) {
                __m512d a_reg1 = _mm512_loadu_pd(&A[k * M + i]);
                __m512d c_reg1 = _mm512_loadu_pd(&C[j * lda + i]);
                 c_reg1 = _mm512_fmadd_pd(a_reg1, b_reg, c_reg1);
                _mm512_storeu_pd(&C[j * lda + i], c_reg1);
            }
            for (; i < M; ++i)
            {
                C[j * lda + i] += A[k * M + i] * b;
            }

        }
    }
}


void dgemm_ref(const int lda, const int M, const int N, const int K,
               const double *A, const double *B, double *C)
{
    for (int j = 0; j < N; ++j) {
        int i = 0;
        for (; i < M - 8; i += 8) { // Process 8 rows at a time
            __m512d c_reg = _mm512_loadu_pd(&C[j * lda + i]); // Load 8 elements of C

            for (int k = 0; k < K; ++k) {
                double b = B[j * K + k];
                __m512d b_reg = _mm512_set1_pd((double)b); // Broadcast B value
                // __m512d a_reg = _mm512_loadu_pd(&A[k * M + i]); // Load A values
                c_reg = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i]), b_reg, c_reg); // Accumulate
            }

            _mm512_storeu_pd(&C[j * lda + i], c_reg); // Store the accumulated C values
        }
        
        // Handle any remaining elements
        for (; i < M; ++i) {
            double c_value = C[j * lda + i];
            for (int k = 0; k < K; ++k) {
                c_value += A[k * M + i] * B[j * K + k];
            }
            C[j * lda + i] = c_value; // Store the final accumulated value
        }
    }
}

void dgemm_ref_unroll32(const int lda, const int M, const int N, const int K,
               const double *A, const double *B, double *C)
{
    for (int j = 0; j < N; ++j) {
        int i = 0;

        // Process 32 elements at a time
        for (; i < M - 32; i += 32) {
            // Load initial values of C for 8 rows at once
            __m512d c_reg0 = _mm512_loadu_pd(&C[j * lda + i + 0]);
            __m512d c_reg1 = _mm512_loadu_pd(&C[j * lda + i + 8]);
            __m512d c_reg2 = _mm512_loadu_pd(&C[j * lda + i + 16]);
            __m512d c_reg3 = _mm512_loadu_pd(&C[j * lda + i + 24]);

            for (int k = 0; k < K; ++k) {
                double b = B[j * K + k];
                __m512d b_reg = _mm512_set1_pd(b); // Broadcast B value

                // Load A values and perform FMADD directly to the C registers
                c_reg0 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 0]), b_reg, c_reg0);
                c_reg1 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 8]), b_reg, c_reg1);
                c_reg2 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 16]), b_reg, c_reg2);
                c_reg3 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 24]), b_reg, c_reg3);
            }

            // Store the accumulated values back to C
            _mm512_storeu_pd(&C[j * lda + i + 0], c_reg0);
            _mm512_storeu_pd(&C[j * lda + i + 8], c_reg1);
            _mm512_storeu_pd(&C[j * lda + i + 16], c_reg2);
            _mm512_storeu_pd(&C[j * lda + i + 24], c_reg3);
        }

        for (; i < M - 8; i += 8) { // Process 8 rows at a time
            __m512d c_reg = _mm512_loadu_pd(&C[j * lda + i]); // Load 8 elements of C

            for (int k = 0; k < K; ++k) {
                double b = B[j * K + k];
                __m512d b_reg = _mm512_set1_pd((double)b); // Broadcast B value
                // __m512d a_reg = _mm512_loadu_pd(&A[k * M + i]); // Load A values
                c_reg = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i]), b_reg, c_reg); // Accumulate
            }

            _mm512_storeu_pd(&C[j * lda + i], c_reg); // Store the accumulated C values
        }
        
        // Handle any remaining elements
        for (; i < M; ++i) {
            double c_value = C[j * lda + i];
            for (int k = 0; k < K; ++k) {
                c_value += A[k * M + i] * B[j * K + k];
            }
            C[j * lda + i] = c_value; // Store the final accumulated value
        }
    }
}

#include <immintrin.h>  // For AVX512 intrinsics

void dgemm_ref_unroll(const int lda, const int M, const int N, const int K,
                      const double *A, const double *B, const double *C_buffer, double *C)
{
    for (int j = 0; j < N; ++j) {
        int i = 0;

        // Process 64 elements at a time
        // for (; i < M - 64; i += 64) {
        //     // Load initial values of C for 8 rows at once
        //     __m512d c_reg0 = _mm512_loadu_pd(&C_buffer[j * lda + i + 0]);
        //     __m512d c_reg1 = _mm512_loadu_pd(&C_buffer[j * lda + i + 8]);
        //     __m512d c_reg2 = _mm512_loadu_pd(&C_buffer[j * lda + i + 16]);
        //     __m512d c_reg3 = _mm512_loadu_pd(&C_buffer[j * lda + i + 24]);
        //     __m512d c_reg4 = _mm512_loadu_pd(&C_buffer[j * lda + i + 32]);
        //     __m512d c_reg5 = _mm512_loadu_pd(&C_buffer[j * lda + i + 40]);
        //     __m512d c_reg6 = _mm512_loadu_pd(&C_buffer[j * lda + i + 48]);
        //     __m512d c_reg7 = _mm512_loadu_pd(&C_buffer[j * lda + i + 56]);

        //     for (int k = 0; k < K; ++k) {
        //         __m512d b_reg = _mm512_set1_pd(B[j * K + k]); // Broadcast B value
        //         // Load A values and perform FMADD directly to the C registers
        //         c_reg0 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 0]), b_reg, c_reg0);
        //         c_reg1 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 8]), b_reg, c_reg1);
        //         c_reg2 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 16]), b_reg, c_reg2);
        //         c_reg3 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 24]), b_reg, c_reg3);
        //         c_reg4 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 32]), b_reg, c_reg4);
        //         c_reg5 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 40]), b_reg, c_reg5);
        //         c_reg6 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 48]), b_reg, c_reg6);
        //         c_reg7 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 56]), b_reg, c_reg7);
        //     }

        //     // Store the accumulated values back to C
        //     _mm512_storeu_pd(&C[j * lda + i + 0], c_reg0);
        //     _mm512_storeu_pd(&C[j * lda + i + 8], c_reg1);
        //     _mm512_storeu_pd(&C[j * lda + i + 16], c_reg2);
        //     _mm512_storeu_pd(&C[j * lda + i + 24], c_reg3);
        //     _mm512_storeu_pd(&C[j * lda + i + 32], c_reg4);
        //     _mm512_storeu_pd(&C[j * lda + i + 40], c_reg5);
        //     _mm512_storeu_pd(&C[j * lda + i + 48], c_reg6);
        //     _mm512_storeu_pd(&C[j * lda + i + 56], c_reg7);
        // }

        // Handle remaining elements in chunks of 32
        for (; i < M - 32; i += 32) {
            __m512d c_reg0 = _mm512_loadu_pd(&C_buffer[j * M + i + 0]);
            __m512d c_reg1 = _mm512_loadu_pd(&C_buffer[j * M + i + 8]);
            __m512d c_reg2 = _mm512_loadu_pd(&C_buffer[j * M + i + 16]);
            __m512d c_reg3 = _mm512_loadu_pd(&C_buffer[j * M + i + 24]);

            for (int k = 0; k < K; ++k) {
                double b = B[j * K + k];
                __m512d b_reg = _mm512_set1_pd(b);

                c_reg0 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 0]), b_reg, c_reg0);
                c_reg1 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 8]), b_reg, c_reg1);
                c_reg2 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 16]), b_reg, c_reg2);
                c_reg3 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 24]), b_reg, c_reg3);
            }

            _mm512_storeu_pd(&C[j * lda + i + 0], c_reg0);
            _mm512_storeu_pd(&C[j * lda + i + 8], c_reg1);
            _mm512_storeu_pd(&C[j * lda + i + 16], c_reg2);
            _mm512_storeu_pd(&C[j * lda + i + 24], c_reg3);
        }

        // // Handle remaining elements in chunks of 16
        // for (; i < M - 16; i += 16) {
        //     __m512d c_reg1 = _mm512_loadu_pd(&C[j * lda + i]);
        //     __m512d c_reg2 = _mm512_loadu_pd(&C[j * lda + i + 8]);
            

        //     for (int k = 0; k < K; ++k) {
        //         double b = B[j * K + k];
        //         __m512d b_reg = _mm512_set1_pd(b);
        //         c_reg1 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i]), b_reg, c_reg1);
        //         c_reg2 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i + 8]), b_reg, c_reg2);
        //     }

        //     _mm512_storeu_pd(&C[j * lda + i], c_reg1);
        //     _mm512_storeu_pd(&C[j * lda + i + 8], c_reg2);
        // }

        // Handle remaining elements in chunks of 8
        for (; i < M - 8; i += 8) {
            __m512d c_reg = _mm512_loadu_pd(&C[j * lda + i]);

            for (int k = 0; k < K; ++k) {
                __m512d b_reg = _mm512_set1_pd(B[j * K + k]);
                c_reg = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * M + i]), b_reg, c_reg);
            }

            _mm512_storeu_pd(&C[j * lda + i], c_reg);
        }

        // Handle any remaining elements (less than 8)
        for (; i < M; ++i) {
            double c_value = C[j * lda + i];
            for (int k = 0; k < K; ++k) {
                c_value += A[k * M + i] * B[j * K + k];
            }
            C[j * lda + i] = c_value; // Store the final accumulated value
        }
    }
}

void dgemm_ref_unroll_const(const int lda, 
                      const double *A, const double *B, const double *C_buffer, double *C)
{
    for (int j = 0; j < BLOCK_SIZE; ++j) {
        int i = 0;

        // Handle remaining elements in chunks of 32
        for (; i < BLOCK_SIZE - 32; i += 32) {
            __m512d c_reg0 = _mm512_loadu_pd(&C_buffer[j * BLOCK_SIZE + i + 0]);
            __m512d c_reg1 = _mm512_loadu_pd(&C_buffer[j * BLOCK_SIZE + i + 8]);
            __m512d c_reg2 = _mm512_loadu_pd(&C_buffer[j * BLOCK_SIZE + i + 16]);
            __m512d c_reg3 = _mm512_loadu_pd(&C_buffer[j * BLOCK_SIZE + i + 24]);

            for (int k = 0; k < BLOCK_SIZE; ++k) {
                double b = B[j * BLOCK_SIZE + k];
                __m512d b_reg = _mm512_set1_pd(b);

                c_reg0 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * BLOCK_SIZE + i + 0]), b_reg, c_reg0);
                c_reg1 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * BLOCK_SIZE + i + 8]), b_reg, c_reg1);
                c_reg2 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * BLOCK_SIZE + i + 16]), b_reg, c_reg2);
                c_reg3 = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * BLOCK_SIZE + i + 24]), b_reg, c_reg3);
            }

            _mm512_storeu_pd(&C[j * lda + i + 0], c_reg0);
            _mm512_storeu_pd(&C[j * lda + i + 8], c_reg1);
            _mm512_storeu_pd(&C[j * lda + i + 16], c_reg2);
            _mm512_storeu_pd(&C[j * lda + i + 24], c_reg3);
        }

        // Handle remaining elements in chunks of 8
        for (; i < BLOCK_SIZE - 8; i += 8) {
            __m512d c_reg = _mm512_loadu_pd(&C[j * lda + i]);

            for (int k = 0; k < BLOCK_SIZE; ++k) {
                __m512d b_reg = _mm512_set1_pd(B[j * BLOCK_SIZE + k]);
                c_reg = _mm512_fmadd_pd(_mm512_loadu_pd(&A[k * BLOCK_SIZE + i]), b_reg, c_reg);
            }

            _mm512_storeu_pd(&C[j * lda + i], c_reg);
        }

        // Handle any remaining elements (less than 8)
        for (; i < BLOCK_SIZE; ++i) {
            double c_value = C[j * lda + i];
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                c_value += A[k * BLOCK_SIZE + i] * B[j * BLOCK_SIZE + k];
            }
            C[j * lda + i] = c_value; // Store the final accumulated value
        }
    }
}



void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k,
              const int i_block_size, const int j_block_size, const int k_block_size,
              double *A_buffer, double *B_buffer, double *C_buffer)
{
    col_to_col(lda, i_block_size, k_block_size, A + i + k * lda, A_buffer);
    col_to_col(lda, k_block_size, j_block_size, B + k + j * lda, B_buffer);
    col_to_col(lda, i_block_size, j_block_size, C + i + j * lda, C_buffer);
    dgemm_ref_unroll(lda, i_block_size, j_block_size, k_block_size, A_buffer, B_buffer, C_buffer, C + i + j * lda);
    // dgemm_ref_unroll(lda, i_block_size, j_block_size, k_block_size, A_buffer, B_buffer, C_buffer);
    // col_to_col_add(lda, i_block_size, j_block_size, C_buffer, C + i + j * lda);
}

void do_block_const(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k,
              double *A_buffer, double *B_buffer, double *C_buffer)
{
    col_to_col(lda, BLOCK_SIZE, BLOCK_SIZE, A + i + k * lda, A_buffer);
    col_to_col(lda, BLOCK_SIZE, BLOCK_SIZE, B + k + j * lda, B_buffer);
    col_to_col(lda, BLOCK_SIZE, BLOCK_SIZE, C + i + j * lda, C_buffer);
    dgemm_ref_unroll_const(lda, A_buffer, B_buffer, C_buffer, C + i + j * lda);
    // dgemm_ref_unroll(lda, i_block_size, j_block_size, k_block_size, A_buffer, B_buffer, C_buffer);
    // col_to_col_add(lda, i_block_size, j_block_size, C_buffer, C + i + j * lda);
}

void square_dgemm(const int M, const double * restrict A, const double * restrict B, double * restrict C)
{
    const int n_super_blocks = M / SUPER_BLOCK_SIZE + (M % SUPER_BLOCK_SIZE ? 1 : 0);
    const int n_blocks = M / BLOCK_SIZE + (M % BLOCK_SIZE ? 1 : 0);
    int sbi, sbj, sbk;
    int bi, bj, bk;
    int i,j, k;

    double *A_buffer = (double *)aligned_alloc(64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    double *B_buffer = (double *)aligned_alloc(64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    double *C_buffer = (double *)aligned_alloc(64, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    
    // const int i_block_size = (i + BLOCK_SIZE > M) ? (M - i) : BLOCK_SIZE;
    // const int j_block_size = (j + BLOCK_SIZE > M) ? (M - j) : BLOCK_SIZE;
    // const int k_block_size = (k + BLOCK_SIZE > M) ? (M - k) : BLOCK_SIZE;
    
    // for (bi = 0; bi < n_blocks; ++bi)
    // {
    //     const int i = bi * BLOCK_SIZE;
    //     for (bj = 0; bj < n_blocks; ++bj)
    //     {
    //         const int j = bj * BLOCK_SIZE;
    //         for (bk = 0; bk < n_blocks; ++bk)
    //         {
    //             const int k = bk * BLOCK_SIZE;
    //             if (i + BLOCK_SIZE < M & j + BLOCK_SIZE < M & k + BLOCK_SIZE < M){
    //                 do_block_const(M, A, B, C, i, j, k, A_buffer, B_buffer, C_buffer);
    //             }
    //             else{
    //                 const int i_block_size = (i + BLOCK_SIZE > M) ? (M - i) : BLOCK_SIZE;
    //                 const int j_block_size = (j + BLOCK_SIZE > M) ? (M - j) : BLOCK_SIZE;
    //                 const int k_block_size = (k + BLOCK_SIZE > M) ? (M - k) : BLOCK_SIZE;
    //                 do_block(M, A, B, C, i, j, k, i_block_size, j_block_size, k_block_size, A_buffer, B_buffer, C_buffer);
    //             }
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
                        // const int i_block_size = (i + BLOCK_SIZE > M) ? (M - i) : BLOCK_SIZE;

                        for (int bj = 0; bj < j_super_size / BLOCK_SIZE + (j_super_size % BLOCK_SIZE ? 1 : 0); ++bj)
                        {
                            const int j = j_super + bj * BLOCK_SIZE;
                            // const int j_block_size = (j + BLOCK_SIZE > M) ? (M - j) : BLOCK_SIZE;

                            for (int bk = 0; bk < k_super_size / BLOCK_SIZE + (k_super_size % BLOCK_SIZE ? 1 : 0); ++bk)
                            {
                                const int k = k_super + bk * BLOCK_SIZE;
                                // const int k_block_size = (k + BLOCK_SIZE > M) ? (M - k) : BLOCK_SIZE;
                                if (i + BLOCK_SIZE < M & j + BLOCK_SIZE < M & k + BLOCK_SIZE < M){
                                    do_block(M, A, B, C, i, j, k, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE,
                                            A_buffer, B_buffer, C_buffer);
                                }
                                else {
                                    const int i_block_size = (i + BLOCK_SIZE > M) ? (M - i) : BLOCK_SIZE;
                                    const int j_block_size = (j + BLOCK_SIZE > M) ? (M - j) : BLOCK_SIZE;
                                    const int k_block_size = (k + BLOCK_SIZE > M) ? (M - k) : BLOCK_SIZE;
                                    // const int i_block_size = (M - i);
                                    // const int j_block_size = (M - j);
                                    // const int k_block_size = (M - k);
                                    do_block(M, A, B, C, i, j, k,
                                            i_block_size, j_block_size, k_block_size,
                                            A_buffer, B_buffer, C_buffer);
                                }
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