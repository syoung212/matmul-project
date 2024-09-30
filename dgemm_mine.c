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

void col_to_col_add(const int lda, const int rows, const int cols, const double * restrict src, double * restrict buffer)
{
    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {

            buffer[j * rows + i] += src[i + j * lda];
        }
    }
}

void dgemm_ref(const int lda, const int M, const int N, const int K,
               const double *A, const double *B, double *C)
{
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            double b = B[j * K + k];
            __m256d b_reg = _mm256_broadcast_sd(&B[j * K + k]);
            int i = 0;
            for (; i < M - 4; i+=4) {
                __m256d a_reg1 = _mm256_loadu_pd(&A[k * M + i]);
                __m256d c_reg1 = _mm256_loadu_pd(&C[j * lda + i]);
                 c_reg1 = _mm256_fmadd_pd(a_reg1, b_reg, c_reg1);
                _mm256_storeu_pd(&C[j * lda + i], c_reg1);
            }
            for (; i < M; ++i)
            {
                C[j * lda + i] += A[k * M + i] * b;
            }

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
    // col_to_col(lda, i_block_size, j_block_size, C + i + j * lda, C_buffer);
    dgemm_ref(lda, i_block_size, j_block_size, k_block_size, A_buffer, B_buffer, C + i + j * lda);
    // dgemm_ref(lda, i_block_size, j_block_size, k_block_size, A_buffer, B_buffer, C_buffer);
    // col_to_col_add(lda, i_block_size, j_block_size, C_buffer, C + i + j * lda);
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

                                do_block(M, A, B, C, i, j, k,
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