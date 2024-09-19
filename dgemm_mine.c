#include <emmintrin.h>
#include <xmmintrin.h>
#include <stdlib.h>

const char *dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int)32)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void cpy_to_bufferA(const int lda, const int rows, const int cols, const double *src, double *buffer)
{
    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            buffer[i * cols + j] = src[i + j * lda];
        }
    }
}

void cpy_to_bufferB(const int lda, const int rows, const int cols, const double *src, double *buffer)
{
    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            buffer[j * rows + i] = src[i + j * lda];
        }
    }
}

void simd_dgemm(const int lda, const int M, const int N, const int K,
                const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            __m128d c_reg = _mm_setzero_pd();
            for (k = 0; k < K - 1; k += 2)
            {

                __m128d a_reg = _mm_load_pd(&A[i * K + k]);
                __m128d b_reg = _mm_load_pd(&B[k * N + j]);

                c_reg = _mm_add_pd(c_reg, _mm_mul_pd(a_reg, b_reg));
            }
            if (K % 2 != 0)
            {
                __m128d a_reg = _mm_load_sd(&A[i * K + K - 1]); // Load single double
                __m128d b_reg = _mm_load_sd(&B[(K - 1) * N + j]);
                c_reg = _mm_add_pd(c_reg, _mm_mul_pd(a_reg, b_reg));
            }

            _mm_store_pd(&C[i * N + j], c_reg);
        }
    }
}

void loop_unroll_dgemm(const int lda, const int M, const int N, const int K,
                       const double *A, const double *B, double *C)
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

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    const int M = (i + BLOCK_SIZE > lda ? lda - i : BLOCK_SIZE);
    const int N = (j + BLOCK_SIZE > lda ? lda - j : BLOCK_SIZE);
    const int K = (k + BLOCK_SIZE > lda ? lda - k : BLOCK_SIZE);

    double *A_buffer = (double *)aligned_alloc(16, M * K * sizeof(double)); // store in aligned row major
    double *B_buffer = (double *)aligned_alloc(16, K * N * sizeof(double)); // store in aligned column major

    cpy_to_bufferA(lda, M, K, A + i + k * lda, A_buffer);
    cpy_to_bufferB(lda, K, N, B + k + j * lda, B_buffer);

    // simd_dgemm(lda, M, N, K,
    //            A_buffer, B + k + j * lda, C + i + j * lda);
    loop_unroll_dgemm(lda, M, N, K,
                      A_buffer, B_buffer, C + i + j * lda);

    free(A_buffer);
    free(B_buffer);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M % BLOCK_SIZE ? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi)
    {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj)
        {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk)
            {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}