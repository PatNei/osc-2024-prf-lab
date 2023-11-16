/********************************************************
 * Kernels to be optimized for the OS&C prflab.
 * Acknowledgment: This lab is an extended version of the
 * CS:APP Performance Lab
 ********************************************************/

#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include "defs.h"
#include "smooth.h" // helper functions for naive_smooth
#include "blend.h"  // helper functions for naive_blend

/*
 * Please fill in the following struct
 */
student_t student = {
    "paab",             /* ITU alias */
    "Patrick A. Neira", /* Full name */
    "paab@itu.dk",      /* Email address */
};

/******************************************************************************
 * ROTATE KERNEL
 *****************************************************************************/

// Your different versions of the rotate kernel go here

/*
 * naive_rotate - The naive baseline version of rotate
 */
/* stride pattern, visualization (we recommend that you draw this for your functions):
    dst         src
    3 7 B F     0 1 2 3
    2 6 A E     4 5 6 7
    1 5 9 D     8 9 A B
    0 4 8 C     C D E F
 */
char naive_rotate_descr[] = "naive_rotate: Naive baseline implementation";
void naive_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j;

    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            dst[RIDX(dim - 1 - j, i, dim)] = src[RIDX(i, j, dim)];
}
void naive_2_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j, dim2;
    dim2 = dim - 1;               // Code motion
    for (j = 0; j < dim; j++)     // loop interchange
        for (i = 0; i < dim; i++) // loop interchange
            dst[RIDX(dim2 - j, i, dim)] = src[RIDX(i, j, dim)];
}
void naive_3_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j, dim2, dim3;
    dim2 = dim - 1;
    for (j = 0; j < dim; j++)
    {
        dim3 = dim2 - j; // Code motion
        for (i = 0; i < dim; i++)
        {
            dst[RIDX(dim3, i, dim)] = src[RIDX(i, j, dim)];
        }
    }
}
void naive_4_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j, dim2, dim3;
    dim2 = dim - 1;
    for (j = 0; j < dim; j++)
    {
        dim3 = (dim2 - j) * dim; // Strength reduction for RIDX

        for (i = 0; i < dim; i++)
        {
            dst[dim3 + i] = src[RIDX(i, j, dim)]; // Strength reduction for RIDX
        }
    }
}
void naive_5_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j, dim2, dim3;
    dim2 = dim - 1;
    pixel *acc = malloc(sizeof(pixel) * dim * dim); // accumulator bad implementation
    for (j = 0; j < dim; j++)
    {
        dim3 = (dim2 - j) * dim;

        for (i = 0; i < dim; i++)
        {
            acc[dim3 + i] = src[RIDX(i, j, dim)];
        }
    }
    memcpy(dst, acc, sizeof(pixel) * dim * dim);
}
void naive_6_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j, dim2, dim3;
    pixel acc;
    dim2 = dim - 1;
    for (j = 0; j < dim; j++)
    {
        dim3 = (dim2 - j) * dim;

        for (i = 0; i < dim; i++)
        {
            acc = src[RIDX(i, j, dim)]; // accumulator improved
            dst[dim3 + i] = acc;
        }
    }
}
void naive_7_rotate(int dim, pixel *src, pixel *dst)
{
    // Loop unrolling
    int i, j, dim2, dim3, dim4;
    dim2 = dim - 1;
    for (j = 0; j < dim; j++)
    {
        dim3 = (dim2 - j) * dim;
        for (i = 0; i < dim; i = i + 8)
        {
            dim4 = dim3 + i; // Code motion
            dst[dim4] = src[RIDX(i, j, dim)];
            dst[dim4 + 1] = src[RIDX(1 + i, j, dim)];
            dst[dim4 + 2] = src[RIDX(2 + i, j, dim)];
            dst[dim4 + 3] = src[RIDX(3 + i, j, dim)];
            dst[dim4 + 4] = src[RIDX(4 + i, j, dim)];
            dst[dim4 + 5] = src[RIDX(5 + i, j, dim)];
            dst[dim4 + 6] = src[RIDX(6 + i, j, dim)];
            dst[dim4 + 7] = src[RIDX(7 + i, j, dim)];
        }
    }
}
void naive_8_rotate(int dim, pixel *src, pixel *dst)
{

    int i, j, dim2, dim3, dim4;
    pixel acc, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    dim2 = dim - 1;
    for (j = 0; j < dim; j++)
    {
        dim3 = (dim2 - j) * dim;
        for (i = 0; i < dim; i = i + 8)
        {
            // accumulator. Not faster at all.
            acc = src[RIDX(i, j, dim)];
            acc1 = src[RIDX(1 + i, j, dim)];
            acc2 = src[RIDX(2 + i, j, dim)];
            acc3 = src[RIDX(3 + i, j, dim)];
            acc4 = src[RIDX(4 + i, j, dim)];
            acc5 = src[RIDX(5 + i, j, dim)];
            acc6 = src[RIDX(6 + i, j, dim)];
            acc7 = src[RIDX(7 + i, j, dim)];
            dim4 = dim3 + i;
            dst[dim4] = acc;
            dst[dim4 + 1] = acc1;
            dst[dim4 + 2] = acc2;
            dst[dim4 + 3] = acc3;
            dst[dim4 + 4] = acc4;
            dst[dim4 + 5] = acc5;
            dst[dim4 + 6] = acc6;
            dst[dim4 + 7] = acc7;
        }
    }
}
void naive_9_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j, dim2, dim3, dim4;
    dim2 = dim - 1;
    for (j = 0; j < dim; j++)
    {
        dim3 = (dim2 - j) * dim;
        for (i = 0; i < dim; i = i + 16)
        {
            dim4 = dim3 + i;
            dst[dim4] = src[RIDX(i, j, dim)];
            dst[dim4 + 1] = src[RIDX(1 + i, j, dim)];
            dst[dim4 + 2] = src[RIDX(2 + i, j, dim)];
            dst[dim4 + 3] = src[RIDX(3 + i, j, dim)];
            dst[dim4 + 4] = src[RIDX(4 + i, j, dim)];
            dst[dim4 + 5] = src[RIDX(5 + i, j, dim)];
            dst[dim4 + 6] = src[RIDX(6 + i, j, dim)];
            dst[dim4 + 7] = src[RIDX(7 + i, j, dim)];
            dst[dim4 + 8] = src[RIDX(8 + i, j, dim)];
            dst[dim4 + 9] = src[RIDX(9 + i, j, dim)];
            dst[dim4 + 10] = src[RIDX(10 + i, j, dim)];
            dst[dim4 + 11] = src[RIDX(11 + i, j, dim)];
            dst[dim4 + 12] = src[RIDX(12 + i, j, dim)];
            dst[dim4 + 13] = src[RIDX(13 + i, j, dim)];
            dst[dim4 + 14] = src[RIDX(14 + i, j, dim)];
            dst[dim4 + 15] = src[RIDX(15 + i, j, dim)];
        }
    }
    for (; j < dim; j++)
    {
        dim3 = (dim2 - j) * dim;
        for (; i < dim; i++)
        {
            dim4 = dim3 + i;
            dst[dim4] = src[RIDX(i, j, dim)];
        }
    }
}
void naive_10_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j, k, dim2, dim3;
    dim2 = dim - 1;
    for (j = 0; j < dim; j += 8)
    {
        for (i = 0; i < dim; i += 16)
        {
            for (k = j; k < (j + 8); k++)
            {
                dim3 = (dim2 - k) * dim + i;
                dst[dim3] = src[RIDX(i, k, dim)];
                dst[dim3 + 1] = src[RIDX(1 + i, k, dim)];
                dst[dim3 + 2] = src[RIDX(2 + i, k, dim)];
                dst[dim3 + 3] = src[RIDX(3 + i, k, dim)];
                dst[dim3 + 4] = src[RIDX(4 + i, k, dim)];
                dst[dim3 + 5] = src[RIDX(5 + i, k, dim)];
                dst[dim3 + 6] = src[RIDX(6 + i, k, dim)];
                dst[dim3 + 7] = src[RIDX(7 + i, k, dim)];
                dst[dim3 + 8] = src[RIDX(8 + i, k, dim)];
                dst[dim3 + 9] = src[RIDX(9 + i, k, dim)];
                dst[dim3 + 10] = src[RIDX(10 + i, k, dim)];
                dst[dim3 + 11] = src[RIDX(11 + i, k, dim)];
                dst[dim3 + 12] = src[RIDX(12 + i, k, dim)];
                dst[dim3 + 13] = src[RIDX(13 + i, k, dim)];
                dst[dim3 + 14] = src[RIDX(14 + i, k, dim)];
                dst[dim3 + 15] = src[RIDX(15 + i, k, dim)];
            }
        }
    }
}

/*
 * rotate - Your current working version of rotate
 * IMPORTANT: This is the version you will be graded on
 */

void rotate(int dim, pixel *src, pixel *dst)
{
    naive_10_rotate(dim, src, dst);
}

/*
 * register_rotate_functions - Register all of your different versions
 *     of the rotate kernel with the driver by calling the
 *     add_rotate_function() for each test function.
 */
char rotate_descr_1[] = "rotate: 1 (raw)";
char rotate_descr_2[] = "rotate: 2 (code motion + loop interchange)";
char rotate_descr_3[] = "rotate: 3 (code motion)";
char rotate_descr_4[] = "rotate: 4 (strength reduction)";
char rotate_descr_5[] = "rotate: 5 (accumulator bad)";
char rotate_descr_6[] = "rotate: 6 (accumulator)";
char rotate_descr_7[] = "rotate: 7 (loop unrolling)";
char rotate_descr_8[] = "rotate: 8 (loop unrolling and accumulator)";
char rotate_descr_9[] = "rotate: 9 (maximized loop unrolling)";
char rotate_descr_10[] = "rotate: 10 (extreme loop unrolling)";
void register_rotate_functions()
{
    // add_rotate_function(&naive_rotate, rotate_descr_1);
    // add_rotate_function(&naive_2_rotate, rotate_descr_2);
    // add_rotate_function(&naive_3_rotate, rotate_descr_3);
    // add_rotate_function(&naive_4_rotate, rotate_descr_4);
    // add_rotate_function(&naive_5_rotate, rotate_descr_5);
    // add_rotate_function(&naive_6_rotate, rotate_descr_6);
    // add_rotate_function(&naive_7_rotate, rotate_descr_7);
    // add_rotate_function(&naive_8_rotate, rotate_descr_8);
    add_rotate_function(&naive_9_rotate, rotate_descr_9);
    add_rotate_function(&naive_10_rotate, rotate_descr_10);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * ROTATE_T KERNEL
 *****************************************************************************/

typedef struct
{
    int dim;
    int jmin;
    int jmax;
    pixel *src;
    pixel *dst;
} threadArgs;

// Your different versions of the rotate_t kernel go here
// (i.e. rotate with multi-threading)

void *rotate_t_helper(void *arg)
{
    int i, j, k, dim, dim2, dim3, jmin, jmax;
    threadArgs *args = arg;
    pixel *dst, *src;
    dst = args->dst;
    src = args->src;
    dim = args->dim;
    jmin = args->jmin;
    jmax = args->jmax;
    dim2 = args->dim - 1;
    for (j = jmin; j < jmax; j += 8)
    {
        for (i = 0; i < dim; i += 16)
        {
            for (k = j; k < (j + 8); k++)
            {
                dim3 = (dim2 - k) * dim + i;
                dst[dim3] = src[RIDX(i, k, dim)];
                dst[dim3 + 1] = src[RIDX(1 + i, k, dim)];
                dst[dim3 + 2] = src[RIDX(2 + i, k, dim)];
                dst[dim3 + 3] = src[RIDX(3 + i, k, dim)];
                dst[dim3 + 4] = src[RIDX(4 + i, k, dim)];
                dst[dim3 + 5] = src[RIDX(5 + i, k, dim)];
                dst[dim3 + 6] = src[RIDX(6 + i, k, dim)];
                dst[dim3 + 7] = src[RIDX(7 + i, k, dim)];
                dst[dim3 + 8] = src[RIDX(8 + i, k, dim)];
                dst[dim3 + 9] = src[RIDX(9 + i, k, dim)];
                dst[dim3 + 10] = src[RIDX(10 + i, k, dim)];
                dst[dim3 + 11] = src[RIDX(11 + i, k, dim)];
                dst[dim3 + 12] = src[RIDX(12 + i, k, dim)];
                dst[dim3 + 13] = src[RIDX(13 + i, k, dim)];
                dst[dim3 + 14] = src[RIDX(14 + i, k, dim)];
                dst[dim3 + 15] = src[RIDX(15 + i, k, dim)];
            }
        }
    }
    free(args);
    return NULL;
}

/*
 * rotate_t - Your current working version of rotate_t
 * IMPORTANT: This is the version you will be graded on
 */
char rotate_t_descr[] = "rotate_t: Current working version";
void rotate_t(int dim, pixel *src, pixel *dst)
{
    naive_rotate(dim, src, dst);
}

char rotate_t_descr_my[] = "Multi threaded for every value over 256";
void rotate_t_my(int dim, pixel *src, pixel *dst)
{
    if (dim <= 256)
    {
        rotate(dim, src, dst);
        return;
    }
    int i, max, dimSplit;
    max = 32;
    dimSplit = dim / max;
    pthread_t tid[max]; // Void pointers
    // should split the workload into the pieces for jmin and jmax
    for (i = 0; i < max; i++) // should divide the dim into equal pieces
    {
        threadArgs *args = malloc(sizeof(threadArgs));
        args->dim = dim;
        args->dst = dst;
        args->jmin = dimSplit * i;
        args->jmax = args->jmin + dimSplit;
        args->src = src;
        pthread_create(&tid[i], NULL, rotate_t_helper, args);
    }
    for (i = 0; i < max; i++)
    {
        pthread_join(tid[i], NULL);
    }
}

/*********************************************************************
 * register_rotate_t_functions - Register all of your different versions
 *     of the rotate_t kernel with the driver by calling the
 *     add_rotate_t_function() for each test function. When you run the
 *     driver program, it will test and report the performance of each
 *     registered test function.
 *********************************************************************/

void register_rotate_t_functions()
{
    add_rotate_t_function(&rotate_t, rotate_t_descr);
    add_rotate_t_function(&rotate_t_my, rotate_t_descr_my);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * BLEND KERNEL
 *****************************************************************************/

// Your different versions of the blend kernel go here.

char naive_blend_descr[] = "naive_blend: Naive baseline implementation";
void naive_blend(int dim, pixel *src, pixel *dst) // reads global variable `pixel bgc`
{
    int i, j;

    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            blend_pixel(&src[RIDX(i, j, dim)], &dst[RIDX(i, j, dim)], &bgc); // `blend_pixel` defined in blend.c
}

char blend_descr[] = "blend: Current working version";
void blend(int dim, pixel *src, pixel *dst)
{
    naive_blend(dim, src, dst);
}

/*
 * register_blend_functions - Register all of your different versions
 *     of the blend kernel with the driver by calling the
 *     add_blend_function() for each test function.
 */
void register_blend_functions()
{
    add_blend_function(&blend, blend_descr);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * BLEND_V KERNEL
 *****************************************************************************/

// Your different versions of the blend_v kernel go here
// (i.e. with vectorization, aka. SIMD).

char blend_v_descr[] = "blend_v: Current working version";
void blend_v(int dim, pixel *src, pixel *dst)
{
    naive_blend(dim, src, dst);
}

/*
 * register_blend_v_functions - Register all of your different versions
 *     of the blend_v kernel with the driver by calling the
 *     add_blend_function() for each test function.
 */
void register_blend_v_functions()
{
    add_blend_v_function(&blend_v, blend_v_descr);
    /* ... Register additional test functions here */
}

/******************************************************************************
 * SMOOTH KERNEL
 *****************************************************************************/

// Your different versions of the smooth kernel go here

/*
 * naive_smooth - The naive baseline version of smooth
 */
char naive_smooth_descr[] = "naive_smooth: Naive baseline implementation";
void naive_smooth(int dim, pixel *src, pixel *dst)
{
    int i, j;

    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            dst[RIDX(i, j, dim)] = avg(dim, i, j, src); // `avg` defined in smooth.c
}

char smooth_descr[] = "smooth: Current working version";
void smooth(int dim, pixel *src, pixel *dst)
{
    naive_smooth(dim, src, dst);
}

/*
 * register_smooth_functions - Register all of your different versions
 *     of the smooth kernel with the driver by calling the
 *     add_smooth_function() for each test function.
 */

void register_smooth_functions()
{
    add_smooth_function(&smooth, smooth_descr);
    /* ... Register additional test functions here */
}
