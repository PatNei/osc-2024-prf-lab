/********************************************************
 * Kernels to be optimized for the OS&C prflab.
 * Acknowledgment: This lab is an extended version of the
 * CS:APP Performance Lab
 ********************************************************/
#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
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

/*
 * rotate_t - Your current working version of rotate_t
 * IMPORTANT: This is the version you will be graded on
 */
char rotate_t_descr[] = "rotate_t: Current working version";
void rotate_t(int dim, pixel *src, pixel *dst)
{
    rotate_t_my(dim, src, dst);
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

char blend_descr_my[] = "blend: removed procedure call and added code motion where applicaple ";
void blend_my(int dim, pixel *src, pixel *dst)
{
    float a, a1;
    int i, dummy;
    dummy = dim * dim;

    for (i = 0; i < dummy; i++)
    {
        a = ((float)(src[i].alpha)) / USHRT_MAX;
        a1 = 1 - a;
        dst[i].red = (a * src[i].red) + (a1 * bgc.red);
        dst[i].green = (a * src[i].green) + (a1 * bgc.green);
        dst[i].blue = (a * src[i].blue) + (a1 * bgc.blue);
        dst[i].alpha = USHRT_MAX; // opaques
    }
}

char blend_descr[] = "blend: Current working version";
void blend(int dim, pixel *src, pixel *dst)
{
    blend_my(dim, src, dst);
}

/*
 * register_blend_functions - Register all of your different versions
 *     of the blend kernel with the driver by calling the
 *     add_blend_function() for each test function.
 */
void register_blend_functions()
{
    add_blend_function(&blend, blend_descr);
    add_blend_function(&blend_my, blend_descr_my);
    /* ... Register additional test functions here */
}
/******************************************************************************
 * BLEND_V KERNEL
 *****************************************************************************/

void print_pix(__m256i *pix)
{
    unsigned short shorts[16];
    memcpy(shorts, pix, sizeof(shorts)); // Copy the data that pix points to, not the address of pix
    printf("loaded from src:\n");
    for (int z = 0; z < 8; z++)
    {
        printf("%*d:%*d, ", 2, z, 5, shorts[z]);
    }
    printf("\n");
    for (int z = 8; z < 16; z++)
    {
        printf("%*d:%*d, ", 2, z, 5, shorts[z]);
    }
    printf("\n");
}

void print_float(char *mess, __m256 flt8)
{
    float floats[8];
    memcpy(floats, &flt8, sizeof(flt8));
    printf("Message: %s\n", mess);
    for (int z = 0; z < 8; z++)
    {
        printf("%*d:%f, ", 2, z, floats[z]);
    }
    printf("\n");
}

#define m256 __m256
#define m256i __m256i
#define createVectorAs_float _mm256_set1_ps
#define createVectorWith_float _mm256_setr_ps // reverse because of little endian bs
#define createVectorWith_integer _mm256_set_epi32
#define createVectorAsZero_float _mm256_setzero_ps
#define createVectorAsZero_integer _mm256_setzero_si256
#define createMask_float _mm256_setr_ps
#define pickAndChooseFromVectors_float _mm256_blendv_ps
#define load4PixelToVector_integer(x) _mm256_load_si256((m256i *)x) // does implicit type casting to m256i
#define storeVectorToPixel(x, j) _mm256_store_si256((m256i *)x, j)  // does implicit type casting to m256i
#define convertFrom256IntegerTo128BitFloatVector _mm256_cvtepi32_ps
#define splitVectorToFloat_lo _mm256_unpacklo_epi16
#define splitVectorToFloat_hi _mm256_unpackhi_epi16
#define multiplyVectors_float _mm256_mul_ps
#define subtractVectors_float _mm256_sub_ps
#define addVectors_float _mm256_add_ps
#define orderValuesBy_float _mm256_permutevar8x32_ps // Not as fast as shuffle
#define orderShuffle _MM_SHUFFLE
#define orderValuesBy_shuff_float(a, b, c) _mm256_shuffle_ps(a, b, ((int)c)) // Way faster than using permute
#define combineTwoFloatVectorstoIntegerVector _mm256_packus_epi32
#define convertFVectorToIVector _mm256_cvtps_epi32

char blend_v_descr_dev[] = "blend_v: dev version";
void blend_v_dev(int dim, pixel *src, pixel *dst)
{
    int i, dummy;
    dummy = dim * dim;
    const m256 bgcV = createVectorWith_float(bgc.red, bgc.green, bgc.blue, 0.0, bgc.red, bgc.green, bgc.blue, 0.0);
    const m256 ushrt_max = createVectorAs_float(1.0 / USHRT_MAX); // Vector for 1 / USHORTMAX
    const m256 oneVector = createVectorAs_float(1.0);             // Vector for 1
    const m256 uShortMAX = createVectorAs_float(USHRT_MAX);       // Vector for USHRT_MAX
    const m256i zeroes = createVectorAsZero_integer();            // Vector with zeroes
    const m256 combineWithAlphaMask = createMask_float(0, 0, 0, -1, 0, 0, 0, -1);
    // const m256i orderPattern = createVectorWith_integer(7, 7, 7, 7, 3, 3, 3, 3);
    for (i = 0; i < dummy; i += 4, src += 4, dst += 4)
    {
        // [p1_r,p1_g,p1_b,p1_a,p2_r,p1_g,p1_b,p1_a,p3_r,p3_g,p3_b,p3_a,p4_r,p4_g,p4_b,p4_a]
        m256i pix4 = load4PixelToVector_integer(src);
        // print_pix(&pix4);
        // convert to float vector where x % 2 = 0 is equal to 0
        // [p1_r,0,p1_g,0,p1_b,0,p1_a,0,p3_r,0,p3_g,0,p3_b,0,p3_a,0]
        m256 loPixF = convertFrom256IntegerTo128BitFloatVector(splitVectorToFloat_lo(pix4, zeroes));
        // [p2_r,0,p2_g,0,p2_b,0,p2_a,0,p4_r,0,p4_g,0,p4_b,0,p4_a,0]
        m256 hiPixF = convertFrom256IntegerTo128BitFloatVector(splitVectorToFloat_hi(pix4, zeroes));
        // print_float("Convert to float lo", loPixF);
        // print_float("Convert to float hi", hiPixF);
        // calculate alpha for vector
        /* Example:
        [
            p1_a * 1 / USHRT_MAX,
            p1_a * 1 / USHRT_MAX,
            p1_a * 1 / USHRT_MAX,
            p1_a * 1 / USHRT_MAX,
            p3_a * 1 / USHRT_MAX,
            p3_a * 1 / USHRT_MAX,
            p3_a * 1 / USHRT_MAX,
            p3_a * 1 / USHRT_MAX
        ]
        */
        // print_float("calculate a low", multiplyVectors_float(loPixF, ushrt_max));
        // print_float("calculate a hi", multiplyVectors_float(hiPixF, ushrt_max));
        // print_float("replicate Alpha low", orderValuesBy_float(loPixF, orderPattern(7, 7, 7, 7, 3, 3, 3, 3)));
        // print_float("replicate Alpha High", orderValuesBy_float(hiPixF, orderPattern(7, 7, 7, 7, 3, 3, 3, 3)));
        m256 a_low = multiplyVectors_float(loPixF, ushrt_max);
        m256 a_high = multiplyVectors_float(hiPixF, ushrt_max);
        m256 alpha_vec_low = orderValuesBy_shuff_float(a_low, a_low, orderShuffle(3, 3, 7, 7));
        m256 alpha_vec_high = orderValuesBy_shuff_float(a_high, a_high, orderShuffle(3, 3, 7, 7));
        m256 alphaTimesColor_lo = multiplyVectors_float(alpha_vec_low, loPixF);
        m256 alphaTimesColor_hi = multiplyVectors_float(alpha_vec_high, hiPixF);
        // print_float("copy Alpha low with calculation", alpha_low);
        // print_float("copy Alpha High with calculation", alpha_high);
        //  need to fill a vector with alpha values
        //  calculate 1 minus alpha AKA a1
        /*
      [
          1 - a,
          1 - a,
          1 - a,
          1 - a,
          1 - a,
          1 - a,
          1 - a,
          1 - a
      ]
      */
        m256 oneminusa_lo = subtractVectors_float(oneVector, alpha_vec_low);
        m256 oneminusa_hi = subtractVectors_float(oneVector, alpha_vec_high);
        // print_float("oneminusa lo", oneminusa_lo);
        // print_float("oneminusa hi", oneminusa_hi);

        // calculate alpha * color
        /*
        [
            a * p1_r,
            a * p1_g,
            a * p1_b,
            a * p1_a,
            a * p2_r,
            a * p2_g,
            a * p2_b,
            a * p2_a
        ]
        */

        // print_float("alphaTimesColor", alphaTimesColor_lo);
        // print_float("alphaTimesColor", alphaTimesColor_hi);

        /*
        [
            a1 * bgcV_r,
            a1 * bgcV_g,
            a1 * bgcV_b,
            a1 * bgcV_a,
            a1 * bgcV_r,
            a1 * bgcV_g,
            a1 * bgcV_b,
            a1 * bgcV_a
        ]
        */
        // calculate 1 minus alpha * bgc_color
        m256 oneMinusAlphaTimesColor_lo = multiplyVectors_float(oneminusa_lo, bgcV);
        m256 oneMinusAlphaTimesColor_hi = multiplyVectors_float(oneminusa_hi, bgcV);

        // add oneminusAlphaTimesColor to AlphaTimesColor
        /*
            [
                (a * p1_r) + (a1 * bgc.red),
                (a * p1_g) + (a1 * bgc.green),
                (a * p1_b) + (a1 * bgc.blue),
                (a1 * p1_a) + (a1 * bgcV_b),
                (a * p2_r) + (a1 * bgc.red),
                (a * p2_g) + (a1 * bgc.green),
                (a * p2_b) + (a1 * bgc.blue),
                (a1 * p2_a) + (a1 * bgcV_b),
            ]
        */
        m256 preDst_lo = addVectors_float(alphaTimesColor_lo, oneMinusAlphaTimesColor_lo);
        m256 preDst_hi = addVectors_float(alphaTimesColor_hi, oneMinusAlphaTimesColor_hi);

        // Find a way to get all values from USHORTMAX VECTOR
        /*
            [
                (a * p1_r) + (a1 * bgc.red),
                (a * p1_g) + (a1 * bgc.green),
                (a * p1_b) + (a1 * bgc.blue),
                uShortMAX),
                (a * p2_r) + (a1 * bgc.red),
                (a * p2_g) + (a1 * bgc.green),
                (a * p2_b) + (a1 * bgc.blue),
                uShortMAX,
            ]
        */
        m256 preDstWithAlpha_lo = pickAndChooseFromVectors_float(preDst_lo, uShortMAX, combineWithAlphaMask);
        m256 preDstWithAlpha_hi = pickAndChooseFromVectors_float(preDst_hi, uShortMAX, combineWithAlphaMask);

        // convert two float vectors into integer vector
        m256i result = combineTwoFloatVectorstoIntegerVector(convertFVectorToIVector(preDstWithAlpha_lo), convertFVectorToIVector(preDstWithAlpha_hi));
        // printf("This is the result\n");
        // void print_pix(__m256i * result);

        // store whatever we did into the dist
        storeVectorToPixel(dst, result);
    }

    /*
        a = ((float)(src[i].alpha)) / USHRT_MAX;
        a1 = 1 - a;
        dst[i].red = (a * src[i].red) + (a1 * bgc.red);
        dst[i].green = (a * src[i].green) + (a1 * bgc.green);
        dst[i].blue = (a * src[i].blue) + (a1 * bgc.blue);
        dst[i].alpha = USHRT_MAX; // opaques

    */
}
char blend_v_descr_my[] = "blend_v: my working version";
void blend_v_my(int dim, pixel *src, pixel *dst)
{
    int i, dummy;
    dummy = dim * dim;
    const m256 bgcV = createVectorWith_float(bgc.red, bgc.green, bgc.blue, 0.0, bgc.red, bgc.green, bgc.blue, 0.0);
    const m256 ushrt_max = createVectorAs_float(1.0 / USHRT_MAX); // Vector for 1 / USHORTMAX
    const m256 oneVector = createVectorAs_float(1.0);             // Vector for 1
    const m256 uShortMAX = createVectorAs_float(USHRT_MAX);       // Vector for USHRT_MAX
    const m256i zeroes = createVectorAsZero_integer();            // Vector with zeroes
    const m256 combineWithAlphaMask = createMask_float(0, 0, 0, -1, 0, 0, 0, -1);
    // const m256i orderPattern = createVectorWith_integer(7, 7, 7, 7, 3, 3, 3, 3);
    for (i = 0; i < dummy; i += 4, src += 4, dst += 4)
    {
        m256i pix4 = load4PixelToVector_integer(src);

        // ordered like this for locality? No speed up tho.
        m256 loPixF = convertFrom256IntegerTo128BitFloatVector(splitVectorToFloat_lo(pix4, zeroes));
        m256 a_low = multiplyVectors_float(loPixF, ushrt_max);
        m256 alpha_vec_low = orderValuesBy_shuff_float(a_low, a_low, orderShuffle(3, 3, 7, 7));
        m256 oneminusa_lo = subtractVectors_float(oneVector, alpha_vec_low);
        m256 alphaTimesColor_lo = multiplyVectors_float(alpha_vec_low, loPixF);
        m256 oneMinusAlphaTimesColor_lo = multiplyVectors_float(oneminusa_lo, bgcV);
        m256 preDst_lo = addVectors_float(alphaTimesColor_lo, oneMinusAlphaTimesColor_lo);
        m256 preDstWithAlpha_lo = pickAndChooseFromVectors_float(preDst_lo, uShortMAX, combineWithAlphaMask);

        // ordered like this for locality? No speed up tho.
        m256 hiPixF = convertFrom256IntegerTo128BitFloatVector(splitVectorToFloat_hi(pix4, zeroes));
        m256 a_high = multiplyVectors_float(hiPixF, ushrt_max);
        m256 alpha_vec_high = orderValuesBy_shuff_float(a_high, a_high, orderShuffle(3, 3, 7, 7));
        m256 oneminusa_hi = subtractVectors_float(oneVector, alpha_vec_high);
        m256 alphaTimesColor_hi = multiplyVectors_float(alpha_vec_high, hiPixF);
        m256 oneMinusAlphaTimesColor_hi = multiplyVectors_float(oneminusa_hi, bgcV);
        m256 preDst_hi = addVectors_float(alphaTimesColor_hi, oneMinusAlphaTimesColor_hi);
        m256 preDstWithAlpha_hi = pickAndChooseFromVectors_float(preDst_hi, uShortMAX, combineWithAlphaMask);

        m256i result = combineTwoFloatVectorstoIntegerVector(convertFVectorToIVector(preDstWithAlpha_lo), convertFVectorToIVector(preDstWithAlpha_hi));

        storeVectorToPixel(dst, result);
    }
}
// Your different versions of the blend_v kernel go here
// (i.e. with vectorization, aka. SIMD).

char blend_v_descr[] = "blend_v: Current working version";
void blend_v(int dim, pixel *src, pixel *dst)
{
    blend_v_my(dim, src, dst);
}
/*
 * register_blend_v_functions - Register all of your different versions
 *     of the blend_v kernel with the driver by calling the
 *     add_blend_function() for each test function.
 */
void register_blend_v_functions()
{
    add_blend_v_function(&blend_v, blend_v_descr);
    add_blend_v_function(&blend_v_dev, blend_v_descr_dev);
    add_blend_v_function(&blend_v_my, blend_v_descr_my);
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
