
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>//AVX: -mavx


void vec_add(const size_t n, double *z, const double *x, const double *y);
void vec_sub(const size_t n, double *z, const double *x, const double *y);
void vec_mul(const size_t n, double *z, const double *x, const double *y);
void vec_div(const size_t n, double *z, const double *x, const double *y);
void vec_setzero(const size_t n, double *z);

int main(void)
{
	const size_t n = 1024;
	double *x, *y, *ad, *sb, *dv, *ml;
	
	//アラインメントは32=128bit
	//double * nを32区切りで分割して_mm_mallocで確保

	x = (double *)_mm_malloc(sizeof(double) * n, 32);
	y = (double *)_mm_malloc(sizeof(double) * n, 32);


	ad = (double *)_mm_malloc(sizeof(double) * n, 32);
	dv = (double *)_mm_malloc(sizeof(double) * n, 32);
	sb = (double *)_mm_malloc(sizeof(double) * n, 32);
	ml = (double *)_mm_malloc(sizeof(double) * n, 32);
	
	for(size_t i=0; i<n; ++i) x[i] = i;
	for(size_t i=0; i<n; ++i) y[i] = i+1;
	
	vec_setzero(n, ad);
	vec_setzero(n, sb);
	vec_setzero(n, ml);
	vec_setzero(n, dv);
	printf("Set zero check");
	for(size_t i=0; i<n; ++i) 
		printf("%g\t%g\t%g\t%g\n", ad[i], sb[i], ml[i], dv[i]);
	vec_add(n, ad, x, y);
	vec_sub(n, sb, x, y);
	vec_mul(n, ml, x, y);
	vec_div(n, dv, x, y);

	
	printf("\n\nCalculate check\n");
	for(size_t i=0; i<n; ++i) 
		printf("%g\t%g\t%g\t%g\n", ad[i], sb[i], ml[i], dv[i]);

	//_mm_mallocで確保したものは_mm_freeで開放
	_mm_free(x);
	_mm_free(y);
	_mm_free(ad);
	_mm_free(sb);
	_mm_free(ml);
	_mm_free(dv);

	return 0;
}


void vec_setzero(const size_t n, double *z)
{	
	static const size_t double_size = 4;
	const size_t end = n / double_size;
	
	__m256d *vz = (__m256d *)z;
	
	for(size_t i=0; i<end; ++i)
		vz[i] = _mm256_setzero_pd();
}

void vec_add(const size_t n, double *z, const double *x, const double *y)
{	
	static const size_t double_size = 4;
	const size_t end = n / double_size;
	
	__m256d *vz = (__m256d *)z;
	__m256d *vx = (__m256d *)x;
	__m256d *vy = (__m256d *)y;
	
	for(size_t i=0; i<end; ++i)
		vz[i] = _mm256_add_pd(vx[i], vy[i]);
}

void vec_sub(const size_t n, double *z, const double *x, const double *y)
{	
	static const size_t double_size = 4;
	const size_t end = n / double_size;
	
	__m256d *vz = (__m256d *)z;
	__m256d *vx = (__m256d *)x;
	__m256d *vy = (__m256d *)y;
	
	for(size_t i=0; i<end; ++i)
		vz[i] = _mm256_sub_pd(vx[i], vy[i]);
}

void vec_mul(const size_t n, double *z, const double *x, const double *y)
{	
	static const size_t double_size = 4;
	const size_t end = n / double_size;
	
	__m256d *vz = (__m256d *)z;
	__m256d *vx = (__m256d *)x;
	__m256d *vy = (__m256d *)y;
	
	for(size_t i=0; i<end; ++i)
		vz[i] = _mm256_mul_pd(vx[i], vy[i]);
}


void vec_div(const size_t n, double *z, const double *x, const double *y)
{	
	static const size_t double_size = 4;
	const size_t end = n / double_size;
	
	__m256d *vz = (__m256d *)z;
	__m256d *vx = (__m256d *)x;
	__m256d *vy = (__m256d *)y;
	
	for(size_t i=0; i<end; ++i)
		vz[i] = _mm256_div_pd(vx[i], vy[i]);
}

