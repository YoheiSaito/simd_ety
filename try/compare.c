
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>//AVX: -mavx
#include <sys/time.h>

void vec_add(const size_t n, double *z, const double *x, const double *y);
void vec_sub(const size_t n, double *z, const double *x, const double *y);
void vec_mul(const size_t n, double *z, const double *x, const double *y);
void vec_div(const size_t n, double *z, const double *x, const double *y);
void vec_setzero(const size_t n, double *z);

int main(void)
{
	struct timeval st,ed;
	const size_t n = 1024;
	double *x, *y, *ad, *sb, *dv, *ml;
	double *s,*d,*a,*m, *xx, *yy;
	a = (double*) malloc(sizeof(double) * n);
	d = (double*) malloc(sizeof(double) * n);
	s = (double*) malloc(sizeof(double) * n);
	m = (double*) malloc(sizeof(double) * n);
	
	xx = (double*) malloc(sizeof(double) * n);
	yy = (double*) malloc(sizeof(double) * n);
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
	for(size_t i=0; i<n; ++i) xx[i] = i;
	for(size_t i=0; i<n; ++i) yy[i] = i+1;
	
	gettimeofday(&st, NULL);
	for(unsigned long i = 0; i < 5000000; i++){
		vec_setzero(n, ad);
		vec_setzero(n, sb);
		vec_setzero(n, ml);
		vec_setzero(n, dv);
	
		vec_add(n, ad, x, y);
		vec_sub(n, sb, x, y);
		vec_mul(n, ml, x, y);
		vec_div(n, dv, x, y);
	}

	gettimeofday(&ed, NULL);
	printf("simd : %lf\n",(ed.tv_sec - st.tv_sec) + (ed.tv_usec - st.tv_usec)*1.0E-6);
	
	gettimeofday(&st, NULL);
	for(unsigned long i = 0; i < 5000000; i++){
		for(int j = 0; j < n; j++){
			a[j] = 0;
			s[j] = 0;
			d[j] = 0;
			m[j] = 0;
		}
		for(int j = 0; j < n; j++){
			a[j] = xx[j] + yy[j];
			s[j] = xx[j] - yy[j];
			d[j] = xx[j] / yy[j];
			m[j] = xx[j] * yy[j];
		}
	}
	
	gettimeofday(&ed, NULL);
	printf("normal : %lf\n",(ed.tv_sec - st.tv_sec) + (ed.tv_usec - st.tv_usec)*1.0E-6);
	
	_mm_free(x);
	_mm_free(y);
	_mm_free(ad);
	_mm_free(sb);
	_mm_free(ml);
	_mm_free(dv);
	
	free(a);
	free(s);
	free(d);
	free(m);
	free(xx);
	free(yy);
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

