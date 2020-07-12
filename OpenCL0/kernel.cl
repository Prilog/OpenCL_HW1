__kernel void prefix_sum_local(__global const float *a, __global float *b, unsigned int N, unsigned int n, __local float *buf) {   
	int x = get_local_id(0);   
	int y = get_global_id(0);
	buf[x] = a[y];
	float t = buf[x];
	float current = t;
	int i;
	for (i = 1; i < n; i *= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if ((x + 1) % (2 * i) == 0) {
			current = buf[x] + buf[x - i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		buf[x] = current;
	}
	if (x == n - 1) {
		buf[x] = 0;
	}
	for (i = i / 2; i > 0; i /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if ((x + 1) % (2 * i) == 0) {
			current = buf[x] + buf[x - i];
		} else if ((x + 1) % i == 0) {
			current = buf[x + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		buf[x] = current;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (x == n - 1) {
		current = buf[x] + t;
	} else {
		current = buf[x + 1];
	}
	b[y] = current;
}

__kernel void prefix_sum_global(__global const float *a, __global float *b, unsigned int N, unsigned int n) {
	int x = get_local_id(0);
	int y = get_global_id(0);
	__local float last;
	if (x == 0) {
		last = 0;
		for (int i = n - 1; i < y; i += n) {
			last += a[i];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	b[y] = a[y] + last;
}