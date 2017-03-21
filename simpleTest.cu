#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>


static const int ROW=1024;
static const int COL=1024;
void printArr(const char * name, float * arr, unsigned int len)
{
	for (int i=0;i<len;i++)
		printf("arr %s [%u]=%5.5f\n", name, i, arr[i]);
}
void initData(float *ip, int size)
{
	sleep(1);
	time_t t;
	srand((unsigned ) time(&t));
	for (int i=0;i<size;i++)
	{
		ip[i] = (float) (rand() & 0xFF)/10.0f;
	}
	//printArr("initData", ip, size);
}

unsigned int getTimeInUs()
{
	struct timespec tm;
	clock_gettime(CLOCK_REALTIME, &tm);
	return (unsigned int)(tm.tv_sec*1000*1000*1000+tm.tv_nsec);
}
void sumArraysOnHost(float *A, float *B, float *C, int N)
{
	for (int idx=0;idx<N;idx++)
	{
		C[idx]=A[idx]+B[idx];
	}
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C)
{
	int i=blockIdx.x*COL+threadIdx.x;
	//printf("[gpu]:gridDim.x=%u, gridDim.y=%u, gridDim.z=%u, blockDim.x=%u, blockDim.y=%u, blockDim.z=%u, blockIdx.x=%u, blockIdx.y=%u, blockIdx.z=%u,threadIdx.x=%u, threadIdx.y=%u, threadIdx.z=%u\n",
		//gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.x, blockIdx.y, blockIdx.z,threadIdx.x, threadIdx.y, threadIdx.z);
	C[i]=A[i]+B[i];
	//printf("sum[%u][%u]: A[%5.5f]+B[%5.5f]=C[%5.5f]\n",blockIdx.x, threadIdx.x, A[i], B[i], C[i]);
}

int main(int argc, char * argv[])
{
	printf("%s Starting\n", argv[0]);
	int dev=0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Using Device %d, %s\n", dev, deviceProp.name);
	//return 0;
	cudaSetDevice(dev);
	int nElem = ROW*COL;
	//int rpCount = 1000;
	printf("Vector size= %d\n", nElem);
	
	size_t nBytes = nElem * sizeof(float);
	
	float * hA = (float*)malloc(nBytes);
	float * hB = (float*)malloc(nBytes);
	float * hostRef = (float*)malloc(nBytes);
	float * gpuRef = (float*)malloc(nBytes);
	initData(hA, nElem);
	initData(hB, nElem);
	
	//printArr("hA", hA, nElem);
	//printArr("hB", hB, nElem);
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	float * dA;
	cudaMalloc((float**)&dA, nBytes);
	float * dB;
	cudaMalloc((float**)&dB, nBytes);
	float * dC;
	cudaMalloc((float**)&dC, nBytes);

	unsigned int startTime=getTimeInUs();

	printf("=======start data copying [%u]==========\n", startTime);
	cudaMemcpy(dA, hA, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, nBytes, cudaMemcpyHostToDevice);

	unsigned int workStartTime=getTimeInUs();
	printf("=======data copy finish, start working   [%u]   delta[%u]=======\n", workStartTime, workStartTime-startTime);
	dim3 block(COL);
	dim3 grid(ROW);

	sumArraysOnGPU<<<grid, block>>>(dA, dB, dC);
	cudaDeviceSynchronize();
	
	unsigned int workEndTime=getTimeInUs();
	printf("=======work on GPU finish                [%u]   delta[%u]==========\n", workEndTime, workEndTime-workStartTime);
	cudaMemcpy(gpuRef, dC, nBytes, cudaMemcpyDeviceToHost);
	unsigned int cpEndTime=getTimeInUs();
	printf("=======copy back to CPU finish           [%u]   delta[%u]==========\n", cpEndTime, cpEndTime-workEndTime);

	workStartTime=getTimeInUs();
	printf("=======work on CPU started               [%u]   ==========\n", workStartTime);
	sumArraysOnHost(hA, hB, hostRef, nElem);
	workEndTime=getTimeInUs();
	printf("=======work on CPU finished              [%u]   delta[%u]==========\n", workEndTime, workEndTime-workStartTime);

	//validation
	for (int i=0; i<nElem;i++)
	{
		if (hostRef[i]!=gpuRef[i])
		{
			printf("invalid result: host: %5.5f device: %5.5f\n", hostRef[i], gpuRef[i]);
		}
	}
	printf("=======finish==========\n");
	
	return 0;
}
