# include "../include/SHTimer.cuh"

SHTimer::SHTimer() : timeElapsed(0.0f)
{
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
}
SHTimer::~SHTimer()
{
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
void SHTimer::startTimer()
{
	cudaEventRecord(start, 0);
}
float SHTimer::stopTimer()
{
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	checkCudaErrors(cudaEventElapsedTime(&timeElapsed, start, stop));
	return timeElapsed;
}