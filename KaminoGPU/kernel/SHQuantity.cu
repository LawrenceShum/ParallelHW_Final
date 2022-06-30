# include "../include/SHQuantity.cuh"

void SHQuantity::copyToGPU()
{
	/* 
	Pitch : nPhi * sizeof(fReal)
	Width : nPhi * sizeof(fReal)
	Height: nTheta
	*/
	checkCudaErrors(cudaMemcpy2D(gpuThisStep, thisStepPitch, cpuBuffer, 
		nPhi * sizeof(fReal), nPhi * sizeof(fReal), nTheta, cudaMemcpyHostToDevice));
}

void SHQuantity::copyBackToCPU()
{
	checkCudaErrors(cudaMemcpy2D((void*)this->cpuBuffer, nPhi * sizeof(fReal), (void*)this->gpuThisStep,
	this->thisStepPitch, nPhi * sizeof(fReal), nTheta, cudaMemcpyDeviceToHost));
}

SHQuantity::SHQuantity(std::string attributeName, size_t nPhi, size_t nTheta,
	fReal phiOffset, fReal thetaOffset)
	: nPhi(nPhi), nTheta(nTheta), gridLen(M_2PI / nPhi), invGridLen(1.0 / gridLen),
	attrName(attributeName), phiOffset(phiOffset), thetaOffset(thetaOffset)
{
	cpuBuffer = new fReal[nPhi * nTheta];
	cpuBufferNext = new fReal[nPhi * nTheta];
	checkCudaErrors(cudaMallocPitch((void**)&gpuThisStep, &thisStepPitch, nPhi * sizeof(fReal), nTheta));
	checkCudaErrors(cudaMallocPitch((void**)&gpuNextStep, &nextStepPitch, nPhi * sizeof(fReal), nTheta));
}

SHQuantity::~SHQuantity()
{
	delete[] cpuBuffer;
	delete[] cpuBufferNext;
	checkCudaErrors(cudaFree(gpuThisStep));
	checkCudaErrors(cudaFree(gpuNextStep));
}

std::string SHQuantity::getName()
{
	return this->attrName;
}

size_t SHQuantity::getNPhi()
{
	return this->nPhi;
}

size_t SHQuantity::getNTheta()
{
	return this->nTheta;
}

void SHQuantity::swapCPUBuffer()
{
	fReal* tempPtr = this->cpuBuffer;
	this->cpuBuffer = this->cpuBufferNext;
	this->cpuBufferNext = tempPtr;
}

void SHQuantity::swapGPUBuffer()
{
	fReal* tempPtr = this->gpuThisStep;
	this->gpuThisStep = this->gpuNextStep;
	this->gpuNextStep = tempPtr;
}

fReal SHQuantity::getCPUValueAt(size_t phi, size_t theta)
{
	return this->accessCPUValueAt(phi, theta);
}

void SHQuantity::setCPUValueAt(size_t phi, size_t theta, fReal val)
{
	this->accessCPUValueAt(phi, theta) = val;
}

fReal& SHQuantity::accessCPUValueAt(size_t phi, size_t theta)
{
	return this->cpuBuffer[theta * nPhi + phi];
}

fReal SHQuantity::getThetaOffset()
{
	return this->thetaOffset;
}

fReal SHQuantity::getPhiOffset()
{
	return this->phiOffset;
}

fReal* SHQuantity::getGPUThisStep()
{
	return this->gpuThisStep;
}

fReal* SHQuantity::getGPUNextStep()
{
	return this->gpuNextStep;
}

size_t SHQuantity::getThisStepPitchInElements()
{
	return this->thisStepPitch / sizeof(fReal);
}

size_t SHQuantity::getNextStepPitchInElements()
{
	return this->nextStepPitch / sizeof(fReal);
}

fReal* SHQuantity::getCPUThisStep()
{
	return this->cpuBuffer;
}

fReal* SHQuantity::getCPUNextStep()
{
	return this->cpuBufferNext;
}