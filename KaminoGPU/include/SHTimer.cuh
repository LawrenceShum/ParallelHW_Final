# pragma once

# include "../include/SHHeader.cuh"

class SHTimer
{
private:
	cudaEvent_t start;
	cudaEvent_t stop;
	float timeElapsed;
public:
	SHTimer();
	~SHTimer();

	void startTimer();
	float stopTimer();
};
