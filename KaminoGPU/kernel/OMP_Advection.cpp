# include "omp.h"
# include "KaminoQuantity.cuh"
# include "KaminoHeader.cuh"
# include "KaminoGPU.cuh"

using namespace std;

size_t num_Phi = 128;
size_t num_Theta = 64;

fReal validate(fReal& phi, fReal& theta)
{
	fReal ret = 1.0f;
	theta = theta - static_cast<int>(floorf(theta / M_2PI)) * M_2PI;
	if (theta > M_PI)
	{
		theta = M_2PI - theta;
		phi += M_PI;
		ret = -ret;
	}
	if (theta < 0)
	{
		theta = -theta;
		phi += M_PI;
		ret = -ret;
	}
	phi = phi - static_cast<int>(floorf(phi / M_2PI)) * M_2PI;
	return ret;
}

fReal Lerp(fReal from, fReal to, fReal alpha)
{
	return (1.0 - alpha) * from + alpha * to;
}

fReal samplePhi(fReal* input, fReal phiRaw, fReal thetaRaw, fReal gridLen)
{
	fReal phi = phiRaw - gridLen * vPhiPhiOffset;
	fReal theta = thetaRaw - gridLen * vPhiThetaOffset;
	// Phi and Theta are now shifted back to origin

	fReal invGridLen = 1.0 / gridLen;
	fReal isFlippedPole = validate(phi, theta);
	fReal normedPhi = phi * invGridLen;
	fReal normedTheta = theta * invGridLen;

	int phiIndex = static_cast<int>(floorf(normedPhi));
	int thetaIndex = static_cast<int>(floorf(normedTheta));
	fReal alphaPhi = normedPhi - static_cast<fReal>(phiIndex);
	fReal alphaTheta = normedTheta - static_cast<fReal>(thetaIndex);

	if ((thetaIndex == 0 && isFlippedPole == -1.0f)
		|| thetaIndex == num_Theta - 1)
	{
		size_t phiLower = (phiIndex) % num_Phi;
		size_t phiHigher = (phiLower + 1) % num_Phi;
		fReal lowerBelt = Lerp(input[phiLower + num_Phi * thetaIndex],
			input[phiHigher + num_Phi * thetaIndex], alphaPhi);

		phiLower = (phiIndex + num_Phi / 2) % num_Phi;
		phiHigher = (phiLower + 1) % num_Phi;

		fReal higherBelt = Lerp(input[phiLower + num_Phi * thetaIndex],
			input[phiHigher + num_Phi * thetaIndex], alphaPhi);

		fReal lerped = Lerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
	else
	{
		size_t phiLower = phiIndex % num_Phi;
		size_t phiHigher = (phiLower + 1) % num_Phi;
		size_t thetaLower = thetaIndex;
		size_t thetaHigher = thetaIndex + 1;

		fReal lowerBelt = Lerp(input[phiLower + num_Phi * thetaLower],
			input[phiHigher + num_Phi * thetaLower], alphaPhi);
		fReal higherBelt = Lerp(input[phiLower + num_Phi * thetaHigher],
			input[phiHigher + num_Phi * thetaHigher], alphaPhi);

		fReal lerped = Lerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
}

fReal sampleTheta(fReal* input, fReal phiRaw, fReal thetaRaw, size_t gridLen)
{
	fReal phi = phiRaw - gridLen * vThetaPhiOffset;
	fReal theta = thetaRaw - gridLen * vThetaThetaOffset;
	// Phi and Theta are now shifted back to origin

	fReal invGridLen = 1.0 / gridLen;
	fReal isFlippedPole = validate(phi, theta);
	fReal normedPhi = phi * invGridLen;
	fReal normedTheta = theta * invGridLen;

	int phiIndex = static_cast<int>(floorf(normedPhi));
	int thetaIndex = static_cast<int>(floorf(normedTheta));
	fReal alphaPhi = normedPhi - static_cast<fReal>(phiIndex);
	fReal alphaTheta = normedTheta - static_cast<fReal>(thetaIndex);

	if ((thetaIndex == 0 && isFlippedPole == -1.0f) ||
		thetaIndex == num_Theta - 2)
	{
		size_t phiLower = phiIndex % num_Phi;
		size_t phiHigher = (phiLower + 1) % num_Phi;
		fReal lowerBelt = Lerp(input[phiLower + num_Phi * thetaIndex],
			input[phiHigher + num_Phi * thetaIndex], alphaPhi);

		phiLower = (phiLower + num_Phi / 2) % num_Phi;
		phiHigher = (phiHigher + num_Phi / 2) % num_Phi;
		fReal higherBelt = Lerp(input[phiLower + num_Phi * thetaIndex],
			input[phiHigher + num_Phi * thetaIndex], alphaPhi);

		alphaTheta = 0.5 * alphaTheta;
		fReal lerped = Lerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	}
	else
	{
		size_t phiLower = phiIndex % num_Phi;
		size_t phiHigher = (phiLower + 1) % num_Phi;
		size_t thetaLower = thetaIndex;
		size_t thetaHigher = thetaIndex + 1;

		fReal lowerBelt = Lerp(input[phiLower + num_Phi * thetaLower],
			input[phiHigher + num_Phi * thetaLower], alphaPhi);
		fReal higherBelt = Lerp(input[phiLower + num_Phi * thetaHigher],
			input[phiHigher + num_Phi * thetaHigher], alphaPhi);

		fReal lerped = Lerp(lowerBelt, higherBelt, alphaTheta);
		return lerped;
	} 
}

void OMP_advect_phi(fReal* velPhi, fReal* velTheta, fReal* outputVel, fReal r, fReal dt)
{
	omp_set_num_threads(8);
	fReal gridLen = M_PI / num_Theta;
	std::cout << "fuck" << std::endl;
	fReal* temp = new fReal[num_Phi * num_Theta];
#pragma omp parallel
	{
#pragma omp for
		for (int j = 0; j < num_Theta; j++)
		{
			
			for (int i = 0; i < num_Phi; i++)
			{
				int phiId = i;
				int thetaId = j;
				// Coord in phi-theta space
				fReal gPhi = ((fReal)phiId + vPhiPhiOffset) * gridLen;
				fReal gTheta = ((fReal)thetaId + vPhiThetaOffset) * gridLen;

				// Sample the speed
				fReal guPhi = samplePhi(velPhi, gPhi, gTheta, gridLen);
				fReal guTheta = sampleTheta(velTheta, gPhi, gTheta, gridLen);
				//std::cout << guPhi << std::endl;
				std::cout << guTheta << std::endl;
				fReal latRadius = r * sinf(gTheta);
				fReal cofPhi = dt / latRadius;
				fReal cofTheta = dt / r;

				fReal deltaPhi = guPhi * cofPhi;
				fReal deltaTheta = guTheta * cofTheta;

				fReal pPhi = gPhi - deltaPhi;
				fReal pTheta = gTheta - deltaTheta;

				fReal advectedVal = samplePhi(velPhi, pPhi, pTheta, gridLen);
				outputVel[thetaId * num_Phi + phiId] = advectedVal;
				//std::cout << advectedVal << std::endl;
			}
			
		}
	}
}

void OMP_advect_Theta(fReal* velPhi, fReal* velTheta, fReal* outputVel, fReal r, fReal dt)
{
	omp_set_num_threads(8);
	fReal gridLen = M_PI / num_Theta;
	std::cout << "oh my god" << std::endl;
#pragma omp parallel
	{
#pragma omp for
		for (int j = 0; j < num_Theta - 1; j++)
		{
			for (int i = 0; i < num_Phi; i++)
			{
				int phiId = i;
				int thetaId = j;
				// Coord in phi-theta space
				fReal gPhi = ((fReal)phiId + vThetaPhiOffset) * gridLen;
				fReal gTheta = ((fReal)thetaId + vThetaThetaOffset) * gridLen;

				// Sample the speed
				fReal guPhi = samplePhi(velPhi, gPhi, gTheta, gridLen);
				fReal guTheta = sampleTheta(velTheta, gPhi, gTheta, gridLen);

				fReal latRadius = r * sinf(gTheta);
				fReal cofPhi = dt / latRadius;
				fReal cofTheta = dt / r;

				fReal deltaPhi = guPhi * cofPhi;
				fReal deltaTheta = guTheta * cofTheta;

				fReal pPhi = gPhi - deltaPhi;
				fReal pTheta = gTheta - deltaTheta;

				fReal advectedVal = sampleTheta(velTheta, pPhi, pTheta, gridLen);

				outputVel[thetaId * num_Phi + phiId] = advectedVal;
				//std::cout << advectedVal << std::endl;
			}
		}
	}
}