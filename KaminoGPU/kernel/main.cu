# include "../include/SHGPU.cuh"
# include <fstream>

int main(int argc, char** argv)
{
	if (argc == 2)
	{
		std::string configFile = argv[1];
		std::fstream fin;
		fin.open(configFile, std::ios::in);
		fReal radius; size_t nTheta; fReal particleDensity;
		float dt; float DT; int frames;
		float A; int B; int C; int D; int E;
		std::string gridPath; std::string particlePath;
		std::string densityImage; std::string solidImage; std::string colorImage;

		fin >> radius;
		fin >> nTheta;
		fin >> particleDensity;
		fin >> dt;
		fin >> DT;
		fin >> frames;
		fin >> A;
		fin >> B;
		fin >> C;
		fin >> D;
		fin >> E;
		fin >> gridPath;
		fin >> particlePath;

		fin >> densityImage;
		if (densityImage == "null")
		{
			densityImage = "";
		}
		fin >> solidImage;
		if (solidImage == "null")
		{
			solidImage == "";
		}
		fin >> colorImage;
		if (colorImage == "null")
		{
			colorImage = "";
		}

		SH SHInstance(radius, nTheta, particleDensity, dt, DT, frames,
			A, B, C, D, E,
			gridPath, particlePath, densityImage, solidImage, colorImage);
		SHInstance.run();
		return 0;
	}
	else
	{
		//std::cout << "Please provide the path to configSH.txt as an argument." << std::endl;
		//std::cout << "Usage example: ./SH.exe ./configSH.txt" << std::endl;
		//std::cout << "Configuration file was missing, exiting." << std::endl;
		//return -1;
		fReal radius; size_t nTheta; fReal particleDensity;
		float dt; float DT; int frames;
		float A; int B; int C; int D; int E;
		std::string gridPath; std::string particlePath;
		std::string densityImage; std::string solidImage; std::string colorImage;

		radius = 10.0;
		nTheta = 512;
		particleDensity = 2.0;
		dt = 0.001;
		DT = 1.0 / 24;
		frames = 100;
		A = 0;
		B = 0;
		C = 0;
		D = 0;
		E = 0;
		gridPath = "/data/grids";
		particlePath = "/data/particles";
		densityImage = "";
		solidImage == "";
		colorImage = "";

		SH SHInstance(radius, nTheta, particleDensity, dt, DT, frames,
			A, B, C, D, E,
			gridPath, particlePath, densityImage, solidImage, colorImage);
		SHInstance.run();
		return 0;
	}
}