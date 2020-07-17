#include "volume.h"

#include <fstream>
#include <stdexcept>
#include <cuMat/src/Errors.h>
#include <experimental/filesystem>

#include "halton_sampler.h"

#ifdef RENDERER_HAS_RENDERER
BEGIN_RENDERER_NAMESPACE

std::unique_ptr<Volume> TheVolume;

const int Volume::BytesPerType[Volume::_TypeCount_] = {
	1, 2, 4 
};

Volume::MipmapLevel::MipmapLevel(Volume* parent, size_t sizeX, size_t sizeY, size_t sizeZ)
	: dataCpu_(new char[sizeX * sizeY * sizeZ * BytesPerType[parent->type()]])
    , dataGpu_(nullptr)
	, sizeX_(sizeX), sizeY_(sizeY), sizeZ_(sizeZ)
	, cpuDataCounter_(0), gpuDataCounter_(0)
	, dataTex_(0)
	, parent_(parent)
{
}

Volume::MipmapLevel::~MipmapLevel()
{
	delete[] dataCpu_;
	if (dataTex_ != 0)
		CUMAT_SAFE_CALL(cudaDestroyTextureObject(dataTex_));
	if (dataGpu_)
		CUMAT_SAFE_CALL(cudaFreeArray(dataGpu_));
}

void Volume::MipmapLevel::copyCpuToGpu()
{
	if (gpuDataCounter_ == cpuDataCounter_ && dataGpu_)
		return; //nothing changed
	gpuDataCounter_ = cpuDataCounter_;

	//create array
	cudaExtent extent = make_cudaExtent(sizeX_, sizeY_, sizeZ_);
	if (!dataGpu_) {
		cudaChannelFormatDesc channelDesc;
		switch (parent_->type()) {
		case TypeUChar:
			channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
			break;
		case TypeUShort:
			channelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
			break;
		case TypeFloat:
			channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
			break;
		default:
			throw std::runtime_error("unknown enum constant");
		}
		CUMAT_SAFE_CALL(cudaMalloc3DArray(&dataGpu_, &channelDesc, extent));
		std::cout << "Cuda array allocated" << std::endl;
	}
	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = make_cudaPitchedPtr(dataCpu_,
		BytesPerType[parent_->type()] * sizeX_, sizeX_, sizeY_);
	params.dstArray = dataGpu_;
	params.extent = extent;
	params.kind = cudaMemcpyHostToDevice;
	CUMAT_SAFE_CALL(cudaMemcpy3D(&params));

	//create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = dataGpu_;
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = parent_->type() == TypeFloat ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 0;
	if (dataTex_ == 0)
		CUMAT_SAFE_CALL(cudaDestroyTextureObject(dataTex_));
	dataTex_ = 0;
	CUMAT_SAFE_CALL(cudaCreateTextureObject(&dataTex_, &resDesc, &texDesc, NULL));
}

Volume::Volume()
	: worldSizeX_(1), worldSizeY_(1), worldSizeZ_(1)
	, type_(DataType::TypeUChar)
{
}

Volume::Volume(DataType type, size_t sizeX, size_t sizeY, size_t sizeZ)
	: worldSizeX_(1), worldSizeY_(1), worldSizeZ_(1)
	, type_(type)
{
	levels_.push_back(std::unique_ptr<MipmapLevel>(new MipmapLevel(this, sizeX, sizeY, sizeZ)));
}

Volume::~Volume()
{
}

static const char MAGIC[] = "cvol";

/*
 * FORMAT:
 * magic number "cvol", 4Bytes
 * sizeXYZ, 3*8 Bytes
 * voxelSizeXYZ, 3*8 Bytes,
 * datatype, 4Bytes
 * 8 bytes padding
 * ==> 64 Bytes header
 * Then follows the raw data
 */

void Volume::save(const std::string& filename,
	const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging,
	const VolumeErrorCallback_t& error) const
{
	assert(sizeof(size_t) == 8);
	assert(sizeof(double) == 8);
	std::ofstream s(filename, std::fstream::binary);

	const MipmapLevel* data = getLevel(0);

	//header
	double voxelSizeX = worldSizeX_ / data->sizeX_;
	double voxelSizeY = worldSizeY_ / data->sizeY_;
	double voxelSizeZ = worldSizeZ_ / data->sizeZ_;
	s.write(MAGIC, 4);
	s.write(reinterpret_cast<const char*>(&data->sizeX_), 8);
	s.write(reinterpret_cast<const char*>(&data->sizeY_), 8);
	s.write(reinterpret_cast<const char*>(&data->sizeZ_), 8);
	s.write(reinterpret_cast<const char*>(&voxelSizeX), 8);
	s.write(reinterpret_cast<const char*>(&voxelSizeY), 8);
	s.write(reinterpret_cast<const char*>(&voxelSizeZ), 8);
	int type = static_cast<int>(type_);
	s.write(reinterpret_cast<const char*>(&type), 4);
	char padding[8] = { 0 };
	s.write(padding, 8);

	//body
	progress(0.0f);
	size_t dataToWrite = BytesPerType[type_] * data->sizeX_ * data->sizeY_;
	for (int z = 0; z < data->sizeZ_; ++z)
	{
		s.write(data->dataCpu_ + z * dataToWrite, dataToWrite);
		if (z % 10 == 0)
			progress(z / float(data->sizeZ_));
	}
	progress(1.0f);
}

Volume::Volume(const std::string& filename,
	const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging,
	const VolumeErrorCallback_t& error)
	: Volume()
{
	assert(sizeof(size_t) == 8);
	assert(sizeof(double) == 8);
	std::ifstream s(filename, std::fstream::binary);

	//header
	char magic[4];
	s.read(magic, 4);
	if (memcmp(MAGIC, magic, 4) != 0)
	{
		error("Illegal magic number", -1);
	}
	size_t sizeX, sizeY, sizeZ;
	double voxelSizeX, voxelSizeY, voxelSizeZ;
	s.read(reinterpret_cast<char*>(&sizeX), 8);
	s.read(reinterpret_cast<char*>(&sizeY), 8);
	s.read(reinterpret_cast<char*>(&sizeZ), 8);
	s.read(reinterpret_cast<char*>(&voxelSizeX), 8);
	s.read(reinterpret_cast<char*>(&voxelSizeY), 8);
	s.read(reinterpret_cast<char*>(&voxelSizeZ), 8);
	int type;
	s.read(reinterpret_cast<char*>(&type), 4);
	s.ignore(8);
	type_ = static_cast<DataType>(type);

	//create level
	levels_.push_back(std::unique_ptr<MipmapLevel>(new MipmapLevel(this, sizeX, sizeY, sizeZ)));
	MipmapLevel* data = levels_[0].get();
	worldSizeX_ = voxelSizeX * sizeX;
	worldSizeY_ = voxelSizeY * sizeY;
	worldSizeZ_ = voxelSizeZ * sizeZ;

	//body
	progress(0.0f);
	size_t dataToRead = BytesPerType[type_] * data->sizeX_ * data->sizeY_;
	for (int z = 0; z < data->sizeZ_; ++z)
	{
		s.read(data->dataCpu_ + z * dataToRead, dataToRead);
		if (z % 10 == 0)
			progress(z / float(data->sizeZ_));
	}
	progress(1.0f);
}

namespace
{
	//copied and adapted from Pytorch: ATen/native/AdaptiveAveragePooling3d.cpp

	inline int start_index(int a, int b, int c) {
		return (int)std::floor((float)(a * c) / b);
	}

	inline int end_index(int a, int b, int c) {
		return (int)std::ceil((float)((a + 1) * c) / b);
	}
	
	template<typename T>
    void adaptive_avg_pool3d(const Volume::MipmapLevel* in, Volume::MipmapLevel* out)
	{
		const T* dataIn = in->dataCpu<T>();
		T* dataOut = out->dataCpu<T>();
		//fetch sizes
		const int inSizeX  = static_cast<int>(in->sizeX());
		const int inSizeY  = static_cast<int>(in->sizeY());
		const int inSizeZ  = static_cast<int>(in->sizeZ());
		const int outSizeX = static_cast<int>(out->sizeX());
		const int outSizeY = static_cast<int>(out->sizeY());
		const int outSizeZ = static_cast<int>(out->sizeZ());
		//loop over output
#pragma omp parallel for
		for (int oz = 0; oz < outSizeZ; ++oz)
		{
			const int iStartZ = start_index(oz, outSizeZ, inSizeZ);
			const int iEndZ = end_index(oz, outSizeZ, inSizeZ);
			const int kZ = iEndZ - iStartZ;
			for (int oy = 0; oy < outSizeY; ++oy)
			{
				const int iStartY = start_index(oy, outSizeY, inSizeY);
				const int iEndY = end_index(oy, outSizeY, inSizeY);
				const int kY = iEndY - iStartY;
				for (int ox = 0; ox < outSizeX; ++ox)
				{
					const int iStartX = start_index(ox, outSizeX, inSizeX);
					const int iEndX = end_index(ox, outSizeX, inSizeX);
					const int kX = iEndX - iStartX;

					//compute local average
					float sum = 0;
					for (int iz = iStartZ; iz < iEndZ; ++iz)
						for (int iy = iStartY; iy < iEndY; ++iy)
							for (int ix = iStartX; ix < iEndX; ++ix)
								sum += static_cast<float>(dataIn[in->idx(ix, iy, iz)]);
					dataOut[out->idx(ox, oy, oz)] = static_cast<T>(sum / (kX*kY*kZ));
				}
			}
		}
	}

	//Halton-sampling the pixels to use.
	//It uses base 3, 5, 7 for the x,y,z axis
	template<typename T>
	void adaptive_halton_pool3d(const Volume::MipmapLevel* in, Volume::MipmapLevel* out)
	{
		const T* dataIn = in->dataCpu<T>();
		T* dataOut = out->dataCpu<T>();
		//fetch sizes
		const int inSizeX = static_cast<int>(in->sizeX());
		const int inSizeY = static_cast<int>(in->sizeY());
		const int inSizeZ = static_cast<int>(in->sizeZ());
		const int outSizeX = static_cast<int>(out->sizeX());
		const int outSizeY = static_cast<int>(out->sizeY());
		const int outSizeZ = static_cast<int>(out->sizeZ());
		//loop over output
#pragma omp parallel for
		for (int oz = 0; oz < outSizeZ; ++oz)
		{
			const int iStartZ = start_index(oz, outSizeZ, inSizeZ);
			const int iEndZ = end_index(oz, outSizeZ, inSizeZ);
			const int kZ = iEndZ - iStartZ;
			for (int oy = 0; oy < outSizeY; ++oy)
			{
				const int iStartY = start_index(oy, outSizeY, inSizeY);
				const int iEndY = end_index(oy, outSizeY, inSizeY);
				const int kY = iEndY - iStartY;
				for (int ox = 0; ox < outSizeX; ++ox)
				{
					const int iStartX = start_index(ox, outSizeX, inSizeX);
					const int iEndX = end_index(ox, outSizeX, inSizeX);
					const int kX = iEndX - iStartX;

					//get sample index
					const uint64_t sampleIdx = uint64_t(out->idx(ox, oy, oz));
					const int ix = iStartX + int(kX * HaltonSampler::Sample<3, float>(sampleIdx));
					const int iy = iStartY + int(kY * HaltonSampler::Sample<5, float>(sampleIdx));
					const int iz = iStartZ + int(kZ * HaltonSampler::Sample<7, float>(sampleIdx));
					dataOut[out->idx(ox, oy, oz)] = dataIn[in->idx(ix, iy, iz)];
				}
			}
		}
	}
}

void Volume::createMipmapLevel(int level, MipmapFilterMode filter)
{
	switch (filter)
	{
	case MipmapFilterMode::AVERAGE:
		createMipmapLevelAverage(level);
		break;
	case MipmapFilterMode::HALTON:
		createMipmapLevelHalton(level);
		break;
	default:
		throw std::runtime_error("Unknown enum constant");
	}
}

bool Volume::mipmapCheckOrCreate(int level)
{
	TORCH_CHECK(level >= 0, "level has to be non-zero, but is ", level);
	if (level < levels_.size() && levels_[level]) return false; //already available

	//create storage
	if (level >= levels_.size()) levels_.resize(level + 1);
	size_t newSizeX = std::max(size_t(1), levels_[0]->sizeX_ / (level + 1));
	size_t newSizeY = std::max(size_t(1), levels_[0]->sizeY_ / (level + 1));
	size_t newSizeZ = std::max(size_t(1), levels_[0]->sizeZ_ / (level + 1));
	levels_[level] = std::unique_ptr<MipmapLevel>(new MipmapLevel(this, newSizeX, newSizeY, newSizeZ));
	return true;
}

void Volume::createMipmapLevelAverage(int level)
{
	if (!mipmapCheckOrCreate(level)) return; //already available
	auto data = levels_[level].get();

	//perform area filtering
	switch (type_)
	{
	case TypeUChar:
		adaptive_avg_pool3d<unsigned char>(levels_[0].get(), data);
		break;
	case TypeUShort:
		adaptive_avg_pool3d<unsigned short>(levels_[0].get(), data);
		break;
	case TypeFloat:
		adaptive_avg_pool3d<float>(levels_[0].get(), data);
		break;
	default:
		throw std::runtime_error("Unknown enum constant");
	}
}

void Volume::createMipmapLevelHalton(int level)
{
	if (!mipmapCheckOrCreate(level)) return; //already available
	auto data = levels_[level].get();

	//perform area filtering
	switch (type_)
	{
	case TypeUChar:
		adaptive_halton_pool3d<unsigned char>(levels_[0].get(), data);
		break;
	case TypeUShort:
		adaptive_halton_pool3d<unsigned short>(levels_[0].get(), data);
		break;
	case TypeFloat:
		adaptive_halton_pool3d<float>(levels_[0].get(), data);
		break;
	default:
		throw std::runtime_error("Unknown enum constant");
	}
}

void Volume::deleteAllMipmapLevels()
{
	levels_.resize(1); //just keep the first level = original data
}

const Volume::MipmapLevel* Volume::getLevel(int level) const
{
	TORCH_CHECK(level >= 0, "level has to be non-zero, but is ", level);
	if (level >= levels_.size()) return nullptr;
	return levels_[level].get();
}

Volume::MipmapLevel* Volume::getLevel(int level)
{
	TORCH_CHECK(level >= 0, "level has to be non-zero, but is ", level);
	if (level >= levels_.size()) return nullptr;
	return levels_[level].get();
}

void CloseVolume()
{
	TheVolume.reset();
	std::cout << "Volume closed and memory freed" << std::endl;
}

static void printProgress(const std::string& prefix, float progress)
{
	int barWidth = 50;
	std::cout << prefix << " [";
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progress * 100.0) << " %\r";
	std::cout.flush();
	if (progress >= 1) std::cout << std::endl;
}

Volume* loadVolumeFromRaw(const std::string& filename, const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging, const VolumeErrorCallback_t& error, int64_t downsampling)
{
	auto filename_path = std::experimental::filesystem::path(filename);
	//read descriptor file
	if (filename_path.extension() == "dat")
	{
		error("Unrecognized extension, .dat expected : ." + filename_path.extension().string(), -1);
		return nullptr;
	}
	std::ifstream file(filename);
	if (!file.is_open())
	{
		error("Unable to open file " + filename, -1);
		return nullptr;
	}
	std::string line;
	std::string objectFileName = "";
	size_t resolutionX = 0;
	size_t resolutionY = 0;
	size_t resolutionZ = 0;
	double sliceThicknessX = 1;
	double sliceThicknessY = 1;
	double sliceThicknessZ = 1;
	std::string datatype = "";
	const std::string DATATYPE_UCHAR = "UCHAR";
	const std::string DATATYPE_USHORT = "USHORT";
	const std::string DATATYPE_BYTE = "BYTE";
	while (std::getline(file, line))
	{
		if (line.empty()) continue;
		std::istringstream iss(line);
		std::string token;
		iss >> token;
		if (!iss) continue; //no token in the current line
		if (token == "ObjectFileName:")
			iss >> objectFileName;
		else if (token == "Resolution:")
			iss >> resolutionX >> resolutionY >> resolutionZ;
		else if (token == "SliceThickness:")
			iss >> sliceThicknessX >> sliceThicknessY >> sliceThicknessZ;
		else if (token == "Format:")
			iss >> datatype;
		if (!iss)
		{
			error("Unable to parse line with token " + token, -2);
			return nullptr;
		}
	}
	file.close();
	if (objectFileName.empty() || resolutionX == 0 || datatype.empty())
	{
		error("Descriptor file does not contain ObjectFileName, Resolution and Format", -3);
		return nullptr;
	}
	if (!(datatype == DATATYPE_UCHAR || datatype == DATATYPE_USHORT || datatype == DATATYPE_BYTE))
	{
		error("Unknown format " + datatype, -4);
		return nullptr;
	}
	logging(std::string("Descriptor file read")
		+ "\nObjectFileName: " + objectFileName
		+ "\nResolution: " + std::to_string(resolutionX) + ", " + std::to_string(resolutionY) + ", " + std::to_string(resolutionZ)
		+ "\nFormat: " + datatype);

	//downsampling
	if ((resolutionX % downsampling != 0) || 
		(resolutionY % downsampling != 0) ||
		(resolutionZ % downsampling != 0))
	{
		error("Resolution is not divisible by the downsampling factor, can't downsample", -6);
		return nullptr;
	}
	resolutionX /= downsampling;
	resolutionY /= downsampling;
	resolutionZ /= downsampling;
	
	// open volume
	size_t bytesPerEntry = 0;
	if (datatype == DATATYPE_UCHAR) bytesPerEntry = 1;
	if (datatype == DATATYPE_BYTE) bytesPerEntry = 1;
	if (datatype == DATATYPE_USHORT) bytesPerEntry = 2;
	size_t bytesToRead = resolutionX * resolutionY * resolutionZ * bytesPerEntry;
	std::string bfilename = filename_path.replace_filename(objectFileName).generic_string();

	if (bytesToRead > 4096ll * 4096ll * 4096ll * 16)
	{
		error("Files is too large", -10);
		return nullptr;
	}

	std::cout << "Load " << bytesToRead << " bytes from " << bfilename << std::endl;
	std::ifstream bfile(bfilename, std::ifstream::binary | std::ifstream::ate);
	if (!bfile.is_open())
	{
		error("Unable to open file " + bfilename, -5);
		return nullptr;
	}
	size_t filesize = bfile.tellg();
	int headersize = static_cast<int>(filesize - static_cast<long long>(bytesToRead));
	if (headersize < 0)
	{
		error("File is too small, " + std::to_string(-headersize) + " bytes missing", -6);
		return nullptr;
	}
	std::cout << "Skipping header of length " << headersize << std::endl;
	bfile.seekg(std::ifstream::pos_type(headersize));

	// create output volume and read the data
	bytesToRead = resolutionX * resolutionY * bytesPerEntry * downsampling* downsampling* downsampling;
	std::vector<char> data(bytesToRead);

	std::unique_ptr<Volume> vol;
	if (datatype == DATATYPE_UCHAR || datatype == DATATYPE_BYTE) {
		vol = std::make_unique<Volume>(
			Volume::TypeUChar, resolutionX, resolutionY, resolutionZ);
		Volume::MipmapLevel* level = vol->getLevel(0);
		unsigned char* volumeData = level->dataCpu<unsigned char>();
		const unsigned char* raw = reinterpret_cast<unsigned char*>(data.data());
		for (int z = 0; z < resolutionZ; ++z)
		{
			bfile.read(&data[0], bytesToRead);
			if (!bfile)
			{
				error("Loading data file failed, attempted to read "+std::to_string(bytesToRead)+" bytes", -7);
				if (bfile.eof()) error("eofbit is set", -7);
				if (bfile.bad()) error("badbit is set", -7);
				if (bfile.fail()) error("failbit is set", -7);
				return nullptr;
			}
			if (z % 10 == 0)
				progress(z / float(resolutionZ));
#pragma omp parallel for
			for (int y = 0; y < resolutionY; ++y)
				for (int x = 0; x < resolutionX; ++x)
				{
					float value = 0;
					for (int iz=0; iz<downsampling; ++iz)
						for (int iy=0; iy<downsampling; ++iy)
							for (int ix=0; ix<downsampling; ++ix)
							{
								long long idx2 = (ix + downsampling * x)
									+ (resolutionX*downsampling) * ((iy + downsampling * y) 
										+ resolutionY * downsampling * iz);
								unsigned char val = raw[idx2];
								value += float(val) / 255.0;
							}
					value /= downsampling * downsampling * downsampling;
					volumeData[level->idx(x, y, z)] = static_cast<unsigned char>(
						std::max(0, std::min(255, static_cast<int>(value*255))));
				}
		}
	}
	else if (datatype == DATATYPE_USHORT) {
		vol = std::make_unique<Volume>(
			Volume::TypeUShort, resolutionX, resolutionY, resolutionZ);
		Volume::MipmapLevel* level = vol->getLevel(0);
		unsigned short* volumeData = level->dataCpu<unsigned short>();
		const unsigned short* raw = reinterpret_cast<unsigned short*>(data.data());
		for (int z = 0; z < resolutionZ; ++z)
		{
			bfile.read(&data[0], bytesToRead);
			if (!bfile)
			{
				error("Loading data file failed, attempted to read " + std::to_string(bytesToRead) + " bytes", -7);
				if (bfile.eof()) error("eofbit is set", -7);
				if (bfile.bad()) error("badbit is set", -7);
				if (bfile.fail()) error("failbit is set", -7);
				return nullptr;
			}
			if (z % 10 == 0)
				progress(z / float(resolutionZ));
#pragma omp parallel for
			for (int y = 0; y < resolutionY; ++y)
				for (int x = 0; x < resolutionX; ++x)
				{
					//unsigned short val = raw[x + resolutionX * y];
					//volumeData[level->idx(x, y, z)] = val;
					float value = 0;
					for (int iz = 0; iz < downsampling; ++iz)
						for (int iy = 0; iy < downsampling; ++iy)
							for (int ix = 0; ix < downsampling; ++ix)
							{
								long long idx2 = (ix + downsampling * x)
									+ (resolutionX * downsampling) * ((iy + downsampling * y)
										+ resolutionY * downsampling * iz);
								unsigned short val = raw[idx2];
								value += float(val) / 65535.0f;
							}
					value /= downsampling * downsampling * downsampling;
					volumeData[level->idx(x, y, z)] = static_cast<unsigned short>(
						std::max(0, std::min(65535, static_cast<int>(value * 65535))));
				}
		}
	}
	progress(1.0f);

	// set voxel size, scale so that a box of at most 1x1x1 is occupied
	double maxSize = std::max({
		sliceThicknessX * resolutionX,
		sliceThicknessY * resolutionY,
		sliceThicknessZ * resolutionZ
	});
	vol->setWorldSizeX(sliceThicknessX / maxSize * resolutionX);
	vol->setWorldSizeY(sliceThicknessY / maxSize * resolutionY);
	vol->setWorldSizeZ(sliceThicknessZ / maxSize * resolutionZ);

	//done
	std::stringstream s;
	s << "Reading done, resolution=(" << resolutionX <<
		"," << resolutionY << "," << resolutionZ <<
		"), size=(" << vol->worldSizeX() <<
		"," << vol->worldSizeY() << "," << vol->worldSizeZ() <<
		")";
	logging(s.str());

	return vol.release();
}


int64_t LoadVolumeFromRaw(const std::string& filename, int64_t downsampling)
{
	CloseVolume();

	VolumeProgressCallback_t progress = [](float v)
	{
		printProgress("Load", v);
	};
	VolumeLoggingCallback_t logging = [](const std::string& msg)
	{
		std::cout << msg << std::endl;
	};
	int errorCode = 1;
	VolumeErrorCallback_t error = [&errorCode](const std::string& msg, int code)
	{
		errorCode = code;
		std::cerr << msg << std::endl;
	};

	TheVolume.reset(loadVolumeFromRaw(filename, progress, logging, error, downsampling));
	return errorCode;
}

Volume* loadVolumeFromXYZ(const std::string& filename, const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging, const VolumeErrorCallback_t& error)
{
	std::ifstream in(filename, std::ifstream::in | std::ifstream::binary);
	unsigned int sizeX, sizeY, sizeZ;
	double voxelSizeX, voxelSizeY, voxelSizeZ;
	in.read(reinterpret_cast<char*>(&sizeX), sizeof(unsigned int));
	in.read(reinterpret_cast<char*>(&sizeY), sizeof(unsigned int));
	in.read(reinterpret_cast<char*>(&sizeZ), sizeof(unsigned int));
	in.read(reinterpret_cast<char*>(&voxelSizeX), sizeof(double));
	in.read(reinterpret_cast<char*>(&voxelSizeY), sizeof(double));
	in.read(reinterpret_cast<char*>(&voxelSizeZ), sizeof(double));
	unsigned int maxSize = std::max({ sizeX, sizeY, sizeZ });
	voxelSizeX = 1.0 / maxSize;
	voxelSizeY = 1.0 / maxSize;
	voxelSizeZ = 1.0 / maxSize;

	std::unique_ptr<Volume> vol = std::make_unique<Volume>(Volume::TypeFloat, sizeX, sizeY, sizeZ);
	vol->setWorldSizeX(voxelSizeX * sizeX);
	vol->setWorldSizeY(voxelSizeY * sizeY);
	vol->setWorldSizeZ(voxelSizeZ * sizeZ);
	Volume::MipmapLevel* level = vol->getLevel(0);
	float* volumeData = level->dataCpu<float>();

	size_t floatsToRead = sizeZ * sizeY;
	std::vector<float> data(floatsToRead);
	for (int x = 0; x < sizeX; ++x)
	{
		in.read(reinterpret_cast<char*>(&data[0]), sizeof(float)*floatsToRead);
		if (!in)
		{
			error("Loading data file failed", -7);
			return nullptr;
		}
		if (x % 10 == 0)
			progress(x / float(sizeX));

#pragma omp parallel for
		for (int y = 0; y < sizeY; ++y)
			for (int z = 0; z < sizeZ; ++z)
				volumeData[level->idx(x, y, z)] = data[z + sizeZ * y];
	}
	progress(1.0f);

	//done
	std::stringstream s;
	s << "Reading done, resolution=(" << sizeX <<
		"," << sizeY << "," << sizeZ <<
		"), size=(" << vol->worldSizeX() <<
		"," << vol->worldSizeY() << "," << vol->worldSizeZ() <<
		")" << std::endl;
	logging(s.str());

	return vol.release();
}


int64_t LoadVolumeFromXYZ(const std::string& filename)
{
	CloseVolume();

	VolumeProgressCallback_t progress = [](float v)
	{
		printProgress("Load", v);
	};
	VolumeLoggingCallback_t logging = [](const std::string& msg)
	{
		std::cout << msg << std::endl;
	};
	int errorCode = 1;
	VolumeErrorCallback_t error = [&errorCode](const std::string& msg, int code)
	{
		errorCode = code;
		std::cerr << msg << std::endl;
	};

	TheVolume.reset(loadVolumeFromXYZ(filename, progress, logging, error));
	return errorCode;
}

int64_t LoadVolumeFromBinary(const std::string& file)
{
	CloseVolume();

	VolumeProgressCallback_t progress = [](float v)
	{
		printProgress("Load", v);
	};
	VolumeLoggingCallback_t logging = [](const std::string& msg)
	{
		std::cout << msg << std::endl;
	};
	int errorCode = 1;
	VolumeErrorCallback_t error = [&errorCode](const std::string& msg, int code)
	{
		errorCode = code;
		std::cerr << msg << std::endl;
	};
	TheVolume = std::make_unique<Volume>(file, progress, logging, error);
	return errorCode;
}

std::tuple<int64_t, int64_t, int64_t> GetVolumeResolution()
{
	return std::make_tuple(
		static_cast<int64_t>(TheVolume->getLevel(0)->sizeX()),
		static_cast<int64_t>(TheVolume->getLevel(0)->sizeY()),
		static_cast<int64_t>(TheVolume->getLevel(0)->sizeZ()));
}

int64_t SaveVolumeToBinary(const std::string& file)
{
	VolumeProgressCallback_t progress = [](float v)
	{
		printProgress("Save", v);
	};
	VolumeLoggingCallback_t logging = [](const std::string& msg)
	{
		std::cout << msg << std::endl;
	};
	int errorCode = 1;
	VolumeErrorCallback_t error = [&errorCode](const std::string& msg, int code)
	{
		errorCode = code;
		std::cerr << msg << std::endl;
	};
	TheVolume->save(file, progress, logging, error);
	return errorCode;
}

int64_t CreateMipmapLevel(int64_t level, const std::string& filter)
{
	if (!TheVolume)
	{
		std::cerr << "No volume loaded!" << std::endl;
	}
	Volume::MipmapFilterMode mode;
	if (filter == "average")
		mode = Volume::MipmapFilterMode::AVERAGE;
	else if (filter == "halton")
		mode = Volume::MipmapFilterMode::HALTON;
	else
	{
		std::cerr << "Unrecognized filter mode \"" << filter <<
			"\", only \"average\" and \"halton\" supported!" << std::endl;
		return -1;
	}
	TheVolume->createMipmapLevel(level, mode);
	std::cout << "Mipmap level " << level << " created with mode \""
		<< filter << "\"" << std::endl;
	return 1;
}

int64_t CreateVolumeFromPytorch(const torch::Tensor& data)
{
	TORCH_CHECK(data.dim() == 3, "The input tensor must be three-dimensional");
	Volume::DataType type;
	if (data.scalar_type() == c10::ScalarType::Byte)
		type = Volume::TypeUChar;
	else if (data.scalar_type() == c10::ScalarType::Short)
		type = Volume::TypeUShort;
	else if (data.scalar_type() == c10::ScalarType::Float)
		type = Volume::TypeFloat;
	else
		C10_THROW_ERROR(Error, std::string("unsupported scalar type, expected Byte, Short or Float, but got ") + toString(data.scalar_type()));
	int64_t X = data.size(0);
	int64_t Y = data.size(1);
	int64_t Z = data.size(2);
	
	torch::Tensor data2 = data
		.contiguous(c10::MemoryFormat::Contiguous)
		.to(c10::kCPU);

	//create new volume
	TheVolume = std::make_unique<Volume>(type, X, Y, Z);
	memcpy(
		TheVolume->getLevel(0)->dataCpu<void>(),
		data2.data_ptr(),
		Volume::BytesPerType[type] * X * Y * Z);

	return 0;
}

torch::Tensor ConvertVolumeToPytorch()
{
	TORCH_CHECK(TheVolume, "No volume loaded");
	c10::ScalarType type;
	switch (TheVolume->type())
	{
	case Volume::TypeUChar:
		type = c10::ScalarType::Byte;
		break;
	case Volume::TypeUShort:
		type = c10::ScalarType::Short;
		break;
	case Volume::TypeFloat:
		type = c10::ScalarType::Float;
		break;
	default:
		C10_THROW_ERROR(Error, std::string("Unknown type ") + std::to_string(TheVolume->type()));
	}
	int64_t X = TheVolume->getLevel(0)->sizeX();
	int64_t Y = TheVolume->getLevel(0)->sizeY();
	int64_t Z = TheVolume->getLevel(0)->sizeZ();
	void* data = TheVolume->getLevel(0)->dataCpu<void>();

	auto t = torch::from_blob(
		data, 
		at::IntArrayRef{ X, Y, Z }, 
		at::TensorOptions().dtype(type).device(c10::kCPU));
	return t.clone(); //return a clone as to not modify the original data
}

void ConvertVolumeToDtype(const std::string& dtype)
{
	Volume::DataType newType;
	if (dtype == "uchar")
		newType == Volume::TypeUChar;
	else if (dtype == "ushort")
		newType == Volume::TypeUShort;
	else if (dtype == "float")
		newType == Volume::TypeFloat;
	else
		C10_THROW_ERROR(Error, "Unknown dtype " + dtype);

	size_t X = TheVolume->getLevel(0)->sizeX();
	size_t Y = TheVolume->getLevel(0)->sizeY();
	size_t Z = TheVolume->getLevel(0)->sizeZ();

	auto newVolume = std::make_unique<Volume>(newType, X, Y, Z);
	const auto levelIn = TheVolume->getLevel(0);
	auto levelOut = newVolume->getLevel(0);
	for (int z=0; z<Z; ++z)
	{
		if (z % 10 == 0)
			printProgress("Convert", z / float(Z));
#pragma omp parallel for
		for (int y=0; y<Y; ++y)
			for (int x=0; x<X; ++x)
			{
				//load
				float value = 0;
				if (TheVolume->type() == Volume::TypeUChar)
					value = levelIn->dataCpu<unsigned char>()[levelIn->idx(x, y, z)] / 255.0f;
				else if (TheVolume->type() == Volume::TypeUShort)
					value = levelIn->dataCpu<unsigned short>()[levelIn->idx(x, y, z)] / 65535.0f;
				else if (TheVolume->type() == Volume::TypeFloat)
					value = levelIn->dataCpu<float>()[levelIn->idx(x, y, z)];
				//save
				if (newType == Volume::TypeUChar)
					levelOut->dataCpu<unsigned char>()[levelOut->idx(x, y, z)] = static_cast<unsigned char>(
						std::max(0, std::min(255, static_cast<int>(value * 255))));
				else if (newType == Volume::TypeUChar)
					levelOut->dataCpu<unsigned char>()[levelOut->idx(x, y, z)] = static_cast<unsigned short>(
						std::max(0, std::min(255, static_cast<int>(value * 65535.0f))));
				else if (newType == Volume::TypeFloat)
					levelOut->dataCpu<float>()[levelOut->idx(x, y, z)] = value;
			}
	}
	TheVolume.swap(newVolume);
	std::cout << std::endl << "Done" << std::endl;
}

MY_API std::vector<double> GetHistogram()
{
	TheVolume->getLevel(0)->copyCpuToGpu();
	auto histogram = TheVolume->extractHistogram();

	std::vector<double> ret(2 + std::size(histogram.bins));
	ret[0] = histogram.minDensity;
	ret[1] = histogram.maxDensity;
	std::copy(std::begin(histogram.bins), std::end(histogram.bins), ret.begin() + 2);
	
	return ret;
}

END_RENDERER_NAMESPACE
#endif
