#pragma once

#include <torch/types.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>

#include "commons.h"

#ifdef RENDERER_HAS_RENDERER
BEGIN_RENDERER_NAMESPACE

typedef std::function<void(const std::string&)> VolumeLoggingCallback_t;
typedef std::function<void(float)> VolumeProgressCallback_t;
typedef std::function<void(const std::string&, int)> VolumeErrorCallback_t;

class MY_API Volume
{
public:
	template<int numOfBins>
	struct VolumeHistogram
	{
		float bins[numOfBins]{ 0.0f };
		float minDensity{ FLT_MAX };
		float maxDensity{ 0.0f };
		float maxFractionVal{ 1.0f };
		unsigned int numOfNonzeroVoxels{ 0 };

		constexpr int getNumOfBins() { return numOfBins; }
	};
	using Histogram = VolumeHistogram<512>;

public:
	enum DataType
	{
		TypeUChar,
		TypeUShort,
		TypeFloat,
		_TypeCount_
	};
	static const int BytesPerType[_TypeCount_];

	class MY_API MipmapLevel
	{
	private:
		size_t sizeX_, sizeY_, sizeZ_;
		char* dataCpu_;
		cudaArray_t dataGpu_;
		cudaTextureObject_t dataTex_;
		int cpuDataCounter_;
		int gpuDataCounter_;

		Volume* parent_;
		friend class Volume;

		MipmapLevel(Volume* parent, size_t sizeX, size_t sizeY, size_t sizeZ);
	public:
		~MipmapLevel();

		MipmapLevel(const MipmapLevel& other) = delete;
		MipmapLevel(MipmapLevel&& other) noexcept = delete;
		MipmapLevel& operator=(const MipmapLevel& other) = delete;
		MipmapLevel& operator=(MipmapLevel&& other) noexcept = delete;

		size_t sizeX() const { return sizeX_; }
		size_t sizeY() const { return sizeY_; }
		size_t sizeZ() const { return sizeZ_; }

		size_t idx(int x, int y, int z) const
		{
			assert(x >= 0 && x < sizeX_);
			assert(y >= 0 && y < sizeY_);
			assert(z >= 0 && z < sizeZ_);
			return x + sizeX_ * (y + sizeY_ * z);
		}

		template<typename T>
		const T* dataCpu() const { return reinterpret_cast<T*>(dataCpu_); }
		template<typename T>
		T* dataCpu() { cpuDataCounter_++; return reinterpret_cast<T*>(dataCpu_); }
		cudaArray_const_t dataGpu() const { return dataGpu_; }
		/**
		 * Returns the data as a 3d texture with un-normalized coordinates
		 * and linear interpolation.
		 */
		cudaTextureObject_t dataTexGpu() const { return dataTex_; }

		void copyCpuToGpu();
	};
	
private:
	
	double worldSizeX_, worldSizeY_, worldSizeZ_;
	DataType type_;
	std::vector<std::unique_ptr<MipmapLevel>> levels_;
	
public:
	Volume();
	Volume(DataType type, size_t sizeX, size_t sizeY, size_t sizeZ);
	~Volume();

	Volume(const Volume& other) = delete;
	Volume(Volume&& other) noexcept = delete;
	Volume& operator=(const Volume& other) = delete;
	Volume& operator=(Volume&& other) noexcept = delete;

	/**
	 * Saves the volume to the file
	 */
	void save(const std::string& filename,
		const VolumeProgressCallback_t& progress,
		const VolumeLoggingCallback_t& logging,
		const VolumeErrorCallback_t& error) const;
	/**
	 * Loads and construct the volume
	 */
	Volume(const std::string& filename,
		const VolumeProgressCallback_t& progress,
		const VolumeLoggingCallback_t& logging,
		const VolumeErrorCallback_t& error);

	/**
	 * Creates the histogram of the volume.
	 */
	Volume::Histogram extractHistogram() const;

	double worldSizeX() const { return worldSizeX_; }
	double worldSizeY() const { return worldSizeY_; }
	double worldSizeZ() const { return worldSizeZ_; }
	void setWorldSizeX(double s) { worldSizeX_ = s; }
	void setWorldSizeY(double s) { worldSizeY_ = s; }
	void setWorldSizeZ(double s) { worldSizeZ_ = s; }
	DataType type() const { return type_; }

	enum class MipmapFilterMode
	{
		/**
		 * Average filtering
		 */
		AVERAGE,
		/**
		 * A random sample is taken
		 */
		HALTON
	};
	
	/**
	 * \brief Creates the mipmap level specified by the given index.
	 * The level zero is always the original data.
	 * Level 1 is 2x downsampling, level 2 is 3x downsampling, level n is (n+1)x downsampling.
	 * This function does nothing if that level is already created.
	 * \param level the mipmap level
	 * \param filter the filter mode
	 */
	void createMipmapLevel(int level, MipmapFilterMode filter);

private:
	bool mipmapCheckOrCreate(int level);
	void createMipmapLevelAverage(int level);
	void createMipmapLevelHalton(int level);

public:
	/**
	 * \brief Deletes all mipmap levels.
	 */
	void deleteAllMipmapLevels();

	/**
	 * \brief Returns the mipmap level specified by the given index.
	 * The level zero is always the original data.
	 * If the level is not created yet, <code>nullptr</code> is returned.
	 * Level 1 is 2x downsampling, level 2 is 3x downsampling, level n is (n+1)x downsampling
	 */
	const MipmapLevel* getLevel(int level) const;
	/**
	 * \brief Returns the mipmap level specified by the given index.
	 * The level zero is always the original data.
	 * If the level is not created yet, <code>nullptr</code> is returned.
	 * Level 1 is 2x downsampling, level 2 is 3x downsampling, level n is (n+1)x downsampling
	 */
	MipmapLevel* getLevel(int level);
};

MY_API extern std::unique_ptr<Volume> TheVolume;

/**
 * Closes the volume and releases the memory
 */
MY_API void CloseVolume();

/**
 * Loads the volume from a raw file.
 * The specified file points to the .dat file specifying the format.
 */
MY_API Volume* loadVolumeFromRaw(const std::string& file,
	const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging,
	const VolumeErrorCallback_t& error,
	int64_t downsampling = 1);
/**
 * Loads the volume from a raw file.
 * The specified file points to the .dat file specifying the format
 */
MY_API int64_t LoadVolumeFromRaw(const std::string& file, int64_t downsampling = 1);

/**
 * Loads the volume from an .xyz file
 * Returns 1 on success, a negative value on failure
 */
MY_API Volume* loadVolumeFromXYZ(const std::string& file,
	const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging,
	const VolumeErrorCallback_t& error);
/**
 * Loads the volume from an .xyz file
 * Returns 1 on success, a negative value on failure
 */
MY_API int64_t LoadVolumeFromXYZ(const std::string& file);

/**
 * Loads the volume from a custom binary format
 * Returns 1 on success, a negative value on failure
 */
MY_API int64_t LoadVolumeFromBinary(const std::string& file);

MY_API std::tuple<int64_t, int64_t, int64_t> GetVolumeResolution();

/**
 * Saves the volume to a custom binary format
 * Returns 1 on success, a negative value on failure
 */
MY_API int64_t SaveVolumeToBinary(const std::string& file);

/**
 * Python API to Volume::createMipmapLevel(int, Volume::MipmapFilterMode)
 * \param level the mipmap level
 * \param filter the filter mode, can be "average" or "halton"
 * Returns 1 on success, a negative value on failure
 */
MY_API int64_t CreateMipmapLevel(int64_t level, const std::string& filter);

/**
 * Creates a new volume from the given pytorch tensor.
 * The tensor must be a 3D tensor of type uchar, ushort or float.
 * The dimension will be set as the size of the volume (X,Y,Z).
 * The bounding box size is the unit cube.
 */
MY_API int64_t CreateVolumeFromPytorch(const torch::Tensor& data);

/**
 * Returns the data of the current volume as a pytorch tensor.
 * The returned tensor is of type uchar, ushort or float.
 */
MY_API torch::Tensor ConvertVolumeToPytorch();

/**
 * Converts the current volume to a new dtype.
 * dtype can be 'uchar', 'ushort' or 'float'
 */
MY_API void ConvertVolumeToDtype(const std::string& dtype);

/**
 * Extracts the histogram of the volume and saves it to a file.
 */
MY_API std::vector<double> GetHistogram();

END_RENDERER_NAMESPACE
#endif
