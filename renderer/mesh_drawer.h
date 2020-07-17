#pragma once

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <vector>
#include <functional>

#include "commons.h"

#ifdef RENDERER_HAS_MESH_DRAWER
BEGIN_RENDERER_NAMESPACE

/**
 * \brief If the mesh drawer is used stand-alone within PyTorch,
 * an offscreen context has to be created with this function.
 * It is save to call this function multiple times, only the first time
 * has an effect.
 * 
 * If used in the GUI, the mesh drawer uses the current context from that GUI
 * and this function is not needed.
 */
MY_API int64_t CreateOffscreenContext();

/**
 * \brief Draws a triangular mesh in screen space
 * into two textures.
 * The vertices are defined by a buffer of six floats:
 *  2x position, 4x color.
 * The vertex buffer can be written to from CUDA.
 * The index buffer is fixed.
 *
 * Pipeline:
 * 1. modifyVertexBuffer(...)
 * 2. draw(...)
 * 3. grab
 */
class MY_API MeshDrawer
{
public:
	/**
	 * \brief Creates the mesh drawer
	 * \param numVertices the number of vertices, data filled by
	 *	\ref modifyVertexBuffer() afterwards
	 * \param triangles the triangle indices as a
	 *	M*3 32-bit integer tensor on the CPU.
	 */
	MeshDrawer(GLuint numVertices, const torch::Tensor& triangles);
	~MeshDrawer();

	GLuint getNumVertices() const { return numVertices_; }
	
	/**
	 * \brief Modifies the vertex buffer.
	 * It binds the buffer to CUDA, calls the function on the buffer
	 * and unbinds the buffer.
	 * The number of vertices is given by \ref getNumVertices()
	 * \param f the function that modifies the vertex buffer.
	 *  Parameters: position (2x float), color/data (8x float)
	 */
	void modifyVertexBuffer(const std::function<void(float*, float*)>& f);

	/**
	 * Modifies the vertex buffers directly from PyTorch tensors.
	 * \param samplePositions 2*N float tensor on the GPU
	 * \param sampleData 8*N float tensor on the GPU
	 */
	void modifyVertexBuffer(
		const torch::Tensor& samplePositions,
		const torch::Tensor& sampleData);
	
	/**
	 * \brief Draws the triangles.
	 * The vertex position 'x' is transformed by
	 * $ x*scale + offset $.
	 * Remember that OpenGL's viewport is in [-1,+1]^2.
	 * \param scale 
	 * \param offset
	 * \param size the size (width, height) of the framebuffer
	 * \param points true: only points are drawn, false: triangles
	 * \param defaultValues: 8-element vector with the clear color for the texture
	 * or empty to clear with black
	 */
	void draw(float2 scale, float2 offset, int2 size, bool points,
		const std::vector<double>& defaultValues = {});

	/**
	 * \brief Copies the drawn points/triangles back to PyTorch.
	 * \result the output tensor of shape 8*H*W where
	 *  H, W are the sizes used in the previous draw call
	 */
	torch::Tensor grabResult();

private:
	GLuint numVertices_;
	GLuint numTriangles_;

	GLuint outputFBO_;
	GLuint outputTexture_[2];
	cudaGraphicsResource_t outputTextureRes_[2];
	int2 size_;
	
	GLuint vertexArray_[2];
	GLuint vertexVAO_;
	GLuint indexArray_;
	cudaGraphicsResource_t vertexRes_[2];

	void deleteFBO();
	void resize(int2 size);
};

//PyTorch API


/**
 * \brief Creates the Mesh Drawer to be used from PyTorch Python-API
 * \param numVertices the number of vertices (to be specified later)
 * \param triangles the triangle indices as a M*3 32-bit integer tensor on the CPU.
 */
MY_API int64_t CreateMeshDrawer(int64_t numVertices, torch::Tensor triangles);

/**
 * \brief Renders the mesh from PyTorch's Python-API.
 *
 * The vertex position 'x' is transformed by
 * $ x*scale + offset $.
 * Remember that OpenGL's viewport is in [-1,+1]^2.
 * 
 * \param samplePositions the positions of the samples as a 2*N float tensor on the GPU
 * \param sampleData the data of the samples as a 8*N float tensor on the GPU
 * \param scaleX scales the positions, see description
 * \param scaleY scales the positions, see description
 * \param offsetX moves the positions, see description
 * \param offsetY moves the positions, see description
 * \param sizeX the size of the output texture
 * \param sizeY the size of the output texture
 * \param points true: points are rendered, false: triangles
 * \param default_values 8-element vector with the clear color
 * \return a float tensor of shape 8*sizeY*sizeX with the rendered output
 */
MY_API torch::Tensor RenderMesh(
	const torch::Tensor& samplePositions,
	const torch::Tensor& sampleData,
	double scaleX, double scaleY,
	double offsetX, double offsetY,
	int64_t sizeX, int64_t sizeY,
	bool points,
	std::vector<double> default_values);

MY_API extern std::unique_ptr<MeshDrawer> TheMeshDrawer;

END_RENDERER_NAMESPACE
#endif
