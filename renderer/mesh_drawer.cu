#include "mesh_drawer.h"

#include <memory>
#include <stdexcept>
#include <cuda_gl_interop.h>
#include <cuMat/src/Errors.h>
#include <cuMat/src/Context.h>
#include <GLFW/glfw3.h>

#ifdef RENDERER_HAS_MESH_DRAWER
BEGIN_RENDERER_NAMESPACE

namespace
{
	class MeshDrawerStatic
	{
	public:
		GLuint program_ = 0;
		GLuint scaleLoc_ = 0;
		GLuint offsetLoc_ = 0;

		MeshDrawerStatic()
		{
			static const char* VERTEX_SHADER = R"shader(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec4 aColor0;
layout (location = 2) in vec4 aColor1;

out vec4 Color0;
out vec4 Color1;

uniform vec2 scale;
uniform vec2 offset;

void main()
{
    Color0 = aColor0;
	Color1 = aColor1;
    gl_Position = vec4(aPos * scale + offset, 0.0, 1.0);
}
)shader";

			static const char* FRAGMENT_SHADER = R"shader(
#version 330 core
layout(location=0) out vec4 FragColor0;
layout(location=1) out vec4 FragColor1;

in vec4 Color0;
in vec4 Color1;

void main()
{             
    FragColor0 = Color0;
	FragColor1 = Color1;
}
)shader";

			// compile shaders
			GLuint vertex, fragment;
			int success;
			char infoLog[512];

			// vertex Shader
			vertex = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertex, 1, &VERTEX_SHADER, NULL);
			glCompileShader(vertex);
			glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(vertex, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
				return;
			}

			// fragment Shader
			fragment = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragment, 1, &FRAGMENT_SHADER, NULL);
			glCompileShader(fragment);
			glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(fragment, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
				return;
			}

			program_ = glCreateProgram();
			glAttachShader(program_, vertex);
			glAttachShader(program_, fragment);
			glLinkProgram(program_);
			// print linking errors if any
			glGetProgramiv(program_, GL_LINK_STATUS, &success);
			if (!success)
			{
				glGetProgramInfoLog(program_, 512, NULL, infoLog);
				std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
				return;
			}

			//get uniforms
			scaleLoc_ = glGetUniformLocation(program_, "scale");
			offsetLoc_ = glGetUniformLocation(program_, "offset");

			glDeleteShader(vertex);
			glDeleteShader(fragment);
		}

		~MeshDrawerStatic()
		{
			glDeleteProgram(program_);
		}
	};
	std::unique_ptr<MeshDrawerStatic> TheMeshDrawerStatic;
	
	struct DestroyOffscreenContextFunctor
	{
		void operator()(GLFWwindow* p)
		{
			TheMeshDrawerStatic = nullptr;
			if (p)
			{
				glfwDestroyWindow(p);
				glfwTerminate();
			}
		}
	};
	
	std::unique_ptr<GLFWwindow, DestroyOffscreenContextFunctor> offscreen_context;
}

static void glfw_error_callback(int error, const char* description)
{
	std::string msg = std::string("GLFW Error ") +
		std::to_string(error) + ": " + description;
	throw std::runtime_error(msg.c_str());
}

int64_t CreateOffscreenContext()
{
	if (offscreen_context)
		return 0; //already initialized
	
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		throw std::runtime_error("Unable to initialize GLFW");
	
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
	glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
	GLFWwindow* ctx = glfwCreateWindow(640, 480, "", NULL, NULL);
	if (ctx == nullptr)
		throw std::runtime_error("Unable to create offscreen context");
	
	offscreen_context.reset(ctx);
	glfwMakeContextCurrent(offscreen_context.get());

	bool err = glewInit() != GLEW_OK;
	if (err)
		throw std::runtime_error("Unable to initialize GLEW");
	return 0;
}

MeshDrawer::MeshDrawer(GLuint numVertices, const torch::Tensor& triangles)
	: numVertices_(numVertices)
	, numTriangles_(triangles.size(0))
	, outputFBO_(0)
	, outputTexture_{0,0}
	, size_{0,0}
	, vertexArray_{0,0}
	, vertexVAO_(0)
	, indexArray_(0)
	, vertexRes_{nullptr, nullptr}
{
	//create shader
	if (!TheMeshDrawerStatic)
		TheMeshDrawerStatic = std::make_unique<MeshDrawerStatic>();
	
	//create vertex buffer
	glGenVertexArrays(1, &vertexVAO_);
	glGenBuffers(2, vertexArray_);
	glBindVertexArray(vertexVAO_);
	glBindBuffer(GL_ARRAY_BUFFER, vertexArray_[0]);
	GLsizeiptr size = sizeof(float) * 2 * numVertices;
	glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexArray_[1]);
	size = sizeof(float) * 8 * numVertices;
	glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(4 * sizeof(float)));
	glBindVertexArray(0);

	AT_ASSERTM(triangles.dim() == 2, "triangles must be an M*3 tensor");
	int M = triangles.size(0);
	AT_ASSERTM(triangles.size(1) == 3, "triangles must be an M*3 tensor");
	AT_ASSERTM(triangles.is_contiguous(), "triangles must be contiguous");
	AT_ASSERTM((triangles.dtype() == at::kInt), "triangles must be an integer tensor");
	AT_ASSERTM(!triangles.type().is_cuda(), "triangles must be a CPU tensor");

	//create index buffer
	glGenBuffers(1, &indexArray_);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexArray_);
	size = sizeof(int) * 3 * M;
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, triangles.data<int>(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	//register vertex buffer with CUDA
	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&vertexRes_[0], vertexArray_[0], cudaGraphicsMapFlagsWriteDiscard));
	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&vertexRes_[1], vertexArray_[1], cudaGraphicsMapFlagsWriteDiscard));
}

MeshDrawer::~MeshDrawer()
{
	CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(vertexRes_[0]));
	CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(vertexRes_[1]));
	glDeleteVertexArrays(1, &vertexVAO_);
	glDeleteBuffers(2, vertexArray_);
	glDeleteBuffers(1, &indexArray_);
	deleteFBO();
}

void MeshDrawer::modifyVertexBuffer(const std::function<void(float*, float*)>& f)
{
	float* ptr1;
	float* ptr2;
	size_t num_bytes;
	
	CUMAT_SAFE_CALL(cudaGraphicsMapResources(2, vertexRes_, 0));
	CUMAT_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&ptr1, &num_bytes, vertexRes_[0]));
	CUMAT_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&ptr2, &num_bytes, vertexRes_[1]));
	
	f(ptr1, ptr2);
	
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(2, vertexRes_, 0));
}

void MeshDrawer::modifyVertexBuffer(const torch::Tensor& samplePositions, const torch::Tensor& sampleData)
{
	torch::Tensor p = samplePositions.t().contiguous();
	torch::Tensor d = sampleData.t().contiguous();
	int64_t N = samplePositions.size(1);
	TORCH_CHECK(N == numVertices_,
		"Number of entries in samplePositions must match numVertices set in the constructor, ",
		"expected ", numVertices_, " but got ", N);
	TORCH_CHECK(sampleData.size(1) == N,
		"sampleData.size(1) does not match in size to samplePositions, ",
		"expected ", N, " but got ", sampleData.size(1));
	TORCH_CHECK(samplePositions.size(0) == 2, 
		"samplePositions must have two entries in the first dimension, ",
		"but got ", samplePositions.size(0));
	TORCH_CHECK(sampleData.size(0) == 8, 
		"sampleData must have eight entries in the first dimension, ",
		"but got ", sampleData.size(0));
	TORCH_CHECK(sampleData.type().is_cuda(), "sampleData must be a CUDA tensor");
	TORCH_CHECK(samplePositions.type().is_cuda(), "samplePositions must be a CUDA tensor");
	modifyVertexBuffer([p, d, N](float* pp, float* dp)
	{
		cudaMemcpy(pp, p.data<float>(), sizeof(float) * 2 * N, cudaMemcpyDeviceToDevice);
		cudaMemcpy(dp, d.data<float>(), sizeof(float) * 8 * N, cudaMemcpyDeviceToDevice);
	});
}

void MeshDrawer::draw(float2 scale, float2 offset, int2 size, bool points,
	const std::vector<double>& defaultValues)
{
	resize(size);
	
	glUseProgram(TheMeshDrawerStatic->program_);
	glUniform2f(TheMeshDrawerStatic->scaleLoc_, scale.x, scale.y);
	glUniform2f(TheMeshDrawerStatic->offsetLoc_, offset.x, offset.y);

	glBindFramebuffer(GL_FRAMEBUFFER, outputFBO_);

	GLint oldViewport[4];
	glGetIntegerv(GL_VIEWPORT, oldViewport);
	
	glViewport(0, 0, size_.x, size_.y);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	if (defaultValues.empty())
	{
		static const float transparent[] = { 0, 0, 0, 0 };
		glClearBufferfv(GL_COLOR, 0, transparent);
		glClearBufferfv(GL_COLOR, 1, transparent);
	} else
	{
		AT_ASSERTM(defaultValues.size() == 8, "Default values must be of length 8 or empty");
		const float col1[] = { defaultValues[0], defaultValues[1], defaultValues[2], defaultValues[3] };
		const float col2[] = { defaultValues[4], defaultValues[5], defaultValues[6], defaultValues[7] };
		glClearBufferfv(GL_COLOR, 0, col1);
		glClearBufferfv(GL_COLOR, 1, col2);
	}
	//glClear(GL_COLOR_BUFFER_BIT);

	if (points)
	{
		glBindVertexArray(vertexVAO_);
		glDrawArrays(GL_POINTS, 0, numVertices_);
		glBindVertexArray(0);
	} else
	{
		glBindVertexArray(vertexVAO_);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexArray_);
		glDrawElements(GL_TRIANGLES, 3 * numTriangles_, GL_UNSIGNED_INT, (GLvoid*)0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	glViewport(oldViewport[0], oldViewport[1], oldViewport[2], oldViewport[3]);
	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

texture<float4, cudaTextureType2D, cudaReadModeElementType> tex1;
texture<float4, cudaTextureType2D, cudaReadModeElementType> tex2;
__global__ void CopyTexturesToPyTorch(dim3 virtual_size,
	torch::PackedTensorAccessor<float, 3, torch::RestrictPtrTraits, size_t> output)
{
	CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)
		float4 col1 = tex2D(tex1, x, y);
		float4 col2 = tex2D(tex2, x, y);
		output[0][y][x] = col1.x;
		output[1][y][x] = col1.y;
		output[2][y][x] = col1.z;
		output[3][y][x] = col1.w;
		output[4][y][x] = col2.x;
		output[5][y][x] = col2.y;
		output[6][y][x] = col2.z;
		output[7][y][x] = col2.w;
	CUMAT_KERNEL_2D_LOOP_END
}

torch::Tensor MeshDrawer::grabResult()
{
	//allocate output
	torch::Tensor output = torch::empty({ 8, size_.y, size_.x }, 
		at::dtype(at::kFloat).device(at::kCUDA));

	//bind textures to CUDA
	cudaArray_t array[2];
	CUMAT_SAFE_CALL(cudaGraphicsMapResources(2, outputTextureRes_));
	CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array[0], outputTextureRes_[0], 0, 0));
	CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array[1], outputTextureRes_[1], 0, 0));
	CUMAT_SAFE_CALL(cudaBindTextureToArray(tex1, (cudaArray_t)array[0]));
	CUMAT_SAFE_CALL(cudaBindTextureToArray(tex2, (cudaArray_t)array[1]));
	tex1.filterMode = cudaFilterModePoint;
	tex2.filterMode = cudaFilterModePoint;

	//call kernel
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(size_.x, size_.y, CopyTexturesToPyTorch);
	CopyTexturesToPyTorch
		<<<cfg.block_count, cfg.thread_per_block >>>
		(cfg.virtual_size, output.packed_accessor<float, 3, torch::RestrictPtrTraits, size_t>());
	CUMAT_CHECK_ERROR();
	
	//unbind textures
	CUMAT_SAFE_CALL(cudaUnbindTexture(tex1));
	CUMAT_SAFE_CALL(cudaUnbindTexture(tex2));
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(2, outputTextureRes_));

	return output;
}

void MeshDrawer::deleteFBO()
{
	if (outputFBO_)
	{
		glDeleteFramebuffers(1, &outputFBO_);
		outputFBO_ = 0;
	}
	if (outputTexture_[0])
	{
		CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(outputTextureRes_[0]));
		CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(outputTextureRes_[1]));
		glDeleteTextures(2, outputTexture_);
		outputTexture_[0] = 0;
		outputTexture_[1] = 0;
	}
}

void MeshDrawer::resize(int2 size)
{
	if (outputFBO_ != 0 && size_.x == size.x && size_.y == size.y)
		return; //nothing to do
	//std::cout << "resize framebuffer to " << size.x << ", " << size.y << std::endl;

	deleteFBO();

	glGenFramebuffers(1, &outputFBO_);
	glGenTextures(2, outputTexture_);
	for (int i=0; i<2; ++i)
	{
		glBindTexture(GL_TEXTURE_2D, outputTexture_[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		cudaGraphicsGLRegisterImage(&outputTextureRes_[i], outputTexture_[i], GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, outputFBO_);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, outputTexture_[0], 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, outputTexture_[1], 0);
	GLenum DrawBuffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, DrawBuffers);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		throw std::runtime_error("unknown error occured in creating the framebuffer");
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	size_ = size;
}

std::unique_ptr<MeshDrawer> TheMeshDrawer;

int64_t CreateMeshDrawer(int64_t numVertices, torch::Tensor triangles)
{	
	TheMeshDrawer = std::make_unique<MeshDrawer>(numVertices, triangles);
	return 0;
}

torch::Tensor RenderMesh(
	const torch::Tensor& samplePositions, const torch::Tensor& sampleData, 
	double scaleX, double scaleY, double offsetX, double offsetY, 
	int64_t sizeX, int64_t sizeY, bool points, std::vector<double> default_values)
{
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	TheMeshDrawer->modifyVertexBuffer(samplePositions, sampleData);
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	TheMeshDrawer->draw(
		make_float2(scaleX, scaleY), make_float2(offsetX, offsetY),
		make_int2(sizeX, sizeY), points, default_values);
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	return TheMeshDrawer->grabResult();
}

END_RENDERER_NAMESPACE
#endif
