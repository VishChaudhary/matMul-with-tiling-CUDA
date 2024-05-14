#define TILE_WIDTH 16

__global__ void matrixMulKernel (float* M, float * N, float* P, int width){
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x;  int ty = treadIdx.y;

  // Identify the row and column of the P element to work on
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  //Loop over the M and N tiles required to compute P element
  float Pvalue = 0;

  for( int ph = 0; ph<(width/(float)TILE_WIDTH); ++ph){
    if ((row < width) && (ph*TILE_WIDTH) < width){
      Mds[ty][tx] = M[row*width + ph*TILE_WIDTH +tx];
    }
    else {
      Mds[ty][tx] = 0.0f;
    }

    if ((ph*TILE_WIDTH+ty) < width && col < width){
      Nds[ty][tx] = N[(ph*TILE_WIDTH +ty)*width + col];
    }
    else Nds[ty][tx] = 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k){
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  if((row<width) && (col<width)){
    P[row*width+col] = Pvalue;
  }
}
