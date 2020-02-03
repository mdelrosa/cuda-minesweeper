#ifndef MINESWEEPER_CU
#define MINESWEEPER_CU

#include <nvfunctional>
#include "minesweeperUtilsGPU.cuh"
#include <stdio.h>
using namespace std;

// define constants as per minesweeperUtilsCPU.h
#define MINE -1

#define COVERED -51
#define FLAG    -1
#define DUMMY    100

#define MISSED_MINE     -101
#define CLICKED_MINE    -102
#define WRONG_FLAG      -103

# define DEBUG_FLAG true

# define BLOCK_SIZE 40
# define BLOCK_EDGE 10

// Functions to be pointed to
__device__ float Plus (float a, float b) {return a+b;}

__device__
inline bool isOutOfRange(int x, int y, int width, int height) {
    if (x < 0 || y < 0 || x >= width || y >= height)
        return true;
    else
        return false;
}

__device__
void foreachNeighbor(int width, int height, int x, int y, nvstd::function<void (int,int, int)> func) {
    for (int ny = y - 1; ny <= y + 1; ny++) {
        for (int nx = x - 1; nx <= x + 1; nx++) {
            if (isOutOfRange(nx, ny, width, height) || (nx == x && ny == y))
                continue;
            func(nx, ny, ny * width + nx);
        }
    }
}

__device__
void foreachNeighborExt(int width, int height, int x, int y, nvstd::function<void (int,int, int)> func) {
    for (int ny = y - 1; ny <= y + 1; ny++) {
        for (int nx = x - 1; nx <= x + 1; nx++) {
            if (nx == x && ny == y)
	      continue; 
            func(nx, ny, ny * width + nx);
        }
    }
}

__device__
void getNeighborInfo(int* board, int width, int height, int x, int y, int* numFlagged, int* numCovered) {
    int f = 0;
    int c = 0;
    foreachNeighbor(width, height, x, y, [=, &f, &c](int nx, int ny, int neighborIdx) {
        if (board[neighborIdx] == FLAG)
            f += 1;
        else if (board[neighborIdx] == COVERED)
            c += 1;
    });
    *numFlagged = f;
    *numCovered = c;
}

// gpu code for printing board
__device__
void printBoardGPU(int* board, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (board[idx] == COVERED)
                printf(". ");
	    else if (board[idx] == DUMMY) 
		printf("? ");
            else if (board[idx] == FLAG)
                printf("F ");
            else if (board[idx] == WRONG_FLAG)
                printf("W ");
            else if (board[idx] == MINE || board[idx] == MISSED_MINE)
                printf("* ");
            else if (board[idx] == CLICKED_MINE)
                printf("& ");
            else
                printf("%d ", board[idx]);
        }
        printf("\n");
    }
    printf("\n");
}

// CPU function for printing board
void printBoard(int* board, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (board[idx] == COVERED)
                printf(". ");
            else if (board[idx] == FLAG)
                printf("F ");
            else if (board[idx] == WRONG_FLAG)
                printf("W ");
            else if (board[idx] == MINE || board[idx] == MISSED_MINE)
                printf("* ");
            else if (board[idx] == CLICKED_MINE)
                printf("& ");
            else
                printf("%d ", board[idx]);
        }
        printf("\n");
    }
    printf("\n");
}

// push tiles to process onto stack
__device__
int uncoverTileImpl(int* board, int width, int height, int stackPos, int* tileStackX, int* tileStackY) {
    int x = tileStackX[stackPos];
    int y = tileStackY[stackPos];
    int idx = y * width + x;
    // printf("Before clicking on (%d, %d) value is %d\n", x, y, board[idx]);
    int value = clickTile(x, y);
    // printf("Clicked on (%d, %d) with value %d at idx=%d\n", x, y, value, idx);
    board[idx] = value;

    if (value != 0)
        return stackPos;

    for (int ny = y - 1; ny <= y + 1; ny++) {
        for (int nx = x - 1; nx <= x + 1; nx++) {
            if (isOutOfRange(nx, ny, width, height) || (nx == x && ny == y))
                continue;

            int neighborIdx = ny * width + nx;
            if (board[neighborIdx] == COVERED) {
                // Temporarily store a nonsense value here, which will get replace when the queue is processed
                board[neighborIdx] = DUMMY;
                // tilesToUncover->push(nx, ny);
                tileStackX[stackPos] = nx;
                tileStackY[stackPos] = ny;
 		// printf("Push (%d, %d) with depth %d\n", nx, ny, stackPos);
		stackPos += 1;
            }
        }
    }
    return stackPos;
}

// init stack and process vals until stack exhausted
__device__
void uncoverTile(int* board, int width, int height, int x, int y, int* tileStackX, int* tileStackY) {
    // TileStack tilesToUncover;
    // tilesToUncover.push(x, y);
    tileStackX[0] = x;
    tileStackY[0] = y;
    int stackPos = 0;

    // while (!tilesToUncover.empty()) {
        // auto tile = tilesToUncover.pop();
         
    while (stackPos != -1) {
        stackPos = uncoverTileImpl(board, width, height, stackPos, tileStackX, tileStackY);
	stackPos -= 1; // get latest value off stack
 	// printf("Pop to depth %d\n", stackPos);
    }
}

// first click kernel 
__global__
void firstClick(int startX, int startY, int width, int height, int *output, int* tileStackX, int* tileStackY)
{
  int i = startY * width + startX;
  // output[i] = clickTile(startX, startY);
  uncoverTile(output, width, height, startX, startY, tileStackX, tileStackY);
  if (DEBUG_FLAG)
    printf("startX: %d, startY: %d, startValue: %d\n", startX, startY, output[i]);
}

// first click kernel for global mem implementation
__global__
void firstClickGlobal(int startX, int startY, int width, int height, int *output, int* tileStackX, int* tileStackY)
{
  uncoverTile(output, width, height, startX, startY, tileStackX, tileStackY);
  int value = clickTile(startX, startY);
  output[startY*width+startX] = value;
  if (DEBUG_FLAG)
    printf("startX: %d, startY: %d, startValue: %d\n", startX, startY, value);
}

__device__ 
bool performIteration(int *board, int width, int height, int* tileStackX, int* tileStackY) {
  bool moveMade = false;
  for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
          int idx = y * width + x;
          int value = board[idx];
          if (value == COVERED || value == FLAG || value==0) { 
	      if (DEBUG_FLAG) {
	        printf("Tile (%d, %d): Skipping\n", x, y);
	      }
              continue;
	  }

          int numFlagged, numCovered;
          getNeighborInfo(board, width, height, x, y, &numFlagged, &numCovered);

          // If this tile has all of it's mines identified,
          // then uncover it's remaining covered neighbors.
          if (value == numFlagged) {
	      if (DEBUG_FLAG)
	        printf("Tile (%d, %d): NumFlagged=%d\n", x, y, numFlagged);
              board[idx] = 0;
              moveMade = true;
              foreachNeighbor(width, height, x, y, [board, width, height, tileStackX, tileStackY](int nx, int ny, int neighborIdx) {
                  int neighborValue = board[neighborIdx];
                  if (neighborValue == COVERED)
		    uncoverTile(board, width, height, nx, ny, tileStackX, tileStackY);
              });
          }
          // If the rest of the covered neighbors are all mines, mark them as such.
          else if (value - numFlagged == numCovered) {
	      if (DEBUG_FLAG)
	        printf("Tile (%d, %d): Marking Mines\n", x, y);
              board[idx] = 0;
              moveMade = true;
              foreachNeighbor(width, height, x, y, [board, width, height](int nx, int ny, int neighborIdx) {
                  int neighborValue = board[neighborIdx];
                  if (neighborValue == COVERED)
                      board[neighborIdx] = FLAG;
              });
          }
      }
  }

  return moveMade;
}

// simple solver with no optimization
__global__
void simpleSolver(int width, int height, int *output, int *tileStackX, int *tileStackY)
{
  int numIterations = 0;
  bool moveMade = performIteration(output, width, height, tileStackX, tileStackY);
  // int lim = 10;
  // while(numIterations < lim && moveMade) {
  while(moveMade) {
    numIterations += 1;
    moveMade = performIteration(output, width, height, tileStackX, tileStackY);
  }
  if (DEBUG_FLAG)
    printf("numIterations: %d\n", numIterations);
}

__device__ 
int performGlobalIteration(int *board, int idx, int width, int height, int *flagCounter) {
  int value = board[idx];
  int x = idx % width;
  int y = idx / width;
  int minesFlagged (0);
  if (value == COVERED || value == FLAG) { 
      if (DEBUG_FLAG) {
        printf("Tile (%d, %d): Skipping\n", x, y);
      }
      return minesFlagged;
  }

  int numFlagged, numCovered;
  getNeighborInfo(board, width, height, x, y, &numFlagged, &numCovered);
  // if (DEBUG_FLAG)
  //   printf("Tile (%d, %d): value=%d - numFlagged=%d - numCovered=%d)\n", x, y, value, numFlagged, numCovered);

  // If this tile has all of it's mines identified,
  // then uncover it's remaining covered neighbors.
  // in the global memory case, this should handle 0-valued tiles as well
  if (value == numFlagged) {
      // if (DEBUG_FLAG)
      //   printf("Tile (%d, %d): NumFlagged=%d\n", x, y, numFlagged);
      board[idx] = 0;
      foreachNeighbor(width, height, x, y, [board, width, height](int nx, int ny, int neighborIdx) {
          int neighborValue = board[neighborIdx];
          if (neighborValue == COVERED)
	    board[ny*width+nx] = clickTile(nx, ny);
      });
  }
  // If the rest of the covered neighbors are all mines, mark them as such.
  else if (value - numFlagged == numCovered) {
    // if (DEBUG_FLAG)
    //   printf("Tile (%d, %d): Marking Mines\n", x, y);
    board[idx] = 0;
    foreachNeighbor(width, height, x, y, [board, width, height, &minesFlagged](int nx, int ny, int neighborIdx) {
        int neighborValue = board[neighborIdx];
        if (neighborValue == COVERED) {
            board[neighborIdx] = FLAG;
	    minesFlagged += 1;
	}
    });
  }
  if (DEBUG_FLAG) {
      printf("Tile (%d, %d): minesFlagged = %d\n", x, y, minesFlagged);
  }
  return minesFlagged;
}

// global solver with no optimization
__global__
void globalSolver(int numMines, int width, int height, int *output, int *flagCounter)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x; // contains index of current thread
  int stride = blockDim.x * gridDim.x; // contains number of threads in block
  // int iterLim = 100; 
  int iter = 0;
  int minesFlagged;
  while (flagCounter[0] < numMines) {
  // while (flagCounter[0] < numMines && iter < iterLim) {
    minesFlagged = 0;
    for (int i = index; i < width*height; i += stride)
      minesFlagged += performGlobalIteration(output, i, width, height, flagCounter);
    atomicAdd(&flagCounter[0], minesFlagged);
    __syncthreads();
    if (DEBUG_FLAG)
      printf("Iteration #%d - %d Mines Flagged\n", iter, flagCounter[0]);
    iter += 1; 
  }
}

__device__
int getEdgeValue(int row, int col, int width, int height, int neighborIdx, int *output) {
  if(isOutOfRange(col, row, width, height)) {
    return DUMMY;
  }
  else {
    return output[neighborIdx];
  }
}

// shared memory helper function - get tiles that are adjacent to tiles in thread block for gather operation
__device__
void populateSharedMemory(int subRow, int subCol, int subWidth, int blockY, int blockX, int row, int col, int width, int height, int *output, int *subBoard) {
  subBoard[subRow*subWidth+subCol] = output[row*width+col];
  foreachNeighborExt(width, height, col, row, [width, height, col, row, blockY, blockX, subRow, subCol, &subBoard, &output](int nx, int ny, int neighborIdx) { 
    int val = getEdgeValue(row, col, width, height, neighborIdx, output); 
    // populate corners of subBoard
    if (subRow == 1 && subCol == 1 && nx == -1 && ny == -1) {
      subBoard[0] = val; 
    }
    else if (subRow == 1 && subCol == blockX && nx == 1 && ny == -1) {
      subBoard[subCol+1] = val;
    }
    else if (subRow == blockY && subCol == 1 && nx == -1 && ny == 1) {
      subBoard[(subRow+1)*width] = val;
    }
    else if (subRow == blockY && subCol == blockX && nx == 1 && ny == 1) {
      subBoard[(subRow+1)*width+subCol+1] = val;
    }
    // populate edges of subBoard
    else if (subRow==1 && ny == -1 && nx == 0) {
        subBoard[subCol] = val;
    }
    else if (subRow==blockY && ny == 1 && nx == 0) {
        subBoard[(subRow+1)*width+subCol] = val;
    }
    else if (subCol==1 && ny == 0 && nx == -1) {
        subBoard[subRow*width] = val;
    }
    else if (subCol==blockX && ny == 0 && nx == 1) {
        subBoard[(subRow*width)+subCol+1] = val;
    }
  });
}

// shared memory helper function - write out boards from shared memory to global memory 
__device__
void syncWithGlobalMemory(int subRow, int subCol, int subWidth, int blockY, int blockX, int row, int col, int width, int height, int *output, int *subBoard) {
  output[row*width+col] = subBoard[subRow*subWidth+subCol];
  foreachNeighborExt(width, height, col, row, [width, height, col, row, blockY, blockX, subRow, subCol, &subBoard, &output](int nx, int ny, int neighborIdx) { 
    // int val = getEdgeValue(row, col, width, height, neighborIdx, output); 
    // populate corners of subBoard
    if (subRow == 1 && subCol == 1 && nx == -1 && ny == -1) {
      output[neighborIdx] = subBoard[0]; 
    }
    else if (subRow == 1 && subCol == blockX && nx == 1 && ny == -1) {
      output[neighborIdx] = subBoard[subCol+1];
    }
    else if (subRow == blockY && subCol == 1 && nx == -1 && ny == 1) {
      output[neighborIdx] = subBoard[(subRow+1)*width];
    }
    else if (subRow == blockY && subCol == blockX && nx == 1 && ny == 1) {
      output[neighborIdx] = subBoard[(subRow+1)*width+subCol+1];
    }
    // populate edges of subBoard
    else if (subRow==1 && ny == -1 && nx == 0) {
        output[neighborIdx] = subBoard[subCol];
    }
    else if (subRow==blockY && ny == 1 && nx == 0) {
        output[neighborIdx] = subBoard[(subRow+1)*width+subCol];
    }
    else if (subCol==1 && ny == 0 && nx == -1) {
        output[neighborIdx] = subBoard[subRow*width];
    }
    else if (subCol==blockX && ny == 0 && nx == 1) {
        output[neighborIdx] = subBoard[(subRow*width)+subCol+1];
    }
  });
}

__device__ 
int performSharedIteration(int *board, int idx, int width, int height, int *flagCounter) {
  int value = board[idx];
  int x = idx % width;
  int y = idx / width;
  int minesFlagged (0);
  if (value == COVERED || value == FLAG) { 
      if (DEBUG_FLAG) {
        printf("Tile (%d, %d): Skipping\n", x, y);
      }
      return minesFlagged;
  }

  int numFlagged, numCovered;
  getNeighborInfo(board, width, height, x, y, &numFlagged, &numCovered);
  // if (DEBUG_FLAG)
  //   printf("Tile (%d, %d): value=%d - numFlagged=%d - numCovered=%d)\n", x, y, value, numFlagged, numCovered);

  // If this tile has all of it's mines identified,
  // then uncover it's remaining covered neighbors.
  // in the global memory case, this should handle 0-valued tiles as well
  if (value == numFlagged) {
      // if (DEBUG_FLAG)
      //   printf("Tile (%d, %d): NumFlagged=%d\n", x, y, numFlagged);
      board[idx] = 0;
      foreachNeighbor(width, height, x, y, [board, width, height](int nx, int ny, int neighborIdx) {
          int neighborValue = board[neighborIdx];
          if (neighborValue == COVERED)
	    board[ny*width+nx] = clickTile(nx, ny);
      });
  }
  // If the rest of the covered neighbors are all mines, mark them as such.
  else if (value - numFlagged == numCovered) {
    // if (DEBUG_FLAG)
    //   printf("Tile (%d, %d): Marking Mines\n", x, y);
    board[idx] = 0;
    foreachNeighbor(width, height, x, y, [board, width, height, &minesFlagged](int nx, int ny, int neighborIdx) {
        int neighborValue = board[neighborIdx];
        if (neighborValue == COVERED) {
            board[neighborIdx] = FLAG;
	    minesFlagged += 1;
	}
    });
  }
  if (DEBUG_FLAG) {
      printf("Tile (%d, %d): minesFlagged = %d\n", x, y, minesFlagged);
  }
  return minesFlagged;
}

// shared memory solver 
__global__
void sharedSolver(int numMines, int width, int height, int *output, int *flagCounter)
{
  int minesFlagged; // global counter for mines flagged; use with atomic adds
  // int subMinesFlagged; // global counter for mines flagged; use with atomic adds
  // int index = blockIdx.x * blockDim.x + threadIdx.x; // contains index of current thread
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Block: (%d, %d) - Index: (%d, %d)\n", blockIdx.y, blockIdx.x, row, col);
  // rows and cols of subBoard include additional tile from edge  
  int subRow = row % blockDim.y + 1;
  int subCol = col % blockDim.x + 1;
  int subHeight = blockDim.y + 2;
  int subWidth  = blockDim.x + 2;
  int subIndex = subRow*subWidth+subCol;
  extern __shared__ int subBoard[]; 
  // TO-DO: This is not working perfectly right now. Do some root cause analysis //
  populateSharedMemory(subRow, subCol, subWidth, blockIdx.y, blockIdx.x, row, col, width, height, output, subBoard);
  // calculate extra rows in current block
  __syncthreads();
  int iterLim = 40; 
  int iter = 0;
  if (DEBUG_FLAG && threadIdx.y == 0 && blockIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1) {
    printf("--- threadIdx: (%d, %d) - blockIdx: (%d, %d) ---\n", threadIdx.y, threadIdx.x, blockIdx.y, blockIdx.x);
    printf("--- After %d iterations ---\n", iter);
    printBoardGPU(subBoard, subWidth, subHeight);
  }
  // testing subBoard; add random value to known position
  // subBoard[subWidth+5] = 7;
  // while (flagCounter[0] < numMines) {
  // while (flagCounter[0] < numMines && iter < iterLim) {
  //   subMinesFlagged = 0;
  //   subMinesFlagged += performSharedIteration(subBoard, subIndex, subWidth, subHeight, flagCounter);
  //   // if (DEBUG_FLAG)
  //   //   printf("Iteration #%d - %d Mines Flagged\n", iter, flagCounter[0]);
  //   iter += 1; 
  //   if (subMinesFlagged > 0)
  //     atomicAdd(&flagCounter[0], subMinesFlagged);
  //   // after certain # of iterations, sync with global memory 
  //   syncWithGlobalMemory(subRow, subCol, subWidth, blockIdx.y, blockIdx.x, row, col, width, height, output, subBoard); 
  //   output[row*width+col] = subBoard[subRow*subWidth+subCol];
  //   __syncthreads();
  //   populateSharedMemory(subRow, subCol, subWidth, blockIdx.y, blockIdx.x, row, col, width, height, output, subBoard); 
  //   __syncthreads();
  // }

  // if (DEBUG_FLAG && threadIdx.y == 0 && blockIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1) {
  //   printf("--- threadIdx: (%d, %d) - blockIdx: (%d, %d) ---\n", threadIdx.y, threadIdx.x, blockIdx.y, blockIdx.x);
  //   printf("--- After %d iterations ---\n", iter);
  //   printBoardGPU(subBoard, subWidth, subHeight);
  // }
}

// TODO Implement this function, as well as GPU kernel(s) and any other functions
//      needed for your GPU minesweeper solver.
void minesweeperGPU(int width, int height, int numMines, int startX, int startY, int* output) { 
  // setup board 
  for (int i=0; i< width*height; i++)
    output[i] = COVERED;
  // testing subBoard; add random value to known position
  // output[width+5] = 7;
  // Dynamically determine size of shared memory per thread block 
  // int blockHeight = BLOCK_SIZE / width;
  // int subBoardDim = (blockHeight+2)*width; // need to pull in extra row on top and bottom for gather operations
  int* tileStackX;
  int* tileStackY;
  int blockHeight = min(BLOCK_EDGE, height);
  int blockWidth  = min(BLOCK_EDGE, width);
  dim3 block(blockWidth, blockHeight,1); 
  int gridHeight = height / blockHeight;
  int gridWidth  = width  / blockHeight;
  dim3 grid(gridWidth, gridHeight, 1); 
  int subBoardDim = (blockHeight+2)*(blockWidth+2); // need to pull in extra row on top and bottom for gather operations
  int* flagCounter;
  int numSMs;
  int devId = 0;
  cudaDeviceGetAttribute(&numSMs, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, devId); 
  // int numBlocks = 32*numSMs;
  cudaMallocManaged(&tileStackX, width*height*sizeof(int));
  cudaMallocManaged(&tileStackY, width*height*sizeof(int));
  cudaMallocManaged(&flagCounter, sizeof(int));
  firstClickGlobal<<<1, 1>>>(startX, startY, width, height, output, tileStackX, tileStackY);
  cudaDeviceSynchronize();
  if (DEBUG_FLAG) {
    printf("--- Board after first click ---\n");
    printBoard(output, width, height);
    printf("--- Init Shared Solver with grid size = (%d, %d), blockSize = (%d, %d)\n", grid.y, grid.x, block.y, block.x);
  }
  // simpleSolver<<<1, 1>>>(width, height, output, tileStackX, tileStackY);
  // globalSolver<<<32*numSMs, BLOCK_SIZE>>>(numMines, width, height, output, flagCounter);
  sharedSolver<<<grid, block, subBoardDim*sizeof(int)>>>(numMines, width, height, output, flagCounter);
  // sharedSolver<<<32*numSMs, BLOCK_SIZE, subBoardDim*sizeof(int)>>>(numMines, width, height, output, flagCounter);
  // simpleSolver<<<numBlocks, blockSize>>>(width, height, output);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  if (DEBUG_FLAG) {
    printf("--- Board after iterations ---\n");
    printBoard(output, width, height);
  }

}

#endif // MINESWEEPER_CU
