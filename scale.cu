#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <semaphore.h>
#include <errno.h>
#include <unistd.h>
//#include <pthread.h>
#include <cuda_runtime.h>
#include <time.h>

#include "scale.h"
#include "stats.h"
pthread_cond_t scale_start;
pthread_cond_t scale_finish;
pthread_mutex_t main_lock;
//static int scale_done_count=0;

struct scaler_info_t s_info[ROW_NUM][COLUMN_NUM];
pthread_t pid[ROW_NUM][COLUMN_NUM];

struct video_buffer_t g_v_in_buf;
struct video_buffer_t g_v_out_buf;

struct video_buffer_t *d_v_in_buf;
struct video_buffer_t *d_v_out_buf;
uint8_t * d_in_buffer;
uint8_t * d_out_buffer;


extern unsigned int camera_width;
extern unsigned int camera_height;
struct block_param_t
{
	uint32_t m_width_start_pos;
	uint32_t m_height_start_pos;
	uint32_t m_width_len;
	uint32_t m_height_len;
};

__global__ void do_scale(uint8_t * inBuffer, uint8_t * outBuffer, 
			uint32_t inWidth, uint32_t inHeight,
			uint32_t outWidth, uint32_t outHeight,
			double ratioHeight, double ratioWidth,
			double xRatio, double yRatio,
			uint32_t outHeightOffset, uint32_t outWidthOffset,
			uint32_t inStep, uint32_t outStep)
{
	uint32_t outRowIndex=blockIdx.x+outHeightOffset;
	uint32_t outColIndex=threadIdx.x+outWidthOffset;
	uint32_t outIndex=(outRowIndex)*outWidth+outColIndex;

	uint32_t inX00row=outRowIndex/ratioHeight;
	uint32_t inX00col=outColIndex/ratioWidth;
	//if (inX00row >= inHeight-2){ 
		//printf("==edge inX00row[%u]inHeight[%u]==\n", inX00row, inHeight);
		//inX00row=inHeight-2;
	//}
	//if (inX00col >= inWidth-2){ 
		//printf("==edge inX00col[%u]inWidth[%u]==\n", inX00col, inWidth);
		//inX00col=inWidth-2;
	//}
	uint8_t inX00=inBuffer[(inX00row*inWidth+inX00col)*inStep];
	uint8_t inX01=inBuffer[(inX00row*inWidth+inX00col+1)*inStep];
	uint8_t inX10=inBuffer[((inX00row+1)*inWidth+inX00col)*inStep];
	uint8_t inX11=inBuffer[((inX00row+1)*inWidth+inX00col+1)*inStep];
	outBuffer[outIndex*outStep]=yRatio*(xRatio*(inX00)+(1-xRatio)*(inX01))+(1-yRatio)*(xRatio*(inX10)+(1-xRatio)*(inX11));
}


void do_bilineary_scale(uint32_t inWidth, uint32_t inHeight, uint32_t outWidth, uint32_t outHeight,
						uint32_t inStep, uint32_t outStep, 
						uint32_t inBufferOffset, uint32_t outBufferOffset)
{
        double ratioWidth=(double)outWidth/(double)inWidth;
        double ratioHeight=(double)outHeight/(double)inHeight;
        double rWidthX=floor(ratioWidth);
        double rHeightY=floor(ratioHeight);
        double xRatio=ratioWidth-rWidthX;
        double yRatio=ratioHeight-rHeightY;
	uint32_t outRowIndex=0;
        uint32_t outColIndex=0;

	dim3 block(COLUMN_NUM);
	dim3 grid(ROW_NUM);
	for (outRowIndex=0;outRowIndex<outHeight;outRowIndex+=ROW_NUM) {
		for (outColIndex=0;outColIndex<outWidth;outColIndex+=COLUMN_NUM) {
			do_scale<<<grid, block>>>(d_in_buffer+inBufferOffset, d_out_buffer+outBufferOffset,
						  inWidth, inHeight,
						  outWidth, outHeight,
						  ratioHeight, ratioWidth,
						  xRatio, yRatio,
						  outRowIndex, outColIndex,
						  inStep, outStep);
		}
	}
}
/*use thread to scale*/
void do_image_scale(uint8_t * in_buf, uint32_t inWidth, uint32_t inHeight, uint8_t * out_buf, uint32_t outWidth, uint32_t outHeight)
{
	uint32_t inImgSize=inWidth*inHeight;
	uint32_t outImgSize=outWidth*outHeight;
	//handle Y
	do_bilineary_scale(inWidth, inHeight, outWidth, outHeight,1,1, 0,0);
	//handle U

	do_bilineary_scale(inWidth, inHeight, outWidth, outHeight,2,2,inImgSize, outImgSize);
	//handle V
	do_bilineary_scale(inWidth, inHeight, outWidth, outHeight,2,2,inImgSize+1, outImgSize+1);

	return ;
}
__device__ void bilineary_scale(struct video_buffer_t * pInBuffer, struct video_buffer_t * pOutBuffer, 
						uint32_t inStep, uint32_t outStep, 
						struct block_param_t * outBlockParam)
{
	/*if (inWidth ==0 || inHeight==0 || outWidth==0 || outHeight==0 || inBuffer==NULL || outBuffer==NULL || inStep==0 || outStep==0) {
		printf("bilineary_scale: invalide arg: inBuffer %p, outBuffer %p, inWidth %u, inHeight %u, outWidth %u, outHeight %u, instep %u outstep %u\n"
			, inBuffer , outBuffer , inWidth , inHeight , outWidth , outHeight, inStep, outStep);
		return;
	}*/
	uint32_t outWidth=pOutBuffer->m_width;
	uint32_t outHeight=pOutBuffer->m_height;
	uint32_t inWidth=pInBuffer->m_width;
	uint32_t inHeight=pInBuffer->m_height;
	uint8_t * inBuffer=pInBuffer->m_buffer;
	uint8_t * outBuffer=pOutBuffer->m_buffer;

	uint32_t outHeightOffset = outBlockParam->m_height_start_pos;
	uint32_t outWidthOffset = outBlockParam->m_width_start_pos;

	uint32_t outHeightLen = outBlockParam->m_height_len;
	uint32_t outWidthLen = outBlockParam->m_width_len;

	double ratioWidth=(double)outWidth/(double)inWidth;
	double ratioHeight=(double)outHeight/(double)inHeight;
	double rWidthX=floor(ratioWidth);
	double rHeightY=floor(ratioHeight);
	double xRatio=ratioWidth-rWidthX;
	double yRatio=ratioHeight-rHeightY;
	uint32_t outRowIndex=0;
	uint32_t outColIndex=0;
	uint32_t outIndex=0;
	uint32_t inX00row=0;
	uint32_t inX00col=0;
	uint8_t inX00=0;
	uint8_t inX01=0;
	uint8_t inX10=0;
	uint8_t inX11=0;
	//printf("==== start bilineary_scale ===\n");
	//printf("enter with: inBuffer %p, outBuffer %p, inWidth %u, inHeight %u, outWidth %u, outHeight %u, outWidthOffset=%u, outHeightOffset=%u, outWidthLen=%u, outHeightLen=%u\n"
	//		, inBuffer , outBuffer , inWidth , inHeight , outWidth , outHeight, outWidthOffset, outHeightOffset, outWidthLen, outHeightLen);
	//printf("ratioWidth=%f, ratioHeight=%f,rWidthX=%f,rHeightY=%f  xRatio=%f, yRatio=%f\n"
	//		,ratioWidth, ratioHeight,rWidthX, rHeightY,  xRatio, yRatio);
	for (outRowIndex=outHeightOffset;outRowIndex<outHeightOffset+outHeightLen;outRowIndex++) {
		for (outColIndex=outWidthOffset;outColIndex<outWidthOffset+outWidthLen;outColIndex++) {
			outIndex=(outRowIndex)*outWidth+outColIndex;

            inX00row = outRowIndex / ratioHeight;
            inX00col = outColIndex / ratioWidth;

            if (outColIndex % 2 == 0) {
                if (inX00col % 2 == 1) inX00col -= 1;
            }

            if (outColIndex % 2 == 1) {
                if (inX00col % 2 == 0) inX00col += 1;
            }

            if (inX00row >= inHeight - 1) {
                //printf("==edge inX00row[%u]inHeight[%u]==\n", inX00row, inHeight);
                inX00row = inHeight - 1;
            }

            if (inX00col >= inWidth - 1) {
                //printf("==edge inX00col[%u]inWidth[%u]==\n", inX00col, inWidth);
                inX00col = inWidth - 1;
            }

            /*uint32_t inX00index = (inX00row * inWidth + inX00col) * inStep;
            uint32_t inX01index = (inX00row * inWidth + inX00col + 1) * inStep;
            uint32_t inX10index = ((inX00row + 1) * inWidth + inX00col) * inStep;
            uint32_t inX11index = ((inX00row + 1) * inWidth + inX00col + 1) * inStep;*/

            uint32_t inX00index = (inX00row * inWidth) + (inX00col);// * inStep;
            uint32_t inX01index = (inX00row * inWidth) + (inX00col + inStep);// * inStep;
            uint32_t inX10index = ((inX00row + 1) * inWidth) + (inX00col);// * inStep;
            uint32_t inX11index = ((inX00row + 1) * inWidth) + (inX00col + inStep);// * inStep;

            uint32_t inRange = inWidth * inHeight;

            if (inX00index > inRange || inX01index > inRange || inX10index > inRange || inX11index > inRange) {
                //printf("index out of range inX00index[%u]=(inX00row[%u]*inWidth[%u]+inX00col[%u])*inStep[%u]\n", inX00index, inX00row, inWidth, inX00col, inStep);
                //printf("index out of range[%u]: [%u] [%u] [%u] [%u]\n", inRange, inX00index, inX01index, inX10index, inX11index);
            } else {
                inX00 = inBuffer[inX00index];
                inX01 = inBuffer[inX01index];
                inX10 = inBuffer[inX10index];
                inX11 = inBuffer[inX11index];
            }

            outBuffer[outIndex/* * outStep*/] = yRatio * (xRatio * (inX00) + (1 - xRatio) * (inX01)) + (1 - yRatio) * (xRatio * (inX10) + (1 - xRatio) * (inX11));
            //printf("outBuffer[%u]=[%u]%p, inX00row=%u, inX00col=%u, outRowIndex=%u, outColIndex=%u\n"
            //      ,outIndex,outBuffer[outIndex/**outStep*/], &outBuffer[outIndex/**outStep*/],inX00row, inX00col, outRowIndex, outColIndex   );
        }
    }

    //printf("==== end bilineary_scale %u===\n", outIndex);

}
/*use thread to scale*/
__global__ void image_scale(uint8_t * in_buf, uint32_t inWidth, uint32_t inHeight, uint8_t * out_buf, uint32_t outWidth, uint32_t outHeight)
{
	uint32_t row_index=blockIdx.x;
	uint32_t column_index=threadIdx.x;
	struct video_buffer_t v_in_buf;
	struct video_buffer_t v_out_buf;
	v_in_buf.m_buffer = in_buf;
	v_in_buf.m_width=inWidth;
	v_in_buf.m_height=inHeight;
	v_out_buf.m_buffer = out_buf;
	v_out_buf.m_width=outWidth;
	v_out_buf.m_height=outHeight;

		struct block_param_t v_out_blk_par;
		//uint32_t inWidth    = v_in_buf.m_width;
		//uint32_t inHeight   = v_in_buf.m_height;
		//uint32_t outWidth   = v_out_buf.m_width;
		//uint32_t outHeight  = v_out_buf.m_height;

		//handle Y
		v_out_blk_par.m_width_start_pos=column_index*v_out_buf.m_width/COLUMN_NUM;
		v_out_blk_par.m_width_len=v_out_buf.m_width/COLUMN_NUM;
		v_out_blk_par.m_height_start_pos=row_index*v_out_buf.m_height/ROW_NUM;
		v_out_blk_par.m_height_len=v_out_buf.m_height/ROW_NUM;
		bilineary_scale(&v_in_buf,&v_out_buf,1,1,&v_out_blk_par);
		//handle U

		v_in_buf.m_buffer=(in_buf)+inWidth*inHeight;
		v_in_buf.m_height=inHeight/2;
		v_out_buf.m_buffer=out_buf+outWidth*outHeight;
		v_out_buf.m_height=outHeight/2;
		v_out_blk_par.m_width_start_pos=column_index*outWidth/COLUMN_NUM;
		v_out_blk_par.m_width_len=outWidth/COLUMN_NUM;
		v_out_blk_par.m_height_start_pos=row_index*v_out_buf.m_height/ROW_NUM;
		v_out_blk_par.m_height_len=v_out_buf.m_height/ROW_NUM;
		bilineary_scale(&v_in_buf,&v_out_buf,2,2,&v_out_blk_par);
		//handle V
		v_in_buf.m_buffer=in_buf+inWidth*inHeight+1;
		v_out_buf.m_buffer=out_buf+outWidth*outHeight+1;
		v_out_blk_par.m_width_start_pos=column_index*outWidth/COLUMN_NUM;
		v_out_blk_par.m_width_len=outWidth/COLUMN_NUM;
		v_out_blk_par.m_height_start_pos=row_index*v_out_buf.m_height/ROW_NUM;
		v_out_blk_par.m_height_len=v_out_buf.m_height/ROW_NUM;
		bilineary_scale(&v_in_buf,&v_out_buf,2,2,&v_out_blk_par);

		//scale_done_count++;
	//}
	return ;
}
//#define MULTI_THREAD 1
unsigned long cp2DevTime1=0;
unsigned long cp2DevDelta=0;
unsigned long calcTime1=0;
unsigned long calcDelta=0;
unsigned long cp2HostTime1=0;
unsigned long cp2HostDelta=0;
void scaleNV12(uint8_t * inBuffer, unsigned int inWidth, unsigned int inHeight, uint8_t * outBuffer, uint32_t outWidth, uint32_t outHeight)
{
#ifdef MULTI_THREAD
	g_v_in_buf.m_buffer=inBuffer;
	g_v_in_buf.m_width=inWidth;
	g_v_in_buf.m_height=inHeight;
	g_v_out_buf.m_buffer=outBuffer;
	g_v_out_buf.m_width=outWidth;
	g_v_out_buf.m_height=outHeight;
	//printf("start scaling\n");
	//if (cp2DevTime1==0)
	cp2DevTime1=getTimeInMicroS();
	//cudaMemcpy(d_in_buffer, inBuffer, inWidth*inHeight*3/2, cudaMemcpyHostToDevice);
	cp2DevDelta+=getTimeInMicroS() - cp2DevTime1;
	//cudaMemcpy(d_in_v_buf, g_in_v_buf, sizeof(struct video_buffer_t), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_out_v_buf, g_out_v_buf, sizeof(struct video_buffer_t), cudaMemcpyHostToDevice);
	dim3 block(COLUMN_NUM);
	dim3 grid(ROW_NUM);
	calcTime1=getTimeInMicroS();
	image_scale<<<grid, block>>>(d_in_buffer, inWidth, inHeight, d_out_buffer, outWidth, outHeight);
	//do_image_scale(d_in_buffer, inWidth, inHeight, d_out_buffer, outWidth, outHeight);
	cudaDeviceSynchronize();	
	calcDelta+=getTimeInMicroS() - calcTime1;
	cp2HostTime1=getTimeInMicroS();
	cudaMemcpy(outBuffer, d_out_buffer, outWidth*outHeight*3/2, cudaMemcpyDeviceToHost);
	cp2HostDelta+=getTimeInMicroS() - cp2HostTime1;
	if (frame_count%P_COUNT==0) {
		printf( "GPU process %u frame, cp 2 dev takes %lu ms, "
			"calc takes %lums, "
			"cp 2 host takes %lu ms\n", 
			frame_count, cp2DevDelta/1000,
			calcDelta/1000,
			cp2HostDelta/1000);
		cp2DevDelta=0;
		calcDelta=0;
		cp2HostDelta=0;
	}
/************* CPU multi threading*****
	pthread_cond_broadcast(&scale_start);
	while(scale_done_count<ROW_NUM*COLUMN_NUM && loop_cout<1000){
		usleep(100);
		loop_cout++;
	}
	if (loop_cout>1000) 
		printf("timeout \n");
	loop_cout=0;
	//printf("done  scaling\n");
	scale_done_count=0;
***************************************/
#endif
#if 0
	uint32_t column=0;
	uint32_t row=0;
	struct video_buffer_t v_in_buf;
	struct video_buffer_t v_out_buf;

	v_in_buf.m_buffer=inBuffer;
	v_in_buf.m_width=inWidth;
	v_in_buf.m_height=inHeight;
	v_out_buf.m_buffer=outBuffer;
	v_out_buf.m_width=outWidth;
	v_out_buf.m_height=outHeight;

	struct block_param_t v_out_blk_par;

	for (row=0;row<ROW_NUM;row++) {
		for (column=0;column<COLUMN_NUM;column++) {
				v_out_blk_par.m_width_start_pos=column*v_out_buf.m_width/COLUMN_NUM;
				v_out_blk_par.m_width_len=v_out_buf.m_width/COLUMN_NUM;
				v_out_blk_par.m_height_start_pos=row*v_out_buf.m_height/ROW_NUM;
				v_out_blk_par.m_height_len=v_out_buf.m_height/ROW_NUM;
				bilineary_scale(&v_in_buf,&v_out_buf,1,1,&v_out_blk_par);
		}
	}
	v_in_buf.m_buffer=inBuffer+inWidth*inHeight;
	v_in_buf.m_height=inHeight/2;
	v_out_buf.m_buffer=outBuffer+outWidth*outHeight;
	v_out_buf.m_height=outHeight/2;
	for (row=0;row<ROW_NUM;row++) {
		for (column=0;column<COLUMN_NUM;column++) {
				v_out_blk_par.m_width_start_pos=column*outWidth/COLUMN_NUM;
				v_out_blk_par.m_width_len=outWidth/COLUMN_NUM;
				v_out_blk_par.m_height_start_pos=row*v_out_buf.m_height/ROW_NUM;
				v_out_blk_par.m_height_len=v_out_buf.m_height/ROW_NUM;
				bilineary_scale(&v_in_buf,&v_out_buf,2,2,&v_out_blk_par);
		}
	}
	v_in_buf.m_buffer=inBuffer+inWidth*inHeight+1;
	v_out_buf.m_buffer=outBuffer+outWidth*outHeight+1;
	for (row=0;row<ROW_NUM;row++) {
		for (column=0;column<COLUMN_NUM;column++) {
				v_out_blk_par.m_width_start_pos=column*outWidth/COLUMN_NUM;
				v_out_blk_par.m_width_len=outWidth/COLUMN_NUM;
				v_out_blk_par.m_height_start_pos=row*v_out_buf.m_height/ROW_NUM;
				v_out_blk_par.m_height_len=v_out_buf.m_height/ROW_NUM;
				bilineary_scale(&v_in_buf,&v_out_buf,2,2,&v_out_blk_par);
		}
	}
	
	
	//bilineary_scale(inBuffer, inWidth/2, inHeight/2, outBuffer, outWidth/2, outHeight/2, 1,1,0,0,0,0);
	//bilineary_scale(inBuffer, inWidth/2, inHeight/2, outBuffer, outWidth/2, outHeight/2, 1,1, inWidth/2, 0, outWidth/2,0);
	//bilineary_scale(inBuffer, inWidth/2, inHeight/2, outBuffer, outWidth/2, outHeight/2, 1,1,0,inHeight/2,0,outHeight/2);
	//bilineary_scale(inBuffer, inWidth/2, inHeight/2, outBuffer, outWidth/2, outHeight/2, 1,1,inWidth/2,inHeight/2,outWidth/2,outHeight/2);
	//printf("====scaling y done==========\n");
	//printf("===scaling u==========\n");
	//bilineary_scale(inBuffer+inWidth*inHeight, inWidth, inHeight/2, outBuffer+outWidth*outHeight, outWidth, outHeight/2, 2,2,0,0,0,0);
	//printf("===scaling u done==========\n");
	//printf("===scaling v==========\n");
	//bilineary_scale(inBuffer+inWidth*inHeight+1, inWidth, inHeight/2, outBuffer+(outWidth*outHeight)+1, outWidth, outHeight/2, 2,2,0,0,0,0);
#endif
}

void NV12toI420(uint8_t * inBuffer, unsigned int inWidth, unsigned int inHeight, uint8_t * outBuffer, uint32_t outWidth, uint32_t outHeight)
{
    if (inBuffer == NULL) {
        printf("NV12toI420: can't convert null pointer\n");
        return;
    }
	if (outWidth != inWidth || outHeight != inHeight) {
		printf("NV12toI420: don't support scale in this function!inWidth[%u]inHeight[%u]outWidth[%u]outHeight[%u]\n"
			,inWidth, inHeight, outWidth, outHeight);
        return;
	}
    unsigned int img_size = inWidth * inHeight * 3 / 2;
    unsigned int index = 0;
    unsigned int uvIndex = inWidth * inHeight;
	memcpy(outBuffer, inBuffer, img_size);
    for (index = inWidth * inHeight; index < img_size; index += 2) {
        outBuffer[uvIndex++] = inBuffer[index];
    }

    for (index = inWidth * inHeight + 1; index < img_size; index += 2) {
        outBuffer[uvIndex++] = inBuffer[index];
    }

}
void NV12toI420scale(uint8_t * inBuffer, unsigned int inWidth, unsigned int inHeight, uint8_t * outBuffer, uint32_t outWidth, uint32_t outHeight)
{
    if (inBuffer == NULL) {
        printf("NV12toI420: can't convert null pointer\n");
        return;
    }

    unsigned int buf_size = inWidth * inHeight * 3 / 2;
    uint8_t *tempBuf = (uint8_t*)malloc(buf_size);
#if 0
    unsigned int img_size = inWidth * inHeight;
	unsigned int windex = 0;
    unsigned int hindex = 0;
    unsigned int outYIndex = 0;
    unsigned int outImg_size = outWidth*outHeight;
    unsigned int outUIndex = outImg_size;
    unsigned int outVIndex = outImg_size + outImg_size / 4;
    for (hindex = 0; hindex < inHeight; hindex += 2) {
        for (windex = 0; windex < inWidth; windex += 2) {
            outBuffer[outYIndex++] = inBuffer[hindex * inWidth + windex];
        }
    }

    //printf("NV12toI420scale: windex=%u, hindex=%u, outIndex=%u, buf_size=%u\n", windex, hindex, outIndex, buf_size);
    for (hindex = 0; hindex < inHeight; hindex += 4) {
        for (windex = 0; windex < inWidth; windex += 4) {
            outBuffer[outUIndex++] = (inBuffer[img_size + hindex * inWidth / 2 + windex] + inBuffer[img_size + hindex * inWidth / 2 + windex + 2]) / 2;
            outBuffer[outVIndex++] = (inBuffer[img_size + hindex * inWidth / 2 + windex + 1] + inBuffer[img_size + hindex * inWidth / 2 + windex + 3]) / 2;
        }
    }
#else
	scaleNV12(inBuffer, inWidth, inHeight, tempBuf, outWidth, outHeight);
	NV12toI420(tempBuf,outWidth,outHeight, outBuffer, outWidth,outHeight);
#endif

    free(tempBuf);

}

void NV12toYUY2(uint8_t * inBuffer, unsigned int inWidth, unsigned int inHeight, uint8_t * outBuffer, uint32_t outWidth, uint32_t outHeight)
{
    if (inBuffer == NULL) {
        printf("NV12toYUY2: can't convert null pointer\n");
        return;
    }

	if (outWidth != inWidth || outHeight != inHeight) {
		printf("NV12toYUY2: don't support scale in this function!inWidth[%u]inHeight[%u]outWidth[%u]outHeight[%u]\n"
			,inWidth, inHeight, outWidth, outHeight);
        return;
	}

    unsigned int img_size = inWidth * inHeight;
    unsigned int windex = 0;
    unsigned int hindex = 0;
    unsigned int outIndex = 0;
    unsigned int uvIndex = img_size;
#if 0
    for (hindex = 0; hindex < inHeight; hindex++) {
        if (hindex % 2 == 1) {
            uvIndex -= inWidth;
        }

        for (windex = 0; windex < inWidth; windex++) {
            //printf("NV12toI420scale:outIndex=%u, uvIndex=%u, hindex[%u] x inWidth[%u]+windex[%u]=%u\n", outIndex,  uvIndex, hindex, inWidth, windex, hindex*inWidth+windex);
            outBuffer[outIndex] = inBuffer[hindex * inWidth + windex];
            outIndex++;
            outBuffer[outIndex] = inBuffer[uvIndex];
            outIndex++;
            uvIndex++;
        }
    }
#else
	//following MS recommended  interpolation algo
	char inY;
	char inCbrA;
	char inCbrB;
	char inCbrC;
	char inCbrD;
	unsigned int curIndex;
	unsigned int uvWidth=inWidth;
	unsigned int uvHeight=inHeight/2;
	unsigned int uvHindex=0;
    for (hindex = 0; hindex < inHeight; hindex++) {
		uvHindex=hindex/2;
		if (hindex%2==1)
			uvIndex-=inWidth;
        for (windex = 0; windex < inWidth; windex++) {
			curIndex=hindex * inWidth + windex;
			inY=inBuffer[curIndex];
			outBuffer[outIndex] = inY;//Y=Y
            outIndex++;

			if (hindex%2==0) {//even line
				outBuffer[outIndex] = inBuffer[uvIndex];
			}
			else {//odd line
				if (uvHindex==0) {//first line
					inCbrA=inBuffer[uvIndex];
				}
				else {
					inCbrA=inBuffer[uvIndex-uvWidth];
				}
				inCbrB=inBuffer[uvIndex];
				if (uvHindex <uvHeight-2) {
					inCbrC=inBuffer[uvIndex+uvWidth];
					inCbrD=inBuffer[uvIndex+uvWidth*2];
				}
				else if (uvHindex == uvHeight-2) {
					inCbrC=inBuffer[uvIndex+uvWidth];
					inCbrD=inBuffer[uvIndex+uvWidth];
				}
				else {
					inCbrC=inBuffer[uvIndex];
					inCbrD=inBuffer[uvIndex];
				}
				//now calc U or V
				outBuffer[outIndex] = ((9*(inCbrB+inCbrC)-(inCbrA+inCbrD)+8)>>4)&0xff;

			}
			outIndex++;
			uvIndex++;
        }
    }
#endif

}

void NV12toYUY2scale(uint8_t * inBuffer, unsigned int inWidth, unsigned int inHeight, uint8_t * outBuffer, uint32_t outWidth, uint32_t outHeight)
{
    if (inBuffer == NULL) {
        printf("NV12toI420: can't convert null pointer\n");
        return;
    }

    unsigned int out_img_size = outWidth*outHeight;
    unsigned int out_buf_size = out_img_size * 2;
    uint8_t *tempBuf = (uint8_t *)malloc(out_buf_size);
#if 0
    unsigned int img_size = inWidth * inHeight;
	unsigned int windex = 0;
    unsigned int hindex = 0;
    unsigned int outIndex = 0;
    unsigned int uvIndex = img_size;

    for (hindex = 0; hindex < inHeight; hindex += 2) {
        for (windex = 0; windex < inWidth; windex += 2) {
            //printf("NV12toI420scale:outIndex=%u, uvIndex=%u, hindex[%u] x inWidth[%u]+windex[%u]=%u\n", outIndex,  uvIndex, hindex, inWidth, windex, hindex*inWidth+windex);
            tempBuf[outIndex] = inBuffer[hindex * inWidth + windex];
            outIndex++;
            tempBuf[outIndex] = inBuffer[uvIndex];
            outIndex++;

            if (windex % 4 == 0)
                uvIndex++;
            else
                uvIndex += 3;
        }
    }
	memcpy(inBuffer, tempBuf, out_buf_size);
#else
	scaleNV12(inBuffer, inWidth, inHeight, tempBuf, outWidth, outHeight);
	NV12toYUY2(tempBuf,outWidth,outHeight, outBuffer, outWidth, outHeight);
#endif
    free(tempBuf);
}

/* convert a YUV set to a rgb set - thanks to MartinS and
   http://www.efg2.com/lab/Graphics/Colors/YUV.htm */
void yuvtorgb(unsigned int Y, unsigned int U, unsigned int V, unsigned int *r_ptr, unsigned int *g_ptr, unsigned int *b_ptr)
{
    int r, g, b;
    static short L1[256], L2[256], L3[256], L4[256], L5[256];
    static int initialised = 0;

    if (!initialised) {
        int i;
        initialised = 1;

        for (i = 0; i < 256; i++) {
            L1[i] = 1.164 * (i - 16);
            L2[i] = 1.596 * (i - 128);
            L3[i] = -0.813 * (i - 128);
            L4[i] = 2.018 * (i - 128);
            L5[i] = -0.391 * (i - 128);
        }
    }

    r = L1[Y] + L2[V];
    g = L1[Y] + L3[U] + L5[V];
    b = L1[Y] + L4[U];

    if (r < 0) r = 0;

    if (g < 0) g = 0;

    if (b < 0) b = 0;

    if (r > 255) r = 255;

    if (g > 255) g = 255;

    if (b > 255) b = 255;

    *r_ptr = r;
    *g_ptr = g;
    *b_ptr = b;

    return;
}

void get_rgb(unsigned char *src,
                    unsigned int x,
                    unsigned int  y,
                    unsigned int width,
                    unsigned int height,
                    unsigned int *r_value,
                    unsigned int *g_value,
                    unsigned int *b_value,
                    unsigned int format)
{
    unsigned int y_value = 0;
    unsigned int u_value = 0;
    unsigned int v_value = 0;
    unsigned char *y_buffer = src;
    unsigned char *uv_buffer = (unsigned char *)(src + (width * height));
    unsigned int yOffset = y * width + x ;
    unsigned int uvOffset = (y >> 1) * (width >> 1) + (x >> 1) ;
    y_value = *(y_buffer +  yOffset);

    //0:NV12, 1:YUV422 Planar
    if (format == 0) {
        u_value = *(uv_buffer + 2 * uvOffset);
        v_value =  *(uv_buffer + 2 * uvOffset  + 1);
    } else if (format == 1) {
        //TODO:Support YUV422 Planar
    }

    yuvtorgb(y_value, u_value, v_value, r_value, g_value, b_value);

    return ;
}

int NV12toRGBA(unsigned char *yuv_buffer,
                      unsigned int width, unsigned int height,
                      unsigned char *rgb_buffer)
{
    unsigned int _width = width;
    unsigned int _height = height;
    unsigned int x = 0, y = 0;
    unsigned int r_value = 0, g_value = 0, b_value = 0;
    unsigned long location = 0;

    if (!yuv_buffer || !rgb_buffer)
        return -1;

    for (y = 0; y < _height; y++) {
        for (x = 0; x < _width; x++) {
            /* Figure out where in memory to put the pixel */

            location = (x) * (COLOR_COMPONENTS) + (y) * (_width * COLOR_COMPONENTS);

            get_rgb(yuv_buffer, x, y, _width, _height, &r_value, &g_value, &b_value, 0);

            *(rgb_buffer + location) = r_value; //Paint a black pixel.
            *(rgb_buffer + location + 1) = g_value;
            *(rgb_buffer + location + 2) = b_value;
        }
    }

    return 0;
}

/*use thread to scale*/
void *image_scale_thread(void * attr)
{
#if 0
	struct scaler_info_t * pScalerInfo = (struct scaler_info_t *)attr;
	
	if (pScalerInfo==NULL) {
		printf("NULL param: %p\n", attr);
		return NULL;
	}
	uint32_t row_index=pScalerInfo->m_row_id;
	uint32_t column_index=pScalerInfo->m_column_id;
	struct video_buffer_t v_in_buf;
	struct video_buffer_t v_out_buf;
	while (1) {
		pthread_mutex_lock(&main_lock);
		//printf("image_scale_thread[%u][%u] started \n", row_index, column_index);
		pthread_cond_wait(&scale_start,&main_lock);
		pthread_mutex_unlock(&main_lock);

		//printf("image_scale_thread[%u][%u] get lock\n", row_index, column_index);

		memcpy(&v_in_buf, &g_v_in_buf, sizeof(struct video_buffer_t));
		memcpy(&v_out_buf, &g_v_out_buf, sizeof(struct video_buffer_t));
		//if(v_in_buf==NULL || v_out_buf==NULL) {
		//	printf("image_scale_thread[%u][%u] inbuf[%p] outbuf[%p] invalid\n", row_index, column_index, v_in_buf, v_out_buf);
		//}
		struct block_param_t v_out_blk_par;
		uint32_t inWidth    = v_in_buf.m_width;
		uint32_t inHeight   = v_in_buf.m_height;
		uint32_t outWidth   = v_out_buf.m_width;
		uint32_t outHeight  = v_out_buf.m_height;

		//handle Y
		v_out_blk_par.m_width_start_pos=column_index*v_out_buf.m_width/COLUMN_NUM;
		v_out_blk_par.m_width_len=v_out_buf.m_width/COLUMN_NUM;
		v_out_blk_par.m_height_start_pos=row_index*v_out_buf.m_height/ROW_NUM;
		v_out_blk_par.m_height_len=v_out_buf.m_height/ROW_NUM;
		bilineary_scale(&v_in_buf,&v_out_buf,1,1,&v_out_blk_par);
		//handle U
		v_in_buf.m_buffer=g_v_in_buf.m_buffer+inWidth*inHeight;
		v_in_buf.m_height=inHeight/2;
		v_out_buf.m_buffer=g_v_out_buf.m_buffer+outWidth*outHeight;
		v_out_buf.m_height=outHeight/2;
		v_out_blk_par.m_width_start_pos=column_index*outWidth/COLUMN_NUM;
		v_out_blk_par.m_width_len=outWidth/COLUMN_NUM;
		v_out_blk_par.m_height_start_pos=row_index*v_out_buf.m_height/ROW_NUM;
		v_out_blk_par.m_height_len=v_out_buf.m_height/ROW_NUM;
		bilineary_scale(&v_in_buf,&v_out_buf,2,2,&v_out_blk_par);
		//handle V
		v_in_buf.m_buffer=g_v_in_buf.m_buffer+inWidth*inHeight+1;
		v_out_buf.m_buffer=g_v_out_buf.m_buffer+outWidth*outHeight+1;
		v_out_blk_par.m_width_start_pos=column_index*outWidth/COLUMN_NUM;
		v_out_blk_par.m_width_len=outWidth/COLUMN_NUM;
		v_out_blk_par.m_height_start_pos=row_index*v_out_buf.m_height/ROW_NUM;
		v_out_blk_par.m_height_len=v_out_buf.m_height/ROW_NUM;
		bilineary_scale(&v_in_buf,&v_out_buf,2,2,&v_out_blk_par);

		scale_done_count++;
	}
#endif
	return NULL;
}

void create_scaler_thread(uint32_t colNum, uint32_t rowNum)
{
	//uint32_t i=0;
	//uint32_t j=0;
	//uint32_t finished_count=0;
	//pthread_cond_t scaler_start;
	//pthread_mutex_t scaler_lock;
	int dev=0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
        printf("Using Device %d, %s\n", dev, deviceProp.name);
        //return 0;
        cudaSetDevice(dev);
	cudaMalloc((uint8_t**)&d_in_buffer, (1920*1080*3/2));
	cudaMalloc((uint8_t**)&d_out_buffer, (1280*720*3/2));
	cudaMalloc((struct video_buffer_t**)&d_v_in_buf, sizeof(struct video_buffer_t));
	cudaMalloc((struct video_buffer_t**)&d_v_out_buf, sizeof(struct video_buffer_t));
/************CPU multi thread************
	pthread_cond_init(&scale_start, NULL);
	pthread_cond_init(&scale_finish, NULL);
	pthread_mutex_init(&main_lock, NULL);

	for (i=0;i<rowNum;i++) {
		for (j=0;j<colNum;j++) {

			s_info[i][j].m_row_id=i;
			s_info[i][j].m_column_id=j;
			
			if (0!=pthread_create(&(pid[i][j]), NULL, image_scale_thread, &(s_info[i][j]))) {
				printf("[%u][%u]create scaler thread failed: %d, %s\n",i,j, errno, strerror(errno));
				return;
			}
		}
	}
	printf("thread create done\n");
******************************************/


}


