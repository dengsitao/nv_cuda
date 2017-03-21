#include "stdlib.h"
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#define BMP_HEADER 14
#define DIB_HEADER 40
#define IMAGE_WIDTH  4608
#define IMAGE_HEIGHT 3456
#define IMAGE_BYTES_PER_PIXEL 3
#define IMAGE_SIZE (IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_BYTES_PER_PIXEL)
#define BUF_SIZE (IMAGE_SIZE+BMP_HEADER+DIB_HEADER)

void ReadBMP(const char * pPath, uint8_t * imageData, uint32_t bufferLen, uint32_t * imageLen)
{
	if (pPath==NULL || imageData==NULL) {
		printf("[ReadBMP]: Invalid args: pPath=%p, %s, imageData=%p\n"
			, pPath, pPath==NULL?"NULL":pPath, imageData);
		return;
	}
	FILE * pFile=fopen(pPath,  "r");
	if (NULL==pFile) {
		printf("[ReadBMP]Open file %s failed! %d %s\n", pPath, errno, strerror(errno));
		return;
	}
	int ret = fread(imageData, 1, bufferLen, pFile);
	if (-1==ret) {
		printf("[ReadBMP]Read file %s failed %d, %s\n", pPath, errno, strerror(errno));
	}	
	*imageLen=ret;
	fclose(pFile);
	return;
}


int main()
{
	uint8_t imageBuf[BUF_SIZE];
	memset(imageBuf, 0, BUF_SIZE);
	uint32_t imageLen;
	ReadBMP("./material/cap_1.bmp", imageBuf, BUF_SIZE, &imageLen);
	printf("simple read test:  len=%u", imageLen);
	return 0;
}
