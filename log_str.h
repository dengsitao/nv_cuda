#ifndef KINGKONG_LOG_STR
#define KINGKONG_LOG_STR

#include "stdio.h"

/* A.9.7. VideoStreaming Interface Control Selectors */
static const char * VSIntfControlSelectorStr[] = {
"[UVC_VS_CONTROL_UNDEFINED_0x00]",
"[UVC_VS_PROBE_CONTROL_0x01]",
"[UVC_VS_COMMIT_CONTROL_0x02]",
"[UVC_VS_STILL_PROBE_CONTROL_0x03]",
"[UVC_VS_STILL_COMMIT_CONTROL_0x04]",
"[UVC_VS_STILL_IMAGE_TRIGGER_CONTROL_0x05]",
"[UVC_VS_STREAM_ERROR_CODE_CONTROL_0x06]",
"[UVC_VS_GENERATE_KEY_FRAME_CONTROL_0x07]",
"[UVC_VS_UPDATE_FRAME_SEGMENT_CONTROL_0x08]",
"[UVC_VS_SYNC_DELAY_CONTROL_0x09]"
};

inline const char * getVSIntfControlSelectorStr(unsigned int cs){
	if (cs<UVC_VS_SYNC_DELAY_CONTROL)
		return VSIntfControlSelectorStr[cs];
	else 
		return VSIntfControlSelectorStr[UVC_VS_CONTROL_UNDEFINED];
};

static const char * UVCEventStr[] = {
"[UVC_EVENT_CONNECT]",
"[UVC_EVENT_DISCONNECT]",
"[UVC_EVENT_STREAMON]",
"[UVC_EVENT_STREAMOFF]",
"[UVC_EVENT_SETUP]",
"[UVC_EVENT_DATA]"
};

inline const char * getUVCEventStr(unsigned int uEvent){
	if (uEvent>V4L2_EVENT_PRIVATE_START)
		return UVCEventStr[uEvent-V4L2_EVENT_PRIVATE_START];
	else 
		return UVCEventStr[UVC_EVENT_LAST-1];
};

#define UVC_RC_LOG_INDEX_BASE 0x80

static const char * UVCOpsStr [] = {
	"[UVC_RC_UNDEFINED_0x00]",
	"[UVC_SET_CUR_0x01]",
	"[UVC_GET_CUR_0x81]",
	"[UVC_GET_MIN_0x82]",
	"[UVC_GET_MAX_0x83]",
	"[UVC_GET_RES_0x84]",
	"[UVC_GET_LEN_0x85]",
	"[UVC_GET_INFO_0x86]",
	"[UVC_GET_DEF_0x87]"
};
inline const char * getUVCOpStr(unsigned int uOp){
	if (uOp>UVC_RC_LOG_INDEX_BASE)
		return UVCOpsStr[uOp-UVC_RC_LOG_INDEX_BASE+1];
	else 
		return UVCOpsStr[uOp];
};

static const char * V4L2PixFormatStr [] = {
	"[V4L2_PIX_FMT_YUYV]",
	"[V4L2_PIX_FMT_MJPEG]",
	"[V4L2_PIX_FMT_H264]",
	"[V4L2_PIX_FMT_NV12]",
	"[V4L2_PIX_FMT_YUV420]",
	"[Unknown Format]"
};

inline const char * getV4L2FormatStr(unsigned int fcc){
	switch (fcc) {
         case V4L2_PIX_FMT_YUYV:
            return V4L2PixFormatStr[0];
         case V4L2_PIX_FMT_MJPEG:
            return V4L2PixFormatStr[1];;
		 case V4L2_PIX_FMT_H264:
            return V4L2PixFormatStr[2];
		 case V4L2_PIX_FMT_NV12:
		 	return V4L2PixFormatStr[3];
		case V4L2_PIX_FMT_YUV420:
		 	return V4L2PixFormatStr[4];
		 default:
		 	return "[Unknown Format]";
         }
	
};

inline void printHexData(char * data, unsigned int len, const char * name_str)
{

	printf("=======dump hex data of %s begin======\n", name_str);
	int row=len/16+1;
	char row_buf[128];
	int row_ind=0;
	int column_ind=0;
	//memset(row_buf, 0, sizeof(row_buf));
	for (row_ind=0;row_ind<row;row_ind++)
	{
		memset(row_buf, 0, sizeof(row_buf));
		for (column_ind=0;column_ind<16;column_ind++)
		{
			unsigned int index=(row_ind)*16+column_ind;
			if(index<len)
				sprintf(row_buf, "%s %2x ",row_buf, data[index]);
			else
				break;
		}
		if (row_buf[0]!=0)
			printf("%s \n", row_buf);
	}
	printf("=======dump hex data of %s end  ======\n", name_str);
	
}

#endif
