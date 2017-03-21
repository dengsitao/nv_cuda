/*
 * UVC gadget test application
 *
 * Copyright (C) 2010 Ideas on board SPRL <laurent.pinchart@ideasonboard.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 */

#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/select.h>

#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <math.h>
#include <linux/usb/ch9.h>
#include <linux/usb/video.h>
#include <linux/videodev2.h>

//#include <turbojpeg.h>

#include "uvc.h"

#include "log_str.h"
#include "scale.h"
#include "stats.h"


#define clamp(val, min, max) ({                 \
        typeof(val) __val = (val);              \
        typeof(min) __min = (min);              \
        typeof(max) __max = (max);              \
        (void) (&__val == &__min);              \
        (void) (&__val == &__max);              \
        __val = __val < __min ? __min: __val;   \
        __val > __max ? __max: __val; })

#define ARRAY_SIZE(a)   ((sizeof(a) / sizeof(a[0])))

#define UVC_CAMERA_TERMINAL_CONTROL_UNIT_ID     (1)
#define UVC_PROCESSING_UNIT_CONTROL_UNIT_ID     (2)

#define WEBCAM_DEVICE_SYS_PATH      "/sys/class/plcm_usb/plcm0/f_webcam/webcam_device"
#define WEBCAM_MAXPACKET_SYS_PATH   "/sys/class/plcm_usb/plcm0/f_webcam/webcam_maxpacket"
#define WEBCAM_HEADERSIZE_SYS_PATH  "/sys/class/plcm_usb/plcm0/f_webcam/webcam_headersize"
#define WEBCAM_BULKMODE_SYS_PATH    "/sys/class/plcm_usb/plcm0/f_webcam/webcam_bulkmode"
#define WEBCAM_MAXPAYLOAD_SYS_PATH  "/sys/class/plcm_usb/plcm0/f_webcam/webcam_maxpayload"

#define JPEG_QUALITY        80
#define COLOR_COMPONENTS    3

#define VISAGE_LED_NOTIFICATION 1

struct uvc_device {
    int fd;

    struct uvc_streaming_control probe;
    struct uvc_streaming_control commit;

    int control;
    int unit;

    unsigned int fcc;
    unsigned int width;
    unsigned int height;

    void **mem;
    unsigned int nbufs;
    unsigned int bufsize;

    unsigned int bulk;
    uint8_t color;
    unsigned int imgsize;
    void *imgdata;

    int v4ldevnum;
    unsigned int maxpacketsize;
    unsigned int headersize;
    unsigned int maxpayloadsize;
};

extern uint8_t * d_in_buffer;
unsigned long startStreamTime=0;
unsigned int frame_count=0;
unsigned int stream_on=0;
/*inline long GetTimeInMicroSec()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    long ret = tv.tv_sec*1000*1000 + tv.tv_usec;
    return ret;
}
inline unsigned long GetTimeInMilliSec()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (tv.tv_sec*1000)+(tv.tv_usec/1000);
}*/
//extern unsigned long GetTimeInUs();

unsigned long frameTime1=0;
unsigned long frameTime2=0;
unsigned long frameDelta=0;
int framePrintFlag=0;

static int
uvc_read_value_from_file(const char* filename,
                         const char* format,
                         int *value)
{
    int fd = 0;
    char buf[8];
    int len;

    fd = open(filename, O_RDONLY);

    if (fd < 0) {
        fprintf(stderr, "Can not open the file: %s, error: %d(%s)\n",
                filename, errno, strerror(errno));
        return -1;
    }

    len = read(fd, buf, sizeof(buf));

    if (len <= 0) {
        fprintf(stderr, "Can not read data from file: %s, error: %d(%s)\n",
                filename, errno, strerror(errno));
        close(fd);
        return -1;
    }

    len = sscanf(buf, format, value);

    if (len <= 0) {
        fprintf(stderr, "Can not parse the value from %s\n", filename);
        close(fd);
        return -1;
    }

    close(fd);
    return 0;
}

static int
uvc_video_init(struct uvc_device *dev)
{
    uvc_read_value_from_file(WEBCAM_DEVICE_SYS_PATH,
                             "%d\n",
                             &dev->v4ldevnum);

    uvc_read_value_from_file(WEBCAM_MAXPACKET_SYS_PATH,
                             "%d\n",
                             (int *)&dev->maxpacketsize);

    uvc_read_value_from_file(WEBCAM_HEADERSIZE_SYS_PATH,
                             "%d\n",
                             (int *)&dev->headersize);

    uvc_read_value_from_file(WEBCAM_BULKMODE_SYS_PATH,
                             "%d\n",
                             (int *)&dev->bulk);

    uvc_read_value_from_file(WEBCAM_MAXPAYLOAD_SYS_PATH,
                             "%x\n",
                             (int *)&dev->maxpayloadsize);

    return 0;
}

static struct uvc_device *
uvc_open(const char *devname)
{
    struct uvc_device *dev;
    struct v4l2_capability cap;
    char v4ldevname[64];
    int ret = -1;
    int fd = 0;

    dev = (struct uvc_device *)malloc(sizeof * dev);

    if (dev == NULL) {
        close(fd);
        return NULL;
    }

    memset(dev, 0, sizeof * dev);
    dev->v4ldevnum = -1;
    dev->maxpacketsize = 1024;
    dev->headersize = 2;
    dev->bulk = 0;

    uvc_video_init(dev);

    if (dev->v4ldevnum != -1) {
        snprintf(v4ldevname, sizeof(v4ldevname), "/dev/video%d", dev->v4ldevnum);
    } else {
        snprintf(v4ldevname, sizeof(v4ldevname), "%s", devname);
    }

    printf("We are trying to open the dev: %s\n", v4ldevname);

    fd = open(v4ldevname, O_RDWR | O_NONBLOCK);

    if (fd == -1) {
        printf("v4l2 open failed: %s (%d)\n", strerror(errno), errno);
        return NULL;
    }

    printf("open succeeded, file descriptor = %d\n", fd);
    ret = ioctl(fd, VIDIOC_QUERYCAP, &cap);

    if (ret < 0) {
        printf("unable to query device: %s (%d)\n", strerror(errno),
               errno);
        close(fd);
        return NULL;
    }

    printf("device is %s on bus %s\n", cap.card, cap.bus_info);

    printf("The config values are as below\n");
    printf("\t\tv4ldevnum: %d\n", dev->v4ldevnum);
    printf("\t\tmaxpacketsize(iso): %d\n", dev->maxpacketsize);
    printf("\t\theadersize: %d\n", dev->headersize);
    printf("\t\tbulkmode: %d\n", dev->bulk);
    printf("\t\tmaxpayloadsize: 0x%x\n", dev->maxpayloadsize);
    dev->fd = fd;
    return dev;
}

static void
uvc_close(struct uvc_device *dev)
{
    close(dev->fd);
    free(dev->imgdata);
    free(dev->mem);
    free(dev);
}

/*******************for camera capture begin*********/
struct cam_buffer {
    unsigned int size;
    void *mem;
};
struct cam_buffer *buffers = NULL;
int fd = -1;
unsigned int nbufs = 10;
unsigned char *pattern = NULL;
unsigned int bSelectIO = 1;
enum v4l2_buf_type buftype = V4L2_BUF_TYPE_VIDEO_CAPTURE;
unsigned int pixelformat = V4L2_PIX_FMT_YUV420;
//unsigned int pixelformat = V4L2_PIX_FMT_YUYV;
enum v4l2_memory memtype = V4L2_MEMORY_MMAP ;
#define CAM_DEF_WIDTH 1280
#define CAM_DEF_HEITHG 720
unsigned int camera_width = CAM_DEF_WIDTH;
unsigned int camera_height = CAM_DEF_HEITHG;
unsigned int bytesperline = CAM_DEF_WIDTH;
unsigned int imagesize = CAM_DEF_WIDTH * CAM_DEF_HEITHG;

#define VISAGE_CAM_DEV "/dev/video2"

int video_queue_buffer(int index)
{
    struct v4l2_buffer buf;
    int ret = 0;
    memset(&buf, 0, sizeof buf);
    buf.index = index;
    buf.type = buftype;
    buf.memory = memtype;
    buf.length = buffers[index].size;

    if (buftype == V4L2_BUF_TYPE_VIDEO_OUTPUT) {
        buf.bytesused = buf.length;

        if (pattern != NULL) {
            //if(bUseOptimizedMemcpy)
            //  my_memcpy(buffers[buf.index].mem, pattern, buf.bytesused);
            //else
            memcpy(buffers[buf.index].mem, pattern, buf.bytesused);
        } else {
            memset(buffers[buf.index].mem, 0x0, buf.bytesused);
        }
    } else {
        //memset(buffers[buf.index].mem, 0x55, buf.length);
    }

	ret = ioctl(fd, VIDIOC_QBUF, &buf);

    if (ret < 0)
        printf("Unable to queue buffer (%d).\n", errno);

    return ret;
}

int  video_dequeue_buffer(struct v4l2_buffer *buf)
{
    int ret = 0;
    memset(buf, 0, sizeof(struct v4l2_buffer));
    buf->type = buftype;
    buf->memory = memtype;
    ret = ioctl(fd, VIDIOC_DQBUF, buf);
    return ret;
}

void video_int_capture(const char * devname)
{
    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(struct v4l2_capability));

    if (bSelectIO)
        fd = open(devname, O_RDWR | O_NONBLOCK);
    else
        fd = open(devname, O_RDWR);

    ioctl(fd, VIDIOC_QUERYCAP, &cap);

    if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) {
        buftype = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        printf("buf type is V4L2_BUF_TYPE_VIDEO_CAPTURE\n");
    } else if (cap.capabilities & V4L2_CAP_VIDEO_OUTPUT) {
        buftype = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        printf("buf type is V4L2_BUF_TYPE_VIDEO_CAPTURE\n");
    } else {
        printf("unknown buf type:%u\n", cap.capabilities);
    }

#if 0

    if (bDump && (buftype == V4L2_BUF_TYPE_VIDEO_CAPTURE))
        fpDump = fopen(szDump, "wb");
    else if (bDump && (buftype == V4L2_CAP_VIDEO_OUTPUT))
        fpDump = fopen(szDump, "rb");

#endif
    return;
}

int  video_set_format(unsigned int capture_width, unsigned int capture_height)
{
    int ret = 0;
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof fmt);
    fmt.type = buftype;
    fmt.fmt.pix.width = capture_width;
    fmt.fmt.pix.height = capture_height;
    fmt.fmt.pix.pixelformat = pixelformat;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    ret =  ioctl(fd, VIDIOC_S_FMT, &fmt);

    if (ret < 0) {
        printf("Unable to set format: %s (%d).\n", strerror(errno), errno);
        return ret;
    }

    printf("Video format set: width: %u height: %u buffer size: %u\n",
           fmt.fmt.pix.width, fmt.fmt.pix.height, fmt.fmt.pix.sizeimage);
    return ret;
}
int video_get_format()
{
    struct v4l2_format fmt;
    int ret;
    memset(&fmt, 0, sizeof fmt);
    fmt.type = buftype;
    ret = ioctl(fd, VIDIOC_G_FMT, &fmt);

    if (ret < 0) {
        printf("Unable to get format: %s (%d).\n", strerror(errno),
               errno);
        return ret;
    }

    camera_width = fmt.fmt.pix.width;
    camera_height = fmt.fmt.pix.height;
    bytesperline = fmt.fmt.pix.bytesperline;
    imagesize =  fmt.fmt.pix.sizeimage ;
    printf("Video format: %c%c%c%c (%08x) , wxh - ->%ux%u, stride --> %u, imgesize--->%u\n",
           (fmt.fmt.pix.pixelformat >> 0) & 0xff,
           (fmt.fmt.pix.pixelformat >> 8) & 0xff,
           (fmt.fmt.pix.pixelformat >> 16) & 0xff,
           (fmt.fmt.pix.pixelformat >> 24) & 0xff,
           fmt.fmt.pix.pixelformat,
           fmt.fmt.pix.width, fmt.fmt.pix.height, bytesperline, imagesize);
    return 0;
}

int video_prepare_capture()
{
    struct v4l2_requestbuffers rb;
    struct v4l2_buffer buf;
    int i  = 0;
    int ret = 0;
    memset(&rb, 0, sizeof rb);
    rb.count = nbufs;
    rb.type = buftype;
    rb.memory = memtype;
    ret = ioctl(fd, VIDIOC_REQBUFS, &rb);
    buffers = (struct cam_buffer *)malloc(rb.count * sizeof(struct cam_buffer));
    memset(buffers, 0x0, rb.count * sizeof(struct cam_buffer));

    for (i = 0; i < (int)rb.count; ++i) {
        memset(&buf, 0, sizeof buf);
        buf.index = i;
        buf.type = buftype;
        buf.memory = memtype;
        ioctl(fd, VIDIOC_QUERYBUF, &buf);
        printf("length: %u offset: %u\n", buf.length, buf.m.offset);
        buffers[i].mem = mmap(0,  buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);

        if (buffers[i].mem == MAP_FAILED) {
            printf("Unable to map buffer %u (%d)\n", i, errno);
            return ret;
        }

        buffers[i].size = buf.length;
        printf("Buffer %u mapped at address %p size %u\n", i, buffers[i].mem, buffers[i].size);
        memset(buffers[i].mem, 0x0, buffers[i].size);
    }

    nbufs = rb.count;

    /*if (buftype == V4L2_BUF_TYPE_VIDEO_OUTPUT) {
    ret = video_allocate_display_buffer(buffers[0].size);
    if (ret < 0)
       return ret;
    }*/
    for (i = 0; i < (int)nbufs; ++i) {
        ret = video_queue_buffer(i);

        if (ret  != 0) break;
    }

    return ret;
}

void video_enable(int enable)
{
    ioctl(fd, enable ? VIDIOC_STREAMON : VIDIOC_STREAMOFF, &buftype);
    return;
}

int start_camera(uint32_t captureWidth, uint32_t captureHeight)
{
    int ret = 0;
    video_int_capture(VISAGE_CAM_DEV);
    ret = video_set_format(captureWidth, captureHeight);

    if (ret != 0) {
        printf("[start_camera]: set format failed: %d\n", ret);
        return ret;
    } else
        printf("[start_camera]: set format success: %d\n", ret);

    ret = video_get_format();

    if (ret != 0) {
        printf("[start_camera]: get format failed: %d\n", ret);
        return ret;
    } else
        printf("[start_camera]: get format success: %d\n", ret);

    ret = video_prepare_capture();

    if (ret != 0) {
        printf("[start_camera]: prepare capture failed: %d\n", ret);
        return ret;
    } else
        printf("[start_camera]: prepare capture success: %d\n", ret);

    video_enable(1);
    return 0;
}

void stop_camera()
{
    video_enable(0);
}
#define LEN_720P_I420 (1280*720*3/2)
#define LEN_720P_YUYV (1280*720*2)
#define LEN_1080P_YUYV (1920*1080*2)

#define VBUF_LEN LEN_1080P_YUYV
uint8_t vBuf[VBUF_LEN];
//char vBuf[LEN_1080P_YUYV];


int read_one_camera_frame(void * buffer, unsigned int bufferLen, unsigned int * readLen)
{
    struct v4l2_buffer buf;
    int ret = 0;
	int op_ret=0;
	unsigned int count = 0;
#if 0 
	if (bufferLen==0) {
		printf("[read_one_camera_frame]: bufferlen=0\n");
		return -1;
	}
	memset(buffer, (frame_count*4)&0xff, bufferLen);
	*readLen=bufferLen;
	return 0;
#else
    //do
    {
        op_ret = video_dequeue_buffer(&buf);

        if (op_ret < 0) {
            if (errno == EAGAIN) {
                //printf("[%d]Got EAGAIN!! [%u] %u\n", __LINE__, buf.index, count);
            }

            else {// if (errno != EIO) {
                //printf("Unable to dequeue buffer [%u] (%d) (%s).count=%u\n",buf.index, errno, strerror(errno), count);
            }

            buf.type = buftype;
            buf.memory = memtype;
            ret = -1;
        } else {
            if (buf.bytesused > bufferLen) {
				printf("ERROR! buffer not enough: len=%u, need %u\n", bufferLen, buf.bytesused);
                ret = -1;
            } else {
               //printf("Successfully read from camera: len=%u, need %u, buf.index=%u count=%u\n", bufferLen, buf.bytesused, buf.index, count);
                //memcpy(buffer, buffers[buf.index].mem, buf.bytesused);
		cudaMemcpy(d_in_buffer, buffers[buf.index].mem, buf.bytesused, cudaMemcpyHostToDevice);
                *readLen = buf.bytesused;
            }

            op_ret=video_queue_buffer(buf.index);
			if (op_ret < 0) {
				printf("Unable to queue buffer (%d) (%s).count=%u [%u]\n", errno, strerror(errno), count, buf.index);
			}
			count=30;
        }

        count++;
    }//while(count <30);
	count=0;
    return ret;
#endif
}


/*******************for camera capture end*********/

/* ---------------------------------------------------------------------------
 * Video streaming
 */

unsigned long capTime1=0;
unsigned long capDelta=0;
unsigned long imgTime1=0;
unsigned long imgDelta=0;
static void
uvc_video_fill_buffer(struct uvc_device *dev, struct v4l2_buffer *buf)
{
    //unsigned int bpl;
    //unsigned int i;
    int ret;
    unsigned int cam_read_len = 0;
    unsigned int act_len = 0;
    //unsigned char *rgbraw;
    //long unsigned int jpegSize = 0;
    //tjhandle _jpegCompressor;
    //unsigned char *jpegBuffer = NULL;

    switch (dev->fcc) {
        case V4L2_PIX_FMT_YUYV:
            ret = read_one_camera_frame(vBuf, VBUF_LEN, &cam_read_len);

            if (ret == 0) {
				act_len = (dev->width) * (dev->height) * 2;
                if (dev->width == camera_width) {
                    NV12toYUY2(vBuf, dev->width, dev->height, (uint8_t*)(dev->mem[buf->index]), dev->width, dev->height);
                } else {
                    NV12toYUY2scale(vBuf, camera_width, camera_height, (uint8_t*)(dev->mem[buf->index]), dev->width, dev->height);
                }
                buf->bytesused = act_len;
            } else {
            	//printf("fill_buffer[YUYV]: failed to get image from camera\n");
                buf->bytesused = 0;
            }

			if (frame_count%(100)==0)
				printf("sending data for YUYV: len=%u, ret=%u, %u x %u framecount=%u\n", buf->bytesused, ret, dev->width, dev->height, frame_count);
            break;

        case V4L2_PIX_FMT_YUV420:
	    capTime1=getTimeInMicroS();
            ret = read_one_camera_frame(vBuf, VBUF_LEN, &cam_read_len);
	    //capDelta+=getTimeInMicroS();
		sumFrameStats(capTime1, &capDelta, frame_count, "capture");
		
	    
	//if (frame_count%(P_COUNT)==0) {
		//printf("capture %u frame takes %lums\n", frame_count, capDelta/1000/1000);
		//capDelta=0;
	//}
	imgTime1=getTimeInMicroS();
            if (ret == 0) {
				act_len = (dev->width) * (dev->height) * 3 / 2;
                if (dev->width == camera_width) {
                    NV12toI420(vBuf, dev->width, dev->height, (uint8_t*)(dev->mem[buf->index]), dev->width, dev->height);
                } else {
                    NV12toI420scale(vBuf, camera_width, camera_height, (uint8_t*)(dev->mem[buf->index]), dev->width, dev->height);
                }
                buf->bytesused = act_len;
            } else {
            	//printf("fill_bufferp[I420]: failed to get image from camera\n");
                buf->bytesused = 0;
            }
		sumFrameStats(imgTime1, &imgDelta, frame_count, "imaging");
		//imgDelta+=getTimeInMicroS()-imgTime1; 
		//if (frame_count%P_COUNT==0) {
		//	printf("imaging %u frames take %lu ms\n", frame_count, imgDelta/1000/1000);
		//	imgDelta=0;
		//}

			if (frame_count%(100)==0)
				printf("sending data for YUV420: len=%u, ret=%u, %u x %u framecount=%u\n", buf->bytesused, ret, dev->width, dev->height, frame_count);
            break;

        case V4L2_PIX_FMT_NV12:
            ret = read_one_camera_frame(dev->mem[buf->index], buf->length, &(buf->bytesused));

            if (ret)
                buf->bytesused = 0;

			if (frame_count%(30*10)==0)
				printf("sending data for NV12: len=%u, ret=%u\n", buf->bytesused, ret);
            break;

        case V4L2_PIX_FMT_MJPEG:
#if 0
            ret = read_one_camera_frame(vBuf, VBUF_LEN, &cam_read_len);

            if (ret != 0) {
                buf->bytesused = 0;
                return;
            }

            /* Allocate the rgba raw buffer */
            rgbraw = calloc(dev->height * dev->width * COLOR_COMPONENTS, 1);

            if (!rgbraw) {
                buf->bytesused = 0;
                return;
            }

            NV12toRGBA((unsigned char *)vBuf, dev->width, dev->height, rgbraw);

            /* Start to compress */
            _jpegCompressor = tjInitCompress();

            tjCompress2(_jpegCompressor, rgbraw, dev->width, 0, dev->height, TJPF_RGB,
                        &jpegBuffer, &jpegSize, TJSAMP_444, JPEG_QUALITY, TJFLAG_FASTDCT);

            tjDestroy(_jpegCompressor);

            free(rgbraw);

            /* Copy to uvc buffer. */
            memcpy(dev->mem[buf->index], jpegBuffer, jpegSize);

            buf->bytesused = jpegSize;

            tjFree(jpegBuffer);
			if (frame_count%(10*10)==0)
				printf("sending data for MJPEG: len=%u, ret=%u, %u x %u\n", buf->bytesused, ret,dev->width, dev->height);
			frame_count++;

#endif
            buf->bytesused = 0;
            break;
    }
}

static int
uvc_video_process(struct uvc_device *dev)
{
    struct v4l2_buffer buf;
    int ret;
    memset(&buf, 0, sizeof buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    buf.memory = V4L2_MEMORY_MMAP;

    if ((ret = ioctl(dev->fd, VIDIOC_DQBUF, &buf)) < 0) {
#if 1
	if (stream_on)
        printf("Unable to dequeue buffer: %s (%d).\n", strerror(errno),
               errno);
#endif
        return ret;
    }
	if (framePrintFlag==0) {
		printf("record start time at %u frames \n", frame_count);
		frameTime2=getTimeInMicroS();
		framePrintFlag=1;
	}
	frameTime1=getTimeInMicroS();
    uvc_video_fill_buffer(dev, &buf);
	sumFrameStats(frameTime1, &frameDelta, frame_count, "process");
	//frameDelta+=getTimeInMicroS() - frameTime1;
	//if (frame_count%P_COUNT==0) {
	//	printf("process %u frames take %lums\n", frame_count, frameDelta/1000/1000);
	//	frameDelta=0;
	//}
	if (frame_count%P_COUNT==0) {
		unsigned long curTime=getTimeInMicroS();
		printf("receive %u frames request take %lums\n", frame_count, (curTime-frameTime2)/1000);
		framePrintFlag=0;
	}
	frame_count++;

    if ((ret = ioctl(dev->fd, VIDIOC_QBUF, &buf)) < 0) {
#if 1 
	if (stream_on)
        printf("Unable to requeue buffer: %s (%d).\n", strerror(errno),
               errno);
#endif
        return ret;
    }

    return 0;
}

static int
uvc_video_reqbufs(struct uvc_device *dev, int nbufs)
{
    struct v4l2_requestbuffers rb;
    struct v4l2_buffer buf;
    unsigned int i;
    int ret;

    for (i = 0; i < dev->nbufs; ++i)
        munmap(dev->mem[i], dev->bufsize);

    free(dev->mem);
    dev->mem = 0;
    dev->nbufs = 0;

    if (nbufs <= 0)
        return 0;

    memset(&rb, 0, sizeof rb);
    rb.count = nbufs;
    rb.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    rb.memory = V4L2_MEMORY_MMAP;
    ret = ioctl(dev->fd, VIDIOC_REQBUFS, &rb);

    if (ret < 0) {
        printf("Unable to allocate buffers: %s (%d).\n",
               strerror(errno), errno);
        return ret;
    }

    printf("%u buffers allocated.\n", rb.count);
    /* Map the buffers. */
    dev->mem = (void**)malloc(rb.count * sizeof dev->mem[0]);

    for (i = 0; i < rb.count; ++i) {
        memset(&buf, 0, sizeof buf);
        buf.index = i;
        buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        buf.memory = V4L2_MEMORY_MMAP;
        ret = ioctl(dev->fd, VIDIOC_QUERYBUF, &buf);

        if (ret < 0) {
            printf("Unable to query buffer %u: %s (%d).\n", i,
                   strerror(errno), errno);
            return -1;
        }

        printf("length: %u offset: %u\n", buf.length, buf.m.offset);
        dev->mem[i] = mmap(0, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd, buf.m.offset);

        if (dev->mem[i] == MAP_FAILED) {
            printf("Unable to map buffer %u: %s (%d)\n", i,
                   strerror(errno), errno);
            return -1;
        }

        printf("Buffer %u mapped at address %p.\n", i, dev->mem[i]);
    }

    dev->bufsize = buf.length;
    dev->nbufs = rb.count;
    return 0;
}

static int
uvc_video_stream(struct uvc_device *dev, int enable)
{
    struct v4l2_buffer buf;
    unsigned int i;
    int type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    int ret;

    if (!enable) {
        printf("Stopping video stream.\n");
        ioctl(dev->fd, VIDIOC_STREAMOFF, &type);
#ifdef VISAGE_LED_NOTIFICATION
        /* Set the led to red. */
        system("/usr/sbin/commanduC lightLed 1 1");
#endif
	frame_count=0;
	stream_on=0;
        return 0;
    }
    stream_on=1;

    printf("Starting video stream.\n");

    for (i = 0; i < dev->nbufs; ++i) {
        memset(&buf, 0, sizeof buf);
        buf.index = i;
        buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
        buf.memory = V4L2_MEMORY_MMAP;
        uvc_video_fill_buffer(dev, &buf);
        printf("Queueing buffer %u.\n", i);

        if ((ret = ioctl(dev->fd, VIDIOC_QBUF, &buf)) < 0) {
            printf("Unable to queue buffer: %s (%d).\n",
                   strerror(errno), errno);
            break;
        }
    }

    ioctl(dev->fd, VIDIOC_STREAMON, &type);
#ifdef VISAGE_LED_NOTIFICATION
        /* If the host selects the I420 30fp format, we
         * will set the led to Green, Fast blink.
         * Else set the led to Amber, Slow blink.
         */
        if (dev->fcc == V4L2_PIX_FMT_YUV420 &&
            dev->width == 1280 &&
            dev->height == 720)
            system("/usr/sbin/commanduC lightLed 2 3");
        else
            system("/usr/sbin/commanduC lightLed 4 2");
#endif
    return ret;
}

static int
uvc_video_set_format(struct uvc_device *dev)
{
    struct v4l2_format fmt;
    int ret;
    printf("Setting format to 0x%08x[%s] %ux%u\n",
           dev->fcc, getV4L2FormatStr(dev->fcc), dev->width, dev->height);
    memset(&fmt, 0, sizeof fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.fmt.pix.width = dev->width;
    fmt.fmt.pix.height = dev->height;
    fmt.fmt.pix.pixelformat = dev->fcc;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (dev->fcc == V4L2_PIX_FMT_MJPEG) {
        fmt.fmt.pix.sizeimage = dev->width * dev->height * 1.5;
    }

    if ((ret = ioctl(dev->fd, VIDIOC_S_FMT, &fmt)) < 0)
        printf("Unable to set format: %s (%d).\n",
               strerror(errno), errno);

    return ret;
}

/* ---------------------------------------------------------------------------
 * Request processing
 */

struct uvc_frame_info {
    unsigned int width;
    unsigned int height;
    unsigned int intervals[8];
};

struct uvc_format_info {
    unsigned int fcc;
    const struct uvc_frame_info *frames;
};

static const struct uvc_frame_info uvc_frames_yuyv[] = {
    {  640, 360, { 666666, 10000000, 50000000, 0 }, },
    { 1280, 720, { 50000000, 0 }, },
    { 0, 0, { 0, }, },
};

static const struct uvc_frame_info uvc_frames_i420[] = {
    {  640, 360, { 333333, 666666, 10000000, 50000000, 0 }, },
    { 1280, 720, { 333333, 666666, 10000000, 50000000, 0 }, },
    { 0, 0, { 0, }, },
};

static const struct uvc_frame_info uvc_frames_mjpeg[] = {
    {  640, 360, { 666666, 10000000, 50000000, 0 }, },
    { 1280, 720, { 50000000, 0 }, },
	//{ 1920, 1080, { 333333, 666666, 50000000, 0 }, },
    { 0, 0, { 0, }, },
};

static const struct uvc_format_info uvc_formats[] = {
    { V4L2_PIX_FMT_YUYV, uvc_frames_yuyv },
    { V4L2_PIX_FMT_YUV420, uvc_frames_i420 },
    //{ V4L2_PIX_FMT_NV12, uvc_frames_i420 },
    { V4L2_PIX_FMT_MJPEG, uvc_frames_mjpeg },
};

static void
uvc_fill_streaming_control(struct uvc_device *dev,
                           struct uvc_streaming_control *ctrl,
                           int iframe, int iformat)
{
    const struct uvc_format_info *format;
    const struct uvc_frame_info *frame;
    unsigned int nframes;

    if (iformat < 0)
        iformat = ARRAY_SIZE(uvc_formats) + iformat;

    if (iformat < 0 || iformat >= (int)ARRAY_SIZE(uvc_formats))
        return;

    format = &uvc_formats[iformat];
    nframes = 0;

    while (format->frames[nframes].width != 0)
        ++nframes;

    if (iframe < 0)
        iframe = nframes + iframe;

    if (iframe < 0 || iframe >= (int)nframes)
        return;

    frame = &format->frames[iframe];
    printf("uvc_fill_streaming_control:fcc=0x%x, Format=%s, iformat=0x%x, iframe=0x%x\n", format->fcc, getV4L2FormatStr(format->fcc), iformat, iframe);
    memset(ctrl, 0, sizeof * ctrl);
    ctrl->bmHint = 1;
    ctrl->bFormatIndex = iformat + 1;
    ctrl->bFrameIndex = iframe + 1;
    ctrl->dwFrameInterval = frame->intervals[0];

    switch (format->fcc) {
        case V4L2_PIX_FMT_YUYV:
            ctrl->dwMaxVideoFrameSize = frame->width * frame->height * 2;
            break;

        case V4L2_PIX_FMT_YUV420:
        case V4L2_PIX_FMT_NV12:
            ctrl->dwMaxVideoFrameSize = frame->width * frame->height *  3 / 2;
            break;

        case V4L2_PIX_FMT_MJPEG:
            ctrl->dwMaxVideoFrameSize = frame->width * frame->height * 3 / 2;
            break;
    }

    if (!dev->bulk)
        ctrl->dwMaxPayloadTransferSize = dev->maxpacketsize;   /* TODO this should be filled by the driver. */

    ctrl->bmFramingInfo = 3;
    ctrl->bPreferedVersion = 1;
    ctrl->bMaxVersion = 1;
}

static void
uvc_events_process_standard(struct uvc_device *dev, struct usb_ctrlrequest *ctrl,
                            struct uvc_request_data *resp)
{
    printf("standard request\n");
    (void)dev;
    (void)ctrl;
    resp->length = 0;
}

__u16 brightness = 0x0004;

static void
uvc_events_process_control(struct uvc_device *dev, uint8_t req, uint8_t cs,
                           uint8_t unit_id, struct uvc_request_data *resp)
{
#if 0
    printf("control request (req %02x cs %02x)\n", req, cs);
    (void)dev;
    resp->length = 0;
#else
    __u16 *wValuePtr = (__u16 *)(resp->data);
    printf("control request (req %02x cs %02x)\n", req, cs);
    (void)dev;

    switch (cs) {
        case UVC_PU_BRIGHTNESS_CONTROL:
            switch (req) {
                case UVC_GET_INFO:
                    resp->data[0] = 0x03;
                    resp->length = 1;
                    break;

                case UVC_GET_MIN:
                    *wValuePtr = 0x0000;
                    resp->length = 2;
                    break;

                case UVC_GET_MAX:
                    *wValuePtr = 0x0009;
                    resp->length = 2;
                    break;

                case UVC_GET_RES:
                    *wValuePtr = 0x0001;
                    resp->length = 2;
                    break;

                case UVC_GET_DEF:
                    *wValuePtr = 0x0004;
                    resp->length = 2;
                    break;

                case UVC_GET_CUR:
                    *wValuePtr = brightness;
                    resp->length = 2;
                    break;

                case UVC_GET_LEN:
                    *wValuePtr = sizeof(brightness);
                    resp->length = 2;
                    break;

                case UVC_SET_CUR:
                    //TODO
                    dev->control = cs;
                    dev->unit = unit_id;
                    resp->length = 2;
                    break;

                default:
                    resp->length = 0;
                    break;
            }

            break;

        default:
            resp->length = 0;
            break;
    }

#endif
}

static void
uvc_events_process_streaming(struct uvc_device *dev, uint8_t req, uint8_t cs,
                             struct uvc_request_data *resp)
{
    struct uvc_streaming_control *ctrl;
    printf("streaming request (req 0x%02x cs 0x%02x, ) %s\n", req, cs,  getVSIntfControlSelectorStr(cs));

    if (cs != UVC_VS_PROBE_CONTROL && cs != UVC_VS_COMMIT_CONTROL)
        return;

    ctrl = (struct uvc_streaming_control *)&resp->data;
    resp->length = sizeof * ctrl;
    printf("UVC OPS  %s resp->length=%u\n", getUVCOpStr(req), resp->length);

    switch (req) {
        case UVC_SET_CUR:
            dev->control = cs;
            dev->unit = 0;
            resp->length = 34;
            break;

        case UVC_GET_CUR:
            if (cs == UVC_VS_PROBE_CONTROL)
                memcpy(ctrl, &dev->probe, sizeof * ctrl);
            else
                memcpy(ctrl, &dev->commit, sizeof * ctrl);

            break;

        case UVC_GET_MIN:
        case UVC_GET_MAX:
        case UVC_GET_DEF:
            uvc_fill_streaming_control(dev, ctrl, req == UVC_GET_MAX ? -1 : 0,
                                       req == UVC_GET_MAX ? -1 : 0);
            break;

        case UVC_GET_RES:
            memset(ctrl, 0, sizeof * ctrl);
            break;

        case UVC_GET_LEN:
            resp->data[0] = 0x00;
            resp->data[1] = 0x22;
            resp->length = 2;
            break;

        case UVC_GET_INFO:
            resp->data[0] = 0x03;
            resp->length = 1;
            break;
    }
}

static void
uvc_events_process_class(struct uvc_device *dev, struct usb_ctrlrequest *ctrl,
                         struct uvc_request_data *resp)
{
    if ((ctrl->bRequestType & USB_RECIP_MASK) != USB_RECIP_INTERFACE)
        return;

#if 0

    switch (ctrl->wIndex & 0xff) {
        case UVC_INTF_CONTROL:
            uvc_events_process_control(dev, ctrl->bRequest, ctrl->wValue >> 8, ctrl->wIndex >> 8, resp);
            break;

        case UVC_INTF_STREAMING:
            uvc_events_process_streaming(dev, ctrl->bRequest, ctrl->wValue >> 8, resp);
            break;

        default:
            break;
    }

#else

    if ((ctrl->wIndex >> 8) & 0xff) {
        //has unit id. Control event
        uvc_events_process_control(dev, ctrl->bRequest, ctrl->wValue >> 8, ctrl->wIndex >> 8, resp);
    } else {
        uvc_events_process_streaming(dev, ctrl->bRequest, ctrl->wValue >> 8, resp);
    }

#endif
}

static void
uvc_events_process_setup(struct uvc_device *dev, struct usb_ctrlrequest *ctrl,
                         struct uvc_request_data *resp)
{
    dev->control = 0;
    printf("bRequestType %02x bRequest %02x wValue %04x wIndex %04x "
           "wLength %04x\n", ctrl->bRequestType, ctrl->bRequest,
           ctrl->wValue, ctrl->wIndex, ctrl->wLength);

    switch (ctrl->bRequestType & USB_TYPE_MASK) {
        case USB_TYPE_STANDARD:
            uvc_events_process_standard(dev, ctrl, resp);
            break;

        case USB_TYPE_CLASS:
            uvc_events_process_class(dev, ctrl, resp);
            break;

        default:
            printf("Unhandled bRequestType %02x bRequest %02x wValue %04x wIndex %04x "
                   "wLength %04x\n", ctrl->bRequestType, ctrl->bRequest,
                   ctrl->wValue, ctrl->wIndex, ctrl->wLength);
            break;
    }
}

static void
uvc_events_process_data(struct uvc_device *dev, struct uvc_request_data *data)
{
    struct uvc_streaming_control *target;
    struct uvc_streaming_control *ctrl;
    const struct uvc_format_info *format;
    const struct uvc_frame_info *frame;
    const unsigned int *interval;
    unsigned int iformat, iframe;
    unsigned int nframes;

    switch (((dev->unit) << 8) | dev->control) {
        case UVC_VS_PROBE_CONTROL:
            printf("setting probe control, length = %d\n", data->length);
            target = &dev->probe;
            break;

        case UVC_VS_COMMIT_CONTROL:
            printf("setting commit control, length = %d\n", data->length);
            target = &dev->commit;
            break;

        case UVC_PROCESSING_UNIT_CONTROL_UNIT_ID << 8 | UVC_PU_BRIGHTNESS_CONTROL:
            printf("setting UVC_PU_BRIGHTNESS_CONTROL, length = %d\n", data->length);
            brightness = *(__u16 *)(data->data);
            return;

        default:
            printf("setting unknown control, length = %d\n", data->length);
            return;
    }

    ctrl = (struct uvc_streaming_control *)&data->data;
    iformat = clamp((unsigned int)ctrl->bFormatIndex, 1U,
                    (unsigned int)ARRAY_SIZE(uvc_formats));
    format = &uvc_formats[iformat - 1];
    nframes = 0;

    while (format->frames[nframes].width != 0)
        ++nframes;

    iframe = clamp((unsigned int)ctrl->bFrameIndex, 1U, nframes);
    frame = &format->frames[iframe - 1];
    interval = frame->intervals;

    while (interval[0] < ctrl->dwFrameInterval && interval[1])
        ++interval;

    target->bFormatIndex = iformat;
    target->bFrameIndex = iframe;

    switch (format->fcc) {
        case V4L2_PIX_FMT_YUYV:
            target->dwMaxVideoFrameSize = frame->width * frame->height * 2;
            break;

        case V4L2_PIX_FMT_YUV420:
        case V4L2_PIX_FMT_NV12:
            target->dwMaxVideoFrameSize = frame->width * frame->height *  3 / 2;
            break;

        case V4L2_PIX_FMT_MJPEG:
            target->dwMaxVideoFrameSize = frame->width * frame->height;
            target->wCompQuality = JPEG_QUALITY;
            break;
    }

    target->dwFrameInterval = *interval;

    if (dev->control == UVC_VS_COMMIT_CONTROL) {
        dev->fcc = format->fcc;
        dev->width = frame->width;
        dev->height = frame->height;
        uvc_video_set_format(dev);

        if (dev->bulk) {
            /* In bulk mode, we can receive the set_alt
             * request from the host. That means no STREAM_ON
             * or STREAM_OFF event from the gadget driver. So
             * we need to start the transfer immediatly after
             * receiving the COMMIT Set_CUR.
             */
            //Cancel the alarm. Stop the stream if possible
            alarm(0);
            uvc_video_stream(dev, 0);
            uvc_video_reqbufs(dev, 0);

            uvc_video_reqbufs(dev, 4);
            uvc_video_stream(dev, 1);
            alarm(2);
        }
    }
}

static void
uvc_events_process(struct uvc_device *dev)
{
    struct v4l2_event v4l2_event;
    struct uvc_event *uvcEvent = (uvc_event *)&v4l2_event.u.data;
    struct uvc_request_data resp;
    struct usb_ctrlrequest *ctrl_req = &uvcEvent->req;
    int ret;
    ret = ioctl(dev->fd, VIDIOC_DQEVENT, &v4l2_event);

    if (ret < 0) {
        printf("VIDIOC_DQEVENT failed: %s (%d)\n", strerror(errno),
               errno);
        return;
    }

    memset(&resp, 0, sizeof resp);
    resp.length = -EL2HLT;
    //intf=(ctrl_req->wIndex&0xff);
    printf("[BEGIN]uvc_events_process: Receive V4L2 evnet [0x%x] %s\n", v4l2_event.type, getUVCEventStr(v4l2_event.type));

    switch (v4l2_event.type) {
        case UVC_EVENT_CONNECT:
        case UVC_EVENT_DISCONNECT:
            return;

        case UVC_EVENT_SETUP:
            printf("bRequestType %02x bRequest %02x wValue %04x wIndex %04x wLength %04x [intf=]\n",
                   ctrl_req->bRequestType, ctrl_req->bRequest,
                   ctrl_req->wValue, ctrl_req->wIndex, ctrl_req->wLength);
            printHexData((char *)ctrl_req, sizeof(*ctrl_req), getUVCEventStr(v4l2_event.type));
            uvc_events_process_setup(dev, ctrl_req, &resp);
            break;

        case UVC_EVENT_DATA:
            printHexData((char *)&uvcEvent->data, sizeof(uvcEvent->data), getUVCEventStr(v4l2_event.type));
            uvc_events_process_data(dev, &uvcEvent->data);
            return;

        case UVC_EVENT_STREAMON:
            uvc_video_reqbufs(dev, 4);
            uvc_video_stream(dev, 1);
            break;

        case UVC_EVENT_STREAMOFF:
            // Cacel the alarm.
            alarm(0);
            uvc_video_stream(dev, 0);
            uvc_video_reqbufs(dev, 0);
            break;
    }

    //Ignore the SET_CUR event. Because we have triggle the transfer
    //in the kernel side.
    if ((ctrl_req->bRequestType & USB_DIR_IN) != 0 ||
            (ctrl_req->bRequestType & USB_RECIP_MASK) != USB_RECIP_INTERFACE ||
            ctrl_req->bRequest != UVC_SET_CUR) {
        ioctl(dev->fd, UVCIOC_SEND_RESPONSE, &resp);

        if (ret < 0) {
            printf("UVCIOC_S_EVENT failed: %s (%d)\n", strerror(errno),
                   errno);
            return;
        }
    }

    printHexData((char *)(&resp), sizeof(resp), "REPLY");
    printf("[END]uvc_events_process: DONE with V4L2 event [0x%x] %s\n", v4l2_event.type, getUVCEventStr(v4l2_event.type));
}

static void
uvc_events_init(struct uvc_device *dev)
{
    struct v4l2_event_subscription sub;
    uvc_fill_streaming_control(dev, &dev->probe, 0, 0);
    uvc_fill_streaming_control(dev, &dev->commit, 0, 0);

    if (dev->bulk) {
        /* FIXME Crude hack, must be negotiated with the driver. */
        dev->probe.dwMaxPayloadTransferSize = dev->maxpayloadsize;
        dev->commit.dwMaxPayloadTransferSize = dev->maxpayloadsize;
    }

    memset(&sub, 0, sizeof sub);
    sub.type = UVC_EVENT_SETUP;
    ioctl(dev->fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
    sub.type = UVC_EVENT_DATA;
    ioctl(dev->fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
    sub.type = UVC_EVENT_STREAMON;
    ioctl(dev->fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
    sub.type = UVC_EVENT_STREAMOFF;
    ioctl(dev->fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
}

/* ---------------------------------------------------------------------------
 * main
 */

static void image_load(struct uvc_device *dev, const char *img)
{
    int fd = -1;

    if (img == NULL)
        return;

    fd = open(img, O_RDONLY);

    if (fd == -1) {
        printf("Unable to open MJPEG image '%s'\n", img);
        return;
    }

    dev->imgsize = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    dev->imgdata = malloc(dev->imgsize);

    if (dev->imgdata == NULL) {
        printf("Unable to allocate memory for MJPEG image\n");
        dev->imgsize = 0;
        return;
    }

    read(fd, dev->imgdata, dev->imgsize);
    close(fd);
}

static void usage(const char *argv0)
{
    fprintf(stderr, "Usage: %s [options]\n", argv0);
    fprintf(stderr, "Available options are\n");
    fprintf(stderr, " -d device	Video device\n");
    fprintf(stderr, " -h		Print this help screen and exit\n");
    fprintf(stderr, " -i image	MJPEG image\n");
}

static struct uvc_device *global_uvc = NULL;

void sig_handle(int sig)
{
    printf("Received signal: %d(%s)\n", sig, strsignal(sig));

    // Alarm timeout. Stop the video
    if (sig != SIGALRM)
        return;

    if (global_uvc) {
        uvc_video_stream(global_uvc, 0);
        uvc_video_reqbufs(global_uvc, 0);
    }
}

int main(int argc, char *argv[])
{
    char *device = "/dev/video0";
    struct uvc_device *dev;
    char *mjpeg_image = NULL;
    fd_set fds;
    int ret, opt;

    while ((opt = getopt(argc, argv, "d:hi:k:g:")) != -1) {
        switch (opt) {
            case 'd':
                device = optarg;
                break;

            case 'h':
                usage(argv[0]);
                return 0;

            case 'i':
                mjpeg_image = optarg;
                break;
			case 'k':
                camera_width = atoi(optarg);
				bytesperline=camera_width;
                break;
			case 'g':
                camera_height = atoi(optarg);
                break;

            default:
                fprintf(stderr, "Invalid option '-%c'\n", opt);
                usage(argv[0]);
                return 1;
        }
    }
	create_scaler_thread(ROW_NUM, COLUMN_NUM);
	//return;
    /*
     * Setup the signal handler for SIGALRM. It is used
     * for the bulk transfer mode. Because in bulk mode,
     * the driver will not send the STREAM_OFF event when
     * the host stops the video stream. We need to have a
     * timer that if we can not receive the video frame
     * transfer done event in 1 second. We will stop the
     * video and clean the buffer.
     */
	imagesize=camera_width*camera_height;
    signal(SIGALRM, sig_handle);

    /* load camera */
    start_camera(camera_width, camera_height);
    //read_camera();
    //stop_camera();
    //return 0;
    /* load camera end */
    dev = uvc_open(device);

    if (dev == NULL)
        return 1;

    global_uvc = dev;

    image_load(dev, mjpeg_image);
    uvc_events_init(dev);
    FD_ZERO(&fds);
    FD_SET(dev->fd, &fds);

    while (1) {
        fd_set efds = fds;
        fd_set wfds = fds;
        ret = select(dev->fd + 1, NULL, &wfds, &efds, NULL);

        if (ret == -1) {
            if (errno != EINTR) {
                printf("Error in select\n");
                break;
            }
        } else {
            if (FD_ISSET(dev->fd, &efds))
                uvc_events_process(dev);

            if (FD_ISSET(dev->fd, &wfds)) {
                uvc_video_process(dev);
                if (dev->bulk) {
                    // Reset the alarm.
                    alarm(1);
                }
            }
        }
    }

    uvc_close(dev);
    stop_camera();
    return 0;
}
