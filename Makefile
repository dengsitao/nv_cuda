#KERNEL_DIR	?= ../../../kernel_source
#VISAGE_SOURCE_DIR ?= /home/rdeng/code/usb_byoc/l4t_21_3_old
#CROSS_INC_PATH ?= $(VISAGE_SOURCE_DIR)/build-output/_rootfs_/usr/include/
#CROSS_LIB_PATH ?= $(VISAGE_SOURCE_DIR)/build-output/_rootfs_/usr/lib/

CC		:= nvcc
#KERNEL_INCLUDE	:= -I$(KERNEL_DIR)/include -I$(KERNEL_DIR)/arch/$(ARCH)/include
KERNEL_INCLUDE	:=
CFLAGS		:= -g 
LDFLAGS		:= -lm
SOURCE_FILE	:= uvc-gadget.cu \
		scale.cu \

OBJ_FILE	:= $(patsubst %.c,%.o,$(SOURCE_FILE))

TARGET		:=gs_gpu

all: $(TARGET)

$(TARGET): $(OBJ_FILE)
	echo $(OBJ_FILE)
	$(CC) -o $@ $(SOURCE_FILE) $(CFLAGS)  $(LDFLAGS)

clean:
	rm -f *.o
	rm -f $(TARGET)
