import numpy as np
import cv2
import struct
import time
import os
import sys

VBG_ERR_IMG_RETRIEVAL = 1
VBG_ERR_IMG_SIZE = 2
VBG_ERR_DATA_OUT_OF_BOUNDS = 3
VBG_ERR_INVALID_FILE = 4
VBG_MAX_DATA_VALUE = 127
VBG_MAX_IMG_SIZE = 65535
VBG_MAX_QUALITY = 100.0
VBG_MIN_DATA_VALUE = -128
VBG_VERSION = 1.2

def VBG(filename, block_height, block_width, quality):
    # Get the image
    print "Acquiring image..."
    img = get_image(filename)
    
    # Break the image into blocks
    print "Dividing image..."
    start_time = time.time()
    block_data, num_blocks_high, num_blocks_wide = divide_image(img,
                                                   block_height,
                                                   block_width)
    
    # Perform the DCT
    print "Applying DCT..."
    dct_data = dct_blocks(block_data)
    
    # Quantize the DCT
    print "Quantizing data..."
    quant_data, quality_modifier = quantize(dct_data, quality)
    
    # Encode the blocks
    print "Encoding data..."
    vbg_data, vbg_meta = VBG_encode(quant_data, quality,
                                    quality_modifier, num_blocks_high,
                                    num_blocks_wide)
    
    # Write the data
    print "Writing data to file..."
    write_data(filename, vbg_data, vbg_meta, np.shape(img)[0],
               np.shape(img)[1])
    time_to_compress = time.time() - start_time
    
    # Report stats
    print "Adding data to trials.csv"
    return report_stats(filename, img, time_to_compress)
    
def get_image(filename):
    img = cv2.imread("input/" + filename + ".bmp", 0)
    try:
        h,w = np.shape(img)[:2]
    except (ValueError):
        print "ERROR: Retrieval of specified file failed. Check to see that the file exists in the current directory.\n\nVBG_ERR_IMG_RETRIEVAL"
        sys.exit(VBG_ERR_IMG_RETRIEVAL)
    if (w > VBG_MAX_IMG_SIZE) or (h > VBG_MAX_IMG_SIZE):
        print "ERROR: At least one dimension of image is larger than" + str(VBG_MAX_IMG_SIZE) + " pixels. VBG" + str(VBG_VERSION) + " does not support images greaterthan this size.\n\nVBG_ERR_IMG_SIZE"
        sys.exit(VBG_ERR_IMG_SIZE)
    return img
    
def divide_image(img, block_height, block_width):
    height,width = np.shape(img)[:2]
    edge_pixels_wide = width % block_width
    edge_pixels_high = height % block_height
    if edge_pixels_high != 0: num_blocks_high = height/block_height + 1
    else: num_blocks_high = height/block_height
    if edge_pixels_wide != 0: num_blocks_wide = width/block_width + 1
    else: num_blocks_wide = width/block_height
    block_data = np.zeros((block_height,block_width))
    for i in range(num_blocks_high):
        for j in range(num_blocks_wide):
            block = img[(block_height * i):(block_height * (i + 1)),
                        (block_width * j):(block_width * (j + 1))]
            if edge_pixels_high != 0 and i == num_blocks_high - 1:
                px_to_edge = block_height - edge_pixels_high
                addendum = np.zeros((px_to_edge, np.shape(block)[1]))
                block = np.vstack((block, addendum))
            if edge_pixels_wide != 0 and j == num_blocks_wide - 1:
                px_to_edge = block_width - edge_pixels_wide
                addendum = np.zeros((np.shape(block)[0], px_to_edge))
                block = np.hstack((block, addendum))
            block_data = np.dstack((block_data, block))
    block_data = block_data[:,:,1:]
    return block_data, num_blocks_high, num_blocks_wide
    
def reconstruct_image(data, img_height, img_width, num_blocks_high,
                      num_blocks_wide):
    block_height, block_width = np.shape(data)[:2]
    new_img = np.empty((num_blocks_high * block_height,
                        num_blocks_wide * block_width),
                        dtype = np.float32)
    k = 0
    for i in range(num_blocks_high):
        for j in range(num_blocks_wide):
            new_img[(block_height * i):(block_height * (i + 1)),
                    (block_width * j):(block_width * (j + 1))
                   ] = data[:,:,k]
            k += 1
    return new_img[:img_height, :img_width]
                    
    
def dct_blocks(block_data):
    dct_data = np.zeros(np.shape(block_data)[:2])
    for i in range(np.shape(block_data)[2]):
        dct_block = cv2.dct(block_data[:,:,i])
        dct_data = np.dstack((dct_data, dct_block))
    dct_data = dct_data[:,:,1:]
    return dct_data
    
def quantize(dct_data, quality):
    quality = np.float64(quality)
    Q = np.empty(np.shape(dct_data)[:2], dtype = np.float64)
    Q[:,:] = float(VBG_MAX_QUALITY + 1 - quality)
    quant_data = np.zeros(np.shape(dct_data)[:2], dtype = np.float64)
    for i in range(np.shape(dct_data)[2]):
        current_data = dct_data[:,:,i] / Q
        quant_data = np.dstack((quant_data, current_data))
    quant_data = quant_data[:,:,1:]
    data_max = np.max(np.abs(dct_data))
    if data_max >= VBG_MAX_DATA_VALUE + 1:
        quality_modifier = data_max / np.float64(VBG_MAX_DATA_VALUE)
    else: quality_modifier = 1.0
    quant_data /= quality_modifier
    if (np.max(quant_data) >= VBG_MAX_DATA_VALUE + 1 or
       np.min(quant_data) <= VBG_MIN_DATA_VALUE - 1):
        print "ERROR: The attempt to reduce the range of values to conform to VBG standards failed. Some values exceeded" + str(VBG_MAX_DATA_VALUE) + " or " + str(VBG_MIN_DATA_VALUE) + ". Please file a bug report.",
        print "\nERR_VBG_DATA_OUT_OF_BOUNDS"
        print np.min(quant_data)
        print np.max(quant_data)
        sys.exit(VBG_ERR_DATA_OUT_OF_BOUNDS)
    else: quant_data = np.int8(quant_data)
    return quant_data, quality_modifier

def inverse_quantize(data, quality, quality_modifier):
    quality = np.float64(quality)
    quality_modifier = np.float64(quality_modifier)
    Q = np.empty(np.shape(data)[:2], dtype = np.float64)
    Q[:,:] = float(VBG_MAX_QUALITY + 1 - quality)
    Q[:,:] *= quality_modifier
    px_data = np.zeros(np.shape(data)[:2], dtype = np.float64)
    for i in range(np.shape(data)[2]):
        current_block = data[:,:,i] * Q
        ready_block = cv2.dct(current_block, flags=cv2.DCT_INVERSE)
        px_data = np.dstack((px_data, ready_block))
    px_data = px_data[:,:,1:]
    return px_data
    
def VBG_encode(quant_data, quality, quality_modifier, num_blocks_high,
               num_blocks_wide):
    vbg_data = np.empty(0, dtype = np.int8)
    diff_data = np.zeros(np.shape(quant_data), dtype = np.int8)
    for i in range(np.shape(quant_data)[2]):
        if i != 0: 
            b = quant_data[:,:,i] 
            a = quant_data[:,:,i-1]
            diff_data[:,:,i] = b - a
    diff_data[:,:,0] = quant_data[:,:,0]
    for i in range(np.shape(diff_data)[2]):
        zig_zag_line = zig_zag_scan(diff_data[:,:,i])
        vbg_data = np.append(vbg_data, zig_zag_line)
    vbg_data = run_length_encode(vbg_data, 8)
    vbg_data = np.array(vbg_data)
    vbg_data = vbg_data.ravel()
    vbg_meta = [np.shape(quant_data)[0], np.shape(quant_data)[1],
                num_blocks_high, num_blocks_wide, quality,
                quality_modifier, len(vbg_data)]
    return np.array(vbg_data), np.array(vbg_meta)

def zig_zag_scan(block):
    h,w = np.shape(block)[:2]
    zig_zag_data = np.zeros(h*w, dtype = np.int8)
    row = 0
    col = 0
    index = 0
    zig_zag_data[0] = block[row, col]
    while row < h and col < w:
        if row == 0 and (row + col) % 2 == 0 and col != w - 1:
            zig_zag_data[index] = block[row, col]
            col += 1
            index += 1
        elif row == h - 1 and (row + col) % 2 != 0 and col != w - 1:
            zig_zag_data[index] = block[row, col]
            col += 1
            index += 1
        elif col == 0 and (row + col) % 2 != 0 and row != h - 1:
            zig_zag_data[index] = block[row, col]
            row += 1
            index += 1
        elif col == w - 1 and (row + col) % 2 == 0 and row != h - 1:
            zig_zag_data[index] = block[row, col]
            row += 1
            index += 1
        elif col != 0 and row != w - 1 and (row+col) % 2 != 0:
            zig_zag_data[index] = block[row, col]
            row += 1
            col -= 1
            index += 1
        elif row != 0 and col != w - 1 and (col + row) % 2 == 0:
            zig_zag_data[index] = block[row, col]
            row -= 1
            col += 1
            index += 1
        elif row == w - 1 and col == h - 1:
            zig_zag_data[-1] = block[-1, -1]
            break
    return zig_zag_data

def inv_zig_zag(line, h, w):
    depth = len(line)
    block = np.zeros((h, w), np.float32)
    row = 0
    col = 0
    index = 0
    block[0,0] = line[0]
    while row < h and col < w:
        if row == 0 and (row + col) % 2 == 0 and col != w - 1:
            block[row, col] = line[index]
            col += 1
            index += 1
        elif row == h - 1 and (row + col) % 2 != 0 and col != w - 1:
            block[row, col] = line[index]
            col += 1
            index += 1
        elif col == 0 and (row + col) % 2 != 0 and row != h - 1:
            block[row, col] = line[index]
            row += 1
            index += 1
        elif col == w - 1 and (row + col) % 2 == 0 and row != h - 1:
            block[row, col] = line[index]
            row += 1
            index += 1
        elif col != 0 and row != w - 1 and (row+col) % 2 != 0:
            block[row, col] = line[index]
            row += 1
            col -= 1
            index += 1
        elif row != 0 and col != w - 1 and (col + row) % 2 == 0:
            block[row, col] = line[index]
            row -= 1
            col += 1
            index += 1
        elif index == depth - 1:
            block[-1, -1] = line[-1]
            break
    return block

def run_length_encode(array, run_storage_cap):
    data = np.array(array)
    data = data.ravel()
    source = []
    current = data[0]
    count = 0
    for d in data:
        if d == current:
            count += 1
        else:
            source.append(current)
            source.append(count)
            current = d
            count = 1
    source.append(d)
    source.append(count)
    encoded = []
    i = 0
    for s in source:
        if i % 2 == 0:
            encoded.append(s)
        elif i % 2 == 1:
            reference = source[i-1]
            while s > ((2**run_storage_cap) - 1):
                encoded.append(((2**run_storage_cap) - 1))
                s -= ((2**run_storage_cap) - 1)
                if s != 0:
                    encoded.append(reference)
            encoded.append(s)
        i += 1
    return encoded

def write_data(filename, vbg_data, vbg_meta, img_height, img_width):
    file_meta = vbg_meta.ravel()
    file_data = vbg_data.ravel()
    f = open(filename + ".vbg", "wb")
    f.write(struct.pack("sss", "V", "B", "G"))
    f.write(struct.pack("HH", np.uint16(img_height), np.uint16(img_width)))
    f.write(struct.pack("HH", file_meta[0], file_meta[1]))
    f.write(struct.pack("HH", file_meta[2], file_meta[3]))
    f.write(struct.pack("ff", file_meta[4], file_meta[5]))
    f.write(struct.pack("I", file_meta[6]))
    j = 0
    for i in file_data:
        if j % 2 == 1: f.write(struct.pack("B", i))
        else: f.write(struct.pack("b", i))
        j += 1
    f.close()
    
def report_stats(filename, img, time_to_compress):
    start_time = time.time()
    f = open(filename + ".vbg", "rb")
    check_string = f.read(3)
    try:
        if check_string == "VBG":
            pass
        else:
            print "ERROR: File is of incorrect type or does not exist."
            sys.exit(VBG_ERR_INVALID_FILE)
    except:
        print "ERROR: File is of incorrect type or does not exist."
        sys.exit(VBG_ERR_INVALID_FILE)
    
    img_height = struct.unpack("H", f.read(2))[0]
    img_width = struct.unpack("H", f.read(2))[0]
    block_height = struct.unpack("H", f.read(2))[0]
    block_width = struct.unpack("H", f.read(2))[0]
    num_blocks_high = struct.unpack("H", f.read(2))[0]
    num_blocks_wide = struct.unpack("H", f.read(2))[0]
    quality = struct.unpack("f", f.read(4))[0]
    quality_modifier = struct.unpack("f", f.read(4))[0]
    data_length = struct.unpack("I", f.read(4))[0]

    data = []
    for i in range(data_length):
        if i % 2 == 1: b = struct.unpack("B", f.read(1))[0]
        else: b = struct.unpack("b", f.read(1))[0]
        data.append(b)
    f.close()

    non_rle_data = []
    for i in range(data_length):
        if i % 2 == 0:
            value = data[i]
        else:
            run = data[i]
            for j in range(run):
                non_rle_data.append(value)

    block_depth = len(non_rle_data) / (block_height * block_width)
    non_zig_zag_data = np.zeros((block_height, block_width), np.float32)
    for i in range(block_depth):
        line = non_rle_data[block_height*block_width*i:
                            block_height*block_width*(i + 1)]
        block = inv_zig_zag(line, block_height, block_width)
        non_zig_zag_data = np.dstack((non_zig_zag_data, block))
    diff_data = non_zig_zag_data[:,:,1:]
    quant_data = diff_data
    for i in range(np.shape(diff_data)[2]):
        if i != 0: 
            a = diff_data[:,:,i]
            b = diff_data[:,:,(i - 1)]
            quant_data[:,:,i] = a + b
    vbg_data = inverse_quantize(quant_data, quality, quality_modifier)
    vbg_img = reconstruct_image(vbg_data, img_height, img_width,
                                num_blocks_high, num_blocks_wide)
    vbg_img = np.float32(vbg_img)
    vbg_img -= np.min(vbg_img)
    vbg_img /= np.max(vbg_img)
    vbg_img *= 255.999
    vbg_img = np.uint8(vbg_img)
    data_list = []
    # Report
    data_list.append(str(block_height) + ",")
    data_list.append(str(block_width) + ",")
    data_list.append(str(quality) + ",")
    data_list.append(str(os.stat(filename + ".vbg").st_size) + ",")
    data_list.append(str(cv2.matchTemplate(vbg_img, img, method=1)[0,0]) + ",")
    data_list.append(str(time_to_compress) + ",")
    data_list.append(str(time.time() - start_time) + "\n")
    cv2.imwrite("output/" + filename + "_" + str(block_height) + "_" +
                str(block_width) + ".png", vbg_img)
    return data_list

counter = 0
csv_file = open('trials' + str(VBG_VERSION) + '.csv', 'wb')
for i in range(64,66,2):
    print "\nBlock Size = " + str(i) + "x" + str(i)
    print "Quality = " + str(1000)
    data_list = VBG("test_pattern", i, i, 100)
    np.insert(data_list, 0, counter)
    for k in range(len(data_list)):
        csv_file.write(str(data_list[k]))
    counter += 1
csv_file.close()
