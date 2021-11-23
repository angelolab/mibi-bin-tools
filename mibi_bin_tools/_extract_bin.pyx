from cython.view cimport array as cvarray
from cython cimport cdivision, boundscheck, wraparound

from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy, memset
from libc.stdio cimport fopen, fclose, FILE, EOF, fseek, SEEK_END, SEEK_SET, SEEK_CUR, fread, ftell, clearerr
from posix.unistd cimport usleep

import numpy as np
cimport numpy as np

ctypedef np.uint64_t MAXINDEX_t
ctypedef np.uint32_t INT_t
ctypedef np.uint16_t DTYPE_t
ctypedef np.uint8_t SMALL_t

@boundscheck(False) # Deactivate bounds checking
@wraparound(False)  # Deactivate negative indexing
@cdivision(True) # Ignore modulo/divide by zero warning
cdef inline int _minimum_larger_value_in_sorted(const DTYPE_t[:] low_range, DTYPE_t val):
    """ minimal bianry search impl
    """
    cdef int start, end, ans, mid

    start = 0
    end = low_range.shape[0] - 1
    ans = -1

    while start <= end:
        mid = (start + end) // 2

        if low_range[mid] < val:
            start = mid + 1

        else:
            ans = mid
            end = mid - 1

    return ans


@boundscheck(False) # Deactivate bounds checking
@wraparound(False)  # Deactivate negative indexing
@cdivision(True) # Ignore modulo/divide by zero warning
cdef inline void _fread_downloading(char *file_buffer, size_t num_bytes, MAXINDEX_t buffer_size, 
                                    FILE* fp, int timeout, MAXINDEX_t *curr_file_size,
                                    MAXINDEX_t filesize):
    """ moddification of fread to await file download completion
    """

    cdef MAXINDEX_t offset
    offset = ftell(fp)
    if buffer_size <= curr_file_size[0] - offset or curr_file_size[0] == filesize:
        fread(file_buffer, num_bytes, buffer_size, fp)
        return
    cdef int timer = 0
    while True:
        fseek(fp, 0, SEEK_END)
        curr_file_size[0] = ftell(fp)
        clearerr(fp)
        if buffer_size <= curr_file_size[0] - offset:
            break
        elif timer > timeout and timeout >= 0:
            raise FileNotFoundError('Download timeout was exceeded...')
        timer += 1
        usleep(1)
    fseek(fp, offset, SEEK_SET)
    fread(file_buffer, num_bytes, buffer_size, fp)


@boundscheck(False) # Deactivate bounds checking
@wraparound(False)  # Deactivate negative indexing
@cdivision(True) # Ignore modulo/divide by zero warning
cdef inline void _check_buffer_refill(FILE* fp, char* file_buffer, MAXINDEX_t *buffer_idx,
                                      MAXINDEX_t read_size, MAXINDEX_t BUFFER_SIZE, int timeout, 
                                      MAXINDEX_t *curr_file_size, MAXINDEX_t filesize):
    """ Makes sure requested data is loaded into buffer
    
    Otherwise, the buffer is refilled and the offest pointer (buffer_idx) is updated.

    Args:
        fp (FILE*):
            File pointer
        file_buffer (char *):
            File buffer array
        buffer_idx (uint64_t *):
            Current index in file buffer
        read_size (uint64_t):
            Requested size of buffer read
        BUFFER_SIZE (uint64_t):
            Size of file buffer array
        timeout (int):
            How long to wait for file download (ms)
        curr_file_size (uint64_t *t):
            Current size of the downloading file
        filesize (uint64_t):
            Expected/final size of the file
    """
    # note, `buffer_idx[0]` is a cython altrenative to `*buffer_idx = 0`
    cdef MAXINDEX_t i
    if buffer_idx[0] + read_size >= BUFFER_SIZE:
        for i in range(read_size + 1):
            if buffer_idx[0] + i >= BUFFER_SIZE:
                break
            file_buffer[i] = file_buffer[buffer_idx[0] + i]

        # this might be faster
        #memcpy(file_buffer, file_buffer + buffer_idx[0], sizeof(char) * i)

        _fread_downloading(file_buffer + i, sizeof(char), BUFFER_SIZE - i, fp, timeout,
                           curr_file_size, filesize)

        buffer_idx[0] = 0
    return


@boundscheck(False) # Deactivate bounds checking
@wraparound(False)  # Deactivate negative indexing
@cdivision(True) # Ignore modulo/divide by zero warning
cdef INT_t[:, :, :, :] _extract_bin(const char* filename, MAXINDEX_t filesize,
                                    const DTYPE_t[:] low_range, const DTYPE_t[:] high_range,
                                    const SMALL_t[:] calc_intensity, int timeout):
    """ Extracts bin file to single channel tifs

    Args:
        filename (const char*):
            Name of bin file to extract
        filesize (uint64_t):
            Expected size of bin file
        low_range (uint16_t[]):
            Starting integration ranges for each tif
        high_range (uint16_t[]):
            Stoping integration ranges for each tif
        calc_intensity (bool):
            Calculate intensity and intensity*width images.  Not implemented.
        timeout (int):
            How long to wait for file download (ms)
    """
    cdef DTYPE_t num_x, num_y, num_trig, num_frames, desc_len, trig, num_pulses, pulse, time
    cdef DTYPE_t intensity
    cdef SMALL_t width
    cdef MAXINDEX_t data_start, pix, curr_file_size
    cdef int idx

    # 10MB buffer
    cdef MAXINDEX_t BUFFER_SIZE = 10 * 1024 * 1024
    cdef char* file_buffer = <char*> malloc(BUFFER_SIZE * sizeof(char))
    cdef MAXINDEX_t buffer_idx = 0

    # open file
    cdef FILE* fp
    cdef int timer = 0
    while True:
        fp = fopen(filename, "rb")
        if fp != NULL:
            timer = 0
            break
        elif timer > timeout and timeout >= 0:
            raise FileNotFoundError('Timer exceeded timeout of ' + str(timeout) +' seconds')
        timer += 1
        usleep(1)

    fseek(fp, 0, SEEK_END)
    curr_file_size = ftell(fp)
    fseek(fp, 0, SEEK_SET)

    # note, if cython has packed structs, this would be easier
    # or even macros tbh
    fseek(fp, 0x6, SEEK_SET)
    fread(&num_x, sizeof(DTYPE_t), 1, fp)
    fread(&num_y, sizeof(DTYPE_t), 1, fp)
    fread(&num_trig, sizeof(DTYPE_t), 1, fp)
    fread(&num_frames, sizeof(DTYPE_t), 1, fp)
    fseek(fp, 0x2, SEEK_CUR)
    fread(&desc_len, sizeof(DTYPE_t), 1, fp)

    data_start = \
        <MAXINDEX_t>(num_x) * <MAXINDEX_t>(num_y) * <MAXINDEX_t>(num_frames) * 8 + desc_len + 0x12

    img_data = \
        cvarray(
            shape=(3, <MAXINDEX_t>num_x * <MAXINDEX_t>num_y, low_range.shape[0]),
            itemsize=sizeof(INT_t),
            format='I'
        )
    cdef INT_t[:, :, :] img_data_view = img_data

    fseek(fp, data_start, SEEK_SET)
    fread(file_buffer, sizeof(char), BUFFER_SIZE, fp)
    for pix in range(<MAXINDEX_t>(num_x) * <MAXINDEX_t>(num_y)):
        #if pix % num_x == 0:
        #    print('\rpix done: ' + str(100 * pix / num_x / num_y) + '%...', end='')
        for trig in range(num_trig):
            _check_buffer_refill(fp, file_buffer, &buffer_idx, 0x8 * sizeof(char), BUFFER_SIZE,
                                 timeout, &curr_file_size, filesize)
            memcpy(&num_pulses, file_buffer + buffer_idx + 0x6, sizeof(time))
            buffer_idx += 0x8
            for pulse in range(num_pulses):
                _check_buffer_refill(fp, file_buffer, &buffer_idx, 0x5 * sizeof(char), BUFFER_SIZE,
                                     timeout, &curr_file_size, filesize)
                memcpy(&time, file_buffer + buffer_idx, sizeof(time))
                memcpy(&width, file_buffer + buffer_idx + 0x2, sizeof(width))
                memcpy(&intensity, file_buffer + buffer_idx + 0x3, sizeof(intensity))
                buffer_idx += 0x5
                idx = _minimum_larger_value_in_sorted(low_range, time)
                if idx > 0:
                    if time <= high_range[idx - 1]:
                        img_data_view[0, pix, idx - 1] += 1
                        #if calc_intensity[idx - 1]:
                        img_data_view[1, pix, idx - 1] += intensity
                        img_data_view[2, pix, idx - 1] += intensity * width
                elif idx == -1:
                    if time <= high_range[low_range.shape[0] - 1]:
                        img_data_view[0, pix, low_range.shape[0] - 1] += 1
                        #if calc_intensity[low_range.shape[0] - 1]:
                        img_data_view[1, pix, low_range.shape[0] - 1] += intensity
                        img_data_view[2, pix, low_range.shape[0] - 1] += intensity * width
    fclose(fp)
    free(file_buffer)

    return np.asarray(img_data).reshape((3, num_x, num_y, low_range.shape[0]))

@boundscheck(False) # Deactivate bounds checking
@wraparound(False)  # Deactivate negative indexing
@cdivision(True) # Ignore modulo/divide by zero warning
cdef _extract_no_sum(const char* filename, MAXINDEX_t filesize, DTYPE_t low_range,
                                    DTYPE_t high_range, int timeout=100):
    """ Creates histogram of observed peak widths within specified integration range

    Args:
        filename (const char*):
            Name of bin file to extract.
        filesize (uint64_t):
            Expected size of the bin file
        low_range (uint16_t):
            Low time range for integration
        high_range (uint16_t):
            High time range for integration
        timeout (int):
            How long to wait for downloading file (ms)
    """
    cdef DTYPE_t num_x, num_y, num_trig, num_frames, desc_len, trig, num_pulses, pulse, time
    cdef DTYPE_t intensity
    cdef SMALL_t width
    cdef MAXINDEX_t data_start, pix, curr_file_size
    cdef int idx

    # 10MB buffer
    cdef MAXINDEX_t BUFFER_SIZE = 10 * 1024 * 1024
    cdef char* file_buffer = <char*> malloc(BUFFER_SIZE * sizeof(char))
    cdef MAXINDEX_t buffer_idx = 0

    # open file
    cdef FILE* fp
    cdef int timer = 0
    while True:
        fp = fopen(filename, "rb")
        if fp != NULL:
            timer = 0
            break
        elif timer > timeout:
            raise FileNotFoundError('Timer exceeded timeout of ' + str(timeout) +' seconds')
        timer += 1
        usleep(1)

    fseek(fp, 0, SEEK_END)
    curr_file_size = ftell(fp)
    fseek(fp, 0, SEEK_SET)

    # note, if cython has packed structs, this would be easier
    # or even macros tbh
    fseek(fp, 0x6, SEEK_SET)
    fread(&num_x, sizeof(DTYPE_t), 1, fp)
    fread(&num_y, sizeof(DTYPE_t), 1, fp)
    fread(&num_trig, sizeof(DTYPE_t), 1, fp)
    fread(&num_frames, sizeof(DTYPE_t), 1, fp)
    fseek(fp, 0x2, SEEK_CUR)
    fread(&desc_len, sizeof(DTYPE_t), 1, fp)

    data_start = \
        <MAXINDEX_t>(num_x) * <MAXINDEX_t>(num_y) * <MAXINDEX_t>(num_frames) * 8 + desc_len + 0x12

    cdef MAXINDEX_t widths[256]
    memset(widths, 0, 256*sizeof(MAXINDEX_t))

    fseek(fp, data_start, SEEK_SET)
    fread(file_buffer, sizeof(char), BUFFER_SIZE, fp)
    for pix in range(<MAXINDEX_t>(num_x) * <MAXINDEX_t>(num_y)):
        #if pix % num_x == 0:
        #    print('\rpix done: ' + str(100 * pix / num_x / num_y) + '%...', end='')
        for trig in range(num_trig):
            _check_buffer_refill(fp, file_buffer, &buffer_idx, 0x8 * sizeof(char), BUFFER_SIZE,
                                 timeout, &curr_file_size, filesize)
            memcpy(&num_pulses, file_buffer + buffer_idx + 0x6, sizeof(time))
            buffer_idx += 0x8
            for pulse in range(num_pulses):
                _check_buffer_refill(fp, file_buffer, &buffer_idx, 0x5 * sizeof(char), BUFFER_SIZE,
                                     timeout, &curr_file_size, filesize)
                memcpy(&time, file_buffer + buffer_idx, sizeof(time))
                memcpy(&width, file_buffer + buffer_idx + 0x2, sizeof(width))
                memcpy(&intensity, file_buffer + buffer_idx + 0x3, sizeof(intensity))
                buffer_idx += 0x5
                if time <= high_range and time >= low_range:
                    widths[width] += 1

    fclose(fp)
    free(file_buffer)

    return np.copy(widths, order='C')

@boundscheck(False) # Deactivate bounds checking
@wraparound(False)  # Deactivate negative indexing
@cdivision(True) # Ignore modulo/divide by zero warning
cdef INT_t[:] _extract_spectra(const char* filename, MAXINDEX_t filesize,
                                    const DTYPE_t[:] low_range, const DTYPE_t[:] high_range,
                                    const SMALL_t[:] calc_intensity, int timeout):
    
    cdef DTYPE_t num_x, num_y, num_trig, num_frames, desc_len, trig, num_pulses, pulse, time
    cdef DTYPE_t intensity
    cdef SMALL_t width
    cdef MAXINDEX_t data_start, pix, curr_file_size
    cdef int idx
    cdef int n = 0
    #cdef int capacity = 5000
    
    # 10MB buffer
    cdef MAXINDEX_t BUFFER_SIZE = 10 * 1024 * 1024
    cdef char* file_buffer = <char*> malloc(BUFFER_SIZE * sizeof(char))
    cdef MAXINDEX_t buffer_idx = 0

    # open file
    cdef FILE* fp
    cdef int timer = 0
    while True:
        fp = fopen(filename, "rb")
        if fp != NULL:
            timer = 0
            break
        elif timer > timeout and timeout >= 0:
            raise FileNotFoundError('Timer exceeded timeout of ' + str(timeout) +' seconds')
        timer += 1
        usleep(1)

    fseek(fp, 0, SEEK_END)
    curr_file_size = ftell(fp)
    fseek(fp, 0, SEEK_SET)

    # note, if cython has packed structs, this would be easier
    # or even macros tbh
    fseek(fp, 0x6, SEEK_SET)
    fread(&num_x, sizeof(DTYPE_t), 1, fp)
    fread(&num_y, sizeof(DTYPE_t), 1, fp)
    fread(&num_trig, sizeof(DTYPE_t), 1, fp)
    fread(&num_frames, sizeof(DTYPE_t), 1, fp)
    fseek(fp, 0x2, SEEK_CUR)
    fread(&desc_len, sizeof(DTYPE_t), 1, fp)

    data_start = \
        <MAXINDEX_t>(num_x) * <MAXINDEX_t>(num_y) * <MAXINDEX_t>(num_frames) * 8 + desc_len + 0x12
    
    cdef int capacity =   num_x * num_y * 250 * 350
    arr = \
        cvarray(
            shape=((capacity,)),
            itemsize=sizeof(INT_t),
            format='I'
        )    
    cdef INT_t[:] arr_time = arr
    cdef INT_t[:] arr_amp = arr
    
    fseek(fp, data_start, SEEK_SET)
    fread(file_buffer, sizeof(char), BUFFER_SIZE, fp)
    for pix in range(<MAXINDEX_t>(num_x) * <MAXINDEX_t>(num_y)):
        #if pix % num_x == 0:
        #    print('\rpix done: ' + str(100 * pix / num_x / num_y) + '%...', end='')
        for trig in range(num_trig):
            _check_buffer_refill(fp, file_buffer, &buffer_idx, 0x8 * sizeof(char), BUFFER_SIZE,
                                 timeout, &curr_file_size, filesize)
            memcpy(&num_pulses, file_buffer + buffer_idx + 0x6, sizeof(time))
            buffer_idx += 0x8
            for pulse in range(num_pulses):
                _check_buffer_refill(fp, file_buffer, &buffer_idx, 0x5 * sizeof(char), BUFFER_SIZE,
                                     timeout, &curr_file_size, filesize)
                memcpy(&time, file_buffer + buffer_idx, sizeof(time))
                memcpy(&width, file_buffer + buffer_idx + 0x2, sizeof(width))
                memcpy(&intensity, file_buffer + buffer_idx + 0x3, sizeof(intensity))
                buffer_idx += 0x5
                
                arr_time[n] = time
                n += 1
                
        
    arr = \
        cvarray(
            shape=((n + 1,)),
            itemsize=sizeof(INT_t),
            format='I'
        )    
    cdef INT_t[:] arr_final = arr
    arr_final = arr_time[0 : n + 1]

                        
    fclose(fp)
    free(file_buffer)
    
    #return np.vstack((np.asarray(arr_time),np.asarray(arr_amp)))
    return np.asarray(arr_final)

def c_extract_bin(char* filename, MAXINDEX_t filesize, DTYPE_t[:] low_range,
                  DTYPE_t[:] high_range, SMALL_t[:] calc_intensity, int timeout=100):
    return np.asarray(
        _extract_bin(filename, filesize, low_range, high_range, calc_intensity, timeout)
    )


def c_extract_no_sum(char* filename, MAXINDEX_t filesize, DTYPE_t low_range,
                     DTYPE_t high_range, int timeout=100):
    return np.asarray(
        _extract_no_sum(filename, filesize, low_range, high_range, timeout)
    )

def c_extract_spectra(char* filename, MAXINDEX_t filesize, DTYPE_t[:] low_range,
                  DTYPE_t[:] high_range, SMALL_t[:] calc_intensity, int timeout=100):
    return np.asarray(
        _extract_spectra(filename, filesize, low_range, high_range, calc_intensity, timeout)
    )