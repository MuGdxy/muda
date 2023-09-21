HEADER = '''#pragma once
#include "base.h"
#ifndef __INTELLISENSE__
#include <cub/device/$FILE_NAME$.cuh>
#endif

namespace muda
{
//ref: 

class $CLASS_NAME$ : public CubWrapper<$CLASS_NAME$>
{
  public:
    $CLASS_NAME$(cudaStream_t stream = nullptr)
        : CubWrapper(stream)
    {
    }

    template <typename InputIteratorT, typename OutputIteratorT>
    $CLASS_NAME$& ExampleFunction(device_buffer<std::byte>& external_buffer,
                      InputIteratorT            d_in,
                      OutputIteratorT           d_out,
                      int                       num_items)
    {
        MUDA_CUB_WRAPPER_IMPL(cub::$CLASS_NAME$::ExampleFunction(
            d_temp_storage, temp_storage_bytes, 
            d_in, d_out, num_items, m_stream, false));
    }
    
/*
    BEGIN>>>
	END<<<
*/
    // Translate all the above functions(between BEGIN>>> and END<<<) to the style of ExampleFunction, don't stop until you finish all of them.
};
}  // namespace muda
'''

INCL = '''#pragma once
#ifdef __INTELLISENSE__
#include "../$FILE_NAME$.h"
#endif
#include <cub/device/$FILE_NAME$.cuh>
namespace muda
{
//$CLASS_NAME$& ExampleFunction(device_buffer<std::byte>& external_buffer, ...);
//{
//
//    size_t temp_storage_bytes = 0;
//
//    checkCudaErrors(
//        // cub::$CLASS_NAME$::ExampleFunction(
//        nullptr, 
//        // )
//        );
//
//    prepareBuffer(external_buffer, temp_storage_bytes);
//    auto d_temp_storage = external_buffer.data();
//
//    checkCudaErrors(
//        // cub::$CLASS_NAME$::ExampleFunction(
//        d_temp_storage, 
//        // )
//        );
//
//    return *this;
//
//}
}  // namespace muda
'''

import sys
import os

args = sys.argv[1:]
if(len(args) == 0):
    print('usage: mk.py [class name] [class name] ...')
    
cub_device = open("../cub_device.h")
cub_device_content = cub_device.read()
cub_device.close()

for arg in args:
    print('-' * 20)
    print(arg)
    first = True
    # seperate the name by uppercase
    # eg. AaBb -> Aa_Bb
    file_name = ''
    for c in arg:
        if c.isupper() and not first:
            file_name += '_'
        file_name += c
        first = False
    # to lower case
    file_name = file_name.lower()
    print(f'file name: {file_name}')
    class_name = arg
    print(f'class name: {class_name}')
    # substitute the '$FILE_NAME$' and '$CLASS_NAME$' by file_name and class_name
    header = HEADER.replace('$FILE_NAME$', file_name)
    header = header.replace('$CLASS_NAME$', class_name)
    # incl = INCL.replace('$FILE_NAME$', file_name)
    # incl = incl.replace('$CLASS_NAME$', class_name)
    filename = file_name + ".h"
    if(cub_device_content.find(f'#include "device/{filename}"\n') == -1):
        cub_device_content+=f'#include "device/{filename}"\n'
    
    header_path = f'{file_name}.h'
    if not os.path.exists(header_path):
        f = open(header_path, 'w')
        f.write(header)
        f.close()
    else:
        print(f"{header_path} already exists!")
    
    # incl_path = f'impl/{file_name}.inl'
    # if not os.path.exists(incl_path):
    #     f = open(incl_path, 'w')
    #     f.write(incl)
    #     f.close()
    # else:
    #     print(f"{incl_path} already exists!")

print(cub_device_content)
cub_device = open("../cub_device.h","w")
cub_device.write(cub_device_content)
cub_device.close()
