header = '''#pragma once
#include "base.h"

namespace muda
{
class $CLASS_NAME$ : public AlgBase<$CLASS_NAME$>
{
  public:
    $CLASS_NAME$(cudaStream_t stream = nullptr)
        : AlgBase(stream)
    {
    }

    //$CLASS_NAME$& ExampleFunction(device_buffer<std::byte>& external_buffer, ...);
};
}  // namespace muda

#ifndef __INTELLISENSE__
#include "$FILE_NAME$.inl"
#endif

'''

incl = '''#pragma once
//
#ifdef __INTELLISENSE__
#include "$FILE_NAME$.h"
#endif
namespace muda
{
    //$CLASS_NAME$& ExampleFunction(device_buffer<std::byte>& external_buffer, ...);
    //{
        //prepareBuffer(external_buffer, temp_storage_bytes);
    //}
}  // namespace muda
'''

import sys
import os

args = sys.argv[1:]

for arg in args:
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
    header = header.replace('$FILE_NAME$', file_name)
    header = header.replace('$CLASS_NAME$', class_name)
    incl = incl.replace('$FILE_NAME$', file_name)
    
    filename = file_name + ".h"
    if not os.path.exists(file_name):
        f = open(f'{file_name}.h', 'w')
        f.write(header)
        f.close()
    else:
        print(f"{filename} already exists!")
    
    filename = file_name + ".inl"
    if not os.path.exists(file_name):
        f = open(f'{file_name}.inl', 'w')
        f.write(incl)
        f.close()
    else:
        print(f"{filename} already exists!")
