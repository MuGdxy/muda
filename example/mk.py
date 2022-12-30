import sys

content = '''
#include <catch2/catch.hpp>
#include <muda/muda.h>
#include "../example_common.h"
using namespace muda;

void @NAME@()
{
    example_desc("give a description to @NAME@");
}

CASE("@NAME@", "[@TAG@]")
{
    @NAME@();
}
'''
tag = "default"
name = "example_name"
if(len(sys.argv) > 1):
    name = sys.argv[1]
if(len(sys.argv) > 2):
    tag = sys.argv[2]
content = content.replace("@NAME@", name)
content = content.replace("@TAG@", tag)

f = open(name + ".cu", "w")
f.write(content)
f.close()